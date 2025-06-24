import torch
import onnx
import torch.onnx
import sys
import os
import time
import traceback
import comfy.model_management
import comfy.sd
import folder_paths
from tqdm import tqdm
import comfy
from typing import Any, Optional
import numpy as np
import traceback
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


class ClipEmbedWrapper(torch.nn.Module):
    """Wrapper model that handles layer selection and intermediate outputs properly for CLIP conversion"""
    
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        
    def forward(self, embeds):
        # Create proper attention mask for CLIP sequence
        batch_size, seq_len = embeds.shape[:2]
        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=embeds.device)
        
        # For our sequence: [start_token, end_token, pad, pad, ..., pad]
        # Attention mask: [1, 1, 0, 0, ..., 0]
        attention_mask[:, 0] = 1   # start token
        attention_mask[:, 1] = 1   # end token (immediately after start)
        # Remaining positions remain 0 for padding
        
        # Calculate actual number of non-padded tokens per batch
        num_tokens = attention_mask.sum(dim=1).tolist()
        
        # DEBUG: Print debug info about the forward pass
        print(f"    🔍 ClipEmbedWrapper forward debug:")
        print(f"        embeds shape: {embeds.shape}, dtype: {embeds.dtype}")
        print(f"        attention_mask shape: {attention_mask.shape}, sum: {attention_mask.sum().item()}")
        print(f"        num_tokens: {num_tokens}")
        print(f"        layer: {self.clip_model.layer}")
        print(f"        layer_idx: {self.clip_model.layer_idx}")
        print(f"        enable_attention_masks: {self.clip_model.enable_attention_masks}")
        print(f"        layer_norm_hidden_state: {self.clip_model.layer_norm_hidden_state}")
        
        # Handle layer selection like the original forward method
        if self.clip_model.layer == "all":
            intermediate_output = "all"
        else:
            intermediate_output = self.clip_model.layer_idx
        
        # Use attention mask if the model supports it
        attention_mask_model = None
        if self.clip_model.enable_attention_masks:
            attention_mask_model = attention_mask
        
        print(f"        intermediate_output: {intermediate_output}")
        print(f"        attention_mask_model: {attention_mask_model}")
        
        # Create dummy token tensor that matches the embedding sequence
        # The transformer might need this even when we provide embeds
        batch_size, seq_len = embeds.shape[:2]
        device = embeds.device
        
        # Get the special tokens from the clip model
        special_tokens = self.clip_model.special_tokens
        start_token = special_tokens.get("start", 49406)
        end_token = special_tokens.get("end", 49407)
        pad_token = special_tokens.get("pad", 49407)
        
        # Create the token sequence that matches our embeddings
        dummy_tokens = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        for b in range(batch_size):
            dummy_tokens[b, 0] = start_token  # start token
            dummy_tokens[b, 1] = end_token   # end token  
            dummy_tokens[b, 2:] = pad_token  # pad tokens
        
        outputs = self.clip_model.transformer(
            dummy_tokens, 
            attention_mask_model,
            embeds=embeds, 
            num_tokens=num_tokens, 
            intermediate_output=intermediate_output,
            final_layer_norm_intermediate=self.clip_model.layer_norm_hidden_state,
            dtype=torch.float32
        )
        
        print(f"        outputs length: {len(outputs)}")
        for i, output in enumerate(outputs):
            if output is not None:
                print(f"        output[{i}] shape: {output.shape}, dtype: {output.dtype}")
                print(f"        output[{i}] stats: min={output.min().item():.6f}, max={output.max().item():.6f}, mean={output.mean().item():.6f}")
                nan_count = torch.isnan(output).sum().item()
                print(f"        output[{i}] NaN count: {nan_count}/{output.numel()}")
            else:
                print(f"        output[{i}]: None")
        
        # Handle layer output selection like the original forward method
        if self.clip_model.layer == "last":
            z = outputs[0].float()  # Final layer
        else:
            z = outputs[1].float()  # Intermediate layer (for hidden/all)
        
        # Zero out masked tokens if needed
        if self.clip_model.zero_out_masked:
            z *= attention_mask.unsqueeze(-1).float()

        # Handle pooled output
        pooled_output = None
        if len(outputs) >= 3:
            if not self.clip_model.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()
        
        if pooled_output is None:
            pooled_output = torch.zeros((z.shape[0], z.shape[-1]), dtype=z.dtype, device=z.device)
        
        return z, pooled_output 