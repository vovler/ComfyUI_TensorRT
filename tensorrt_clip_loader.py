import torch
import os
import logging

import comfy.model_management
import comfy.sd
import comfy.sd1_clip
import comfy.sdxl_clip
import folder_paths
from .utils.tensorrt_error_recorder import TrTErrorRecorder
from .utils.trt_datatype_to_torch import trt_datatype_to_torch
from .utils.tensorrt_error_recorder import check_for_trt_errors

if "tensorrt" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tensorrt"][0].append(
        os.path.join(folder_paths.models_dir, "tensorrt"))
    folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
else:
    folder_paths.folder_names_and_paths["tensorrt"] = (
        [os.path.join(folder_paths.models_dir, "tensorrt")], {".engine"})

import tensorrt as trt

trt.init_libnvinfer_plugins(None, "")

logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)
runtime.error_recorder = TrTErrorRecorder()

class TensorRTCompatibleEmbedding(torch.nn.Module):
    """Custom embedding that matches ComfyUI's CLIPEmbeddings interface"""
    def __init__(self, vocab_size, embed_dim, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        # Create a dummy weight for interface compatibility
        self.weight = torch.nn.Parameter(torch.zeros(vocab_size, embed_dim, device=device, dtype=dtype))
        
    def forward(self, input_tokens, out_dtype=torch.float32):
        # For TensorRT, we don't actually use this embedding for inference
        # This is just for interface compatibility
        batch_size, seq_len = input_tokens.shape
        return torch.zeros(batch_size, seq_len, self.embed_dim, device=input_tokens.device, dtype=out_dtype)

class TensorRTCLIPTextModel(torch.nn.Module):
    """
    TensorRT implementation that mimics comfy.clip_model.CLIPTextModel interface
    This can be plugged into existing SDClipModel without any changes
    """
    def __init__(self, engine_path, runtime, hidden_size=768):
        super().__init__()
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.runtime = runtime
        self.hidden_size = hidden_size
        self._size = int(os.stat(engine_path).st_size)
        
        # Create a dummy embedding layer for interface compatibility
        # This won't be used for inference, just for device/dtype detection
        self.dummy_embedding = TensorRTCompatibleEmbedding(49408, hidden_size)
        
    def load(self):
        if self.engine is not None or self.context is not None:
            return
        print(f"Loading TensorRT CLIP engine from: {self.engine_path}")
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            check_for_trt_errors(self.runtime)
            self.context = self.engine.create_execution_context()
            check_for_trt_errors(self.runtime)
        print(f"TensorRT CLIP engine loaded successfully")

    def get_input_embeddings(self):
        """Return dummy embedding for interface compatibility"""
        return self.dummy_embedding
    
    def set_input_embeddings(self, embeddings):
        """No-op for TensorRT"""
        pass
    
    @property
    def size(self) -> int:
        return self._size

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    def forward(self, input_tokens=None, attention_mask=None, embeds=None, num_tokens=None, 
                intermediate_output=None, final_layer_norm_intermediate=True, dtype=torch.float32):
        """
        TensorRT inference that matches CLIPTextModel interface exactly
        """
        self.load()  # Ensure engine is loaded
        
        # For TensorRT, we'll use the embeds if provided, otherwise convert input_tokens
        if embeds is not None:
            # Use provided embeddings (from process_tokens)
            tokens = self._embeds_to_tokens(embeds)
        elif input_tokens is not None:
            tokens = input_tokens
        else:
            raise ValueError("Either input_tokens or embeds must be provided")
        
        print(f"TensorRT CLIP forward - Input shape: {tokens.shape}, dtype: {tokens.dtype}")
        
        # Find the actual input tensor name from the engine
        input_tensor_name = None
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_tensor_name = tensor_name
                break
        
        if input_tensor_name is None:
            raise RuntimeError("No input tensor found in TensorRT CLIP engine")
            
        model_inputs = {input_tensor_name: tokens}
        batch_size = tokens.shape[0]
        
        # Handle batch splitting for dynamic profiles
        dims = self.engine.get_tensor_profile_shape(input_tensor_name, 0)
        min_batch = dims[0][0]
        opt_batch = dims[1][0]
        max_batch = dims[2][0]

        curr_split_batch = 1
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break

        self.set_bindings_shape(model_inputs, curr_split_batch)

        # Convert inputs to appropriate data types
        model_inputs_converted = {}
        for k in model_inputs:
            data_type = self.engine.get_tensor_dtype(k)
            torch_dtype = trt_datatype_to_torch(data_type)
            model_inputs_converted[k] = model_inputs[k].to(dtype=torch_dtype)

        # Get output tensor info - Find outputs dynamically
        output_tensors = []
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                output_tensors.append(tensor_name)
        
        if len(output_tensors) < 2:
            raise RuntimeError(f"Expected 2 output tensors for CLIP, found {len(output_tensors)}")
            
        hidden_states_name = output_tensors[0]  # First output is hidden states
        pooled_output_name = output_tensors[1]  # Second output is pooled output
        
        hidden_shape = list(self.engine.get_tensor_shape(hidden_states_name))
        pooled_shape = list(self.engine.get_tensor_shape(pooled_output_name))

        # Handle dynamic shapes
        for idx in range(len(hidden_shape)):
            if hidden_shape[idx] == -1:
                if idx == 0:  # batch
                    hidden_shape[idx] = tokens.shape[0]
                elif idx == 1:  # sequence
                    hidden_shape[idx] = tokens.shape[1]
                elif idx == 2:  # hidden_size
                    hidden_shape[idx] = self.hidden_size
                else:
                    hidden_shape[idx] = tokens.shape[idx]
            else:
                if idx == 0:
                    hidden_shape[idx] *= curr_split_batch

        for idx in range(len(pooled_shape)):
            if pooled_shape[idx] == -1:
                if idx == 0:  # batch
                    pooled_shape[idx] = tokens.shape[0]
                elif idx == 1:  # pooled dimension
                    pooled_shape[idx] = self.hidden_size
                else:
                    pooled_shape[idx] = tokens.shape[idx]
            else:
                if idx == 0:
                    pooled_shape[idx] *= curr_split_batch

        hidden_out = torch.empty(hidden_shape,
                               device=tokens.device,
                               dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(hidden_states_name)))
        pooled_out = torch.empty(pooled_shape,
                               device=tokens.device,
                               dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(pooled_output_name)))
        
        model_inputs_converted[hidden_states_name] = hidden_out
        model_inputs_converted[pooled_output_name] = pooled_out

        # Execute inference
        stream = torch.cuda.default_stream(tokens.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                tensor = model_inputs_converted[k]
                self.context.set_tensor_address(k, tensor[(tensor.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)

        print(f"TensorRT CLIP inference complete")
        
        # Convert to float32 for compatibility
        hidden_out = hidden_out.to(dtype=torch.float32)
        pooled_out = pooled_out.to(dtype=torch.float32)
        
        # Return in the same format as CLIPTextModel:
        # (last_hidden_state, intermediate_hidden_state, projected_pooled, unprojected_pooled)
        return hidden_out, None, pooled_out, pooled_out

    def _embeds_to_tokens(self, embeds):
        """Convert embeddings back to tokens for TensorRT (simplified)"""
        # This is a simplified approach - in practice you might want to 
        # store the original tokens or use a different approach
        batch_size, seq_len, _ = embeds.shape
        return torch.zeros((batch_size, seq_len), dtype=torch.long, device=embeds.device)

    def unload(self):
        engine_obj = self.engine
        self.engine = None
        if engine_obj is not None:
            del engine_obj
        context_obj = self.context
        self.context = None
        if context_obj is not None:
            del context_obj


class TensorRTCLIPLoader:
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        engine_files = folder_paths.get_filename_list("tensorrt")
        # Filter for CLIP engines
        clip_l_files = ["None"] + [f for f in engine_files if "clip_l" in f.lower()]
        clip_g_files = ["None"] + [f for f in engine_files if "clip_g" in f.lower()]
        
        return {
            "required": {
                "clip_l_name": (clip_l_files,),
                "clip_g_name": (clip_g_files,),
            },
        }

    def load_clip(self, clip_l_name, clip_g_name):
        try:
            clip_l_path = None
            clip_g_path = None
            
            if clip_l_name != "None":
                clip_l_path = folder_paths.get_full_path("tensorrt", clip_l_name)
                if clip_l_path is None or not os.path.isfile(clip_l_path):
                    raise FileNotFoundError(f"CLIP-L file {clip_l_name} does not exist")
                    
            if clip_g_name != "None":
                clip_g_path = folder_paths.get_full_path("tensorrt", clip_g_name)
                if clip_g_path is None or not os.path.isfile(clip_g_path):
                    raise FileNotFoundError(f"CLIP-G file {clip_g_name} does not exist")
            
            if clip_l_path is None and clip_g_path is None:
                raise ValueError("At least one of CLIP-L or CLIP-G must be specified")
            
            # Create the appropriate CLIP model based on what engines we have
            if clip_l_path and clip_g_path:
                # SDXL-style dual CLIP
                clip_model = comfy.sdxl_clip.SDXLClipModel(
                    device=comfy.model_management.get_torch_device(),
                    dtype=torch.float16
                )
                # Replace the transformers with TensorRT versions
                clip_model.clip_l.transformer = TensorRTCLIPTextModel(clip_l_path, runtime, hidden_size=768)
                clip_model.clip_g.transformer = TensorRTCLIPTextModel(clip_g_path, runtime, hidden_size=1280)
                tokenizer = comfy.sdxl_clip.SDXLTokenizer()
                
            elif clip_l_path:
                # CLIP-L only
                clip_model = comfy.sd1_clip.SDClipModel(
                    device=comfy.model_management.get_torch_device(),
                    dtype=torch.float16
                )
                clip_model.transformer = TensorRTCLIPTextModel(clip_l_path, runtime, hidden_size=768)
                tokenizer = comfy.sd1_clip.SDTokenizer()
                
            elif clip_g_path:
                # CLIP-G only  
                clip_model = comfy.sdxl_clip.SDXLClipG(
                    device=comfy.model_management.get_torch_device(),
                    dtype=torch.float16
                )
                clip_model.transformer = TensorRTCLIPTextModel(clip_g_path, runtime, hidden_size=1280)
                tokenizer = comfy.sdxl_clip.SDXLClipGTokenizer()
            
            # Create a new CLIP object using ComfyUI's standard approach
            from comfy.model_patcher import ModelPatcher
            
            patcher = ModelPatcher(clip_model, 
                                load_device=comfy.model_management.get_torch_device(),
                                offload_device=comfy.model_management.text_encoder_offload_device())
            
            new_clip = comfy.sd.CLIP(no_init=True)
            new_clip.cond_stage_model = clip_model
            new_clip.tokenizer = tokenizer
            new_clip.patcher = patcher
            new_clip.layer_idx = None
            new_clip.use_clip_schedule = False
            new_clip.tokenizer_options = {}
            
            return (new_clip,)
            
        except Exception as e:
            print(f"TensorRTCLIPLoader - Error loading clip: {e}")
            import traceback
            traceback.print_exc()
            return (None,)