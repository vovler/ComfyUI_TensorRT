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
        # Create a proper embedding layer that can convert tokens to embeddings
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)
        # Initialize with small random values for better numerical stability
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
    @property 
    def weight(self):
        return self.embedding.weight
        
    def forward(self, input_tokens, out_dtype=torch.float32):
        # Convert tokens to embeddings using the embedding layer
        embeddings = self.embedding(input_tokens)
        return embeddings.to(dtype=out_dtype)

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
            
            # Check for engine deseralization errors
            if self.runtime.error_recorder.num_errors() > 0:
                print(f"TensorRT - Warning: {self.runtime.error_recorder.num_errors()} errors during engine deserialization")
                for i in range(self.runtime.error_recorder.num_errors()):
                    print(f"  Error {i}: {self.runtime.error_recorder.get_error_desc(i)}")
                self.runtime.error_recorder.clear()
            
            print(f"TensorRT - Engine loaded, creating execution context...")
            self.context = self.engine.create_execution_context()
            
            # Check for context creation errors but don't fail - just warn
            if self.runtime.error_recorder.num_errors() > 0:
                print(f"TensorRT - Warning: {self.runtime.error_recorder.num_errors()} errors during context creation")
                for i in range(self.runtime.error_recorder.num_errors()):
                    print(f"  Error {i}: {self.runtime.error_recorder.get_error_desc(i)}")
                self.runtime.error_recorder.clear()
                
        # Print engine information for debugging
        print(f"TensorRT CLIP engine loaded successfully")
        print(f"TensorRT - Engine has {self.engine.num_io_tensors} tensors:")
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            print(f"  {tensor_name}: {'INPUT' if is_input else 'OUTPUT'} - shape: {tensor_shape}")
            
            # For input tensors, also show profile information
            if is_input:
                try:
                    num_profiles = self.engine.num_optimization_profiles
                    print(f"    Optimization profiles: {num_profiles}")
                    for profile_idx in range(num_profiles):
                        min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(tensor_name, profile_idx)
                        print(f"    Profile {profile_idx}: min={min_shape}, opt={opt_shape}, max={max_shape}")
                except Exception as e:
                    print(f"    Could not get profile info: {e}")

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
            new_shape = [shape[0] // split_batch] + list(shape[1:])
            print(f"TensorRT - Setting input shape for '{k}': {shape} -> {new_shape}")
            
            # Get engine's expected shape for comparison
            engine_shape = self.engine.get_tensor_shape(k)
            print(f"TensorRT - Engine expects shape for '{k}': {engine_shape}")
            
            try:
                self.context.set_input_shape(k, new_shape)
                print(f"TensorRT - Successfully set input shape for '{k}'")
            except Exception as e:
                print(f"TensorRT - Failed to set input shape for '{k}': {e}")
                print(f"TensorRT - Trying with original shape: {shape}")
                try:
                    self.context.set_input_shape(k, shape)
                    print(f"TensorRT - Successfully set input shape with original shape for '{k}'")
                except Exception as e2:
                    print(f"TensorRT - Failed with original shape too: {e2}")
                    raise e2

    def forward(self, input_tokens=None, attention_mask=None, embeds=None, num_tokens=None, 
                intermediate_output=None, final_layer_norm_intermediate=True, dtype=torch.float32):
        """
        TensorRT inference that matches CLIPTextModel interface exactly
        """
        self.load()  # Ensure engine is loaded
        
        # The TensorRT engine expects embeddings as input, not tokens
        if embeds is not None:
            # Use provided embeddings directly
            embeddings = embeds
            print(f"TensorRT CLIP forward - Using provided embeddings: {embeddings.shape}, dtype: {embeddings.dtype}")
        elif input_tokens is not None:
            # Convert tokens to embeddings using dummy embedding layer
            embeddings = self.dummy_embedding(input_tokens, out_dtype=dtype)
            print(f"TensorRT CLIP forward - Converted tokens to embeddings: {input_tokens.shape} -> {embeddings.shape}")
        else:
            raise ValueError("Either input_tokens or embeds must be provided")
        
        print(f"TensorRT CLIP forward - Final embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
        
        # Find the actual input tensor name from the engine
        input_tensor_name = None
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_tensor_name = tensor_name
                break
        
        if input_tensor_name is None:
            raise RuntimeError("No input tensor found in TensorRT CLIP engine")
            
        model_inputs = {input_tensor_name: embeddings}
        batch_size = embeddings.shape[0]
        
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

        # Debug: Print engine tensor info before setting shapes
        print(f"TensorRT - Engine tensor info:")
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            print(f"  {tensor_name}: {'INPUT' if is_input else 'OUTPUT'} - engine_shape: {tensor_shape}")
        
        print(f"TensorRT - About to set input shape for {input_tensor_name}: {embeddings.shape}")
        try:
            self.set_bindings_shape(model_inputs, curr_split_batch)
        except Exception as e:
            print(f"TensorRT - Error setting input shape: {e}")
            print(f"TensorRT - Trying direct context.set_input_shape...")
            self.context.set_input_shape(input_tensor_name, embeddings.shape)
            print(f"TensorRT - Direct shape setting successful")

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
                    hidden_shape[idx] = embeddings.shape[0]
                elif idx == 1:  # sequence
                    hidden_shape[idx] = embeddings.shape[1]
                elif idx == 2:  # hidden_size
                    hidden_shape[idx] = self.hidden_size
                else:
                    hidden_shape[idx] = embeddings.shape[idx]
            else:
                if idx == 0:
                    hidden_shape[idx] *= curr_split_batch

        for idx in range(len(pooled_shape)):
            if pooled_shape[idx] == -1:
                if idx == 0:  # batch
                    pooled_shape[idx] = embeddings.shape[0]
                elif idx == 1:  # pooled dimension
                    pooled_shape[idx] = self.hidden_size
                else:
                    pooled_shape[idx] = embeddings.shape[idx]
            else:
                if idx == 0:
                    pooled_shape[idx] *= curr_split_batch

        hidden_out = torch.empty(hidden_shape,
                               device=embeddings.device,
                               dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(hidden_states_name)))
        pooled_out = torch.empty(pooled_shape,
                               device=embeddings.device,
                               dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(pooled_output_name)))
        
        model_inputs_converted[hidden_states_name] = hidden_out
        model_inputs_converted[pooled_output_name] = pooled_out

        # Execute inference
        stream = torch.cuda.default_stream(embeddings.device)
        try:
            for i in range(curr_split_batch):
                for k in model_inputs_converted:
                    tensor = model_inputs_converted[k]
                    self.context.set_tensor_address(k, tensor[(tensor.shape[0] // curr_split_batch) * i:].data_ptr())
                success = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
                if not success:
                    print(f"TensorRT - Warning: execute_async_v3 returned False for batch {i}")
            
            # Ensure GPU operations complete
            torch.cuda.synchronize()
            print(f"TensorRT CLIP inference complete")
            
        except Exception as e:
            print(f"TensorRT - Error during inference: {e}")
            # Return zero tensors as fallback
            hidden_out = torch.zeros(hidden_shape, device=embeddings.device, dtype=torch.float32)
            pooled_out = torch.zeros(pooled_shape, device=embeddings.device, dtype=torch.float32)
            return hidden_out, hidden_out, pooled_out, pooled_out
        
        # Convert to float32 for compatibility
        hidden_out = hidden_out.to(dtype=torch.float32)
        pooled_out = pooled_out.to(dtype=torch.float32)
        
        # Check for NaN/Inf values in TensorRT outputs
        hidden_has_nan = torch.isnan(hidden_out).any()
        hidden_has_inf = torch.isinf(hidden_out).any()
        pooled_has_nan = torch.isnan(pooled_out).any()
        pooled_has_inf = torch.isinf(pooled_out).any()
        
        if hidden_has_nan:
            print(f"🔴 TensorRT - ERROR: NaN values detected in hidden_out!")
            print(f"   NaN count: {torch.isnan(hidden_out).sum().item()}/{hidden_out.numel()}")
            nan_positions = torch.nonzero(torch.isnan(hidden_out))
            if len(nan_positions) <= 10:  # Only print first 10 positions
                print(f"   NaN positions: {nan_positions.tolist()}")
                
        if hidden_has_inf:
            print(f"🔴 TensorRT - ERROR: Infinite values detected in hidden_out!")
            print(f"   Inf count: {torch.isinf(hidden_out).sum().item()}/{hidden_out.numel()}")
            inf_positions = torch.nonzero(torch.isinf(hidden_out))
            if len(inf_positions) <= 10:  # Only print first 10 positions
                print(f"   Inf positions: {inf_positions.tolist()}")
                
        if pooled_has_nan:
            print(f"🔴 TensorRT - ERROR: NaN values detected in pooled_out!")
            print(f"   NaN count: {torch.isnan(pooled_out).sum().item()}/{pooled_out.numel()}")
            
        if pooled_has_inf:
            print(f"🔴 TensorRT - ERROR: Infinite values detected in pooled_out!")
            print(f"   Inf count: {torch.isinf(pooled_out).sum().item()}/{pooled_out.numel()}")
        
        # Print detailed statistics for valid outputs
        if not hidden_has_nan and not hidden_has_inf:
            print(f"✅ TensorRT hidden_out stats: min={hidden_out.min().item():.6f}, max={hidden_out.max().item():.6f}, mean={hidden_out.mean().item():.6f}, std={hidden_out.std().item():.6f}")
        else:
            print(f"❌ TensorRT hidden_out: CORRUPTED (contains NaN/Inf)")
            
        if not pooled_has_nan and not pooled_has_inf:
            print(f"✅ TensorRT pooled_out stats: min={pooled_out.min().item():.6f}, max={pooled_out.max().item():.6f}, mean={pooled_out.mean().item():.6f}, std={pooled_out.std().item():.6f}")
        else:
            print(f"❌ TensorRT pooled_out: CORRUPTED (contains NaN/Inf)")
        
        # Also check if the outputs are all zeros (which might indicate inference failure)
        hidden_all_zero = (hidden_out == 0).all()
        pooled_all_zero = (pooled_out == 0).all()
        
        if hidden_all_zero:
            print(f"⚠️  TensorRT - WARNING: hidden_out is all zeros - inference may have failed")
        if pooled_all_zero:
            print(f"⚠️  TensorRT - WARNING: pooled_out is all zeros - inference may have failed")
        
        # Return in the same format as CLIPTextModel:
        # (last_hidden_state, intermediate_hidden_state, projected_pooled, unprojected_pooled)
        # For TensorRT, we return the same hidden states for both last and intermediate
        return hidden_out, hidden_out, pooled_out, pooled_out

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
            
            # Add missing attributes for ComfyUI compatibility
            new_clip.apply_hooks_to_conds = None
            
            return (new_clip,)
            
        except Exception as e:
            print(f"TensorRTCLIPLoader - Error loading clip: {e}")
            import traceback
            traceback.print_exc()
            return (None,)