# pyright: reportOptionalMemberAccess=false

import torch
import os
import logging

import comfy.model_management
import comfy.sd
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


class TrTCLIPL:
    """TensorRT CLIP-L"""
    def __init__(self, engine_path, runtime):
        self.dtype = torch.float16
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self._size = int(os.stat(engine_path).st_size)
        self.runtime = runtime

    def load(self):
        if self.engine is not None or self.context is not None:
            return
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            check_for_trt_errors(self.runtime)
            self.context = self.engine.create_execution_context()
            check_for_trt_errors(self.runtime)

    @property
    def size(self) -> int:
        return self._size

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    def __call__(self, tokens):
        print(f"TrTCLIPL.__call__ - Input shape: {tokens.shape}, dtype: {tokens.dtype}")
        self.load()  # Ensure engine is loaded
        
        model_inputs = {"tokens": tokens}
        batch_size = tokens.shape[0]
        print(f"TrTCLIPL - batch_size: {batch_size}")
        
        # Handle batch splitting for dynamic profiles
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        min_batch = dims[0][0]
        opt_batch = dims[1][0]
        max_batch = dims[2][0]
        print(f"TrTCLIPL - Profile batches: min={min_batch}, opt={opt_batch}, max={max_batch}")

        curr_split_batch = 1
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break
        print(f"TrTCLIPL - curr_split_batch: {curr_split_batch}")

        self.set_bindings_shape(model_inputs, curr_split_batch)

        # Convert inputs to appropriate data types
        model_inputs_converted = {}
        for k in model_inputs:
            data_type = self.engine.get_tensor_dtype(k)
            model_inputs_converted[k] = model_inputs[k].to(dtype=trt_datatype_to_torch(data_type))
            print(f"TrTCLIPL - Input '{k}' converted to dtype: {trt_datatype_to_torch(data_type)}")

        # Get output tensor info
        output_binding_name = self.engine.get_tensor_name(len(model_inputs))
        out_shape = self.engine.get_tensor_shape(output_binding_name)
        out_shape = list(out_shape)
        print(f"TrTCLIPL - Initial output shape from engine: {out_shape}")

        # Handle dynamic shapes
        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                if idx == 0:  # batch
                    out_shape[idx] = tokens.shape[0]
                elif idx == 1:  # sequence
                    out_shape[idx] = tokens.shape[1]
                elif idx == 2:  # hidden_size
                    out_shape[idx] = 768  # CLIP-L hidden size
                else:
                    out_shape[idx] = tokens.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch
        print(f"TrTCLIPL - Final output shape after dynamic handling: {out_shape}")

        out = torch.empty(out_shape,
                          device=tokens.device,
                          dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(output_binding_name)))
        model_inputs_converted[output_binding_name] = out
        print(f"TrTCLIPL - Created output tensor: shape={out.shape}, dtype={out.dtype}")

        # Execute inference
        stream = torch.cuda.default_stream(tokens.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                tensor = model_inputs_converted[k]
                self.context.set_tensor_address(k, tensor[(tensor.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)

        print(f"TrTCLIPL - Inference complete, returning tensor: {out.shape}")
        return out, None  # CLIP-L doesn't have pooled output

    def unload(self):
        engine_obj = self.engine
        self.engine = None
        if engine_obj is not None:
            del engine_obj
        context_obj = self.context
        self.context = None
        if context_obj is not None:
            del context_obj


class TrTCLIPG:
    """TensorRT CLIP-G"""
    def __init__(self, engine_path, runtime):
        self.dtype = torch.float16
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self._size = int(os.stat(engine_path).st_size)
        self.runtime = runtime

    def load(self):
        if self.engine is not None or self.context is not None:
            return
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            check_for_trt_errors(self.runtime)
            self.context = self.engine.create_execution_context()
            check_for_trt_errors(self.runtime)

    @property
    def size(self) -> int:
        return self._size

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    def __call__(self, tokens):
        print(f"TrTCLIPG.__call__ - Input shape: {tokens.shape}, dtype: {tokens.dtype}")
        self.load()  # Ensure engine is loaded
        
        model_inputs = {"tokens": tokens}
        batch_size = tokens.shape[0]
        print(f"TrTCLIPG - batch_size: {batch_size}")
        
        # Handle batch splitting for dynamic profiles
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        min_batch = dims[0][0]
        opt_batch = dims[1][0]
        max_batch = dims[2][0]
        print(f"TrTCLIPG - Profile batches: min={min_batch}, opt={opt_batch}, max={max_batch}")

        curr_split_batch = 1
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break
        print(f"TrTCLIPG - curr_split_batch: {curr_split_batch}")

        self.set_bindings_shape(model_inputs, curr_split_batch)

        # Convert inputs to appropriate data types
        model_inputs_converted = {}
        for k in model_inputs:
            data_type = self.engine.get_tensor_dtype(k)
            model_inputs_converted[k] = model_inputs[k].to(dtype=trt_datatype_to_torch(data_type))
            print(f"TrTCLIPG - Input '{k}' converted to dtype: {trt_datatype_to_torch(data_type)}")

        # Get output tensor info - CLIP-G has 2 outputs (hidden_states, pooled_output)
        hidden_states_name = self.engine.get_tensor_name(len(model_inputs))
        pooled_output_name = self.engine.get_tensor_name(len(model_inputs) + 1)
        
        hidden_shape = self.engine.get_tensor_shape(hidden_states_name)
        pooled_shape = self.engine.get_tensor_shape(pooled_output_name)
        
        hidden_shape = list(hidden_shape)
        pooled_shape = list(pooled_shape)
        
        print(f"TrTCLIPG - Initial hidden shape from engine: {hidden_shape}")
        print(f"TrTCLIPG - Initial pooled shape from engine: {pooled_shape}")

        # Handle dynamic shapes for hidden states
        for idx in range(len(hidden_shape)):
            if hidden_shape[idx] == -1:
                if idx == 0:  # batch
                    hidden_shape[idx] = tokens.shape[0]
                elif idx == 1:  # sequence
                    hidden_shape[idx] = tokens.shape[1]
                elif idx == 2:  # hidden_size
                    hidden_shape[idx] = 1280  # CLIP-G hidden size
                else:
                    hidden_shape[idx] = tokens.shape[idx]
            else:
                if idx == 0:
                    hidden_shape[idx] *= curr_split_batch

        # Handle dynamic shapes for pooled output
        for idx in range(len(pooled_shape)):
            if pooled_shape[idx] == -1:
                if idx == 0:  # batch
                    pooled_shape[idx] = tokens.shape[0]
                elif idx == 1:  # pooled dimension
                    pooled_shape[idx] = 1280  # CLIP-G pooled dimension
                else:
                    pooled_shape[idx] = tokens.shape[idx]
            else:
                if idx == 0:
                    pooled_shape[idx] *= curr_split_batch

        print(f"TrTCLIPG - Final hidden shape after dynamic handling: {hidden_shape}")
        print(f"TrTCLIPG - Final pooled shape after dynamic handling: {pooled_shape}")

        hidden_out = torch.empty(hidden_shape,
                               device=tokens.device,
                               dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(hidden_states_name)))
        pooled_out = torch.empty(pooled_shape,
                               device=tokens.device,
                               dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(pooled_output_name)))
        
        model_inputs_converted[hidden_states_name] = hidden_out
        model_inputs_converted[pooled_output_name] = pooled_out
        
        print(f"TrTCLIPG - Created hidden tensor: shape={hidden_out.shape}, dtype={hidden_out.dtype}")
        print(f"TrTCLIPG - Created pooled tensor: shape={pooled_out.shape}, dtype={pooled_out.dtype}")

        # Execute inference
        stream = torch.cuda.default_stream(tokens.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                tensor = model_inputs_converted[k]
                self.context.set_tensor_address(k, tensor[(tensor.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)

        print(f"TrTCLIPG - Inference complete, returning tensors: {hidden_out.shape}, {pooled_out.shape}")
        return hidden_out, pooled_out

    def unload(self):
        engine_obj = self.engine
        self.engine = None
        if engine_obj is not None:
            del engine_obj
        context_obj = self.context
        self.context = None
        if context_obj is not None:
            del context_obj


class TrTCLIP:
    """TensorRT CLIP wrapper that combines CLIP-L and CLIP-G"""
    def __init__(self, clip_l_path=None, clip_g_path=None, original_clip=None):
        self.clip_l = TrTCLIPL(clip_l_path, runtime) if clip_l_path else None
        self.clip_g = TrTCLIPG(clip_g_path, runtime) if clip_g_path else None
        self.original_clip = original_clip
        
        # CLIP properties
        self.device = comfy.model_management.get_torch_device()
        self.offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = torch.float16
        self.dtypes = {torch.float16}
        
    def set_clip_options(self, options):
        """Set CLIP options (for compatibility)"""
        pass
        
    def reset_clip_options(self):
        """Reset CLIP options (for compatibility)"""
        pass

    def encode_token_weights(self, token_weight_pairs):
        """Encode token weights using TensorRT CLIP models"""
        
        # Handle SDXL dual-clip structure
        if isinstance(token_weight_pairs, dict) and "l" in token_weight_pairs and "g" in token_weight_pairs:
            token_weight_pairs_l = token_weight_pairs["l"]
            token_weight_pairs_g = token_weight_pairs["g"]
            
            l_out = None
            l_pooled = None
            g_out = None
            g_pooled = None
            
            # Process CLIP-L tokens
            if self.clip_l and len(token_weight_pairs_l) > 0:
                # Convert token weight pairs to tokens tensor
                tokens_l = self._convert_token_weights_to_tokens(token_weight_pairs_l)
                tokens_l = tokens_l.to(device=self.device, dtype=torch.long)
                l_out, l_pooled = self.clip_l(tokens_l)
                l_out = l_out.to(dtype=torch.float32)
                if l_pooled is not None:
                    l_pooled = l_pooled.to(dtype=torch.float32)
                else:
                    l_pooled = torch.zeros((tokens_l.shape[0], 768), device=self.device, dtype=torch.float32)
            else:
                l_pooled = torch.zeros((1, 768), device=self.device, dtype=torch.float32)
                
            # Process CLIP-G tokens  
            if self.clip_g and len(token_weight_pairs_g) > 0:
                # Convert token weight pairs to tokens tensor
                tokens_g = self._convert_token_weights_to_tokens(token_weight_pairs_g)
                tokens_g = tokens_g.to(device=self.device, dtype=torch.long)
                g_out, g_pooled = self.clip_g(tokens_g)
                g_out = g_out.to(dtype=torch.float32)
                g_pooled = g_pooled.to(dtype=torch.float32)
            else:
                g_pooled = torch.zeros((1, 1280), device=self.device, dtype=torch.float32)
                
            # Combine outputs like SDXL
            if l_out is not None and g_out is not None:
                cut_to = min(l_out.shape[1], g_out.shape[1])
                combined_out = torch.cat([l_out[:,:cut_to], g_out[:,:cut_to]], dim=-1)
            elif g_out is not None:
                combined_out = torch.nn.functional.pad(g_out, (768, 0))
            elif l_out is not None:
                combined_out = l_out
            else:
                combined_out = torch.zeros((1, 77, 2048), device=self.device, dtype=torch.float32)
                
            # Combine pooled outputs
            pooled = torch.cat((l_pooled, g_pooled), dim=-1)
            
            return combined_out, pooled
        else:
            # Single CLIP model handling
            if self.clip_l:
                tokens = self._convert_token_weights_to_tokens(token_weight_pairs)
                tokens = tokens.to(device=self.device, dtype=torch.long)
                out, pooled = self.clip_l(tokens)
                return out.to(dtype=torch.float32), pooled.to(dtype=torch.float32) if pooled is not None else None
            elif self.clip_g:
                tokens = self._convert_token_weights_to_tokens(token_weight_pairs)
                tokens = tokens.to(device=self.device, dtype=torch.long)
                out, pooled = self.clip_g(tokens)
                return out.to(dtype=torch.float32), pooled.to(dtype=torch.float32)
            else:
                raise RuntimeError("No TensorRT CLIP models loaded")

    def _convert_token_weights_to_tokens(self, token_weight_pairs):
        """Convert token weight pairs to tokens tensor"""
        # This is a simplified conversion - in practice you'd want to handle weights properly
        if len(token_weight_pairs) == 0:
            return torch.zeros((1, 77), dtype=torch.long, device=self.device)
            
        batch_size = len(token_weight_pairs)
        max_length = max(len(tokens) for tokens, _ in token_weight_pairs)
        max_length = min(max_length, 77)  # CLIP max length
        
        tokens_tensor = torch.zeros((batch_size, max_length), dtype=torch.long, device=self.device)
        
        for i, (tokens, weights) in enumerate(token_weight_pairs):
            length = min(len(tokens), max_length)
            tokens_tensor[i, :length] = torch.tensor(tokens[:length], dtype=torch.long, device=self.device)
            
        return tokens_tensor

    def load_sd(self, sd):
        """Load state dict (no-op for TensorRT)"""
        return {}

    def get_sd(self):
        """Get state dict (empty for TensorRT)"""
        return {}

    @property
    def memory_used_clip_l(self):
        clip_l_size = self.clip_l.size if self.clip_l else 0
        return clip_l_size

    @property
    def memory_used_clip_g(self):
        clip_g_size = self.clip_g.size if self.clip_g else 0
        return clip_g_size

    def unload(self):
        if self.clip_l:
            self.clip_l.unload()
        if self.clip_g:
            self.clip_g.unload()


class TensorRTCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        engine_files = folder_paths.get_filename_list("tensorrt")
        # Filter for CLIP engines
        clip_l_files = ["None"] + [f for f in engine_files if "clip_l" in f.lower()]
        clip_g_files = ["None"] + [f for f in engine_files if "clip_g" in f.lower()]
        
        return {
            "required": {
                "clip": ("CLIP",),
                "clip_l_name": (clip_l_files,),
                "clip_g_name": (clip_g_files,),
            },
        }
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "TensorRT"

    def load_clip(self, clip, clip_l_name, clip_g_name):
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
        
        # Create TensorRT CLIP wrapper
        trt_clip_model = TrTCLIP(clip_l_path, clip_g_path, clip)
        
        # Replace the original CLIP's cond_stage_model with our TensorRT version
        new_clip = comfy.sd.CLIP(no_init=True)
        new_clip.cond_stage_model = trt_clip_model
        new_clip.tokenizer = clip.tokenizer
        new_clip.patcher = clip.patcher
        new_clip.layer_idx = clip.layer_idx
        new_clip.use_clip_schedule = clip.use_clip_schedule
        new_clip.tokenizer_options = clip.tokenizer_options
        
        return (new_clip,)


NODE_CLASS_MAPPINGS = {
    "TensorRTCLIPLoader": TensorRTCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorRTCLIPLoader": "TensorRT CLIP Loader",
} 