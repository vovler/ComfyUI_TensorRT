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
        print(f"Loading TensorRT CLIP-L engine from: {self.engine_path}")
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            check_for_trt_errors(self.runtime)
            self.context = self.engine.create_execution_context()
            check_for_trt_errors(self.runtime)
        print(f"TensorRT CLIP-L engine loaded successfully")

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
        
        # Debug: Print all tensor names in the engine
        print(f"TrTCLIPL - Engine has {self.engine.num_io_tensors} tensors:")
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            print(f"  {i}: {tensor_name} - {'INPUT' if is_input else 'OUTPUT'} - shape: {tensor_shape}")
        
        # Find the actual input tensor name from the engine
        input_tensor_name = None
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_tensor_name = tensor_name
                break
        
        if input_tensor_name is None:
            raise RuntimeError("No input tensor found in TensorRT CLIP-L engine")
            
        model_inputs = {input_tensor_name: tokens}
        batch_size = tokens.shape[0]
        print(f"TrTCLIPL - Using input tensor name: {input_tensor_name}")
        print(f"TrTCLIPL - batch_size: {batch_size}")
        
        # Handle batch splitting for dynamic profiles
        dims = self.engine.get_tensor_profile_shape(input_tensor_name, 0)
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

        # Get output tensor info - CLIP-L now has 2 outputs (hidden_states, pooled_output)
        # Find output tensors dynamically
        output_tensors = []
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                output_tensors.append(tensor_name)
        
        if len(output_tensors) < 2:
            raise RuntimeError(f"Expected 2 output tensors for CLIP-L, found {len(output_tensors)}")
            
        hidden_states_name = output_tensors[0]  # First output is hidden states
        pooled_output_name = output_tensors[1]  # Second output is pooled output
        
        hidden_shape = self.engine.get_tensor_shape(hidden_states_name)
        pooled_shape = self.engine.get_tensor_shape(pooled_output_name)
        
        hidden_shape = list(hidden_shape)
        pooled_shape = list(pooled_shape)
        
        print(f"TrTCLIPL - Initial hidden shape from engine: {hidden_shape}")
        print(f"TrTCLIPL - Initial pooled shape from engine: {pooled_shape}")

        # Handle dynamic shapes for hidden states
        for idx in range(len(hidden_shape)):
            if hidden_shape[idx] == -1:
                if idx == 0:  # batch
                    hidden_shape[idx] = tokens.shape[0]
                elif idx == 1:  # sequence
                    hidden_shape[idx] = tokens.shape[1]
                elif idx == 2:  # hidden_size
                    hidden_shape[idx] = 768  # CLIP-L hidden size
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
                    pooled_shape[idx] = 768  # CLIP-L pooled dimension
                else:
                    pooled_shape[idx] = tokens.shape[idx]
            else:
                if idx == 0:
                    pooled_shape[idx] *= curr_split_batch

        print(f"TrTCLIPL - Final hidden shape after dynamic handling: {hidden_shape}")
        print(f"TrTCLIPL - Final pooled shape after dynamic handling: {pooled_shape}")

        hidden_out = torch.empty(hidden_shape,
                               device=tokens.device,
                               dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(hidden_states_name)))
        pooled_out = torch.empty(pooled_shape,
                               device=tokens.device,
                               dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(pooled_output_name)))
        
        model_inputs_converted[hidden_states_name] = hidden_out
        model_inputs_converted[pooled_output_name] = pooled_out
        
        print(f"TrTCLIPL - Created hidden tensor: shape={hidden_out.shape}, dtype={hidden_out.dtype}")
        print(f"TrTCLIPL - Created pooled tensor: shape={pooled_out.shape}, dtype={pooled_out.dtype}")

        # Execute inference
        stream = torch.cuda.default_stream(tokens.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                tensor = model_inputs_converted[k]
                self.context.set_tensor_address(k, tensor[(tensor.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)

        print(f"TrTCLIPL - Inference complete, returning tensors: {hidden_out.shape}, {pooled_out.shape}")
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
        print(f"Loading TensorRT CLIP-G engine from: {self.engine_path}")
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            check_for_trt_errors(self.runtime)
            self.context = self.engine.create_execution_context()
            check_for_trt_errors(self.runtime)
        print(f"TensorRT CLIP-G engine loaded successfully")

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
        
        # Debug: Print all tensor names in the engine
        print(f"TrTCLIPG - Engine has {self.engine.num_io_tensors} tensors:")
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            print(f"  {i}: {tensor_name} - {'INPUT' if is_input else 'OUTPUT'} - shape: {tensor_shape}")
        
        # Find the actual input tensor name from the engine
        input_tensor_name = None
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_tensor_name = tensor_name
                break
        
        if input_tensor_name is None:
            raise RuntimeError("No input tensor found in TensorRT CLIP-G engine")
            
        model_inputs = {input_tensor_name: tokens}
        batch_size = tokens.shape[0]
        print(f"TrTCLIPG - Using input tensor name: {input_tensor_name}")
        print(f"TrTCLIPG - batch_size: {batch_size}")
        
        # Handle batch splitting for dynamic profiles
        dims = self.engine.get_tensor_profile_shape(input_tensor_name, 0)
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
        # Find output tensors dynamically
        output_tensors = []
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                output_tensors.append(tensor_name)
        
        if len(output_tensors) < 2:
            raise RuntimeError(f"Expected 2 output tensors for CLIP-G, found {len(output_tensors)}")
            
        hidden_states_name = output_tensors[0]  # First output is hidden states
        pooled_output_name = output_tensors[1]  # Second output is pooled output
        
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


class TrTCLIPLWrapper:
    """Wrapper that makes TrTCLIPL look like ComfyUI's SDClipModel"""
    def __init__(self, clip_l_path, runtime):
        self.engine = TrTCLIPL(clip_l_path, runtime)
    
    def encode_token_weights(self, token_weight_pairs):
        # Convert ComfyUI token format to simple tokens tensor
        tokens = self._convert_token_weights_to_tokens(token_weight_pairs)
        tokens = tokens.to(device=torch.device('cuda'), dtype=torch.long)
        hidden_states, pooled_output = self.engine(tokens)
        return hidden_states.to(dtype=torch.float32), pooled_output.to(dtype=torch.float32)
    
    @property
    def size(self):
        return self.engine.size
    
    def load(self):
        return self.engine.load()
    
    def unload(self):
        return self.engine.unload()
    
    def _convert_token_weights_to_tokens(self, token_weight_pairs):
        if len(token_weight_pairs) == 0:
            return torch.zeros((1, 77), dtype=torch.long)
        batch_size = len(token_weight_pairs)
        max_length = 77
        tokens_tensor = torch.zeros((batch_size, max_length), dtype=torch.long)
        for i, item in enumerate(token_weight_pairs):
            if isinstance(item, (list, tuple)):
                tokens_list = []
                for token_weight in item:
                    if isinstance(token_weight, (list, tuple)) and len(token_weight) >= 1:
                        tokens_list.append(int(token_weight[0]))
                    elif isinstance(token_weight, (int, float)):
                        tokens_list.append(int(token_weight))
                if len(tokens_list) < max_length:
                    tokens_list.extend([0] * (max_length - len(tokens_list)))
                elif len(tokens_list) > max_length:
                    tokens_list = tokens_list[:max_length]
                tokens_tensor[i, :] = torch.tensor(tokens_list, dtype=torch.long)
        return tokens_tensor


class TrTCLIPGWrapper:
    """Wrapper that makes TrTCLIPG look like ComfyUI's SDClipModel"""
    def __init__(self, clip_g_path, runtime):
        self.engine = TrTCLIPG(clip_g_path, runtime)
    
    def encode_token_weights(self, token_weight_pairs):
        # Convert ComfyUI token format to simple tokens tensor
        tokens = self._convert_token_weights_to_tokens(token_weight_pairs)
        tokens = tokens.to(device=torch.device('cuda'), dtype=torch.long)
        hidden_states, pooled_output = self.engine(tokens)
        return hidden_states.to(dtype=torch.float32), pooled_output.to(dtype=torch.float32)
    
    @property
    def size(self):
        return self.engine.size
    
    def load(self):
        return self.engine.load()
    
    def unload(self):
        return self.engine.unload()
    
    def _convert_token_weights_to_tokens(self, token_weight_pairs):
        if len(token_weight_pairs) == 0:
            return torch.zeros((1, 77), dtype=torch.long)
        batch_size = len(token_weight_pairs)
        max_length = 77
        tokens_tensor = torch.zeros((batch_size, max_length), dtype=torch.long)
        for i, item in enumerate(token_weight_pairs):
            if isinstance(item, (list, tuple)):
                tokens_list = []
                for token_weight in item:
                    if isinstance(token_weight, (list, tuple)) and len(token_weight) >= 1:
                        tokens_list.append(int(token_weight[0]))
                    elif isinstance(token_weight, (int, float)):
                        tokens_list.append(int(token_weight))
                if len(tokens_list) < max_length:
                    tokens_list.extend([0] * (max_length - len(tokens_list)))
                elif len(tokens_list) > max_length:
                    tokens_list = tokens_list[:max_length]
                tokens_tensor[i, :] = torch.tensor(tokens_list, dtype=torch.long)
        return tokens_tensor


class TrTCLIP(torch.nn.Module):
    """TensorRT CLIP wrapper that reuses ComfyUI's SDXL logic"""
    def __init__(self, clip_l_path=None, clip_g_path=None, original_clip=None):
        super().__init__()
        
        # Create TensorRT CLIP engines as mock CLIP models
        self.clip_l = TrTCLIPLWrapper(clip_l_path, runtime) if clip_l_path else None
        self.clip_g = TrTCLIPGWrapper(clip_g_path, runtime) if clip_g_path else None
        self.original_clip = original_clip
        
        # CLIP properties (match SDXLClipModel)
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
        """Use ComfyUI's exact SDXL logic - much cleaner!"""
        self._force_load_engines()
        
        # Use ComfyUI's SDXLClipModel.encode_token_weights logic exactly
        if isinstance(token_weight_pairs, dict) and "l" in token_weight_pairs and "g" in token_weight_pairs:
            token_weight_pairs_g = token_weight_pairs["g"]
            token_weight_pairs_l = token_weight_pairs["l"]
            
            # Call our TensorRT wrappers that mimic ComfyUI's interface
            g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g) if self.clip_g else (None, None)
            l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l) if self.clip_l else (None, None)
            
            # ComfyUI's exact concatenation logic from sdxl_clip.py line 60-61
            if l_out is not None and g_out is not None:
                cut_to = min(l_out.shape[1], g_out.shape[1])
                combined_out = torch.cat([l_out[:,:cut_to], g_out[:,:cut_to]], dim=-1)
            elif g_out is not None:
                combined_out = torch.nn.functional.pad(g_out, (768, 0))  # Pad CLIP-G to 2048
            elif l_out is not None:
                combined_out = torch.nn.functional.pad(l_out, (0, 1280))  # Pad CLIP-L to 2048  
            else:
                combined_out = torch.zeros((1, 77, 2048), device=self.device, dtype=torch.float32)
            
            # ComfyUI returns g_pooled ONLY (not concatenated)
            pooled = g_pooled if g_pooled is not None else torch.zeros((1, 1280), device=self.device, dtype=torch.float32)
            
            return combined_out, pooled
        else:
            # Single CLIP handling
            if self.clip_l:
                return self.clip_l.encode_token_weights(token_weight_pairs)
            elif self.clip_g:
                return self.clip_g.encode_token_weights(token_weight_pairs)
            else:
                raise RuntimeError("No TensorRT CLIP models loaded")



    def load_sd(self, sd):
        """Load state dict (no-op for TensorRT)"""
        return {}

    def get_sd(self):
        """Get state dict (empty for TensorRT)"""
        return {}

    def state_dict(self):
        """Get state dict (required by ModelPatcher)"""
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
    
    def to(self, device=None, dtype=None, **kwargs):
        """Move model to device (for ModelPatcher compatibility)"""
        if device is not None:
            old_device = self.device
            self.device = device
            
            # If moving to GPU and we have engines, force load them
            if device.type == 'cuda' and old_device.type != 'cuda':
                self._force_load_engines()
            # If moving to CPU, we could implement offloading here
            elif device.type == 'cpu' and old_device.type == 'cuda':
                self._offload_engines()
                
        if dtype is not None:
            self.dtype = dtype
        return self
    
    def _force_load_engines(self):
        """Force load TensorRT engines to GPU"""
        if self.clip_l:
            self.clip_l.load()
        if self.clip_g:
            self.clip_g.load()
    
    def _offload_engines(self):
        """Offload TensorRT engines from GPU"""
        if self.clip_l:
            self.clip_l.unload()
        if self.clip_g:
            self.clip_g.unload()
    
    def cuda(self, device=None):
        """Move to CUDA device (for ModelPatcher compatibility)"""
        if device is None:
            device = torch.device('cuda')
        return self.to(device)
    
    def cpu(self):
        """Move to CPU device (for ModelPatcher compatibility)"""
        return self.to(torch.device('cpu'))
    
    def eval(self):
        """Set to evaluation mode (for ModelPatcher compatibility)"""
        return self
    
    def train(self, mode=True):
        """Set training mode (for ModelPatcher compatibility)"""
        return self
    
    def parameters(self):
        """Return empty parameters iterator (for ModelPatcher compatibility)"""
        return iter([])
    
    def named_parameters(self):
        """Return empty named parameters iterator (for ModelPatcher compatibility)"""
        return iter([])
    
    def named_modules(self):
        """Return empty named modules iterator (for ModelPatcher compatibility)"""
        return iter([])
    
    def modules(self):
        """Return empty modules iterator (for ModelPatcher compatibility)"""
        return iter([self])


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
        trt_clip_model = TrTCLIP(clip_l_path, clip_g_path, None)
        
        # Create a new CLIP object with TensorRT backend
        # Use SDXL tokenizer since we're supporting dual CLIP (L and G)
        from comfy.sdxl_clip import SDXLTokenizer
        from comfy.model_patcher import ModelPatcher
        
        # Create a dummy patcher for compatibility
        dummy_patcher = ModelPatcher(trt_clip_model, 
                                   load_device=comfy.model_management.get_torch_device(),
                                   offload_device=comfy.model_management.text_encoder_offload_device())
        
        new_clip = comfy.sd.CLIP(no_init=True)
        new_clip.cond_stage_model = trt_clip_model
        new_clip.tokenizer = SDXLTokenizer()
        new_clip.patcher = dummy_patcher
        new_clip.layer_idx = None
        new_clip.use_clip_schedule = False
        new_clip.tokenizer_options = {}
        
        # Add missing attributes for ComfyUI compatibility
        new_clip.apply_hooks_to_conds = None  # This should be None as per ComfyUI's CLIP init
        
        return (new_clip,)