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


class TrTVAEEncoder:
    """TensorRT VAE Encoder"""
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

    def __call__(self, x):
        self.load()  # Ensure engine is loaded
        
        model_inputs = {"x": x}
        batch_size = x.shape[0]
        
        # Handle batch splitting for dynamic profiles
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
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
            model_inputs_converted[k] = model_inputs[k].to(dtype=trt_datatype_to_torch(data_type))

        # Get output tensor info
        output_binding_name = self.engine.get_tensor_name(len(model_inputs))
        out_shape = self.engine.get_tensor_shape(output_binding_name)
        out_shape = list(out_shape)

        # Handle dynamic shapes
        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                if idx == 0:
                    out_shape[idx] = x.shape[0]
                elif idx == 2:  # height
                    out_shape[idx] = x.shape[2] // 8  # VAE encoder downsamples by 8
                elif idx == 3:  # width
                    out_shape[idx] = x.shape[3] // 8
                else:
                    out_shape[idx] = x.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch

        out = torch.empty(out_shape,
                          device=x.device,
                          dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(output_binding_name)))
        model_inputs_converted[output_binding_name] = out

        # Execute inference
        stream = torch.cuda.default_stream(x.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                tensor = model_inputs_converted[k]
                self.context.set_tensor_address(k, tensor[(tensor.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)

        return out

    def unload(self):
        engine_obj = self.engine
        self.engine = None
        if engine_obj is not None:
            del engine_obj
        context_obj = self.context
        self.context = None
        if context_obj is not None:
            del context_obj


class TrTVAEDecoder:
    """TensorRT VAE Decoder"""
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

    def __call__(self, x):
        self.load()  # Ensure engine is loaded
        
        model_inputs = {"x": x}
        batch_size = x.shape[0]
        
        # Handle batch splitting for dynamic profiles
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
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
            model_inputs_converted[k] = model_inputs[k].to(dtype=trt_datatype_to_torch(data_type))

        # Get output tensor info
        output_binding_name = self.engine.get_tensor_name(len(model_inputs))
        out_shape = self.engine.get_tensor_shape(output_binding_name)
        out_shape = list(out_shape)

        # Handle dynamic shapes
        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                if idx == 0:
                    out_shape[idx] = x.shape[0]
                elif idx == 2:  # height
                    out_shape[idx] = x.shape[2] * 8  # VAE decoder upsamples by 8
                elif idx == 3:  # width
                    out_shape[idx] = x.shape[3] * 8
                else:
                    out_shape[idx] = x.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch

        out = torch.empty(out_shape,
                          device=x.device,
                          dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(output_binding_name)))
        model_inputs_converted[output_binding_name] = out

        # Execute inference
        stream = torch.cuda.default_stream(x.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                tensor = model_inputs_converted[k]
                self.context.set_tensor_address(k, tensor[(tensor.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)

        return out

    def unload(self):
        engine_obj = self.engine
        self.engine = None
        if engine_obj is not None:
            del engine_obj
        context_obj = self.context
        self.context = None
        if context_obj is not None:
            del context_obj


class TrTVAE:
    """TensorRT VAE wrapper that combines encoder and decoder"""
    def __init__(self, encoder_path=None, decoder_path=None):
        self.encoder = TrTVAEEncoder(encoder_path, runtime) if encoder_path else None
        self.decoder = TrTVAEDecoder(decoder_path, runtime) if decoder_path else None
        
        # VAE properties
        self.device = comfy.model_management.get_torch_device()
        self.offload_device = comfy.model_management.vae_offload_device()
        self.dtype = torch.float16
        
    def encode(self, pixel_samples):
        """Encode images to latents"""
        if self.encoder is None:
            raise RuntimeError("No TensorRT encoder loaded")
        
        # Ensure input is on the right device and dtype
        x = pixel_samples.to(device=self.device, dtype=self.dtype)
        
        # VAE expects input in range [-1, 1], ComfyUI typically passes [0, 1]
        x = 2.0 * x - 1.0
        
        # Call TensorRT encoder
        latents = self.encoder(x)
        
        # Convert to float32 for compatibility with ComfyUI latent processing
        latents = latents.to(dtype=torch.float32)
        
        return latents

    def decode(self, samples):
        """Decode latents to images"""
        if self.decoder is None:
            raise RuntimeError("No TensorRT decoder loaded")
            
        # Ensure input is on the right device and dtype
        x = samples.to(device=self.device, dtype=self.dtype)
        
        # Call TensorRT decoder
        images = self.decoder(x)
        
        # Convert output from [-1, 1] to [0, 1] range
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
        
        # Convert to float32 for compatibility with ComfyUI image processing
        images = images.to(dtype=torch.float32)
        
        return images

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        """Tiled encoding for large images - fallback to regular encode for now"""
        return self.encode(pixel_samples)
        
    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16):
        """Tiled decoding for large images - fallback to regular decode for now"""
        return self.decode(samples)

    def get_sd(self):
        """Return empty state dict - TensorRT engines don't have traditional state dicts"""
        return {}

    def load_state_dict(self, sd):
        """No-op for TensorRT engines"""
        pass

    @property
    def memory_used_encode(self):
        encoder_size = self.encoder.size if self.encoder else 0
        return encoder_size

    @property 
    def memory_used_decode(self):
        decoder_size = self.decoder.size if self.decoder else 0
        return decoder_size

    def unload(self):
        if self.encoder:
            self.encoder.unload()
        if self.decoder:
            self.decoder.unload()


class TensorRTVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        engine_files = folder_paths.get_filename_list("tensorrt")
        # Filter for VAE engines
        vae_encoder_files = ["None"] + [f for f in engine_files if "encoder" in f.lower()]
        vae_decoder_files = ["None"] + [f for f in engine_files if "decoder" in f.lower()]
        
        return {
            "required": {
                "encoder_name": (vae_encoder_files,),
                "decoder_name": (vae_decoder_files,),
            },
        }
    
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "TensorRT"

    def load_vae(self, encoder_name, decoder_name):
        encoder_path = None
        decoder_path = None
        
        if encoder_name != "None":
            encoder_path = folder_paths.get_full_path("tensorrt", encoder_name)
            if encoder_path is None or not os.path.isfile(encoder_path):
                raise FileNotFoundError(f"Encoder file {encoder_name} does not exist")
                
        if decoder_name != "None":
            decoder_path = folder_paths.get_full_path("tensorrt", decoder_name)
            if decoder_path is None or not os.path.isfile(decoder_path):
                raise FileNotFoundError(f"Decoder file {decoder_name} does not exist")
        
        if encoder_path is None and decoder_path is None:
            raise ValueError("At least one of encoder or decoder must be specified")
            
        vae = TrTVAE(encoder_path, decoder_path)
        
        return (vae,)


NODE_CLASS_MAPPINGS = {
    "TensorRTVAELoader": TensorRTVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorRTVAELoader": "TensorRT VAE Loader",
} 