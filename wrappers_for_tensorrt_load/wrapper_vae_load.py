# pyright: reportOptionalMemberAccess=false

import torch
import os
import comfy.model_management
from ..utils.tensorrt_error_recorder import TrTErrorRecorder, check_for_trt_errors
from ..utils.trt_datatype_to_torch import trt_datatype_to_torch
from ..tensorrt_model import get_tensorrt_manager
import tensorrt as trt


class TrTVAEEncoder:
    """TensorRT VAE Encoder"""
    def __init__(self, engine_path, runtime):
        self.dtype = torch.float16
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self._size = int(os.stat(engine_path).st_size)
        self.runtime = runtime
        self.trt_manager = get_tensorrt_manager()

    def load(self):
        if self.engine is not None or self.context is not None:
            return
        
        # Use centralized TensorRT manager for engine loading
        self.engine = self.trt_manager.deserialize_engine(self.engine_path, f"vae_encoder_{id(self)}")
        
        # Create context using centralized manager
        self.context = self.trt_manager.create_execution_context(self.engine, f"vae_encoder_ctx_{id(self)}")

    @property
    def size(self) -> int:
        return self._size

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    def __call__(self, x):
        print(f"TrTVAEEncoder.__call__ - Input shape: {x.shape}, dtype: {x.dtype}")
        self.load()  # Ensure engine is loaded
        
        model_inputs = {"x": x}
        batch_size = x.shape[0]
        print(f"TrTVAEEncoder - batch_size: {batch_size}")
        
        # Handle batch splitting for dynamic profiles
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        min_batch = dims[0][0]
        opt_batch = dims[1][0]
        max_batch = dims[2][0]
        print(f"TrTVAEEncoder - Profile batches: min={min_batch}, opt={opt_batch}, max={max_batch}")

        curr_split_batch = 1
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break
        print(f"TrTVAEEncoder - curr_split_batch: {curr_split_batch}")

        self.set_bindings_shape(model_inputs, curr_split_batch)

        # Convert inputs to appropriate data types
        model_inputs_converted = {}
        for k in model_inputs:
            data_type = self.engine.get_tensor_dtype(k)
            model_inputs_converted[k] = model_inputs[k].to(dtype=trt_datatype_to_torch(data_type))
            print(f"TrTVAEEncoder - Input '{k}' converted to dtype: {trt_datatype_to_torch(data_type)}")

        # Get output tensor info
        output_binding_name = self.engine.get_tensor_name(len(model_inputs))
        out_shape = self.engine.get_tensor_shape(output_binding_name)
        out_shape = list(out_shape)
        print(f"TrTVAEEncoder - Initial output shape from engine: {out_shape}")

        # Handle dynamic shapes
        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                if idx == 0:  # batch
                    out_shape[idx] = x.shape[0]
                elif idx == 1:  # channels - VAE encoder outputs 4 channels (latents)
                    out_shape[idx] = 4
                elif idx == 2:  # height
                    out_shape[idx] = x.shape[2] // 8  # VAE encoder downsamples by 8
                elif idx == 3:  # width
                    out_shape[idx] = x.shape[3] // 8
                else:
                    out_shape[idx] = x.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch
        print(f"TrTVAEEncoder - Final output shape after dynamic handling: {out_shape}")

        out = torch.empty(out_shape,
                          device=x.device,
                          dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(output_binding_name)))
        model_inputs_converted[output_binding_name] = out
        print(f"TrTVAEEncoder - Created output tensor: shape={out.shape}, dtype={out.dtype}")

        # Execute inference
        stream = torch.cuda.default_stream(x.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                tensor = model_inputs_converted[k]
                self.context.set_tensor_address(k, tensor[(tensor.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)

        print(f"TrTVAEEncoder - Inference complete, returning tensor: {out.shape}")
        return out

    def unload(self):
        if self.context is not None:
            self.trt_manager.unload_context(f"vae_encoder_ctx_{id(self)}")
            self.context = None
        if self.engine is not None:
            self.trt_manager.unload_engine(f"vae_encoder_{id(self)}")
            self.engine = None


class TrTVAEDecoder:
    """TensorRT VAE Decoder"""
    def __init__(self, engine_path, runtime):
        self.dtype = torch.float16
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self._size = int(os.stat(engine_path).st_size)
        self.runtime = runtime
        self.trt_manager = get_tensorrt_manager()

    def load(self):
        if self.engine is not None or self.context is not None:
            return
        
        # Use centralized TensorRT manager for engine loading
        self.engine = self.trt_manager.deserialize_engine(self.engine_path, f"vae_decoder_{id(self)}")
        
        # Create context using centralized manager
        self.context = self.trt_manager.create_execution_context(self.engine, f"vae_decoder_ctx_{id(self)}")

    @property
    def size(self) -> int:
        return self._size

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    def __call__(self, x):
        print(f"TrTVAEDecoder.__call__ - Input shape: {x.shape}, dtype: {x.dtype}")
        self.load()  # Ensure engine is loaded
        
        model_inputs = {"x": x}
        batch_size = x.shape[0]
        print(f"TrTVAEDecoder - batch_size: {batch_size}")
        
        # Handle batch splitting for dynamic profiles
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        min_batch = dims[0][0]
        opt_batch = dims[1][0]
        max_batch = dims[2][0]
        print(f"TrTVAEDecoder - Profile batches: min={min_batch}, opt={opt_batch}, max={max_batch}")

        curr_split_batch = 1
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break
        print(f"TrTVAEDecoder - curr_split_batch: {curr_split_batch}")

        self.set_bindings_shape(model_inputs, curr_split_batch)

        # Convert inputs to appropriate data types
        model_inputs_converted = {}
        for k in model_inputs:
            data_type = self.engine.get_tensor_dtype(k)
            model_inputs_converted[k] = model_inputs[k].to(dtype=trt_datatype_to_torch(data_type))
            print(f"TrTVAEDecoder - Input '{k}' converted to dtype: {trt_datatype_to_torch(data_type)}")

        # Get output tensor info
        output_binding_name = self.engine.get_tensor_name(len(model_inputs))
        out_shape = self.engine.get_tensor_shape(output_binding_name)
        out_shape = list(out_shape)
        print(f"TrTVAEDecoder - Initial output shape from engine: {out_shape}")

        # Handle dynamic shapes
        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                if idx == 0:  # batch
                    out_shape[idx] = x.shape[0]
                elif idx == 1:  # channels - VAE decoder outputs 3 channels (RGB)
                    out_shape[idx] = 3
                elif idx == 2:  # height
                    out_shape[idx] = x.shape[2] * 8  # VAE decoder upsamples by 8
                elif idx == 3:  # width
                    out_shape[idx] = x.shape[3] * 8
                else:
                    out_shape[idx] = x.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch
        print(f"TrTVAEDecoder - Final output shape after dynamic handling: {out_shape}")

        out = torch.empty(out_shape,
                          device=x.device,
                          dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(output_binding_name)))
        model_inputs_converted[output_binding_name] = out
        print(f"TrTVAEDecoder - Created output tensor: shape={out.shape}, dtype={out.dtype}")

        # Execute inference
        stream = torch.cuda.default_stream(x.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                tensor = model_inputs_converted[k]
                self.context.set_tensor_address(k, tensor[(tensor.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)

        print(f"TrTVAEDecoder - Inference complete, returning tensor: {out.shape}")
        return out

    def unload(self):
        if self.context is not None:
            self.trt_manager.unload_context(f"vae_decoder_ctx_{id(self)}")
            self.context = None
        if self.engine is not None:
            self.trt_manager.unload_engine(f"vae_decoder_{id(self)}")
            self.engine = None


class TrTVAE:
    """TensorRT VAE wrapper that combines encoder and decoder"""
    def __init__(self, encoder_path=None, decoder_path=None, runtime=None):
        if runtime is None:
            raise ValueError("Runtime parameter is required for TrTVAE")
            
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
        latents = latents.to(dtype=torch.float16)
        
        return latents

    def decode(self, samples):
        """Decode latents to images"""
        if self.decoder is None:
            raise RuntimeError("No TensorRT decoder loaded")
            
        # Ensure input is on the right device and dtype
        x = samples.to(device=self.device, dtype=self.dtype)
        
        # Call TensorRT decoder
        images = self.decoder(x)
        print(f"TensorRT VAE Decoder output - shape: {images.shape}, dtype: {images.dtype}, device: {images.device}")
        
        # Check for invalid values immediately after TensorRT inference
        if torch.isnan(images).any():
            print("WARNING: NaN values detected in TensorRT VAE decoder output!")
        if torch.isinf(images).any():
            print("WARNING: Infinite values detected in TensorRT VAE decoder output!")
        
        # Convert to float32 FIRST to avoid precision issues with fp16
        images = images.to(dtype=torch.float16)
        
        # Handle any remaining invalid values after conversion
        if torch.isnan(images).any():
            print("Warning: NaN values detected after dtype conversion, replacing with zeros")
            images = torch.nan_to_num(images, nan=0.0)
        
        if torch.isinf(images).any():
            print("Warning: Infinite values detected after dtype conversion, clamping")
            images = torch.clamp(images, -10.0, 10.0)
        
        # Convert output from [-1, 1] to [0, 1] range
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
        
        # Final check for invalid values after conversion
        if torch.isnan(images).any() or torch.isinf(images).any():
            print("Warning: Invalid values after conversion, clamping to valid range")
            images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
            images = torch.clamp(images, 0.0, 1.0)
        
        # Convert from (batch, channels, height, width) to (batch, height, width, channels) for ComfyUI
        images = images.permute(0, 2, 3, 1)
        print(f"After permute to BHWC format: {images.shape}")
        
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