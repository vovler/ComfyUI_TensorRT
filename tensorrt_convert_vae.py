import torch
import sys
import os
import time
import comfy.model_management
import comfy.sd
import tensorrt as trt
import folder_paths
from tqdm import tqdm
import comfy
from typing import Any, Optional
from .utils.tensorrt_error_recorder import TrTErrorRecorder
from .utils.tqdm_progress_monitor import TQDMProgressMonitor


# add output directory to tensorrt search path
if "tensorrt" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tensorrt"][0].append(
        os.path.join(folder_paths.get_output_directory(), "tensorrt")
    )
    folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
else:
    folder_paths.folder_names_and_paths["tensorrt"] = (
        [os.path.join(folder_paths.get_output_directory(), "tensorrt")],
        {".engine"},
    )


class VAEWrapper(torch.nn.Module):
    """Wrapper for VAE encoder/decoder to make it compatible with ONNX export"""
    def __init__(self, vae_model, is_encoder=False):
        super().__init__()
        self.vae = vae_model
        self.is_encoder = is_encoder

    def forward(self, x):
        if self.is_encoder:
            return self.vae.encode(x)
        else:
            return self.vae.decode(x)


class TRT_VAE_CONVERSION_BASE:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.timing_cache_path = os.path.normpath(
            os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "timing_cache_vae.trt"))
        )

    @classmethod
    def INPUT_TYPES(cls):
        raise NotImplementedError

    # Sets up the builder to use the timing cache file, and creates it if it does not already exist
    def _setup_timing_cache(self, config: trt.IBuilderConfig):
        buffer = b""
        if os.path.exists(self.timing_cache_path):
            with open(self.timing_cache_path, mode="rb") as timing_cache_file:
                buffer = timing_cache_file.read()
            print("Read {} bytes from VAE timing cache.".format(len(buffer)))
        else:
            print("No VAE timing cache found; Initializing a new one.")
        timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    # Saves the config's timing cache to file
    def _save_timing_cache(self, config: trt.IBuilderConfig):
        timing_cache: trt.ITimingCache = config.get_timing_cache()
        with open(self.timing_cache_path, "wb") as timing_cache_file:
            timing_cache_file.write(memoryview(timing_cache.serialize()))

    def _convert_vae(
        self,
        vae,
        filename_prefix,
        batch_size_min,
        batch_size_opt,
        batch_size_max,
        height_min,
        height_opt,
        height_max,
        width_min,
        width_opt,
        width_max,
        is_encoder: bool,
        is_static: bool,
    ):
        output_onnx = os.path.normpath(
            os.path.join(
                os.path.join(self.temp_dir, "{}".format(time.time())), "vae_model.onnx"
            )
        )

        comfy.model_management.unload_all_models()
        
        # Load VAE to GPU
        vae_model = vae.first_stage_model
        device = comfy.model_management.get_torch_device()
        
        # Force fp16 for all VAE components to ensure consistency and avoid NaN issues
        dtype = torch.float16
        print(f"Forcing VAE dtype to: {dtype}")
        
        # Move VAE to device and ensure fp16 dtype
        vae_model = vae_model.to(device=device, dtype=dtype)

        # Create wrapper for the VAE part we want to convert
        if is_encoder:
            vae_wrapper = VAEWrapper(vae_model, is_encoder=True)
            # For encoder: input is images (3 channels), output is latents (typically 4 channels for SDXL)
            input_channels = 3
            # Input shapes for encoder (image space)
            inputs_shapes_min = (batch_size_min, input_channels, height_min, width_min)
            inputs_shapes_opt = (batch_size_opt, input_channels, height_opt, width_opt)
            inputs_shapes_max = (batch_size_max, input_channels, height_max, width_max)
        else:
            vae_wrapper = VAEWrapper(vae_model, is_encoder=False)
            # For decoder: input is latents (typically 4 channels), output is images (3 channels)
            input_channels = 4  # SDXL VAE latent channels
            # Input shapes for decoder (latent space - 8x smaller than image space)
            inputs_shapes_min = (batch_size_min, input_channels, height_min // 8, width_min // 8)
            inputs_shapes_opt = (batch_size_opt, input_channels, height_opt // 8, width_opt // 8)
            inputs_shapes_max = (batch_size_max, input_channels, height_max // 8, width_max // 8)

        input_names = ["x"]
        output_names = ["output"]
        
        dynamic_axes = {
            "x": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

        # Create input tensor for ONNX export
        input_tensor = torch.zeros(
            inputs_shapes_opt,
            device=device,
            dtype=dtype,
        )

        os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                vae_wrapper,
                (input_tensor,),
                output_onnx,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
                dynamic_axes=dynamic_axes,
            )

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        # TRT conversion starts here
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        builder.error_recorder = TrTErrorRecorder()

        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        success = parser.parse_from_file(output_onnx)
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))

        if not success:
            print("ONNX load ERROR")
            return {}

        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        self._setup_timing_cache(config)
        config.progress_monitor = TQDMProgressMonitor()

        # Set up optimization profile
        min_shape_list = [int(x) for x in inputs_shapes_min]
        opt_shape_list = [int(x) for x in inputs_shapes_opt]
        max_shape_list = [int(x) for x in inputs_shapes_max]
        
        min_shape = trt.Dims(min_shape_list)
        opt_shape = trt.Dims(opt_shape_list)
        max_shape = trt.Dims(max_shape_list)
        
        profile.set_shape("x", min_shape, opt_shape, max_shape)

        config.set_flag(trt.BuilderFlag.FP16)

        config.add_optimization_profile(profile)

        # Generate filename
        vae_type = "encoder" if is_encoder else "decoder"
        if is_static:
            filename_prefix = "{}_{}${}".format(
                filename_prefix,
                vae_type,
                "-".join(
                    (
                        "stat",
                        "b",
                        str(batch_size_opt),
                        "h",
                        str(height_opt),
                        "w",
                        str(width_opt),
                    )
                ),
            )
        else:
            filename_prefix = "{}_{}${}".format(
                filename_prefix,
                vae_type,
                "-".join(
                    (
                        "dyn",
                        "b",
                        str(batch_size_min),
                        str(batch_size_max),
                        str(batch_size_opt),
                        "h",
                        str(height_min),
                        str(height_max),
                        str(height_opt),
                        "w",
                        str(width_min),
                        str(width_max),
                        str(width_opt),
                    )
                ),
            )

        serialized_engine = builder.build_serialized_network(network, config)

        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        )
        output_trt_engine = os.path.join(
            full_output_folder, f"{filename}_{counter:05}_.engine"
        )

        with open(output_trt_engine, "wb") as f:
            f.write(serialized_engine)

        self._save_timing_cache(config)

        vae_type = "encoder" if is_encoder else "decoder"
        print(f"TensorRT VAE {vae_type} conversion complete! Engine saved to: {output_trt_engine}")
        return {}


class DYNAMIC_TRT_VAE_DECODER_CONVERSION(TRT_VAE_CONVERSION_BASE):
    def __init__(self):
        super(DYNAMIC_TRT_VAE_DECODER_CONVERSION, self).__init__()

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_VAE_DEC_DYN"}),
                "batch_size_min": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "batch_size_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "batch_size_max": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "height_min": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "height_opt": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "height_max": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_min": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_opt": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_max": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
            },
        }

    def convert(
        self,
        vae,
        filename_prefix,
        batch_size_min,
        batch_size_opt,
        batch_size_max,
        height_min,
        height_opt,
        height_max,
        width_min,
        width_opt,
        width_max,
    ):
        super()._convert_vae(
            vae,
            filename_prefix,
            batch_size_min,
            batch_size_opt,
            batch_size_max,
            height_min,
            height_opt,
            height_max,
            width_min,
            width_opt,
            width_max,
            is_encoder=False,
            is_static=False,
        )
        return ()


class STATIC_TRT_VAE_DECODER_CONVERSION(TRT_VAE_CONVERSION_BASE):
    def __init__(self):
        super(STATIC_TRT_VAE_DECODER_CONVERSION, self).__init__()

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_VAE_DEC_STAT"}),
                "batch_size_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "height_opt": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_opt": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
            },
        }

    def convert(
        self,
        vae,
        filename_prefix,
        batch_size_opt,
        height_opt,
        width_opt,
    ):
        super()._convert_vae(
            vae,
            filename_prefix,
            batch_size_opt,
            batch_size_opt,
            batch_size_opt,
            height_opt,
            height_opt,
            height_opt,
            width_opt,
            width_opt,
            width_opt,
            is_encoder=False,
            is_static=True,
        )
        return ()


class DYNAMIC_TRT_VAE_ENCODER_CONVERSION(TRT_VAE_CONVERSION_BASE):
    def __init__(self):
        super(DYNAMIC_TRT_VAE_ENCODER_CONVERSION, self).__init__()

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_VAE_ENC_DYN"}),
                "batch_size_min": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "batch_size_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "batch_size_max": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "height_min": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "height_opt": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "height_max": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_min": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_opt": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_max": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
            },
        }

    def convert(
        self,
        vae,
        filename_prefix,
        batch_size_min,
        batch_size_opt,
        batch_size_max,
        height_min,
        height_opt,
        height_max,
        width_min,
        width_opt,
        width_max,
    ):
        super()._convert_vae(
            vae,
            filename_prefix,
            batch_size_min,
            batch_size_opt,
            batch_size_max,
            height_min,
            height_opt,
            height_max,
            width_min,
            width_opt,
            width_max,
            is_encoder=True,
            is_static=False,
        )
        return ()


class STATIC_TRT_VAE_ENCODER_CONVERSION(TRT_VAE_CONVERSION_BASE):
    def __init__(self):
        super(STATIC_TRT_VAE_ENCODER_CONVERSION, self).__init__()

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_VAE_ENC_STAT"}),
                "batch_size_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "height_opt": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_opt": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
            },
        }

    def convert(
        self,
        vae,
        filename_prefix,
        batch_size_opt,
        height_opt,
        width_opt,
    ):
        super()._convert_vae(
            vae,
            filename_prefix,
            batch_size_opt,
            batch_size_opt,
            batch_size_opt,
            height_opt,
            height_opt,
            height_opt,
            width_opt,
            width_opt,
            width_opt,
            is_encoder=True,
            is_static=True,
        )
        return ()