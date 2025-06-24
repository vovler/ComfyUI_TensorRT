import torch
import sys
import os
import time
import comfy.model_management
import comfy.sd
import folder_paths
from tqdm import tqdm
import comfy
import tensorrt as trt
from typing import Any, Optional
from .utils.tqdm_progress_monitor import TQDMProgressMonitor
from .utils.timing_cache import setup_timing_cache, save_timing_cache, get_timing_cache_path
from .tensorrt_model import get_tensorrt_manager
from .wrappers_for_onnx_convert.wrapper_vae_convert import VAEWrapper


def _convert_vae(
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
):
        # Save engines in VAE model folder
        vae_folders = folder_paths.get_folder_paths("vae")
        output_dir = vae_folders[0] if vae_folders else folder_paths.get_output_directory()
        temp_dir = folder_paths.get_temp_directory()
        timing_cache_path = get_timing_cache_path("vae")
        trt_manager = get_tensorrt_manager()
        
        output_onnx = os.path.normpath(
            os.path.join(
                os.path.join(temp_dir, "{}".format(time.time())), "vae_model.onnx"
            )
        )

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

        # TRT conversion starts here - Use centralized TensorRT manager
        network = trt_manager.create_network()
        parser = trt_manager.create_onnx_parser(network)
        success = parser.parse_from_file(output_onnx)
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))

        if not success:
            print("ONNX load ERROR")
            return {}

        config = trt_manager.create_builder_config()
        profile = trt_manager.builder.create_optimization_profile()
        setup_timing_cache(config, timing_cache_path)
        config.progress_monitor = TQDMProgressMonitor()

        # Set up optimization profile
        min_shape_list = [int(x) for x in inputs_shapes_min]
        opt_shape_list = [int(x) for x in inputs_shapes_opt]
        max_shape_list = [int(x) for x in inputs_shapes_max]
        
        min_shape = tuple(min_shape_list)
        opt_shape = tuple(opt_shape_list)
        max_shape = tuple(max_shape_list)
        
        profile.set_shape("x", min_shape, opt_shape, max_shape)


        #config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.set_flag(trt.BuilderFlag.DIRECT_IO)
        config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
        config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)
        config.max_aux_streams = 0
        config.builder_optimization_level = trt.BuilderOptimizationLevel.DEFAULT
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10 * 1024 * 1024 * 1024)
        config.add_optimization_profile(profile)

        # Generate filename
        vae_type = "encoder" if is_encoder else "decoder"

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

        serialized_engine = trt_manager.build_serialized_network(network, config)

        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, output_dir)
        )
        output_trt_engine = os.path.join(
            full_output_folder, f"{filename}_{counter:05}_.engine"
        )

        with open(output_trt_engine, "wb") as f:
            f.write(serialized_engine)

        save_timing_cache(config, timing_cache_path)

        vae_type = "encoder" if is_encoder else "decoder"
        print(f"TensorRT VAE {vae_type} conversion complete! Engine saved to: {output_trt_engine}")
        return


class VAE_DECODER_CONVERTER_TENSORRT_DYNAMIC:
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
        try:
            _convert_vae(
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
            )
        except Exception as e:
            print(f"VAE_DECODER_CONVERTER_TENSORRT_DYNAMIC - Error converting: {e}")
            return ()
        return ()





class VAE_ENCODER_CONVERTER_TENSORRT_DYNAMIC:
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
        try:
            _convert_vae(
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
            )
        except Exception as e:
            print(f"VAE_ENCODER_CONVERTER_TENSORRT_DYNAMIC - Error converting: {e}")
            return ()
        return ()


