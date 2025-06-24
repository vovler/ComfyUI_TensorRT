import torch
import sys
import os
import time
import psutil
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
import math

# Create TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def export_to_onnx(model, onnx_file_path, input_name, output_name, input_shape, dynamic_axes):
    """Export VAE model to ONNX format"""
    if os.path.exists(onnx_file_path):
        print(f"ONNX file already exists at {onnx_file_path}, skipping export.")
        return
    
    print(f"Exporting VAE model to ONNX at: {onnx_file_path}")
    model.eval()
    
    # Create dummy input with the specified shape
    dummy_input = torch.randn(input_shape, requires_grad=False, dtype=torch.float16).cuda()
    model = model.half().cuda()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(onnx_file_path), exist_ok=True)
    
    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_file_path, 
            export_params=True, 
            opset_version=17,
            do_constant_folding=True, 
            input_names=[input_name], 
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
    print("ONNX export complete.")


def build_engine(onnx_file_path, engine_file_path, min_shape, opt_shape, max_shape, input_name, timing_cache_data=None):
    """Builds a TensorRT engine using the centralized TensorRT manager, optimized for the current hardware and TRT version."""
    
    # Use centralized TensorRT manager
    trt_manager = get_tensorrt_manager()
    
    # Create network using centralized manager
    network = trt_manager.create_network()
    
    # Create ONNX parser using centralized manager
    parser = trt_manager.create_onnx_parser(network)
    
    # Create builder config using centralized manager
    config = trt_manager.create_builder_config()

    print("Configuring TensorRT builder for maximum performance on specific target...")

    #config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.DIRECT_IO)
    config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)
    config.hardware_compatibility_level = trt.HardwareCompatibilityLevel.NONE
    
    config.builder_optimization_level = 5
    if hasattr(trt, 'TilingOptimizationLevel'):
        config.tiling_optimization_level = trt.TilingOptimizationLevel.FULL

    # Memory pool limit - ensure it's a power of 2 and above minimum threshold
    try:
        available_system_mem = psutil.virtual_memory().available
        # Use 50% of available memory and round down to nearest power of 2
        target_mem = int(available_system_mem * 0.5)
        
        # Find the largest power of 2 that's <= target_mem
        
        power_of_2_mem = 2 ** int(math.log2(target_mem))
        
        # Ensure minimum size (1GB)
        min_mem_size = 1024 * 1024 * 1024  # 1GB
        tactic_dram_limit = max(power_of_2_mem, min_mem_size)
        
        config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM, tactic_dram_limit)
        print(f"Set memory pool limit to: {tactic_dram_limit / (1024*1024*1024):.1f} GB")
    except Exception as e:
        print(f"Warning: Could not set memory pool limit: {e}")
        # Continue without setting memory pool limit
    
    # Note: Weight streaming budget is set on the engine after loading, not during building
    config.max_aux_streams = 0
    
    # Progress monitor
    config.progress_monitor = TQDMProgressMonitor()
    
    # Create optimization profile using the centralized builder
    profile = trt_manager.builder.create_optimization_profile()
    profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    if timing_cache_data:
        timing_cache = config.create_timing_cache(timing_cache_data)
    else:
        timing_cache = config.create_timing_cache(b"")
    config.set_timing_cache(timing_cache, ignore_mismatch=False)
    
    print(f"Parsing ONNX model from: {onnx_file_path}")
    success = parser.parse_from_file(onnx_file_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        print("ONNX parsing failed")
        return None, None

    print("Building serialized network...")
    # Use centralized manager to build the serialized network
    serialized_engine = trt_manager.build_serialized_network(network, config)
    
    if serialized_engine is None: 
        print("Failed to build serialized network")
        return None, None
        
    print("Engine build successful.")
    
    # Save engine to file
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    # Return timing cache data for saving
    timing_cache_serialized = timing_cache.serialize()
    return engine_file_path, timing_cache_serialized


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

    input_name = "x"
    output_name = "output"
    
    dynamic_axes = {
        input_name: {0: "batch", 2: "height", 3: "width"},
        output_name: {0: "batch", 2: "height", 3: "width"},
    }

    # Export to ONNX using the template function
    export_to_onnx(
        model=vae_wrapper,
        onnx_file_path=output_onnx,
        input_name=input_name,
        output_name=output_name,
        input_shape=inputs_shapes_opt,
        dynamic_axes=dynamic_axes
    )

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

    # Get output engine path
    full_output_folder, filename, counter, subfolder, filename_prefix = (
        folder_paths.get_save_image_path(filename_prefix, output_dir)
    )
    output_trt_engine = os.path.join(
        full_output_folder, f"{filename}_{counter:05}_.engine"
    )

    # Load existing timing cache if available
    timing_cache_data = None
    if os.path.exists(timing_cache_path):
        with open(timing_cache_path, 'rb') as f:
            timing_cache_data = f.read()
            print(f"Loaded timing cache from: {timing_cache_path}")

    # Build engine using the template function with centralized TensorRT manager
    engine_path, timing_cache_serialized = build_engine(
        onnx_file_path=output_onnx,
        engine_file_path=output_trt_engine,
        min_shape=inputs_shapes_min,
        opt_shape=inputs_shapes_opt,
        max_shape=inputs_shapes_max,
        input_name=input_name,
        timing_cache_data=timing_cache_data
    )

    if engine_path and timing_cache_serialized:
        # Save timing cache
        os.makedirs(os.path.dirname(timing_cache_path), exist_ok=True)
        with open(timing_cache_path, 'wb') as f:
            f.write(timing_cache_serialized)
        print(f"Saved timing cache to: {timing_cache_path}")

        vae_type = "encoder" if is_encoder else "decoder"
        print(f"TensorRT VAE {vae_type} conversion complete! Engine saved to: {engine_path}")
    else:
        print(f"Failed to build TensorRT engine for VAE {vae_type}")

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
            print("WE ARE HERE 2")
        except Exception as e:
            print(f"VAE_ENCODER_CONVERTER_TENSORRT_DYNAMIC - Error converting: {e}")
            return ()
        return ()


