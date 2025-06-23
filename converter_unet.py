import torch
import sys
import os
import time
import comfy.model_management
import folder_paths
from tqdm import tqdm
import comfy
from typing import Any, Optional
from .utils.tqdm_progress_monitor import TQDMProgressMonitor
from .utils.timing_cache import setup_timing_cache, save_timing_cache, get_timing_cache_path
from .wrappers_for_onnx_convert.wrapper_unet_convert import UNETWrapper
from .tensorrt_model import get_tensorrt_manager


def _convert_model(
    model,
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
    context_min,
    context_opt,
    context_max,
):
        output_dir = folder_paths.get_output_directory()
        temp_dir = folder_paths.get_temp_directory()
        timing_cache_path = get_timing_cache_path("unet")
        trt_manager = get_tensorrt_manager()
        
        output_onnx = os.path.normpath(
            os.path.join(
                os.path.join(temp_dir, "{}".format(time.time())), "model.onnx"
            )
        )

        comfy.model_management.load_models_gpu([model], force_patch_weights=True, force_full_load=True)
        unet = model.model.diffusion_model

        context_dim = model.model.model_config.unet_config.get("context_dim", None)
        context_len = 77
        context_len_min = context_len
        y_dim = model.model.adm_channels
        extra_input = {}
        dtype = torch.float16

        if context_dim is None:
            raise Exception("Context dimension is not set in the model config")

        
        input_names = ["x", "timesteps", "context"]
        output_names = ["h"]

        dynamic_axes = {
            "x": {0: "batch", 2: "height", 3: "width"},
            "timesteps": {0: "batch"},
            "context": {0: "batch", 1: "num_embeds"},
        }

        

        

        input_channels = model.model.model_config.unet_config.get("in_channels", 4)

        inputs_shapes_min = (
                (batch_size_min, input_channels, height_min // 8, width_min // 8),
                (batch_size_min,),
                (batch_size_min, context_len_min * context_min, context_dim),
        )
        inputs_shapes_opt = (
                (batch_size_opt, input_channels, height_opt // 8, width_opt // 8),
                (batch_size_opt,),
                (batch_size_opt, context_len * context_opt, context_dim),
        )
        inputs_shapes_max = (
                (batch_size_max, input_channels, height_max // 8, width_max // 8),
                (batch_size_max,),
                (batch_size_max, context_len * context_max, context_dim),
        )

        if y_dim > 0:
            input_names.append("y")
            dynamic_axes["y"] = {0: "batch"}
            inputs_shapes_min += ((batch_size_min, y_dim),)
            inputs_shapes_opt += ((batch_size_opt, y_dim),)
            inputs_shapes_max += ((batch_size_max, y_dim),)

        for k in extra_input:
            input_names.append(k)
            dynamic_axes[k] = {0: "batch"}
            inputs_shapes_min += ((batch_size_min,) + extra_input[k],)
            inputs_shapes_opt += ((batch_size_opt,) + extra_input[k],)
            inputs_shapes_max += ((batch_size_max,) + extra_input[k],)


        inputs = ()
        for shape in inputs_shapes_opt:
            inputs += (
                torch.zeros(
                    shape,
                    device=comfy.model_management.get_torch_device(),
                    dtype=dtype,
                ),
            )

        transformer_options = model.model.model_config.unet_config.get("transformer_options", {})
        unet = UNETWrapper(unet, transformer_options, input_names[3:])

        
        os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(
                unet,
                inputs,
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

        prefix_encode = ""
        for k in range(len(input_names)):
            
            min_shape_list = [int(x) for x in inputs_shapes_min[k]]
            opt_shape_list = [int(x) for x in inputs_shapes_opt[k]]
            max_shape_list = [int(x) for x in inputs_shapes_max[k]]
            
            min_shape = tuple(min_shape_list)
            opt_shape = tuple(opt_shape_list)
            max_shape = tuple(max_shape_list)
            
            profile.set_shape(input_names[k], min_shape, opt_shape, max_shape)

            # Encode shapes to filename
            encode = lambda a: ".".join(map(lambda x: str(x), a))
            prefix_encode += "{}#{}#{}#{};".format(
                input_names[k], encode(min_shape_list), encode(opt_shape_list), encode(max_shape_list)
            )

        config.set_flag(trt_manager.builder.BuilderFlag.FP16)

        config.add_optimization_profile(profile)

        
        filename_prefix = "{}_${}".format(
            filename_prefix,
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

        print(f"TensorRT UNet conversion complete! Engine saved to: {output_trt_engine}")
        return {}


class UNET_CONVERTER_TENSORRT_DYNAMIC:
    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_DYN"}),
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
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "height_max": (
                    "INT",
                    {
                        "default": 512,
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
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_max": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "context_min": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                    },
                ),
                "context_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                    },
                ),
                "context_max": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                    },
                ),
            },
        }

    def convert(
        self,
        model,
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
        context_min,
        context_opt,
        context_max,
    ):
        try:
            _convert_model(
                model,
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
                context_min,
                context_opt,
                context_max,
            )
        except Exception as e:
            print(f"DYNAMIC_TRT_MODEL_CONVERSION - Error converting: {e}")
            return ()
        return ()


