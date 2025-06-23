import torch
import sys
import os
import time
import traceback
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


class TRT_CLIP_CONVERSION_BASE:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.timing_cache_path = os.path.normpath(
            os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "timing_cache_clip.trt"))
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
            print("Read {} bytes from CLIP timing cache.".format(len(buffer)))
        else:
            print("No CLIP timing cache found; Initializing a new one.")
        timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    # Saves the config's timing cache to file
    def _save_timing_cache(self, config: trt.IBuilderConfig):
        timing_cache: trt.ITimingCache = config.get_timing_cache()
        with open(self.timing_cache_path, "wb") as timing_cache_file:
            timing_cache_file.write(memoryview(timing_cache.serialize()))

    def _convert_clip(
        self,
        clip,
        filename_prefix,
        batch_size_min,
        batch_size_opt,
        batch_size_max,
        sequence_length_min,
        sequence_length_opt,
        sequence_length_max,
        is_clip_l: bool,
        is_static: bool,
    ):
        output_onnx = os.path.normpath(
            os.path.join(
                os.path.join(self.temp_dir, "{}".format(time.time())), "clip_model.onnx"
            )
        )

        comfy.model_management.unload_all_models()
        comfy.model_management.load_models_gpu([clip.load_model()], force_patch_weights=True, force_full_load=True)
        
        clip_model = clip.cond_stage_model
        device = comfy.model_management.get_torch_device()
        dtype = torch.float16
        
        # Get the correct CLIP component based on model structure
        if is_clip_l:
            if hasattr(clip_model, 'clip_l'):
                clip_component = clip_model.clip_l
            elif hasattr(clip_model, 'clip') and hasattr(clip_model, 'clip_name') and clip_model.clip_name == "l":
                clip_component = getattr(clip_model, clip_model.clip)
            else:
                clip_component = clip_model
        else:
            if hasattr(clip_model, 'clip_g'):
                clip_component = clip_model.clip_g
            elif hasattr(clip_model, 'clip') and hasattr(clip_model, 'clip_name') and clip_model.clip_name == "g":
                clip_component = getattr(clip_model, clip_model.clip)
            else:
                clip_component = clip_model
        
        clip_component = clip_component.to(device=device, dtype=dtype)
        model_to_export = clip_component

        # Input shapes for CLIP (token sequences)
        inputs_shapes_min = (batch_size_min, sequence_length_min)
        inputs_shapes_opt = (batch_size_opt, sequence_length_opt)
        inputs_shapes_max = (batch_size_max, sequence_length_max)

        input_names = ["tokens"]
        output_names = ["hidden_states", "pooled_output"]
        dynamic_axes = {
            "tokens": {0: "batch", 1: "sequence"},
            "hidden_states": {0: "batch", 1: "sequence"},
            "pooled_output": {0: "batch"},
        }

        # Create preprocessed tokens as tensor directly
        # Instead of using the complex token processing, create tensor input directly
        batch_size, seq_len = inputs_shapes_opt
        
        # Create token tensor directly (bypassing the complex token processing)
        token_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        for b in range(batch_size):
            token_tensor[b, 0] = 49406  # start token
            token_tensor[b, 1:-1] = 49407  # pad tokens  
            token_tensor[b, -1] = 49407  # end token

        # Get embeddings directly from the transformer
        input_embeddings = clip_component.transformer.get_input_embeddings()
        embedded_tokens = input_embeddings(token_tensor, out_dtype=torch.float32)

        os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
        
        # Create a wrapper model that takes embeddings directly
        class ClipEmbedWrapper(torch.nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip_model = clip_model
                
            def forward(self, embeds):
                # Call transformer directly with embeddings
                num_tokens = [embeds.shape[1]] * embeds.shape[0]
                outputs = self.clip_model.transformer(
                    None, 
                    None,  # attention_mask_model
                    embeds=embeds, 
                    num_tokens=num_tokens, 
                    intermediate_output=None,
                    final_layer_norm_intermediate=self.clip_model.layer_norm_hidden_state,
                    dtype=torch.float32
                )
                
                z = outputs[0].float()
                pooled_output = None
                if len(outputs) >= 3 and outputs[2] is not None:
                    pooled_output = outputs[2].float()
                elif len(outputs) >= 4 and outputs[3] is not None:
                    pooled_output = outputs[3].float()
                
                if pooled_output is None:
                    pooled_output = torch.zeros((z.shape[0], z.shape[-1]), dtype=z.dtype, device=z.device)
                
                return z, pooled_output

        wrapper_model = ClipEmbedWrapper(clip_component)
        
        with torch.no_grad():
            torch.onnx.export(
                wrapper_model,
                (embedded_tokens,),
                output_onnx,
                verbose=False,
                input_names=["embeddings"],
                output_names=output_names,
                opset_version=17,
                dynamic_axes={
                    "embeddings": {0: "batch", 1: "sequence"},
                    "hidden_states": {0: "batch", 1: "sequence"},
                    "pooled_output": {0: "batch"},
                },
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

        # Set up optimization profile - embeddings have shape [batch, sequence, embedding_dim]
        embedding_dim = embedded_tokens.shape[-1]  # Get embedding dimension from the actual tensor
        min_shape_list = [int(x) for x in inputs_shapes_min] + [embedding_dim]
        opt_shape_list = [int(x) for x in inputs_shapes_opt] + [embedding_dim]
        max_shape_list = [int(x) for x in inputs_shapes_max] + [embedding_dim]
        
        min_shape = trt.Dims(min_shape_list)
        opt_shape = trt.Dims(opt_shape_list)
        max_shape = trt.Dims(max_shape_list)
        
        profile.set_shape("embeddings", min_shape, opt_shape, max_shape)

        config.set_flag(trt.BuilderFlag.FP16)
        config.add_optimization_profile(profile)

        # Generate filename
        clip_type = "clip_l" if is_clip_l else "clip_g"
        if is_static:
            filename_prefix = "{}_{}${}".format(
                filename_prefix,
                clip_type,
                "-".join(
                    (
                        "stat",
                        "b",
                        str(batch_size_opt),
                        "s",
                        str(sequence_length_opt),
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

        clip_type = "CLIP-L" if is_clip_l else "CLIP-G"
        print(f"TensorRT {clip_type} conversion complete! Engine saved to: {output_trt_engine}")
        return {}


class STATIC_TRT_CLIP_L_CONVERSION(TRT_CLIP_CONVERSION_BASE):
    def __init__(self):
        super(STATIC_TRT_CLIP_L_CONVERSION, self).__init__()

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_CLIP_L_STAT"}),
                "batch_size_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "sequence_length_opt": (
                    "INT",
                    {
                        "default": 77,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                    },
                ),
            },
        }

    def convert(
        self,
        clip,
        filename_prefix,
        batch_size_opt,
        sequence_length_opt,
    ):
        try:
            super()._convert_clip(
                clip,
                filename_prefix,
                batch_size_opt,
                batch_size_opt,
                batch_size_opt,
                sequence_length_opt,
                sequence_length_opt,
                sequence_length_opt,
                is_clip_l=True,
                is_static=True,
            )
        except Exception as e:
            print(f"STATIC_TRT_CLIP_L_CONVERSION - Error converting: {e}")
            print(f"STATIC_TRT_CLIP_L_CONVERSION - Full stack trace:")
            traceback.print_exc()
            return ()
        return ()


class STATIC_TRT_CLIP_G_CONVERSION(TRT_CLIP_CONVERSION_BASE):
    def __init__(self):
        super(STATIC_TRT_CLIP_G_CONVERSION, self).__init__()

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_CLIP_G_STAT"}),
                "batch_size_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "sequence_length_opt": (
                    "INT",
                    {
                        "default": 77,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                    },
                ),
            },
        }

    def convert(
        self,
        clip,
        filename_prefix,
        batch_size_opt,
        sequence_length_opt,
    ):
        try:
            super()._convert_clip(
                clip,
                filename_prefix,
                batch_size_opt,
                batch_size_opt,
                batch_size_opt,
                sequence_length_opt,
                sequence_length_opt,
                sequence_length_opt,
                is_clip_l=False,
                is_static=True,
            )
        except Exception as e:
            print(f"STATIC_TRT_CLIP_G_CONVERSION - Error converting: {e}")
            print(f"STATIC_TRT_CLIP_G_CONVERSION - Full stack trace:")
            traceback.print_exc()
            return ()
        return ()