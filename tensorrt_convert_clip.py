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


class CLIPWrapper(torch.nn.Module):
    """Wrapper that calls the underlying transformer directly, bypassing ComfyUI's complex preprocessing"""
    def __init__(self, clip_model, is_clip_l=True):
        super().__init__()
        
        # Extract the underlying transformer from ComfyUI's wrapper
        if hasattr(clip_model, 'transformer'):
            # For SDClipModel, extract the CLIPTextModel
            self.transformer = clip_model.transformer
            print(f"CLIPWrapper: Found transformer in clip_model: {type(self.transformer)}")
        else:
            # Fallback - try to use the model directly
            self.transformer = clip_model
            print(f"CLIPWrapper: Using clip_model directly as transformer: {type(self.transformer)}")
            
        self.is_clip_l = is_clip_l
        
        # Get model-specific info for debugging
        if hasattr(self.transformer, 'text_model'):
            self.hidden_size = self.transformer.text_model.embeddings.token_embedding.weight.shape[1]
            print(f"CLIPWrapper: CLIP {'L' if is_clip_l else 'G'} - Hidden size: {self.hidden_size}")
        else:
            print(f"CLIPWrapper: Warning - could not determine hidden size")

    def forward(self, tokens):
        """
        Call the transformer directly with minimal arguments
        This bypasses all of ComfyUI's complex preprocessing
        """
        
        # Call CLIPTextModel.forward() which calls CLIPTextModel_.forward()
        # CLIPTextModel_.forward() signature:
        # def forward(self, input_tokens=None, attention_mask=None, embeds=None, 
        #            num_tokens=None, intermediate_output=None, 
        #            final_layer_norm_intermediate=True, dtype=torch.float32)
        
        print(f"CLIPWrapper.forward: Input tokens shape: {tokens.shape}, dtype: {tokens.dtype}")
        
        outputs = self.transformer(
            input_tokens=tokens,  # Pass tokens directly
            attention_mask=None,  # No attention masking for ONNX simplicity
            embeds=None,          # Use token embeddings, not pre-computed embeds
            num_tokens=None,      # Let it figure out EOS automatically
            intermediate_output=None,  # No intermediate outputs
            final_layer_norm_intermediate=True,
            dtype=torch.float16
        )
        
        print(f"CLIPWrapper.forward: Transformer output type: {type(outputs)}, length: {len(outputs) if isinstance(outputs, tuple) else 'not tuple'}")
        
        # CLIPTextModel.forward() processes CLIPTextModel_.forward() output and returns:
        # (x[0], x[1], out, x[2]) where:
        # - x[0] = hidden_states (last layer)
        # - x[1] = intermediate (can be None)
        # - out = text_projection(x[2]) = projected pooled output
        # - x[2] = pooled_output (unprojected)
        
        if isinstance(outputs, tuple) and len(outputs) >= 3:
            hidden_states = outputs[0]      # Last hidden states
            # outputs[1] is intermediate (can be None)
            # outputs[2] is pooled_output after text_projection
            # outputs[3] is pooled_output before text_projection (if available)
            
            if len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3]  # Use unprojected pooled output for consistency
                print(f"CLIPWrapper.forward: Using unprojected pooled output")
            else:
                pooled_output = outputs[2]  # Use projected pooled output
                print(f"CLIPWrapper.forward: Using projected pooled output")
        else:
            # Fallback
            if isinstance(outputs, tuple) and len(outputs) > 0:
                hidden_states = outputs[0]
            else:
                hidden_states = outputs
            pooled_output = hidden_states[:, -1, :]  # Use last token
            print(f"CLIPWrapper.forward: Using fallback pooled output (last token)")
        
        print(f"CLIPWrapper.forward: Output shapes - hidden: {hidden_states.shape}, pooled: {pooled_output.shape}")
        
        return hidden_states.float(), pooled_output.float()


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
        
        # Load CLIP to GPU
        clip_model = clip.cond_stage_model
        device = comfy.model_management.get_torch_device()
        
        # Load the CLIP model using the patcher instead of direct model management
        if hasattr(clip, 'patcher') and clip.patcher is not None:
            clip.patcher.patch_model(device_to=device, force_patch_weights=True)
            print(f"CLIP model moved to device: {device}")
        else:
            # Fallback: manually move the CLIP model to device
            clip_model = clip_model.to(device=device)
            print(f"CLIP model moved to device - FALLBACK: {device}")
        
        # Debug: Print CLIP model structure
        print(f"CLIP model type: {type(clip_model)}")
        print(f"CLIP model attributes: {dir(clip_model)}")
        if hasattr(clip_model, 'clip_l'):
            print(f"Has clip_l: {type(clip_model.clip_l)}")
        if hasattr(clip_model, 'clip_g'):
            print(f"Has clip_g: {type(clip_model.clip_g)}")
            
        # Force fp16 for all CLIP components to ensure consistency and avoid NaN issues
        dtype = torch.float16
        print(f"Forcing CLIP dtype to: {dtype}")
        
        if is_clip_l:
            if hasattr(clip_model, 'clip_l'):
                clip_component = clip_model.clip_l
                print(f"Using CLIP-L component: {type(clip_component)}")
            else:
                # For single CLIP models, use the model directly
                clip_component = clip_model
                print(f"Using single CLIP model as CLIP-L: {type(clip_component)}")
        else:
            if hasattr(clip_model, 'clip_g'):
                clip_component = clip_model.clip_g
                print(f"Using CLIP-G component: {type(clip_component)}")
            else:
                # For single CLIP models, use the model directly
                clip_component = clip_model
                print(f"Using single CLIP model as CLIP-G: {type(clip_component)}")
        
        # Ensure all model parameters are on the same device with fp16 dtype
        clip_component = clip_component.to(device=device, dtype=dtype)
        print(f"CLIP component moved to device: {device} with dtype: {dtype}")
        
        # Force all buffers and parameters to be on the same device with fp16 dtype
        for name, param in clip_component.named_parameters():
            if param.device != device or param.dtype != dtype:
                param.data = param.data.to(device=device, dtype=dtype)
        
        for name, buffer in clip_component.named_buffers():
            if buffer.device != device or buffer.dtype != dtype:
                buffer.data = buffer.data.to(device=device, dtype=dtype)

        # Create wrapper for the CLIP part we want to convert
        clip_wrapper = CLIPWrapper(clip_component, is_clip_l=is_clip_l)
        
        # Set model to evaluation mode to avoid any training-specific behavior
        clip_wrapper.eval()
        
        # Ensure wrapper is on the correct device (CUDA)
        clip_wrapper = clip_wrapper.cuda()

        # Input shapes for CLIP (token sequences)
        inputs_shapes_min = (batch_size_min, sequence_length_min)
        inputs_shapes_opt = (batch_size_opt, sequence_length_opt)
        inputs_shapes_max = (batch_size_max, sequence_length_max)

        input_names = ["tokens"]
        # Both CLIP-L and CLIP-G now return two outputs consistently
        output_names = ["hidden_states", "pooled_output"]
        dynamic_axes = {
            "tokens": {0: "batch", 1: "sequence"},
            "hidden_states": {0: "batch", 1: "sequence"},
            "pooled_output": {0: "batch"},
        }

        # Create input tensor for ONNX export (integer tokens)
        # Use realistic token values instead of all zeros to avoid issues
        # Token 0 is typically padding, 1 is start token, 2 is end token
        input_tensor = torch.ones(
            inputs_shapes_opt,
            device=device,
            dtype=torch.long,  # CLIP tokens are integers
        )
        # Set start token (typically 1) at the beginning of each sequence
        input_tensor[:, 0] = 1
        # Set some variety in the middle tokens to make it more realistic
        for i in range(1, min(inputs_shapes_opt[1], 10)):
            input_tensor[:, i] = i + 100  # Use tokens in a safe range

        os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
        
        # Export to ONNX with no_grad context to avoid gradient computation
        with torch.no_grad():
            torch.onnx.export(
                clip_wrapper,
                (input_tensor,),
                output_onnx,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,  # Enable constant folding for optimization
                export_params=True,  # Export model parameters
            )

        # Unload the CLIP model using the patcher
        if hasattr(clip, 'patcher') and clip.patcher is not None:
            clip.patcher.unpatch_model()
        
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
        
        profile.set_shape("tokens", min_shape, opt_shape, max_shape)

        
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
        else:
            filename_prefix = "{}_{}${}".format(
                filename_prefix,
                clip_type,
                "-".join(
                    (
                        "dyn",
                        "b",
                        str(batch_size_min),
                        str(batch_size_max),
                        str(batch_size_opt),
                        "s",
                        str(sequence_length_min),
                        str(sequence_length_max),
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


class DYNAMIC_TRT_CLIP_L_CONVERSION(TRT_CLIP_CONVERSION_BASE):
    def __init__(self):
        super(DYNAMIC_TRT_CLIP_L_CONVERSION, self).__init__()

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_CLIP_L_DYN"}),
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
                        "default": 4,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "sequence_length_min": (
                    "INT",
                    {
                        "default": 77,
                        "min": 1,
                        "max": 512,
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
                "sequence_length_max": (
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
        batch_size_min,
        batch_size_opt,
        batch_size_max,
        sequence_length_min,
        sequence_length_opt,
        sequence_length_max,
    ):
        super()._convert_clip(
            clip,
            filename_prefix,
            batch_size_min,
            batch_size_opt,
            batch_size_max,
            sequence_length_min,
            sequence_length_opt,
            sequence_length_max,
            is_clip_l=True,
            is_static=False,
        )
        return ()


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
        return ()


class DYNAMIC_TRT_CLIP_G_CONVERSION(TRT_CLIP_CONVERSION_BASE):
    def __init__(self):
        super(DYNAMIC_TRT_CLIP_G_CONVERSION, self).__init__()

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_CLIP_G_DYN"}),
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
                        "default": 4,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "sequence_length_min": (
                    "INT",
                    {
                        "default": 77,
                        "min": 1,
                        "max": 512,
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
                "sequence_length_max": (
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
        batch_size_min,
        batch_size_opt,
        batch_size_max,
        sequence_length_min,
        sequence_length_opt,
        sequence_length_max,
    ):
        super()._convert_clip(
            clip,
            filename_prefix,
            batch_size_min,
            batch_size_opt,
            batch_size_max,
            sequence_length_min,
            sequence_length_opt,
            sequence_length_max,
            is_clip_l=False,
            is_static=False,
        )
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
        return ()