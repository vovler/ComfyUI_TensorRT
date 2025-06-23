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
    """Wrapper for CLIP encoder to make it compatible with ONNX export"""
    def __init__(self, clip_model, is_clip_l=True):
        super().__init__()
        self.clip = clip_model
        # is_clip_l determines how we derive the pooled output
        self.is_clip_l = is_clip_l
        
        # We need to know which token is the EOS token to extract the pooled output for CLIP-L
        # For OpenCLIP, this is typically the last non-padded token.
        # We will assume a simple case here: the last token of the sequence.
        # A more robust implementation might find the first padding token and take the one before it.

    def forward(self, tokens):
        # The 'tokens' tensor should be of shape (batch, seq_len) and dtype torch.long
        # The underlying HuggingFace Transformers model expects 'input_ids'
        
        # Directly call the model's forward pass. This is what ONNX tracing needs.
        transformer_outputs = self.clip(input_ids=tokens, output_hidden_states=False)

        # The output 'last_hidden_state' is what we need for cross-attention
        hidden_states = transformer_outputs.last_hidden_state

        if self.is_clip_l:
            # For CLIP-L (CLIPTextModel), the pooled output is not returned directly.
            # We must derive it from the hidden state of the EOS token.
            # Assuming the EOS token is the last one in the sequence for simplicity.
            # A more robust way is to find the actual EOS token id from the input 'tokens'.
            pooled_output = hidden_states[:, -1, :] # Shape: [batch_size, hidden_size]
        else:
            # For CLIP-G (CLIPTextModelWithProjection), it provides a 'pooler_output'
            pooled_output = transformer_outputs.pooler_output # Shape: [batch_size, projection_dim]

        # Ensure outputs are float32 for consistency, as TensorRT might default to float16
        # if the model is in half precision.
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
        else:
            # Fallback: manually move the CLIP model to device
            clip_model = clip_model.to(device=device)
        
        # Debug: Print CLIP model structure
        print(f"CLIP model type: {type(clip_model)}")
        print(f"CLIP model attributes: {dir(clip_model)}")
        if hasattr(clip_model, 'clip_l'):
            print(f"Has clip_l: {type(clip_model.clip_l)}")
        if hasattr(clip_model, 'clip_g'):
            print(f"Has clip_g: {type(clip_model.clip_g)}")
            
        # Get the CLIP model's native dtype and use it for consistency
        # Check the dtype of the first parameter to determine model dtype
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
                
        # Check if the component has parameters
        try:
            model_dtype = next(clip_component.parameters()).dtype
            print(f"Model dtype: {model_dtype}")
        except StopIteration:
            print("No parameters found in CLIP component!")
            model_dtype = torch.float16  # Default fallback
        
        # Ensure all model parameters are on the same device
        clip_component = clip_component.to(device=device, dtype=model_dtype)
        
        # Force all buffers and parameters to be on the same device
        for name, param in clip_component.named_parameters():
            if param.device != device:
                param.data = param.data.to(device=device, dtype=model_dtype)
        
        for name, buffer in clip_component.named_buffers():
            if buffer.device != device:
                buffer.data = buffer.data.to(device=device, dtype=model_dtype)
        
        dtype = model_dtype

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

        if dtype == torch.float16:
            config.set_flag(trt.BuilderFlag.FP16)
        if dtype == torch.bfloat16:
            config.set_flag(trt.BuilderFlag.BF16)

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