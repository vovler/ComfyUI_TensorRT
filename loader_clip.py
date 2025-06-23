import torch
import os
import logging

import comfy.model_management
import comfy.sd
import comfy.sd1_clip
import comfy.sdxl_clip
import folder_paths
from .tensorrt_model import get_runtime
from .wrappers_for_tensorrt_load.wrapper_clip_load import TrTCompatibleEmbedding, TrTCLIP


class CLIP_LOADER_TENSORRT:
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
        try:
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
            
            # Use centralized runtime
            runtime = get_runtime()
            
            # Create the appropriate CLIP model based on what engines we have
            if clip_l_path and clip_g_path:
                # SDXL-style dual CLIP
                clip_model = comfy.sdxl_clip.SDXLClipModel(
                    device=comfy.model_management.get_torch_device(),
                    dtype=torch.float16
                )
                # Replace the transformers with TensorRT versions
                clip_model.clip_l.transformer = TrTCLIP(clip_l_path, runtime, hidden_size=768)
                clip_model.clip_g.transformer = TrTCLIP(clip_g_path, runtime, hidden_size=1280)
                tokenizer = comfy.sdxl_clip.SDXLTokenizer()
                
            elif clip_l_path:
                # CLIP-L only
                clip_model = comfy.sd1_clip.SDClipModel(
                    device=comfy.model_management.get_torch_device(),
                    dtype=torch.float16
                )
                clip_model.transformer = TrTCLIP(clip_l_path, runtime, hidden_size=768)
                tokenizer = comfy.sd1_clip.SDTokenizer()
                
            elif clip_g_path:
                # CLIP-G only  
                clip_model = comfy.sdxl_clip.SDXLClipG(
                    device=comfy.model_management.get_torch_device(),
                    dtype=torch.float16
                )
                clip_model.transformer = TrTCLIP(clip_g_path, runtime, hidden_size=1280)
                tokenizer = comfy.sdxl_clip.SDXLClipGTokenizer()
            
            # Create a new CLIP object using ComfyUI's standard approach
            from comfy.model_patcher import ModelPatcher
            
            patcher = ModelPatcher(clip_model, 
                                load_device=comfy.model_management.get_torch_device(),
                                offload_device=comfy.model_management.text_encoder_offload_device())
            
            new_clip = comfy.sd.CLIP(no_init=True)
            new_clip.cond_stage_model = clip_model
            new_clip.tokenizer = tokenizer
            new_clip.patcher = patcher
            new_clip.layer_idx = None
            new_clip.use_clip_schedule = False
            new_clip.tokenizer_options = {}
            
            # Add missing attributes for ComfyUI compatibility
            new_clip.apply_hooks_to_conds = None
            
            return (new_clip,)
            
        except Exception as e:
            print(f"TensorRTCLIPLoader - Error loading clip: {e}")
            import traceback
            traceback.print_exc()
            return (None,)