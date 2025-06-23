import torch
import os
import traceback

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import folder_paths
from .wrappers_for_tensorrt_load.wrapper_unet_load import TrTUnet, TrTModelManageable
from .tensorrt_model import get_runtime


class UNET_LOADER_TENSORRT:

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("tensorrt"), {
                    "tooltip": "Select a TensorRT engine file (.engine) from the tensorrt folder"
                }),
                "model_type": (["sdxl_base"], {
                    "default": "sdxl_base",
                    "tooltip": "Model architecture type - currently supports SDXL Base"
                }),
            },
        }

    def load_unet(self, unet_name, model_type):

            try:

                if not unet_name or not unet_name.strip():
                    raise ValueError("UNet name cannot be empty")
                
                if not model_type or model_type not in ["sdxl_base"]:
                    raise ValueError(f"Model type '{model_type}' not supported. Available types: ['sdxl_base']")
                
                # Get engine file path
                unet_path = folder_paths.get_full_path("tensorrt", unet_name)
                if unet_path is None:
                    raise FileNotFoundError(f"TensorRT engine file '{unet_name}' not found in tensorrt folder")
                    
                if not os.path.isfile(unet_path):
                    raise FileNotFoundError(f"TensorRT engine file does not exist: {unet_path}")
                
                print(f"Loading TensorRT UNet from: {unet_path}")
                
    
                runtime = get_runtime()
                unet = TrTUnet(unet_path, runtime)
                print(f"TensorRT UNet initialized successfully (size: {unet.size / (1024*1024):.1f} MB)")

                # Configure SDXL model
                conf = comfy.supported_models.SDXL({"adm_in_channels": 2816})
                conf.unet_config["disable_unet_model_creation"] = True
                model = comfy.model_base.SDXL(conf)
                    
                # Set the diffusion_model attribute to our TensorRT UNet
                model.diffusion_model = unet
                print("SDXL model configuration applied successfully")
            
                load_device = comfy.model_management.get_torch_device()
                offload_device = comfy.model_management.unet_offload_device()
                manageable_model = TrTModelManageable(model, unet, load_device, offload_device)
                
                print(f"Model management configured: load_device={load_device}, offload_device={offload_device}")
                print("TensorRT UNet loaded successfully!")
            
            except Exception as e:
                print(f"🔴 UNET_LOADER_TENSORRT - Unexpected error: {e}")
                print(f"🔴 UNET_LOADER_TENSORRT - Full stack trace:")
                traceback.print_exc()
                return (None,)

            return (manageable_model,)
            
