# pyright: reportOptionalMemberAccess=false

import torch
import os
import logging
import traceback
import comfy.model_management
import comfy.sd
import folder_paths
from .wrappers_for_tensorrt_load.wrapper_vae_load import TrTVAE
from .tensorrt_model import get_runtime


class VAE_LOADER_TENSORRT:
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        # Get engine files from VAE folder
        vae_files = folder_paths.get_filename_list("vae")
        
        # Filter for VAE engines
        vae_encoder_files = ["None"] + [f for f in vae_files if f.endswith(".engine") and "encoder" in f.lower()]
        vae_decoder_files = ["None"] + [f for f in vae_files if f.endswith(".engine") and "decoder" in f.lower()]
        
        return {
            "required": {
                "encoder_name": (vae_encoder_files,),
                "decoder_name": (vae_decoder_files,),
            },
        }

    def load_vae(self, encoder_name, decoder_name):
        try:
            encoder_path = None
            decoder_path = None
            
            if encoder_name != "None":
                encoder_path = folder_paths.get_full_path("vae", encoder_name)
                if encoder_path is None or not os.path.isfile(encoder_path):
                    raise FileNotFoundError(f"Encoder file {encoder_name} does not exist in VAE folder")
                    
            if decoder_name != "None":
                decoder_path = folder_paths.get_full_path("vae", decoder_name)
                if decoder_path is None or not os.path.isfile(decoder_path):
                    raise FileNotFoundError(f"Decoder file {decoder_name} does not exist in VAE folder")
            
            if encoder_path is None and decoder_path is None:
                raise ValueError("At least one of encoder or decoder must be specified")
            
            # Use centralized runtime
            runtime = get_runtime()
            vae = TrTVAE(encoder_path, decoder_path, runtime)
            
            
        except Exception as e:
            print(f"TensorRTVAELoader - Error loading VAE: {e}")
            traceback.print_exc()
            return (None,)

        return (vae,)