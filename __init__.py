from .tensorrt_convert import DYNAMIC_TRT_MODEL_CONVERSION
from .tensorrt_loader import TensorRTLoader
from .tensorrt_vae_loader import TensorRTVAELoader
from .tensorrt_convert_vae import DYNAMIC_TRT_VAE_DECODER_CONVERSION
from .tensorrt_convert_vae import DYNAMIC_TRT_VAE_ENCODER_CONVERSION
from .tensorrt_convert_clip import STATIC_TRT_CLIP_L_CONVERSION
from .tensorrt_convert_clip import STATIC_TRT_CLIP_G_CONVERSION
from .tensorrt_clip_loader import TensorRTCLIPLoader
from .conditioning_dummy import ConditioningDebugNode
from .conditioning_dummy import CLIPTextEncodeDebug

NODE_CLASS_MAPPINGS = {
        "DYNAMIC_TRT_MODEL_CONVERSION": DYNAMIC_TRT_MODEL_CONVERSION,
        "TensorRTLoader": TensorRTLoader,
        "TensorRTVAELoader": TensorRTVAELoader,
        "DYNAMIC_TRT_VAE_DECODER_CONVERSION": DYNAMIC_TRT_VAE_DECODER_CONVERSION,
        "DYNAMIC_TRT_VAE_ENCODER_CONVERSION": DYNAMIC_TRT_VAE_ENCODER_CONVERSION,
        "STATIC_TRT_CLIP_L_CONVERSION": STATIC_TRT_CLIP_L_CONVERSION,
        "STATIC_TRT_CLIP_G_CONVERSION": STATIC_TRT_CLIP_G_CONVERSION,
        "TensorRTCLIPLoader": TensorRTCLIPLoader,
        "ConditioningDebugNode": ConditioningDebugNode,
        "CLIPTextEncodeDebug": CLIPTextEncodeDebug,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "DYNAMIC_TRT_MODEL_CONVERSION": "DYNAMIC TRT_MODEL CONVERSION",
    "TensorRTLoader": "TensorRT Loader",
    "TensorRTVAELoader": "TensorRT VAE Loader",
    "DYNAMIC_TRT_VAE_DECODER_CONVERSION": "DYNAMIC TRT VAE DECODER CONVERSION",
    "DYNAMIC_TRT_VAE_ENCODER_CONVERSION": "DYNAMIC TRT VAE ENCODER CONVERSION",
    "STATIC_TRT_CLIP_L_CONVERSION": "STATIC TRT CLIP-L CONVERSION",
    "STATIC_TRT_CLIP_G_CONVERSION": "STATIC TRT CLIP-G CONVERSION",
    "TensorRTCLIPLoader": "TensorRT CLIP Loader",
    "ConditioningDebugNode": "Conditioning Debug",
    "CLIPTextEncodeDebug": "CLIP Text Encode (Debug)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']