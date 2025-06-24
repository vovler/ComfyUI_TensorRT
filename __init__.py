from .loader_unet import UNET_LOADER_TENSORRT
from .loader_vae import VAE_LOADER_TENSORRT

from .converter_unet import UNET_CONVERTER_TENSORRT_DYNAMIC
from .converter_vae import VAE_DECODER_CONVERTER_TENSORRT_DYNAMIC
from .converter_vae import VAE_ENCODER_CONVERTER_TENSORRT_DYNAMIC

from .utils.folder_setup import setup_tensorrt_folder_paths

NODE_CLASS_MAPPINGS = {
        
        "UNET_LOADER_TENSORRT": UNET_LOADER_TENSORRT,
        "VAE_LOADER_TENSORRT": VAE_LOADER_TENSORRT,

        "UNET_CONVERTER_TENSORRT_DYNAMIC": UNET_CONVERTER_TENSORRT_DYNAMIC,
        "VAE_DECODER_CONVERTER_TENSORRT_DYNAMIC": VAE_DECODER_CONVERTER_TENSORRT_DYNAMIC,
        "VAE_ENCODER_CONVERTER_TENSORRT_DYNAMIC": VAE_ENCODER_CONVERTER_TENSORRT_DYNAMIC,
        
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "UNET_LOADER_TENSORRT": "UNET Loader (TensorRT)",
    "VAE_LOADER_TENSORRT": "VAE Loader (TensorRT)",

    "UNET_CONVERTER_TENSORRT_DYNAMIC": "UNET Converter -> ONNX -> TensorRT (Dynamic)",
    "VAE_DECODER_CONVERTER_TENSORRT_DYNAMIC": "VAE Decoder Converter -> ONNX -> TensorRT (Dynamic)",
    "VAE_ENCODER_CONVERTER_TENSORRT_DYNAMIC": "VAE Encoder Converter -> ONNX -> TensorRT (Dynamic)",
    
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


setup_tensorrt_folder_paths()
