#Put this in the custom_nodes folder, put your tensorrt engine files in ComfyUI/models/tensorrt/ (you will have to create the directory)
# pyright: reportAttributeAccessIssue=false

import torch
import os

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import folder_paths
from .utils.tensorrt_error_recorder import TrTErrorRecorder
from .models.unet import TrTUnet
from .models.trtModel import TrTModelManageable

if "tensorrt" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tensorrt"][0].append(
        os.path.join(folder_paths.models_dir, "tensorrt"))
    folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
else:
    folder_paths.folder_names_and_paths["tensorrt"] = (
        [os.path.join(folder_paths.models_dir, "tensorrt")], {".engine"})

import tensorrt as trt

trt.init_libnvinfer_plugins(None, "")

logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)
runtime.error_recorder = TrTErrorRecorder()

# Is there a function that already exists for this?
def trt_datatype_to_torch(datatype):
    if datatype == trt.float16:
        return torch.float16
    elif datatype == trt.float32:
        return torch.float32
    elif datatype == trt.int32:
        return torch.int32
    elif datatype == trt.bfloat16:
        return torch.bfloat16






class TensorRTLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (folder_paths.get_filename_list("tensorrt"), ),
                             "model_type": (["sdxl_base"], ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "TensorRT"

    def load_unet(self, unet_name, model_type):

        unet_path = folder_paths.get_full_path("tensorrt", unet_name)
        if unet_path is None:
            raise FileNotFoundError(f"File {unet_path} does not exist")
            
        if not os.path.isfile(unet_path):
            raise FileNotFoundError(f"File {unet_path} does not exist")

        unet = TrTUnet(unet_path, runtime)

        if model_type == "sdxl_base":
            conf = comfy.supported_models.SDXL({"adm_in_channels": 2816})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.SDXL(conf)
            # Set the diffusion_model attribute to our TensorRT UNet since we disabled automatic creation
            model.diffusion_model = unet
        else:
            raise ValueError(f"Model type {model_type} not supported")
        

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        manageable_model = TrTModelManageable(model, unet, load_device, offload_device)

        return (manageable_model,)

NODE_CLASS_MAPPINGS = {
    "TensorRTLoader": TensorRTLoader,
}
