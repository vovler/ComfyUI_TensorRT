import torch
from comfy.model_patcher import ModelPatcher
from comfy.model_management import LoadedModel, get_torch_device
from .unet import TrTUnet
from pathlib import PurePath
import comfy.model_base

class TrTModelManageable(ModelPatcher):
    def __init__(self, model: comfy.model_base.BaseModel, unet, load_device, offload_device):
        self._unet = unet
        super().__init__(model, load_device, offload_device, size=unet.size)

    @property
    def engine(self):
        return self._unet.engine

    @engine.setter
    def engine(self, value):
        self._unet.engine = value

    @property
    def context(self):
        return self._unet.context

    @context.setter
    def context(self, value):
        self._unet.context = value

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        self._unet.load()
        return super().patch_model(device_to, lowvram_model_memory, load_weights, force_patch_weights)

    def unpatch_model(self, device_to=None, unpatch_weights=False):
        self._unet.unload()
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    def model_size(self):
        return self._unet.size

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        self._unet.load()
        super().load(device_to, lowvram_model_memory, force_patch_weights, full_load)

    def model_dtype(self):
        return self._unet.dtype

    def is_clone(self, other):
        return other is not None and isinstance(other,
                                                TrTModelManageable) and other._unet is self._unet and super().is_clone(
            other)

    def clone_has_same_weights(self, clone):
        return clone is not None and isinstance(clone,
                                                TrTModelManageable) and clone._unet.engine_path == self._unet.engine_path and super().clone_has_same_weights(
            clone)

    def memory_required(self, input_shape):
        return self.model_size()  # This is an approximation

    def __str__(self):
        return f"<TrtModelManageable>"
