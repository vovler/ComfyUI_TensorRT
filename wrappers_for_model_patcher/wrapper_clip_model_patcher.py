import torch
from comfy.model_patcher import ModelPatcher
import comfy.model_management


class TrTCLIPModelPatcher(ModelPatcher):
    """TensorRT CLIP model wrapper that extends ModelPatcher for ComfyUI integration"""
    
    def __init__(self, clip_model, clip_l_engine=None, clip_g_engine=None, load_device=None, offload_device=None):
        self.clip_model = clip_model
        self._clip_l_engine = clip_l_engine
        self._clip_g_engine = clip_g_engine
        
        # Calculate total size from engines
        total_size = 0
        if clip_l_engine:
            total_size += getattr(clip_l_engine, 'size', 0)
        if clip_g_engine:
            total_size += getattr(clip_g_engine, 'size', 0)
        
        # Set default devices if not provided
        if load_device is None:
            load_device = comfy.model_management.get_torch_device()
        if offload_device is None:
            offload_device = comfy.model_management.text_encoder_offload_device()
        
        super().__init__(clip_model, load_device, offload_device, size=total_size)

    @property
    def clip_l_engine(self):
        return self._clip_l_engine

    @property
    def clip_g_engine(self):
        return self._clip_g_engine

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        # Load TensorRT engines
        if self._clip_l_engine:
            self._clip_l_engine.load()
        if self._clip_g_engine:
            self._clip_g_engine.load()
        return super().patch_model(device_to, lowvram_model_memory, load_weights, force_patch_weights)

    def unpatch_model(self, device_to=None, unpatch_weights=False):
        # Unload TensorRT engines
        if self._clip_l_engine:
            self._clip_l_engine.unload()
        if self._clip_g_engine:
            self._clip_g_engine.unload()
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    def model_size(self):
        total_size = 0
        if self._clip_l_engine:
            total_size += getattr(self._clip_l_engine, 'size', 0)
        if self._clip_g_engine:
            total_size += getattr(self._clip_g_engine, 'size', 0)
        return total_size

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        if self._clip_l_engine:
            self._clip_l_engine.load()
        if self._clip_g_engine:
            self._clip_g_engine.load()
        super().load(device_to, lowvram_model_memory, force_patch_weights, full_load)

    def model_dtype(self):
        return torch.float16  # TensorRT CLIP models use float16

    def is_clone(self, other):
        return (other is not None and 
                isinstance(other, TrTCLIPModelPatcher) and 
                other._clip_l_engine is self._clip_l_engine and 
                other._clip_g_engine is self._clip_g_engine and 
                super().is_clone(other))

    def clone_has_same_weights(self, clone):
        if not isinstance(clone, TrTCLIPModelPatcher):
            return False
        
        # Compare engine paths if available
        clip_l_match = True
        clip_g_match = True
        
        if self._clip_l_engine and clone._clip_l_engine:
            clip_l_match = (getattr(self._clip_l_engine, 'engine_path', None) == 
                           getattr(clone._clip_l_engine, 'engine_path', None))
        elif self._clip_l_engine or clone._clip_l_engine:
            clip_l_match = False
            
        if self._clip_g_engine and clone._clip_g_engine:
            clip_g_match = (getattr(self._clip_g_engine, 'engine_path', None) == 
                           getattr(clone._clip_g_engine, 'engine_path', None))
        elif self._clip_g_engine or clone._clip_g_engine:
            clip_g_match = False
        
        return clip_l_match and clip_g_match and super().clone_has_same_weights(clone)

    def memory_required(self, input_shape):
        return self.model_size()  # This is an approximation

    def __str__(self):
        engines = []
        if self._clip_l_engine:
            engines.append("CLIP-L")
        if self._clip_g_engine:
            engines.append("CLIP-G")
        engine_str = "+".join(engines) if engines else "None"
        return f"<TrTCLIPModelPatcher({engine_str})>" 