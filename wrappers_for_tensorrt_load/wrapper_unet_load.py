import torch
import os
import tensorrt as trt
from ..utils.trt_datatype_to_torch import trt_datatype_to_torch
from ..utils.tensorrt_error_recorder import check_for_trt_errors
from comfy.model_patcher import ModelPatcher
import comfy.model_base


class TrTUnet:
    def __init__(self, engine_path, runtime):
        self.dtype = torch.float16
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self._size = int(os.stat(engine_path).st_size)
        self.runtime = runtime

    def load(self):
        if self.engine is not None or self.context is not None:
            return
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            check_for_trt_errors(self.runtime)
            self.context = self.engine.create_execution_context()
            check_for_trt_errors(self.runtime)

    @property
    def size(self) -> int:
        return self._size

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    def __call__(self, x, timesteps, context, y=None, control=None, transformer_options=None, **kwargs):
        # Debug prints for CLIP embeddings
        print(f"TrTUnet.__call__ - Input shapes:")
        print(f"  x (latents): {x.shape}, dtype: {x.dtype}")
        print(f"  timesteps: {timesteps.shape}, dtype: {timesteps.dtype}")
        print(f"  context (CLIP embeddings): {context.shape}, dtype: {context.dtype}")
        if y is not None:
            print(f"  y (pooled): {y.shape}, dtype: {y.dtype}")
        
        # Debug: Check for invalid values in context
        if torch.isnan(context).any():
            print(f"  WARNING: NaN values detected in context!")
        if torch.isinf(context).any():
            print(f"  WARNING: Infinite values detected in context!")
        
        # Debug: Print context statistics
        print(f"  context min: {context.min().item():.6f}, max: {context.max().item():.6f}, mean: {context.mean().item():.6f}")
        if y is not None:
            print(f"  y min: {y.min().item():.6f}, max: {y.max().item():.6f}, mean: {y.mean().item():.6f}")
            
        model_inputs = {"x": x, "timesteps": timesteps, "context": context}

        if y is not None:
            model_inputs["y"] = y

        for i in range(len(model_inputs), self.engine.num_io_tensors - 1):
            name = self.engine.get_tensor_name(i)
            model_inputs[name] = kwargs[name]

        batch_size = x.shape[0]
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        min_batch = dims[0][0]
        opt_batch = dims[1][0]
        max_batch = dims[2][0]

        # Split batch if our batch is bigger than the max batch size the trt engine supports
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break

        self.set_bindings_shape(model_inputs, curr_split_batch)

        model_inputs_converted = {}
        for k in model_inputs:
            data_type = self.engine.get_tensor_dtype(k)
            model_inputs_converted[k] = model_inputs[k].to(dtype=trt_datatype_to_torch(data_type))

        output_binding_name = self.engine.get_tensor_name(len(model_inputs))
        out_shape = self.engine.get_tensor_shape(output_binding_name)
        out_shape = list(out_shape)

        # for dynamic profile case where the dynamic params are -1
        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                out_shape[idx] = x.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch

        out = torch.empty(out_shape,
                          device=x.device,
                          dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(output_binding_name)))
        model_inputs_converted[output_binding_name] = out

        stream = torch.cuda.default_stream(x.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                x = model_inputs_converted[k]
                self.context.set_tensor_address(k, x[(x.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        # stream.synchronize() #don't need to sync stream since it's the default torch one
        return out

    def load_state_dict(self, sd, strict=False):
        pass

    def state_dict(self):
        return {}

    def unload(self):
        engine_obj = self.engine
        self.engine = None
        if engine_obj is not None:
            del engine_obj
        context_obj = self.context
        self.context = None
        if context_obj is not None:
            del context_obj


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