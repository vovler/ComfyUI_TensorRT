import torch
import tensorrt as trt

def trt_datatype_to_torch(datatype):
    print(f"trt_datatype_to_torch called with: {datatype} (type: {type(datatype)})")
    
    if datatype == trt.float16:
        print("Converting trt.float16 -> torch.float16")
        return torch.float16
    elif datatype == trt.float32:
        print("Converting trt.float32 -> torch.float32")
        return torch.float32
    elif datatype == trt.int32:
        print("Converting trt.int32 -> torch.int32")
        return torch.int32
    elif datatype == trt.int64:
        print("Converting trt.int64 -> torch.int64 (torch.long)")
        return torch.int64  # torch.long is an alias for torch.int64
    elif datatype == trt.int8:
        print("Converting trt.int8 -> torch.int8")
        return torch.int8
    elif datatype == trt.uint8:
        print("Converting trt.uint8 -> torch.uint8")
        return torch.uint8
    elif datatype == trt.bool:
        print("Converting trt.bool -> torch.bool")
        return torch.bool
    elif datatype == trt.bfloat16:
        print("Converting trt.bfloat16 -> torch.bfloat16")
        return torch.bfloat16
    else:
        print(f"ERROR: Unsupported TensorRT datatype: {datatype}")
        print(f"Available TensorRT datatypes: {dir(trt)}")
        return None