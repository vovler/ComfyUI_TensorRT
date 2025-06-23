import os
import tensorrt as trt


def setup_timing_cache(config: trt.IBuilderConfig, timing_cache_path: str):
    """
    Sets up the builder to use the timing cache file, and creates it if it does not already exist.
    
    Args:
        config: TensorRT builder config
        timing_cache_path: Path to the timing cache file
    """
    buffer = b""
    if os.path.exists(timing_cache_path):
        with open(timing_cache_path, mode="rb") as timing_cache_file:
            buffer = timing_cache_file.read()
        print("Read {} bytes from timing cache: {}".format(len(buffer), timing_cache_path))
    else:
        print("No timing cache found at: {}. Initializing a new one.".format(timing_cache_path))
    
    timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
    config.set_timing_cache(timing_cache, ignore_mismatch=True)


def save_timing_cache(config: trt.IBuilderConfig, timing_cache_path: str):
    """
    Saves the config's timing cache to file.
    
    Args:
        config: TensorRT builder config
        timing_cache_path: Path to save the timing cache file
    """
    timing_cache: trt.ITimingCache = config.get_timing_cache()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(timing_cache_path), exist_ok=True)
    
    with open(timing_cache_path, "wb") as timing_cache_file:
        timing_cache_file.write(memoryview(timing_cache.serialize()))
    
    print("Saved timing cache to: {}".format(timing_cache_path))


def get_timing_cache_path(model_type: str) -> str:
    """
    Get the timing cache path for a specific model type.
    
    Args:
        model_type: Type of model ('vae', 'unet', 'clip', etc.)
        
    Returns:
        Path to the timing cache file for the specified model type
    """
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.normpath(
        os.path.join(current_dir, f"timing_cache_{model_type}.trt")
    ) 