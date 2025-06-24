import os
import folder_paths


def setup_tensorrt_folder_paths():
    """
    Sets up TensorRT folder paths for ComfyUI.
    Registers .engine files in model folders where they should be saved.
    """
    # Add .engine extension to existing model folders where TensorRT engines will be saved
    model_folders = ["vae", "diffusion_models", "text_encoders"]
    
    for folder_name in model_folders:
        if folder_name in folder_paths.folder_names_and_paths:
            # Add .engine extension to the existing folder
            folder_paths.folder_names_and_paths[folder_name][1].add(".engine")
            print(f"Added .engine extension to {folder_name} folder")
        else:
            print(f"Warning: {folder_name} folder not found in folder_paths")
    
    print(f"TensorRT folder paths configured:")
    for folder_name in model_folders:
        if folder_name in folder_paths.folder_names_and_paths:
            paths, extensions = folder_paths.folder_names_and_paths[folder_name]
            print(f"  {folder_name}: paths={paths}, extensions={extensions}") 