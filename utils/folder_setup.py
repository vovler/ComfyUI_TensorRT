import os
import folder_paths


def setup_tensorrt_folder_paths():
    """
    Sets up TensorRT folder paths for ComfyUI.
    Adds the output directory to tensorrt search path and registers .engine files.
    """
    tensorrt_output_dir = os.path.join(folder_paths.get_output_directory(), "tensorrt")
    
    if "tensorrt" in folder_paths.folder_names_and_paths:
        # Add output directory if not already present
        if tensorrt_output_dir not in folder_paths.folder_names_and_paths["tensorrt"][0]:
            folder_paths.folder_names_and_paths["tensorrt"][0].append(tensorrt_output_dir)
        # Ensure .engine extension is registered
        folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
    else:
        # Create new tensorrt folder path entry
        folder_paths.folder_names_and_paths["tensorrt"] = (
            [tensorrt_output_dir],
            {".engine"},
        )
    
    print(f"TensorRT folder paths configured: {folder_paths.folder_names_and_paths.get('tensorrt', 'Not found')}") 