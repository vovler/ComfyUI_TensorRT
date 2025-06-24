import torch
import onnx
import torch.onnx
import sys
import os
import time
import traceback
import comfy.model_management
import comfy.sd
import folder_paths
from tqdm import tqdm
import comfy
from typing import Any, Optional
from .utils.tqdm_progress_monitor import TQDMProgressMonitor
from .utils.timing_cache import setup_timing_cache, save_timing_cache, get_timing_cache_path
from .tensorrt_model import get_tensorrt_manager
from .wrappers_for_onnx_convert.wrapper_clip_convert import ClipEmbedWrapper
import numpy as np
import traceback
import onnxruntime as ort


def convert_clip(
    clip,
    filename_prefix,
    batch_size_min,
    batch_size_opt,
    batch_size_max,
    sequence_length_min,
    sequence_length_opt,
    sequence_length_max,
    is_clip_l: bool,
):
    # Save engines in CLIP model folder
    clip_folders = folder_paths.get_folder_paths("text_encoders")
    output_dir = clip_folders[0] if clip_folders else folder_paths.get_output_directory()
    temp_dir = folder_paths.get_temp_directory()
    timing_cache_path = get_timing_cache_path("clip")
    trt_manager = get_tensorrt_manager()

    output_onnx = os.path.normpath(
        os.path.join(
            os.path.join(temp_dir, "{}".format(time.time())), "clip_model.onnx"
        )
    )

    clip_model = clip.cond_stage_model
    device = comfy.model_management.get_torch_device()
    dtype = torch.float16
    
    # Get the correct CLIP component based on model structure
    if is_clip_l:
        if hasattr(clip_model, 'clip_l'):
            clip_component = clip_model.clip_l
        elif hasattr(clip_model, 'clip') and hasattr(clip_model, 'clip_name') and clip_model.clip_name == "l":
            clip_component = getattr(clip_model, clip_model.clip)
        else:
            clip_component = clip_model
    else:
        if hasattr(clip_model, 'clip_g'):
            clip_component = clip_model.clip_g
        elif hasattr(clip_model, 'clip') and hasattr(clip_model, 'clip_name') and clip_model.clip_name == "g":
            clip_component = getattr(clip_model, clip_model.clip)
        else:
            clip_component = clip_model
    
    # DEBUG: Print CLIP component settings for comparison
    clip_type_str = "CLIP-L" if is_clip_l else "CLIP-G"
    print(f"\n🔍 {clip_type_str} Model Settings:")
    print(f"  layer: {getattr(clip_component, 'layer', 'N/A')}")
    print(f"  layer_idx: {getattr(clip_component, 'layer_idx', 'N/A')}")
    print(f"  special_tokens: {getattr(clip_component, 'special_tokens', 'N/A')}")
    print(f"  layer_norm_hidden_state: {getattr(clip_component, 'layer_norm_hidden_state', 'N/A')}")
    print(f"  enable_attention_masks: {getattr(clip_component, 'enable_attention_masks', 'N/A')}")
    print(f"  zero_out_masked: {getattr(clip_component, 'zero_out_masked', 'N/A')}")
    print(f"  return_projected_pooled: {getattr(clip_component, 'return_projected_pooled', 'N/A')}")
    print(f"  max_length: {getattr(clip_component, 'max_length', 'N/A')}")
    if hasattr(clip_component, 'transformer'):
        transformer = clip_component.transformer
        print(f"  transformer.num_layers: {getattr(transformer, 'num_layers', 'N/A')}")
        if hasattr(transformer, 'get_input_embeddings'):
            embed_layer = transformer.get_input_embeddings()
            if hasattr(embed_layer, 'weight'):
                print(f"  embedding_weight.shape: {embed_layer.weight.shape}")
                print(f"  embedding_weight.dtype: {embed_layer.weight.dtype}")
    print()
    
    clip_component = clip_component.to(device=device, dtype=dtype)
    model_to_export = clip_component

    # Input shapes for CLIP (token sequences)
    inputs_shapes_min = (batch_size_min, sequence_length_min)
    inputs_shapes_opt = (batch_size_opt, sequence_length_opt)
    inputs_shapes_max = (batch_size_max, sequence_length_max)

    input_names = ["tokens"]
    output_names = ["hidden_states", "pooled_output"]
    dynamic_axes = {
        "tokens": {0: "batch", 1: "sequence"},
        "hidden_states": {0: "batch", 1: "sequence"},
        "pooled_output": {0: "batch"},
    }

    # Create token tensors and convert to embeddings to bypass process_tokens
    batch_size, seq_len = inputs_shapes_opt
    
    # Create token tensor directly using the component's special tokens
    special_tokens = getattr(clip_component, 'special_tokens', {"start": 49406, "end": 49407, "pad": 49407})
    start_token = special_tokens.get("start", 49406)
    end_token = special_tokens.get("end", 49407) 
    pad_token = special_tokens.get("pad", 49407)
    
    print(f"🔍 Using tokens for {clip_type_str}: start={start_token}, end={end_token}, pad={pad_token}")
    
    token_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    for b in range(batch_size):
        token_tensor[b, 0] = start_token  # start token
        token_tensor[b, 1] = end_token    # end token (immediately after start)
        token_tensor[b, 2:] = pad_token   # pad tokens for the rest

    # Get embeddings directly from the transformer
    input_embeddings = clip_component.transformer.get_input_embeddings()
    embedded_tokens = input_embeddings(token_tensor, out_dtype=torch.float32)
    
    print(f"🔍 {clip_type_str} token_tensor shape: {token_tensor.shape}, embedded_tokens shape: {embedded_tokens.shape}")
    print(f"🔍 {clip_type_str} embedded_tokens stats: min={embedded_tokens.min().item():.6f}, max={embedded_tokens.max().item():.6f}, mean={embedded_tokens.mean().item():.6f}")

    os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
    
    # Use the wrapper model from the wrapper file
    wrapper_model = ClipEmbedWrapper(clip_component)
    
    # TEST PYTORCH MODEL BEFORE ONNX EXPORT
    print(f"\n🧪 Testing PyTorch model for {clip_type_str}...")
    with torch.no_grad():
        try:
            pytorch_outputs = wrapper_model(embedded_tokens)
            hidden_states_pytorch = pytorch_outputs[0]
            pooled_output_pytorch = pytorch_outputs[1]
            
            # Check for NaN/Inf values in PyTorch model
            hidden_has_nan = torch.isnan(hidden_states_pytorch).any()
            hidden_has_inf = torch.isinf(hidden_states_pytorch).any()
            pooled_has_nan = torch.isnan(pooled_output_pytorch).any()
            pooled_has_inf = torch.isinf(pooled_output_pytorch).any()
            
            if hidden_has_nan or hidden_has_inf or pooled_has_nan or pooled_has_inf:
                print(f"🔴 PyTorch model validation FAILED for {clip_type_str}!")
                print(f"   Hidden states: NaN={hidden_has_nan}, Inf={hidden_has_inf}")
                print(f"   Pooled output: NaN={pooled_has_nan}, Inf={pooled_has_inf}")
                print(f"   The issue is in the PyTorch model itself, not ONNX/TensorRT!")
                
                # Print more diagnostic info
                if hidden_has_nan:
                    nan_count = torch.isnan(hidden_states_pytorch).sum().item()
                    print(f"   Hidden states NaN count: {nan_count}/{hidden_states_pytorch.numel()}")
                if pooled_has_nan:
                    nan_count = torch.isnan(pooled_output_pytorch).sum().item()
                    print(f"   Pooled output NaN count: {nan_count}/{pooled_output_pytorch.numel()}")
                
                return {}
            else:
                print(f"✅ PyTorch model validation PASSED for {clip_type_str}")
                print(f"   Hidden states stats: min={hidden_states_pytorch.min().item():.6f}, max={hidden_states_pytorch.max().item():.6f}, mean={hidden_states_pytorch.mean().item():.6f}")
                print(f"   Pooled output stats: min={pooled_output_pytorch.min().item():.6f}, max={pooled_output_pytorch.max().item():.6f}, mean={pooled_output_pytorch.mean().item():.6f}")
                
        except Exception as e:
            print(f"🔴 PyTorch model validation ERROR for {clip_type_str}: {e}")
            print(f"   Stopping conversion - PyTorch model failed to run!")
            
            traceback.print_exc()
            return {}

    # EXPORT TO ONNX (only if PyTorch validation passed)
    print(f"\n📦 Exporting {clip_type_str} to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper_model,
            (embedded_tokens,),
            output_onnx,
            verbose=False,
            input_names=["embeddings"],
            output_names=output_names,
            opset_version=17,
            dynamic_axes={
                "embeddings": {0: "batch", 1: "sequence"},
                "hidden_states": {0: "batch", 1: "sequence"},
                "pooled_output": {0: "batch"},
            },
        )

    # TEST ONNX MODEL BEFORE TENSORRT CONVERSION
    print(f"\n🧪 Testing ONNX model for {clip_type_str}...")
    try:
        # Create ONNX Runtime session with GPU if available
        providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        print(f"   ONNX Runtime providers: {providers}")
        ort_session = ort.InferenceSession(output_onnx, providers=providers)
        
        # Check which provider is actually being used
        used_providers = [provider.type for provider in ort_session.get_providers()]
        print(f"   ONNX Runtime active providers: {used_providers}")
        
        # Test with the same input we used for export
        test_input = embedded_tokens.cpu().numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        
        # Run inference
        print(f"   Running ONNX inference...")
        ort_outputs = ort_session.run(None, ort_inputs)
        hidden_states_onnx = ort_outputs[0]
        pooled_output_onnx = ort_outputs[1]
        
        # Compare with PyTorch model output
        print(f"🔄 Comparing PyTorch vs ONNX outputs...")
        with torch.no_grad():
            pytorch_outputs = wrapper_model(embedded_tokens)
            hidden_states_pytorch = pytorch_outputs[0].cpu().numpy()
            pooled_output_pytorch = pytorch_outputs[1].cpu().numpy()
        
        # Calculate differences
        hidden_diff = np.abs(hidden_states_pytorch - hidden_states_onnx)
        pooled_diff = np.abs(pooled_output_pytorch - pooled_output_onnx)
        
        hidden_max_diff = hidden_diff.max()
        pooled_max_diff = pooled_diff.max()
        hidden_mean_diff = hidden_diff.mean()
        pooled_mean_diff = pooled_diff.mean()
        
        print(f"   Hidden states max diff: {hidden_max_diff:.8f}, mean diff: {hidden_mean_diff:.8f}")
        print(f"   Pooled output max diff: {pooled_max_diff:.8f}, mean diff: {pooled_mean_diff:.8f}")
        
        # Check if differences are reasonable (allowing for small numerical differences)
        if hidden_max_diff > 1e-3 or pooled_max_diff > 1e-3:
            print(f"⚠️  Large differences detected between PyTorch and ONNX models!")
            print(f"   This might indicate ONNX export issues")
        else:
            print(f"✅ PyTorch vs ONNX differences are within acceptable range")
        
        # Check for NaN/Inf values
        hidden_has_nan = torch.isnan(torch.from_numpy(hidden_states_onnx)).any()
        hidden_has_inf = torch.isinf(torch.from_numpy(hidden_states_onnx)).any()
        pooled_has_nan = torch.isnan(torch.from_numpy(pooled_output_onnx)).any()
        pooled_has_inf = torch.isinf(torch.from_numpy(pooled_output_onnx)).any()
        
        if hidden_has_nan or hidden_has_inf or pooled_has_nan or pooled_has_inf:
            print(f"🔴 ONNX model validation FAILED for {clip_type_str}!")
            print(f"   Hidden states: NaN={hidden_has_nan}, Inf={hidden_has_inf}")
            print(f"   Pooled output: NaN={pooled_has_nan}, Inf={pooled_has_inf}")
            print(f"   Stopping conversion - ONNX model is corrupted!")
            return {}
        else:
            print(f"✅ ONNX model validation PASSED for {clip_type_str}")
            print(f"   Hidden states shape: {hidden_states_onnx.shape}")
            print(f"   Hidden states stats: min={hidden_states_onnx.min():.6f}, max={hidden_states_onnx.max():.6f}, mean={hidden_states_onnx.mean():.6f}")
            print(f"   Pooled output shape: {pooled_output_onnx.shape}")
            print(f"   Pooled output stats: min={pooled_output_onnx.min():.6f}, max={pooled_output_onnx.max():.6f}, mean={pooled_output_onnx.mean():.6f}")
            
    except ImportError:
        print(f"⚠️  onnxruntime not available, skipping ONNX validation")
    except Exception as e:
        print(f"🔴 ONNX model validation ERROR for {clip_type_str}: {e}")
        print(f"   Stopping conversion - ONNX model failed to run!")
        return {}

    # TEST ONNX MODEL WITH PYTORCH DIRECTLY
    print(f"\n🔥 Testing ONNX model with PyTorch directly for {clip_type_str}...")
    try:
        # Load the ONNX model
        print(f"   Loading ONNX model for inspection...")
        onnx_model_proto = onnx.load(output_onnx)
        
        print(f"   ONNX model loaded successfully")
        print(f"   Model IR version: {onnx_model_proto.ir_version}")
        print(f"   Model opset version: {onnx_model_proto.opset_import[0].version if onnx_model_proto.opset_import else 'Unknown'}")
        
        # Validate the ONNX model
        try:
            onnx.checker.check_model(onnx_model_proto)
            print(f"   ✅ ONNX model structure validation passed")
        except Exception as e:
            print(f"   ⚠️  ONNX model structure validation warning: {e}")
        
        # Try to run ONNX model with different approaches
        print(f"\n   🏃 Running multiple ONNX validation approaches...")
        
        # Approach 1: Test with multiple batch sizes and sequence lengths
        test_shapes = [
            (1, 77),    # Standard
            (2, 77),    # Batch size 2
            (1, 64),    # Different sequence length
        ]
        
        for batch_size, seq_len in test_shapes:
            try:
                print(f"   Testing shape ({batch_size}, {seq_len})...")
                
                # Create test tokens
                test_tokens = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
                for b in range(batch_size):
                    test_tokens[b, 0] = start_token
                    test_tokens[b, 1] = end_token
                    test_tokens[b, 2:] = pad_token
                
                # Get embeddings
                test_embeddings = input_embeddings(test_tokens, out_dtype=torch.float32)
                
                # Test with ONNX Runtime
                test_input_np = test_embeddings.cpu().numpy()
                test_ort_inputs = {ort_session.get_inputs()[0].name: test_input_np}
                test_ort_outputs = ort_session.run(None, test_ort_inputs)
                
                # Check for issues
                test_hidden = test_ort_outputs[0]
                test_pooled = test_ort_outputs[1]
                
                test_hidden_nan = np.isnan(test_hidden).any()
                test_pooled_nan = np.isnan(test_pooled).any()
                
                if test_hidden_nan or test_pooled_nan:
                    print(f"   🔴 Shape ({batch_size}, {seq_len}) produces NaN values!")
                else:
                    print(f"   ✅ Shape ({batch_size}, {seq_len}) produces valid values")
                    print(f"      Hidden range: [{test_hidden.min():.4f}, {test_hidden.max():.4f}]")
                    print(f"      Pooled range: [{test_pooled.min():.4f}, {test_pooled.max():.4f}]")
                    
            except Exception as e:
                print(f"   ⚠️  Shape ({batch_size}, {seq_len}) test failed: {e}")
        
        # Approach 2: Test with different token patterns
        print(f"\n   🎯 Testing different token patterns...")
        token_patterns = [
            ("start_only", [start_token] + [pad_token] * 76),
            ("comfy_standard", [start_token, end_token] + [pad_token] * 75),
            ("all_start", [start_token] * 77),
            ("all_pad", [pad_token] * 77),
            ("all_end", [end_token] * 77),
        ]
        
        for pattern_name, token_list in token_patterns:
            try:
                print(f"   Testing token pattern '{pattern_name}'...")
                
                # Create test tokens
                test_tokens = torch.tensor([token_list], dtype=torch.long, device=device)
                test_embeddings = input_embeddings(test_tokens, out_dtype=torch.float32)
                
                # Test with ONNX Runtime  
                test_input_np = test_embeddings.cpu().numpy()
                test_ort_inputs = {ort_session.get_inputs()[0].name: test_input_np}
                test_ort_outputs = ort_session.run(None, test_ort_inputs)
                
                test_hidden = test_ort_outputs[0]
                test_pooled = test_ort_outputs[1]
                
                test_hidden_nan = np.isnan(test_hidden).any()
                test_pooled_nan = np.isnan(test_pooled).any()
                
                if test_hidden_nan or test_pooled_nan:
                    print(f"   🔴 Pattern '{pattern_name}' produces NaN values!")
                    print(f"      Tokens: {token_list[:10]}{'...' if len(token_list) > 10 else ''}")
                else:
                    print(f"   ✅ Pattern '{pattern_name}' produces valid values")
                    
            except Exception as e:
                print(f"   ⚠️  Pattern '{pattern_name}' test failed: {e}")
                
    except ImportError as e:
        print(f"   ⚠️  Required packages not available: {e}")
    except Exception as e:
        print(f"🔴 PyTorch ONNX validation ERROR for {clip_type_str}: {e}")
        print(f"   Continuing with TensorRT conversion...")

    # TRT conversion starts here - Use centralized TensorRT manager
    network = trt_manager.create_network()
    parser = trt_manager.create_onnx_parser(network)
    success = parser.parse_from_file(output_onnx)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        print("ONNX load ERROR")
        return {}

    config = trt_manager.create_builder_config()
    profile = trt_manager.builder.create_optimization_profile()
    setup_timing_cache(config, timing_cache_path)
    config.progress_monitor = TQDMProgressMonitor()

    # Set up optimization profile - embeddings have shape [batch, sequence, embedding_dim]
    embedding_dim = embedded_tokens.shape[-1]  # Get embedding dimension from actual tensor
    min_shape_list = [int(x) for x in inputs_shapes_min] + [embedding_dim]
    opt_shape_list = [int(x) for x in inputs_shapes_opt] + [embedding_dim]
    max_shape_list = [int(x) for x in inputs_shapes_max] + [embedding_dim]
    
    min_shape = tuple(min_shape_list)
    opt_shape = tuple(opt_shape_list)
    max_shape = tuple(max_shape_list)
    
    profile.set_shape("embeddings", min_shape, opt_shape, max_shape)

    config.set_flag(trt_manager.builder.BuilderFlag.FP16)
    config.add_optimization_profile(profile)

    # Generate filename
    clip_type = "clip_l" if is_clip_l else "clip_g"
   
    filename_prefix = "{}_{}${}".format(
        filename_prefix,
        clip_type,
        "-".join(
            (
                "stat",
                "b",
                str(batch_size_opt),
                "s",
                str(sequence_length_opt),
            )
        ),
    )

    serialized_engine = trt_manager.build_serialized_network(network, config)

    full_output_folder, filename, counter, subfolder, filename_prefix = (
        folder_paths.get_save_image_path(filename_prefix, output_dir)
    )
    output_trt_engine = os.path.join(
        full_output_folder, f"{filename}_{counter:05}_.engine"
    )

    with open(output_trt_engine, "wb") as f:
        f.write(serialized_engine)

    save_timing_cache(config, timing_cache_path)

    clip_type = "CLIP-L" if is_clip_l else "CLIP-G"
    print(f"TensorRT {clip_type} conversion complete! Engine saved to: {output_trt_engine}")
    return {}


class CLIP_L_CONVERTER_TENSORRT_STATIC:
   
    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_CLIP_L_STAT"}),
                "batch_size_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "sequence_length_opt": (
                    "INT",
                    {
                        "default": 77,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                    },
                ),
            },
        }

    def convert(
        self,
        clip,
        filename_prefix,
        batch_size_opt,
        sequence_length_opt,
    ):
        try:
            convert_clip(
                clip,
                filename_prefix,
                batch_size_opt,
                batch_size_opt,
                batch_size_opt,
                sequence_length_opt,
                sequence_length_opt,
                sequence_length_opt,
                is_clip_l=True,
            )
        except Exception as e:
            print(f"CLIP_L_CONVERTER_TENSORRT_STATIC - Error converting: {e}")
            print(f"CLIP_L_CONVERTER_TENSORRT_STATIC - Full stack trace:")
            traceback.print_exc()
            return ()
        return ()


class CLIP_G_CONVERTER_TENSORRT_STATIC:

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_CLIP_G_STAT"}),
                "batch_size_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "sequence_length_opt": (
                    "INT",
                    {
                        "default": 77,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                    },
                ),
            },
        }

    def convert(
        self,
        clip,
        filename_prefix,
        batch_size_opt,
        sequence_length_opt,
    ):
        try:
            convert_clip(
                clip,
                filename_prefix,
                batch_size_opt,
                batch_size_opt,
                batch_size_opt,
                sequence_length_opt,
                sequence_length_opt,
                sequence_length_opt,
                is_clip_l=False,
            )
        except Exception as e:
            print(f"CLIP_G_CONVERTER_TENSORRT_STATIC - Error converting: {e}")
            print(f"CLIP_G_CONVERTER_TENSORRT_STATIC - Full stack trace:")
            traceback.print_exc()
            return ()
        return ()