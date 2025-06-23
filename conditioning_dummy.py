import torch
import numpy as np

class ConditioningDebugNode:
    """
    A debug node that receives conditioning and prints detailed information about it.
    Useful for understanding the structure and content of CLIP outputs.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "debug_conditioning"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT/Debug"
    
    def debug_conditioning(self, conditioning):
        print(f"\n{'='*60}")
        print(f"CONDITIONING DEBUG INFO")
        print(f"{'='*60}")
        
        print(f"📋 Conditioning type: {type(conditioning)}")
        print(f"📋 Conditioning length: {len(conditioning) if hasattr(conditioning, '__len__') else 'No length'}")
        
        if isinstance(conditioning, (list, tuple)):
            print(f"📋 Conditioning is a list/tuple with {len(conditioning)} elements")
            
            for i, cond_item in enumerate(conditioning):
                print(f"\n🔍 Conditioning Item [{i}]:")
                print(f"   Type: {type(cond_item)}")
                
                if isinstance(cond_item, (list, tuple)):
                    print(f"   Length: {len(cond_item)}")
                    
                    for j, sub_item in enumerate(cond_item):
                        print(f"\n   📦 Sub-item [{j}]:")
                        print(f"      Type: {type(sub_item)}")
                        
                        if torch.is_tensor(sub_item):
                            print(f"      Shape: {sub_item.shape}")
                            print(f"      Dtype: {sub_item.dtype}")
                            print(f"      Device: {sub_item.device}")
                            if sub_item.numel() > 0:
                                print(f"      Min: {sub_item.min().item():.6f}")
                                print(f"      Max: {sub_item.max().item():.6f}")
                                print(f"      Mean: {sub_item.mean().item():.6f}")
                                nan_count = torch.isnan(sub_item).sum().item()
                                print(f"      NaN count: {nan_count}/{sub_item.numel()}")
                        elif isinstance(sub_item, dict):
                            print(f"      Dict keys: {list(sub_item.keys())}")
                        else:
                            print(f"      Value: {sub_item}")
        
        print(f"{'='*60}")
        print(f"END DEBUG INFO")
        print(f"{'='*60}\n")
        
        return (conditioning,)


class CLIPTextEncodeDebug:
    """
    A debug version of CLIPTextEncode that shows what the CLIP model produces.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "clip": ("CLIP",),
            },
            "optional": {
                "debug_prefix": ("STRING", {"default": "CLIP_DEBUG", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode_with_debug"
    CATEGORY = "TensorRT/Debug"
    
    def encode_with_debug(self, clip, text, debug_prefix="CLIP_DEBUG"):
        print(f"\n{'='*60}")
        print(f"{debug_prefix} - CLIP ENCODING DEBUG")
        print(f"{'='*60}")
        
        print(f"📝 Input text: '{text}'")
        print(f"📎 CLIP model type: {type(clip)}")
        
        # Get the raw CLIP model to examine its properties
        if hasattr(clip, 'cond_stage_model'):
            cond_model = clip.cond_stage_model
            print(f"📎 Cond stage model type: {type(cond_model)}")
            
            if hasattr(cond_model, 'clip_l'):
                print(f"📎 Has CLIP-L: {type(cond_model.clip_l)}")
                clip_l = cond_model.clip_l
                print(f"   CLIP-L special_tokens: {getattr(clip_l, 'special_tokens', 'N/A')}")
                print(f"   CLIP-L layer: {getattr(clip_l, 'layer', 'N/A')}")
                print(f"   CLIP-L max_length: {getattr(clip_l, 'max_length', 'N/A')}")
                
            if hasattr(cond_model, 'clip_g'):
                print(f"📎 Has CLIP-G: {type(cond_model.clip_g)}")
                clip_g = cond_model.clip_g
                print(f"   CLIP-G special_tokens: {getattr(clip_g, 'special_tokens', 'N/A')}")
                print(f"   CLIP-G layer: {getattr(clip_g, 'layer', 'N/A')}")
                print(f"   CLIP-G max_length: {getattr(clip_g, 'max_length', 'N/A')}")
        
        # Tokenize the text first to see what tokens we get
        print(f"\n🔤 Tokenization Debug:")
        try:
            if hasattr(clip, 'tokenize'):
                tokens = clip.tokenize(text)
                print(f"   Tokenized: {tokens}")
                
                # Print token details if it's a dict
                if isinstance(tokens, dict):
                    for key, token_list in tokens.items():
                        print(f"   {key} tokens: {token_list}")
                        if isinstance(token_list, list) and len(token_list) > 0:
                            first_batch = token_list[0]
                            print(f"   {key} first batch length: {len(first_batch)}")
                            print(f"   {key} first few tokens: {first_batch[:10] if len(first_batch) > 10 else first_batch}")
        except Exception as e:
            print(f"   Tokenization error: {e}")
        
        # Now encode and examine the result
        print(f"\n🎯 Encoding...")
        try:
            conditioning = clip.encode(text)
            
            print(f"✅ Encoding successful!")
            print(f"   Result type: {type(conditioning)}")
            print(f"   Result length: {len(conditioning) if hasattr(conditioning, '__len__') else 'No length'}")
            
            # Use our debug function to examine the conditioning
            debug_node = ConditioningDebugNode()
            debug_node.debug_conditioning(conditioning, prefix=f"{debug_prefix}_RESULT", detailed=True)
            
            return (conditioning,)
            
        except Exception as e:
            print(f"❌ Encoding failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty conditioning on failure
            return ([[torch.zeros((1, 77, 768)), {}]],)
