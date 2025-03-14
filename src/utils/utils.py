"""Utility nodes for ComfyUI Stream Pack."""

import torch
from . import AlwaysEqualProxy

class GetTensorDimensionsNode:
    """Node that extracts dimensions from a tensor."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (AlwaysEqualProxy("*"), {
                    "tooltip": "Input tensor in BHWC format (batch, height, width, channels) or HWC format (height, width, channels)"
                }),  # Accept any input type
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "count")
    FUNCTION = "get_dimensions"
    CATEGORY = "StreamPack/Utils"

    def get_dimensions(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a tensor")
            
        # Handle different tensor shapes
        if len(tensor.shape) == 4:  # BHWC format for regular tensors
            batch, height, width, channels = tensor.shape
            return (width, height, batch)
        elif len(tensor.shape) == 3:  # HWC format
            height, width, channels = tensor.shape
            return (width, height, 1)
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}") 