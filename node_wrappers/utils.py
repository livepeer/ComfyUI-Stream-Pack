"""Node wrappers for utility nodes."""

from ..src.utils.utils import GetTensorDimensionsNode

NODE_CLASS_MAPPINGS = {
    "GetTensorDimensions": GetTensorDimensionsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetTensorDimensions": "Get Tensor Dimensions"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 