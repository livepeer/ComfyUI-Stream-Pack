# Import node classes
from ..src.super_resolution.nodes.sr_nodes import SuperResolutionModelLoader, SuperResolutionUpscale

# Define NODE_CLASS_MAPPINGS for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SuperResolutionModelLoader": SuperResolutionModelLoader,
    "SuperResolutionUpscale": SuperResolutionUpscale,
}

# Define NODE_DISPLAY_NAME_MAPPINGS for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperResolutionModelLoader": "SR Model Loader",
    "SuperResolutionUpscale": "SR Upscale",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]