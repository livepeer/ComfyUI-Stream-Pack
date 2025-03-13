"""ComfyUI Stream Pack Nodes."""

from .node_wrappers import feature_bank
from .node_wrappers import facemesh
from .node_wrappers import batched_denoising
# Collect all NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS from submodules.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import and update mappings from submodules.
for module in [feature_bank, facemesh, batched_denoising]:
    if hasattr(module, "NODE_CLASS_MAPPINGS"):
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS["StreamPack"] = "Stream Pack Nodes"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
