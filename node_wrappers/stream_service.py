"""Register dependency-light service nodes in ordinary ComfyUI workspaces."""

from stream_pack.nodes import NODE_CLASS_MAPPINGS

NODE_DISPLAY_NAME_MAPPINGS = {name: name.removeprefix("StreamPack") for name in NODE_CLASS_MAPPINGS}
