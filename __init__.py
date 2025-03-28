"""Expose all nodes in the Stream Pack."""

import logging
import pathlib
import importlib
import traceback

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_nodes() -> tuple[dict[str, type], dict[str, str]]:
    """Load all nodes in the Stream Pack."""
    nodes_dir = pathlib.Path(__file__).parent / "node_wrappers"
    node_class_mappings, node_display_name_mappings = {}, {}

    # Dynamically import all Python modules in the node_wrappers directory.
    for module_path in nodes_dir.iterdir():
        if (
            module_path.is_file()
            and module_path.suffix == ".py"
            and module_path.stem != "__init__"
        ):
            try:
                module_name = f"{__package__}.node_wrappers.{module_path.stem}"
                module = importlib.import_module(module_name)

                # Update mappings if defined in the module.
                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    node_class_mappings.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    node_display_name_mappings.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception:
                format_exc = traceback.format_exc()
                log.error(f"Failed to load module {module_name}:\n{format_exc}")

    return node_class_mappings, node_display_name_mappings


# Dynamically collect mappings from submodules.
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = load_nodes()
NODE_DISPLAY_NAME_MAPPINGS["StreamPack"] = "Stream Pack Nodes"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
