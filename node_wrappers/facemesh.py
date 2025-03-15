from ..src.media_pipe.facemesh import *

NODE_CLASS_MAPPINGS = {
    "FaceMeshNode": FaceMeshNode,
    "FaceMeshDrawNode": FaceMeshDrawNode,
    "FaceMeshMaskNode": FaceMeshMaskNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceMeshNode": "Face Mesh",
    "FaceMeshDrawNode": "Face Mesh Draw",
    "FaceMeshMaskNode": "Face Region Mask"
}
