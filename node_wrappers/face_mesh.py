"""Provides nodes for facial landmark detection and utilities using MediaPipe FaceMesh.
Inspired by the ComfyUI ControlNet Auxiliary MediaPipe Face node.

References:
    https://github.com/Fannovel16/comfyui_controlnet_aux/blob/main/src/custom_controlnet_aux/mediapipe_face/mediapipe_face_common.py
"""

from ..src.media_pipe.face_mesh import *

NODE_CLASS_MAPPINGS = {
    "FaceMeshNode": FaceMeshNode,
    "FaceMeshDrawNode": FaceMeshDrawNode,
    "FaceMeshMaskNode": FaceMeshMaskNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceMeshNode": "Face Mesh",
    "FaceMeshDrawNode": "Face Mesh Draw",
    "FaceMeshMaskNode": "Face Region Mask",
}

__all__ = ["FaceMeshNode", "FaceMeshDrawNode", "FaceMeshMaskNode"]
