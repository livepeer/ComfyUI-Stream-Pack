"""
Real-time Audio Transcription Node for ComfyUI.

This node buffers audio segments and performs transcription using faster-whisper,
with controlled output timing to prevent message flooding.
"""

# Import node classes
from ..src.audio_transcription.transcription_nodes import AudioTranscriptionNode, SRTGeneratorNode

# Define NODE_CLASS_MAPPINGS for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AudioTranscriptionNode": AudioTranscriptionNode,
    "SRTGeneratorNode": SRTGeneratorNode,
}

# Define NODE_DISPLAY_NAME_MAPPINGS for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioTranscriptionNode": "Audio Transcription (Real-time)",
    "SRTGeneratorNode": "SRT Subtitle Generator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
