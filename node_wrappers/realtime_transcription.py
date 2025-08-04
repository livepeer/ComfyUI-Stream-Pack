from ..src.realtime_transcription.audio_transcription import AudioTranscriptionNode
from ..src.realtime_transcription.srt_generator import SRTGeneratorNode

NODE_CLASS_MAPPINGS = {
    "AudioTranscriptionNode": AudioTranscriptionNode, "SRTGeneratorNode": SRTGeneratorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioTranscriptionNode": "Audio Transcription (Real-time)", "SRTGeneratorNode": "SRT Generator"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]