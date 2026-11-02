"""Original workflow class names, now owned by Stream Pack."""

from stream_pack.legacy.load_tensor import LoadTensor
from stream_pack.legacy.save_tensor import SaveTensor
from stream_pack.legacy.save_text_tensor import SaveTextTensor
from stream_pack.legacy.invert_image import InvertImage
from stream_pack.legacy.load_audio_tensor import LoadAudioTensor
from stream_pack.legacy.save_audio_tensor import SaveAudioTensor
from stream_pack.legacy.pitch_shift import PitchShifter
from stream_pack.legacy.flux_klein_stream import FluxKleinStream

NODE_CLASS_MAPPINGS = {
    cls.__name__: cls
    for cls in (
        LoadTensor,
        SaveTensor,
        SaveTextTensor,
        InvertImage,
        LoadAudioTensor,
        SaveAudioTensor,
        PitchShifter,
        FluxKleinStream,
    )
}
