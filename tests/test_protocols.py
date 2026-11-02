import pytest
from jsonschema import ValidationError

from stream_pack.catalog import load_catalog
from stream_pack.protocols import normalize_wire


def test_image_provider_fields_map_to_workflow_parameters():
    spec = load_catalog()["flux-image"]
    result = normalize_wire(
        spec.document,
        {"prompt": "cat", "size": "512x512", "n": 1, "response_format": "b64_json"},
    )
    assert result == {"prompt": "cat", "width": 512, "height": 512}
    spec.validate(result)
    with pytest.raises(ValidationError):
        normalize_wire(spec.document, {"prompt": "cat", "n": 2})


def test_video_and_transcription_media_aliases():
    catalog = load_catalog()
    video = normalize_wire(
        catalog["wan-video"].document,
        {"prompt": "pan", "input_reference": "file_1", "seconds": "2"},
    )
    assert video["frames"] == 33
    assert video["image"] == "file_1"
    assert normalize_wire(catalog["whisper"].document, {"file": "file_1"}) == {
        "audio": "file_1"
    }
