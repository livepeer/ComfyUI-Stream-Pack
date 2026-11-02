"""Regenerate bundled descriptors from reviewed model pins and public parameters."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stream_pack.native import ace_workflow
from stream_pack.protocols import wire_schema

ROOT = Path(__file__).resolve().parents[1] / "stream_pack" / "capabilities"
MODELS = [
    (
        "qwen-chat",
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "apache-2.0",
        "chat_completions",
        "Chat and image understanding",
        {
            "messages": {"type": "array", "minItems": 1, "items": {"type": "object"}},
            "max_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": 8192,
                "default": 1024,
            },
        },
        ["messages"],
    ),
    (
        "flux-image",
        "black-forest-labs/FLUX.2-klein-4B",
        "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "apache-2.0",
        "images_generations",
        "Generate or edit an image",
        {},
        ["prompt"],
    ),
    (
        "qwen-edit",
        "Qwen/Qwen-Image-Edit-2511",
        "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
        "apache-2.0",
        "images_edits",
        "Edit an image using instructions",
        {},
        ["prompt", "image"],
    ),
    (
        "wan-video",
        "Wan-AI/Wan2.2-I2V-A14B",
        "206a9ee1b7bfaaf8f7e4d81335650533490646a3",
        "apache-2.0",
        "videos",
        "Animate an image into a video",
        {},
        ["prompt", "image"],
    ),
    (
        "qwen-speech",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "fd4b254389122332181a7c3db7f27e918eec64e3",
        "apache-2.0",
        "audio_speech",
        "Synthesize speech from a reference voice",
        {
            "input": {"type": "string", "minLength": 1, "maxLength": 10000},
            "reference_audio": {"type": "string", "minLength": 1},
            "reference_text": {"type": "string", "minLength": 1},
            "language": {"type": "string", "default": "Auto"},
        },
        ["input", "reference_audio", "reference_text"],
    ),
    (
        "birefnet",
        "ZhengPeng7/BiRefNet",
        "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
        "mit",
        "image_transformations",
        "Remove an image background",
        {"image": {"type": "string", "minLength": 1}},
        ["image"],
    ),
    (
        "whisper",
        "openai/whisper-large-v3-turbo",
        "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        "mit",
        "audio_transcriptions",
        "Transcribe recorded speech",
        {
            "audio": {"type": "string", "minLength": 1},
            "language": {"type": "string", "default": ""},
        },
        ["audio"],
    ),
    (
        "ace-music",
        "ACE-Step/Ace-Step1.5",
        "19671f406d603126926c1b7e2adc169acbcade22",
        "mit",
        "music_generations",
        "Generate music from a description and lyrics",
        {
            "prompt": {"type": "string", "minLength": 1, "maxLength": 10000},
            "lyrics": {"type": "string", "default": ""},
            "seconds": {"type": "integer", "minimum": 1, "maximum": 120, "default": 30},
        },
        ["prompt"],
    ),
    (
        "triposr",
        "stabilityai/TripoSR",
        "5b521936b01fbe1890f6f9baed0254ab6351c04a",
        "mit",
        "meshes_generations",
        "Reconstruct a mesh from an object image",
        {"image": {"type": "string", "minLength": 1}},
        ["image"],
    ),
]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for (
        name,
        source,
        revision,
        license_,
        operation,
        description,
        properties,
        required,
    ) in MODELS:
        if not properties:
            properties = {
                "prompt": {"type": "string", "minLength": 1, "maxLength": 10000},
                "seed": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 2147483647,
                    "default": 0,
                },
            }
            if name != "flux-image":
                properties["image"] = {"type": "string", "minLength": 1}
            if name == "wan-video":
                properties["frames"] = {
                    "type": "integer",
                    "minimum": 17,
                    "maximum": 81,
                    "default": 81,
                }
            else:
                properties.update(
                    {
                        key: {
                            "type": "integer",
                            "minimum": 256,
                            "maximum": 1536,
                            "default": 1024,
                        }
                        for key in ("width", "height")
                    }
                )
        node = "StreamPack" + "".join(part.capitalize() for part in name.split("-"))
        defaults = {
            k: json.dumps(v.get("default", []))
            if v["type"] in {"array", "object"}
            else v.get("default", "" if v["type"] == "string" else 0)
            for k, v in properties.items()
        }
        document = {
            "version": 1,
            "id": name,
            "description": description,
            "model": {"source": source, "revision": revision, "license": license_},
            "operation": operation,
            "modes": ["job", "sync"],
            "admission": "candidate",
            "dependency_group": "chat"
            if name == "qwen-chat"
            else "language"
            if name in {"qwen-speech", "whisper"}
            else "media",
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "artifacts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "media_type": {"type": "string"},
                            },
                            "required": ["path", "media_type"],
                            "additionalProperties": False,
                        },
                    },
                    "usage": {"type": "object"},
                },
                "required": ["artifacts"],
                "additionalProperties": False,
            },
            "bindings": {k: {"node": "1", "input": k} for k in properties},
            "output_binding": {"node": "2", "key": "service_result"},
            "workflow": {
                "1": {"class_type": node, "inputs": defaults},
                "2": {"class_type": "StreamPackOutput", "inputs": {"value": ["1", 0]}},
            },
        }
        document["wire_schema"] = wire_schema(operation, document["input_schema"])
        if name == "ace-music":
            document["workflow"] = ace_workflow()
            document["bindings"] = {
                "prompt": {"node": "3", "input": "tags"},
                "lyrics": {"node": "3", "input": "lyrics"},
                "seconds": {"node": "2", "input": "seconds"},
            }
            document["output_binding"] = {"node": "10", "key": "service_result"}
            document["components"] = [
                {
                    "source": "Comfy-Org/ace_step_1.5_ComfyUI_files",
                    "revision": "371bb1680f8c88d8c1083f6ffb6330d3d55bff76",
                    "file": "checkpoints/ace_step_1.5_turbo_aio.safetensors",
                    "license": "apache-2.0",
                }
            ]
        (ROOT / (name + ".json")).write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
