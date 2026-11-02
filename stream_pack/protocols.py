"""Supported provider wire shapes, separate from workflow parameter bindings."""

import copy

from jsonschema import Draft202012Validator


def wire_schema(operation, inputs):
    schema = copy.deepcopy(inputs)
    props = schema["properties"]
    if operation in {"images_generations", "images_edits"}:
        props.pop("width", None)
        props.pop("height", None)
        props.update(
            size={
                "type": "string",
                "enum": ["256x256", "512x512", "1024x1024", "1536x1024", "1024x1536"],
                "default": "1024x1024",
            },
            n={"type": "integer", "const": 1},
            response_format={"type": "string", "const": "b64_json"},
        )
    elif operation == "audio_transcriptions":
        props["file"] = props.pop("audio")
        schema["required"] = ["file"]
        props["response_format"] = {
            "type": "string",
            "enum": ["json", "text"],
            "default": "json",
        }
    elif operation == "audio_speech":
        props.update(
            voice={"type": "string", "minLength": 1},
            response_format={"type": "string", "const": "wav"},
        )
        schema["required"] = ["input"]
        schema["anyOf"] = [
            {"required": ["voice"]},
            {"required": ["reference_audio", "reference_text"]},
        ]
    elif operation == "videos":
        props.pop("frames", None)
        props["input_reference"] = props.pop("image")
        props["seconds"] = {
            "type": "string",
            "enum": ["1", "2", "3", "4", "5"],
            "default": "4",
        }
        schema["required"] = ["prompt", "input_reference"]
    return schema


def normalize_wire(document, values):
    schema = document.get("wire_schema", document["input_schema"])
    Draft202012Validator(schema).validate(values)
    result = dict(values)
    operation = document["operation"]
    result.pop("response_format", None)
    if operation in {"images_generations", "images_edits"}:
        width, height = result.pop("size", "1024x1024").split("x")
        result.update(width=int(width), height=int(height))
        result.pop("n", None)
    elif operation == "audio_transcriptions":
        result["audio"] = result.pop("file")
    elif operation == "videos":
        result["image"] = result.pop("input_reference")
        result["frames"] = int(result.pop("seconds", "4")) * 16 + 1
    return result
