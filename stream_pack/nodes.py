"""Service nodes. Backend imports happen only when a workflow executes."""

import json


class ServiceInput:
    CATEGORY = "StreamPack/Service"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("STRING", {"default": ""})}}

    def execute(self, value):
        return (value,)


class ServiceModel:
    CATEGORY = "StreamPack/Service"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"capability": ("STRING",), "inputs_json": ("STRING",)}}

    def execute(self, capability, inputs_json):
        from .models import execute

        return (json.dumps(execute(capability, json.loads(inputs_json))),)


class ServiceOutput:
    CATEGORY = "StreamPack/Service"
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("STRING",)}}

    def execute(self, value):
        return {"ui": {"service_result": [json.loads(value)]}}


NODE_CLASS_MAPPINGS = {
    "StreamPackInput": ServiceInput,
    "StreamPackModel": ServiceModel,
    "StreamPackOutput": ServiceOutput,
}


def capability_node(capability):
    schema = capability.document["input_schema"]

    class ModelNode:
        CATEGORY = "StreamPack/Models"
        RETURN_TYPES = ("STRING",)
        FUNCTION = "execute"

        @classmethod
        def INPUT_TYPES(cls):
            result = {"required": {}, "optional": {}}
            for name, field in schema["properties"].items():
                kind = {"integer": "INT", "number": "FLOAT", "boolean": "BOOLEAN"}.get(
                    field["type"], "STRING"
                )
                options = {
                    k: field[k] for k in ("default", "minimum", "maximum") if k in field
                }
                for source, target in (("minimum", "min"), ("maximum", "max")):
                    if source in options:
                        options[target] = options.pop(source)
                bucket = (
                    "required" if name in schema.get("required", []) else "optional"
                )
                result[bucket][name] = (kind, options)
            return result

        def execute(self, **inputs):
            from .models import execute

            for name, field in schema["properties"].items():
                if name in inputs and field["type"] in {"array", "object"}:
                    inputs[name] = json.loads(inputs[name])
            capability.validate(inputs)
            result = execute(capability.id, inputs)
            capability.validate_output(result)
            return (json.dumps(result),)

    return ModelNode


from .catalog import load_catalog

for _capability in load_catalog().values():
    NODE_CLASS_MAPPINGS[_capability.document["workflow"]["1"]["class_type"]] = (
        capability_node(_capability)
    )
