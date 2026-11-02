"""Versioned public contracts and explicit workflow bindings."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class Capability:
    document: dict[str, Any]

    def __post_init__(self):
        d = self.document
        if d["version"] != 1 or not re.fullmatch(r"[a-z][a-z0-9_-]*", d["id"]):
            raise ValueError("Invalid capability version or identifier")
        if not re.fullmatch(r"[0-9a-f]{40}", d["model"]["revision"]):
            raise ValueError("Model revision must be an immutable commit")
        if d["model"]["license"] not in {
            "apache-2.0",
            "mit",
            "bsd-3-clause",
            "bsd-2-clause",
        }:
            raise ValueError("Model license is not in the permissive catalog")
        Draft202012Validator.check_schema(d["input_schema"])
        Draft202012Validator.check_schema(d["output_schema"])
        self.validate_workflow()

    @property
    def id(self) -> str:
        return self.document["id"]

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.document, sort_keys=True).encode()
        ).hexdigest()

    def validate(self, inputs: Mapping[str, Any]) -> None:
        Draft202012Validator(self.document["input_schema"]).validate(dict(inputs))

    def validate_output(self, output: Any) -> None:
        Draft202012Validator(self.document["output_schema"]).validate(output)

    def validate_workflow(self, node_classes: Mapping[str, type] | None = None) -> None:
        d, workflow = self.document, self.document["workflow"]
        properties = d["input_schema"].get("properties", {})
        if set(properties) != set(d["bindings"]):
            raise ValueError("Every public parameter must have exactly one binding")
        for field, binding in d["bindings"].items():
            node = workflow.get(binding["node"])
            if node is None or binding["input"] not in node["inputs"]:
                raise ValueError(f"Invalid workflow binding: {field}")
        for node in workflow.values():
            if node_classes is not None:
                cls = node_classes.get(node["class_type"])
                if cls is None:
                    raise ValueError(
                        "Required node is not installed: " + node["class_type"]
                    )
                declared = cls.INPUT_TYPES()
                allowed = {
                    **declared.get("required", {}),
                    **declared.get("optional", {}),
                }
                if set(node["inputs"]) - set(allowed):
                    raise ValueError("Workflow contains undeclared node inputs")
            for value in node["inputs"].values():
                if (
                    isinstance(value, list)
                    and len(value) == 2
                    and isinstance(value[0], str)
                    and value[0] not in workflow
                ):
                    raise ValueError("Workflow references a missing node")
        output = d["output_binding"]
        if output["node"] not in workflow:
            raise ValueError("Missing workflow output node")
        if node_classes is not None:
            cls = node_classes[workflow[output["node"]]["class_type"]]
            if not getattr(cls, "OUTPUT_NODE", False):
                raise ValueError("Public output must bind to an output node")

    def render(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        self.validate(inputs)
        workflow = copy.deepcopy(self.document["workflow"])
        for field, binding in self.document["bindings"].items():
            schema = self.document["input_schema"]["properties"][field]
            if field in inputs or "default" in schema:
                value = inputs.get(field, schema.get("default"))
                if schema.get("type") in {"object", "array"}:
                    value = json.dumps(value)
                workflow[binding["node"]]["inputs"][binding["input"]] = value
        return workflow

    def tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.id,
                "description": self.document["description"],
                "parameters": self.document["input_schema"],
            },
        }


def load_catalog() -> dict[str, Capability]:
    root = files("stream_pack").joinpath("capabilities")
    capabilities = [
        Capability(json.loads(p.read_text(encoding="utf-8")))
        for p in root.iterdir()
        if p.name.endswith(".json")
    ]
    result = {c.id: c for c in capabilities}
    if len(result) != len(capabilities):
        raise ValueError("Duplicate capability identifier")
    return result
