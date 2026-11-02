import json
import subprocess
import sys
from multiprocessing import shared_memory

import pytest
from jsonschema import ValidationError

from stream_pack.catalog import Capability, load_catalog
from stream_pack.ipc import SessionBuffers
from stream_pack.nodes import NODE_CLASS_MAPPINGS


def test_catalog_matches_installed_nodes_and_pins():
    catalog = load_catalog()
    assert len(catalog) == 9
    for capability in catalog.values():
        capability.validate_workflow(NODE_CLASS_MAPPINGS)
        assert capability.document["admission"] == "candidate"
        assert len(capability.digest) == 64


def test_unknown_fields_and_broken_bindings_fail():
    spec = load_catalog()["flux-image"]
    with pytest.raises(ValidationError):
        spec.render({"prompt": "cat", "workflow": {}})
    broken = json.loads(json.dumps(spec.document))
    broken["bindings"]["prompt"]["node"] = "missing"
    with pytest.raises(ValueError):
        Capability(broken)
    assert spec.render({"prompt": "cat"})["1"]["inputs"]["prompt"] == "cat"


def test_shared_buffers_enforce_session_limit_and_cleanup():
    with SessionBuffers(max_bytes=4) as session:
        descriptor = session.put(b"test")
        assert SessionBuffers.read(descriptor, session.session) == b"test"
        with pytest.raises(ValueError):
            SessionBuffers.read(descriptor, "another-customer")
        with pytest.raises(ValueError):
            session.put(b"x")
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=descriptor["name"])


def test_discovery_does_not_import_gpu_libraries():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import stream_pack.nodes, sys; assert not {'torch','diffusers','transformers'} & sys.modules.keys()",
        ],
        check=True,
    )
