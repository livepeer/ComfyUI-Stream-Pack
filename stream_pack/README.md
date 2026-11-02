# Local capability contracts

`stream_pack` can be imported without torch, diffusers or transformers. Install only
the model extras needed by the worker image. `load_catalog()` returns nine immutable
model descriptors; each descriptor includes its workflow, public parameter bindings,
JSON Schemas and an agent tool definition. Registration is candidate-only.

Run `python tools/build_catalog.py` after deliberately updating a reviewed model pin.
The generated JSON files are package resources, not checkout-relative assets.
`Capability.validate_workflow(NODE_CLASS_MAPPINGS)` checks installed node mappings.

All model loading is offline and uses immutable local snapshot paths, including
nested tokenizer/configuration loads. Wan's official Diffusers layout is a declared,
pinned execution component. The media extra installs a pinned TripoSR reference
package; its DINO configuration is another declared component. ACE-Step uses native
core nodes and an explicitly retained checkpoint wrapper, with its own pinned file.
Nodes never fetch these optional artifacts during inference.

Six model integrations have produced representative outputs in candidate GPU tests
on eliteprd1. These checks used mounted working sources and do not certify all nine
models or the final immutable deployment. Chat/vision, image editing and Wan still
require model acceptance. The media/speech and chat profiles are separate: a full
uv resolution fails because they require incompatible Transformers major versions.

ComfyStream injects legacy queues through `legacy_io`. New process integrations can
bind session-scoped queues explicitly; never share a customer's feedback or audio
buffer with another session. `SessionBuffers` owns bounded shared-memory blocks and
must be closed by the allocating process, including after worker failure.

Legacy model behavior, including FLUX feedback and pitch shifting, now lives under
`stream_pack.legacy`. FFmpeg-branch frame/container transformations live under
`stream_pack.video_tools`. ComfyStream retains old module paths and node class names
through aliases. Its candidate live client now uses the shared-memory IPC channels.

Model licenses are recorded per immutable checkpoint. Preserve the repository license
and upstream notices; optional model code and dependencies require their own license
review. Permissive model metadata does not certify generated content rights.
