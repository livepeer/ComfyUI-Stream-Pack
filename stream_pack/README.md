# Local capability contracts

`stream_pack` can be imported without torch, diffusers or transformers. Install only
the model extras needed by the worker image. `load_catalog()` returns nine immutable
model descriptors; each descriptor includes its workflow, public parameter bindings,
JSON Schemas and an agent tool definition. Registration is candidate-only.

Run `python tools/build_catalog.py` after deliberately updating a reviewed model pin.
The generated JSON files are package resources, not checkout-relative assets.
`Capability.validate_workflow(NODE_CLASS_MAPPINGS)` checks installed node mappings.

The adapters in `models.py` are candidate reference integrations, not evidence that
all nine models fit a GPU or have passed inference tests. Wan's original checkpoint
layout requires verification against the selected diffusers release. TripoSR requires
the separately installed `tsr` reference implementation; ACE-Step requires its reference
package and checkpoint directory. These optional dependencies are not silently fetched
by nodes. All model loading is offline and requires a populated immutable HF cache.

ComfyStream injects legacy queues through `legacy_io`. New process integrations can
bind session-scoped queues explicitly; never share a customer's feedback or audio
buffer with another session. `SessionBuffers` owns bounded shared-memory blocks and
must be closed by the allocating process, including after worker failure.

Legacy model behavior, including FLUX feedback and pitch shifting, now lives under
`stream_pack.legacy`. ComfyStream retains imports for old workflow class names. Its
existing live runner has not yet been switched to the new IPC channel mechanism.

Model licenses are recorded per immutable checkpoint. Preserve the repository license
and upstream notices; optional model code and dependencies require their own license
review. Permissive model metadata does not certify generated content rights.
