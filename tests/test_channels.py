import asyncio
from multiprocessing import get_context, shared_memory
from types import SimpleNamespace

import numpy as np
import pytest

from stream_pack.channels import SessionChannels


def echo(io, completed):
    frame = io.image_inputs.get(timeout=5)
    io.image_outputs.put_nowait(frame.side_data.input + 1)
    completed.wait(5)


def test_spawned_media_roundtrip_and_reclamation():
    channels = SessionChannels(limit_bytes=4096)
    context = get_context("spawn")
    completed = context.Event()
    process = context.Process(target=echo, args=(channels.io, completed))
    try:
        process.start()
        values = np.zeros((1, 4, 4, 3), dtype=np.float32)
        channels.io.image_inputs.put(
            SimpleNamespace(side_data=SimpleNamespace(input=values))
        )

        async def output():
            return await asyncio.wait_for(channels.io.image_outputs.get(), timeout=10)

        np.testing.assert_array_equal(asyncio.run(output()), values + 1)
        assert not dict(channels.store.registry)
        completed.set()
        process.join(10)
        assert process.exitcode == 0
    finally:
        completed.set()
        if process.is_alive():
            process.terminate()
            process.join()
        channels.close()


def test_killed_worker_buffers_are_reclaimed():
    channels = SessionChannels(limit_bytes=4096)
    descriptor = channels.store.write(np.ones(16, dtype=np.float32))
    channels.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=descriptor["name"])
