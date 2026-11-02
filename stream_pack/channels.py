"""Session-scoped media transport for spawned embedded workflow workers."""

from __future__ import annotations

import asyncio
import queue
from multiprocessing import get_context, shared_memory
from types import SimpleNamespace
from uuid import uuid4


class ArrayStore:
    def __init__(self, registry, lock, session: str, limit: int):
        self.registry, self.session, self.limit = registry, session, limit
        self.lock = lock
        self.handles = {}

    def __getstate__(self):
        return {
            "registry": self.registry,
            "lock": self.lock,
            "session": self.session,
            "limit": self.limit,
            "handles": {},
        }

    def reap(self):
        for name in list(self.handles):
            if name not in self.registry:
                self.handles.pop(name).close()

    def write(self, value):
        import numpy as np

        self.reap()
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject or array.nbytes <= 0:
            raise ValueError("Unsupported media array")
        # One input producer and one workflow output producer share this budget.
        # Each channel is separately bounded, so allow at most two active allocations.
        with self.lock:
            if (
                array.nbytes > self.limit // 2
                or sum(self.registry.values()) + array.nbytes > self.limit
            ):
                raise ValueError("Session media memory limit exceeded")
            memory = shared_memory.SharedMemory(create=True, size=array.nbytes)
            memory.buf[:] = array.tobytes()
            self.handles[memory.name] = memory
            self.registry[memory.name] = array.nbytes
        return {
            "session": self.session,
            "name": memory.name,
            "shape": array.shape,
            "dtype": array.dtype.str,
        }

    def read(self, value):
        import numpy as np

        if value["session"] != self.session or value["name"] not in self.registry:
            raise ValueError("Media buffer belongs to another session")
        memory = shared_memory.SharedMemory(name=value["name"])
        try:
            dtype = np.dtype(value["dtype"])
            count = 1
            for dimension in value["shape"]:
                if not isinstance(dimension, int) or dimension <= 0:
                    raise ValueError("Invalid media shape")
                count *= dimension
            expected = self.registry[value["name"]]
            if (
                dtype.hasobject
                or count * dtype.itemsize != expected
                or expected > memory.size
            ):
                raise ValueError("Invalid media buffer layout")
            return np.ndarray(value["shape"], dtype=dtype, buffer=memory.buf).copy()
        finally:
            memory.close()
            memory.unlink()
            self.registry.pop(value["name"], None)

    def close(self):
        for name in list(self.registry):
            try:
                memory = shared_memory.SharedMemory(name=name)
            except FileNotFoundError:
                continue
            memory.close()
            memory.unlink()
            self.registry.pop(name, None)
        for memory in self.handles.values():
            memory.close()
        self.handles.clear()


class MediaQueue:
    def __init__(self, channel, store, *, frame=False):
        self.channel, self.store, self.frame = channel, store, frame

    def put(self, value, block=True, timeout=None):
        rate = getattr(value, "sample_rate", None) if self.frame else None
        payload = value.side_data.input if self.frame else value
        descriptor = self.store.write(payload)
        try:
            self.channel.put((descriptor, rate), block, timeout)
        except BaseException:
            self.store.read(descriptor)
            raise

    def put_nowait(self, value):
        self.put(value, block=False)

    def get(self, block=True, timeout=None):
        descriptor, rate = self.channel.get(block, timeout)
        array = self.store.read(descriptor)
        if self.frame:
            return SimpleNamespace(
                sample_rate=rate, side_data=SimpleNamespace(input=array, skipped=False)
            )
        return array

    def get_nowait(self):
        return self.get(block=False)

    def empty(self):
        return self.channel.empty()

    def full(self):
        return self.channel.full()


class AsyncOutput:
    def __init__(self, channel):
        self.channel = channel

    def put_nowait(self, value):
        self.channel.put_nowait(value)

    def get_nowait(self):
        return self.channel.get_nowait()

    def empty(self):
        return self.channel.empty()

    async def get(self):
        while True:
            try:
                return self.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)


class SessionChannels:
    def __init__(self, limit_bytes=128 * 1024 * 1024):
        self.manager = get_context("spawn").Manager()
        store = ArrayStore(
            self.manager.dict(), self.manager.RLock(), uuid4().hex, limit_bytes
        )
        self.store = store
        self.io = SimpleNamespace(
            image_inputs=MediaQueue(self.manager.Queue(1), store, frame=True),
            audio_inputs=MediaQueue(self.manager.Queue(8), store, frame=True),
            image_outputs=AsyncOutput(MediaQueue(self.manager.Queue(2), store)),
            audio_outputs=AsyncOutput(MediaQueue(self.manager.Queue(8), store)),
            text_outputs=AsyncOutput(self.manager.Queue(128)),
        )

    def close(self):
        self.store.close()
        self.manager.shutdown()


def bind_worker_channels(io):
    from stream_pack.legacy_io import bind_default

    bind_default(io)
