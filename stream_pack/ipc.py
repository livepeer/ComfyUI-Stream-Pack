"""Bounded per-session IPC. The creating process owns shared-memory cleanup."""

from __future__ import annotations

from multiprocessing import shared_memory
from uuid import uuid4


class SessionBuffers:
    def __init__(self, max_bytes: int = 64 * 1024 * 1024):
        self.session = uuid4().hex
        self.max_bytes = max_bytes
        self._buffers: dict[str, shared_memory.SharedMemory] = {}
        self._size = 0

    def put(self, data: bytes) -> dict:
        if not data or self._size + len(data) > self.max_bytes:
            raise ValueError("Session buffer limit exceeded or empty buffer")
        memory = shared_memory.SharedMemory(create=True, size=len(data))
        memory.buf[: len(data)] = data
        self._buffers[memory.name] = memory
        self._size += len(data)
        return {"session": self.session, "name": memory.name, "size": len(data)}

    @staticmethod
    def read(descriptor: dict, session: str) -> bytes:
        if (
            descriptor["session"] != session
            or not isinstance(descriptor["size"], int)
            or descriptor["size"] <= 0
        ):
            raise ValueError("Invalid session buffer")
        memory = shared_memory.SharedMemory(name=descriptor["name"])
        try:
            if descriptor["size"] > memory.size:
                raise ValueError("Invalid buffer size")
            return bytes(memory.buf[: descriptor["size"]])
        finally:
            memory.close()

    def close(self):
        for memory in self._buffers.values():
            memory.close()
            memory.unlink()
        self._buffers.clear()
        self._size = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
