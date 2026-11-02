"""Compatibility bridge: the host injects queues, nodes do not import ComfyStream."""

from contextvars import ContextVar

_current = ContextVar("stream_pack_io", default=None)
_default = None


def bind_default(queues):
    global _default
    _default = queues


def bind_session(queues):
    return _current.set(queues)


def reset_session(token):
    _current.reset(token)


class QueueProxy:
    def __getattr__(self, name):
        queues = _current.get() or _default
        if queues is None:
            raise RuntimeError("Stream Pack I/O has not been bound by its host")
        return getattr(queues, name)


tensor_cache = QueueProxy()
