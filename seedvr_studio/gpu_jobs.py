"""One GPU reservation shared by video renders and independently cancellable images."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Event, Lock, RLock

from .media import RenderCancelledError

GPU_LOCK = RLock()
_STATE_LOCK = Lock()
_owner: str | None = None


def active_gpu_owner() -> str | None:
    with _STATE_LOCK:
        return _owner


def cancel_gpu_owner(owner: str, cancel):
    """Do not let ownership move between checking it and stopping its processes."""
    with _STATE_LOCK:
        return cancel() if _owner == owner else None


@contextmanager
def reserve_gpu(owner: str, cancelled: Event):
    """Wait without losing cancellation; only the reservation's owner may stop it."""
    global _owner
    while not GPU_LOCK.acquire(timeout=0.1):
        if cancelled.is_set():
            raise RenderCancelledError("Cancelled while waiting for the GPU")
    try:
        if cancelled.is_set():
            raise RenderCancelledError("Render cancelled")
        with _STATE_LOCK:
            previous, _owner = _owner, owner
        try:
            yield
        finally:
            with _STATE_LOCK:
                _owner = previous
    finally:
        GPU_LOCK.release()
