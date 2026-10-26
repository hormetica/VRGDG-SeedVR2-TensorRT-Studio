"""Bounded asynchronous raw checkpoints with atomic publication."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


class AsyncTensorWriter:
    """At most one or two pinned host batches; never an unbounded save queue.

    submit() orders a copy after the producer stream without blocking that stream
    on disk I/O. Callers must not mutate the submitted video. close() propagates
    disk/copy failures and waits before a render is declared successful.
    """

    def __init__(self, max_pending: int = 1):
        if max_pending not in (1, 2):
            raise ValueError("Checkpoint queue must be one or two batches")
        self._limit = max_pending
        self._pending = deque()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="raw-checkpoint")
        self._stream = None

    @staticmethod
    def _save(path, payload, event):
        if event is not None:
            event.synchronize()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @torch.inference_mode()
    def submit(self, path: Path, payload: dict) -> None:
        while len(self._pending) >= self._limit:
            self._pending.popleft().result()
        video = payload["video"]
        event = None
        if video.is_cuda:
            if self._stream is None:
                self._stream = torch.cuda.Stream(device=video.device)
            host = torch.empty(video.shape, dtype=video.dtype, device="cpu", pin_memory=True)
            self._stream.wait_stream(torch.cuda.current_stream(video.device))
            with torch.cuda.stream(self._stream):
                host.copy_(video, non_blocking=True)
                event = torch.cuda.Event()
                event.record()
            video.record_stream(self._stream)
        else:
            host = video.detach().clone()
        self._pending.append(self._executor.submit(self._save, path, {**payload, "video": host}, event))

    def close(self) -> None:
        try:
            while self._pending:
                self._pending.popleft().result()
        finally:
            self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
