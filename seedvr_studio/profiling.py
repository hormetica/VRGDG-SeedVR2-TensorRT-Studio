"""Small, process-local stage reports; CUDA events are optional and imported lazily."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path


class PerformanceReport:
    def __init__(self, **metadata):
        self.metadata = metadata
        self.stages: dict[str, dict[str, float | int]] = {}

    @contextmanager
    def stage(self, name: str, *, cuda: bool = False):
        events = None
        if cuda:
            import torch
            if torch.cuda.is_available():
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                events[0].record()
        started = time.perf_counter()
        try:
            yield
        finally:
            gpu_ms = None
            if events:
                events[1].record()
                events[1].synchronize()
                gpu_ms = events[0].elapsed_time(events[1])
            elapsed = time.perf_counter() - started
            entry = self.stages.setdefault(name, {"calls": 0, "wall_seconds": 0.0})
            entry["calls"] += 1
            entry["wall_seconds"] += elapsed
            if sys.exc_info()[0] is not None:
                entry["failed_calls"] = entry.get("failed_calls", 0) + 1
            if gpu_ms is not None:
                entry["cuda_ms"] = entry.get("cuda_ms", 0.0) + gpu_ms

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps({
            "version": 1, "metadata": self.metadata, "stages": self.stages,
        }, indent=2), encoding="utf-8")
        temporary.replace(path)
