from __future__ import annotations

import os
from threading import Event

import psutil

from .paths import ROOT, SEEDVR_CLI

_CANCEL_REQUESTED = Event()

def begin_render() -> None:
    _CANCEL_REQUESTED.clear()

def cancellation_requested() -> bool:
    return _CANCEL_REQUESTED.is_set()


def cancel_current_render() -> str:
    """Terminate only SeedVR2 inference processes launched by this workspace."""
    _CANCEL_REQUESTED.set()
    render_paths = [str(SEEDVR_CLI).lower()] + [str(ROOT / "tools" / name).lower() for name in (
        "run_tensorrt_tiled.py", "run_tensorrt_persistent.py", "postprocess_tensor_video.py", "assemble_tensor_video.py",
        "run_seedvr_worker.py",
    )]
    targets: dict[int, psutil.Process] = {}

    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or []).lower()
            if process.pid != os.getpid() and any(path in command for path in render_paths):
                targets[process.pid] = process
                for child in process.children(recursive=True):
                    targets[child.pid] = child
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

    if not targets:
        return "### Nothing to stop\nNo active SeedVR2 inference process was found."

    processes = list(targets.values())
    for process in reversed(processes):
        try:
            process.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

    _, alive = psutil.wait_procs(processes, timeout=3)
    for process in alive:
        try:
            process.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

    return f"### Render stopped\nStopped {len(targets)} SeedVR2 process{'es' if len(targets) != 1 else ''}."
