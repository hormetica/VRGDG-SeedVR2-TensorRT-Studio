"""Persistent TensorRT decoder process integration for the JS Studio."""

from __future__ import annotations

import re
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from pathlib import Path

from .media import MediaError, RenderCancelledError
from .paths import ROOT, VENV_PYTHON

RUNNER = ROOT / "tools" / "run_tensorrt_persistent.py"
ProgressCallback = Callable[[float, str], None]


def _safe_print(value: str) -> None:
    try:
        print(value, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(value.encode(encoding, errors="replace").decode(encoding, errors="replace"), flush=True)


def run_persistent_decoder(
    manifest_path: Path,
    total: int,
    child_env: dict[str, str],
    progress_callback: ProgressCallback | None,
    *,
    combined: bool = False,
) -> None:
    """Run one decoder process and stream its per-batch progress."""
    command = [str(VENV_PYTHON), "-u", str(RUNNER), str(manifest_path)]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=child_env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output_tail: deque[str] = deque(maxlen=80)
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        output_tail.append(line)
        _safe_print(line)
        match = re.search(r"(PERSISTENT_PROGRESS|ASSEMBLY_PROGRESS) (\d+)/(\d+)", line)
        if match and progress_callback:
            current = int(match.group(2))
            if combined:
                fraction = (current - (0.5 if match.group(1) == "PERSISTENT_PROGRESS" else 0)) / max(1, total)
                progress_callback(0.78 + 0.2 * fraction, f"GPU decoding and finishing batch {current} of {total}")
                continue
            progress_callback(
                0.78 + 0.12 * current / max(1, total),
                f"Optimized TensorRT VAE decoding batch {current} of {total}",
            )
    return_code = process.wait()
    if return_code:
        from .cancellation import cancellation_requested
        if cancellation_requested():
            raise RenderCancelledError("Render cancelled")
        raise MediaError(f"Optimized TensorRT VAE decode failed:\n{'\n'.join(output_tail)}")
