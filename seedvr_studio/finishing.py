"""One streaming finishing worker for renders and post-only reprocessing.

The host never loads decoded tensors or initializes CUDA. Raw decoded batches
remain the recovery/reprocessing artifact; processed batches stay in memory in
the worker and are sent directly to the video encoder.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .encoding import ExportSettings
from .media import MediaError, RenderCancelledError
from .paths import ROOT, VENV_PYTHON

ASSEMBLER = ROOT / "tools" / "assemble_tensor_video.py"
ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class PostprocessSettings:
    sharpen_strength: float = 0.0
    grain_intensity: float = 0.0
    grain_saturation: float = 0.5
    microtexture_strength: float = 0.0
    skin_evenness: float = 0.0
    skin_smoothing: float = 0.0
    skin_redness: float = 0.0
    skin_shine: float = 0.0
    blemish_mode: str = "off"
    preserve_marks: bool = True
    seed: int = 42


def finishing_manifest(inputs, output, source, *, fps, seam_mode, seam_frames, effects=None,
                       target_size=(0, 0), export=None, scene_cut_detection=False) -> dict:
    return {
        "version": 1, "inputs": list(map(str, inputs)), "output": str(output), "audio": str(source),
        "fps": fps, "seam_mode": seam_mode, "seam_frames": seam_frames,
        "target_width": target_size[0], "target_height": target_size[1],
        "effects": asdict(effects) if effects is not None else None,
        "export": asdict(export or ExportSettings()), "scene_cut_detection": scene_cut_detection,
    }


def finish_decoded_video(
    inputs: list[Path],
    output: Path,
    source: Path,
    child_env: dict[str, str],
    *,
    fps: float,
    seam_mode: str,
    seam_frames: int,
    effects: PostprocessSettings | None = None,
    target_size: tuple[int, int] = (0, 0),
    progress_callback: ProgressCallback | None = None,
    export: ExportSettings | None = None,
    scene_cut_detection: bool = False,
) -> Path:
    """Finish saved batches in order with one process, reporting progress in [0,1]."""
    if not inputs:
        raise MediaError("No decoded TensorRT batches to finish")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".finishing.json")
    manifest_path.write_text(json.dumps(finishing_manifest(
        inputs, output, source, fps=fps, seam_mode=seam_mode, seam_frames=seam_frames,
        effects=effects, target_size=target_size, export=export, scene_cut_detection=scene_cut_detection,
    ), indent=2), encoding="utf-8")
    command = [
        str(VENV_PYTHON), "-u", str(ASSEMBLER), "--manifest", str(manifest_path),
        "--output", str(output), "--audio", str(source), "--fps", f"{fps:.9g}",
        "--seam-mode", seam_mode, "--seam-frames", str(seam_frames),
        "--target-width", str(target_size[0]), "--target-height", str(target_size[1]),
    ]
    if progress_callback:
        progress_callback(0.0, "Starting streaming post-processing and video assembly")
    process = subprocess.Popen(
        command, cwd=ROOT, env=child_env, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output_tail: deque[str] = deque(maxlen=80)
    assert process.stdout is not None
    with process.stdout:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            output_tail.append(line)
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
                print(line.encode(encoding, errors="replace").decode(encoding, errors="replace"), flush=True)
            match = re.fullmatch(r"ASSEMBLY_PROGRESS (\d+)/(\d+)", line)
            if match and progress_callback:
                current = int(match.group(1))
                progress_callback(
                    0.9 * current / len(inputs),
                    f"Post-processing and encoding batch {current} of {len(inputs)}",
                )
    return_code = process.wait()
    if return_code:
        from .cancellation import cancellation_requested
        if cancellation_requested():
            raise RenderCancelledError("Render cancelled")
        raise MediaError(f"TensorRT video finishing failed:\n{'\n'.join(output_tail)}")
    if not output.exists():
        raise MediaError("TensorRT finishing completed but did not create an output video")
    if progress_callback:
        progress_callback(1.0, "Video finishing complete")
    return output
