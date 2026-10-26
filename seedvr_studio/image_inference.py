"""Owned image-inference subprocess, with visible TensorRT -> SeedVR fallback.

One worker is reused within a batch. It never shares temporal context between
files, and is retired after any failure so a poisoned CUDA cache cannot leak.
"""

from __future__ import annotations

import json
import io
import os
import queue
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Callable

import psutil
from PIL import Image, ImageOps

from .backend import MODEL_FILES, TRT_ENCODER_5F, TRT_ENGINE_5F, _progress_from_log
from .media import MediaError, RenderCancelledError
from .paths import MODELS, ROOT, VENV_PYTHON


def decode_image(data: bytes) -> Image.Image:
    """Normalize visible orientation/channels; never silently reduce 16-bit data."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) > 24 and data[24] > 8:
        raise ValueError("High-bit-depth PNG is not supported; export an 8-bit image first")
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {"JPEG", "PNG"} or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Only static JPEG and PNG images are supported")
        if image.mode in {"I", "F"} or image.mode.startswith("I;16"):
            raise ValueError("High-bit-depth images are not supported; export an 8-bit JPEG or PNG first")
        if image.width * image.height > 64_000_000 or min(image.size) < 2:
            raise ValueError("Source images must be at least 2 pixels per edge and at most 64 megapixels")
        oriented = ImageOps.exif_transpose(image)
        alpha = "A" in oriented.getbands() or "transparency" in oriented.info
        return oriented.convert("RGBA" if alpha else "RGB")


def fatal_gpu_error(error: Exception) -> bool:
    return any(marker in str(error).casefold() for marker in (
        "illegal memory access", "device-side assert", "driver shutting down", "device lost",
    ))


@dataclass(frozen=True)
class ImageSettings:
    engine: str = "auto"
    scale: float = 2.0
    max_edge: int = 4096
    model_label: str = "3B FP16 — best 3B quality"
    seed: int = 42
    attention_mode: str = "sdpa"
    blocks_to_swap: int = 0
    vae_tiling: bool = True

    def __post_init__(self):
        if self.engine not in {"auto", "seedvr", "tensorrt"}:
            raise ValueError("Unknown image inference engine")
        if isinstance(self.scale, bool) or self.scale not in {1.0, 2.0, 4.0}:
            raise ValueError("Image scale must be 1x (enhancement), 2x or 4x")
        if type(self.max_edge) is not int or not 64 <= self.max_edge <= 16384:
            raise ValueError("Maximum image edge must be between 64 and 16384 pixels")
        if self.model_label not in MODEL_FILES:
            raise ValueError("Unknown image model")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("Seed must be between 0 and 4294967295")
        if self.attention_mode not in {"sdpa", "sageattn_2", "sageattn_3", "flash_attn_2", "flash_attn_3"}:
            raise ValueError("Unknown attention mode")
        limit = 36 if MODEL_FILES[self.model_label].startswith("seedvr2_ema_7b") else 32
        if type(self.blocks_to_swap) is not int or not 0 <= self.blocks_to_swap <= limit:
            raise ValueError(f"Blocks to swap must be between 0 and {limit} for this model")
        if type(self.vae_tiling) is not bool:
            raise ValueError("VAE tiling must be true or false")

    def target_size(self, width: int, height: int) -> tuple[int, int]:
        ratio = min(self.scale, self.max_edge / max(width, height))
        size = (round(width * ratio), round(height * ratio))
        if min(size) < 2 or size[0] * size[1] > 64_000_000:
            raise ValueError("Requested image dimensions are too narrow or exceed 64 megapixels")
        return size


class ImageRenderer:
    def __init__(self, *, timeout: float = 1800):
        self.timeout = timeout
        self.process: subprocess.Popen | None = None
        self._worker_process: psutil.Process | None = None
        self.lines: queue.Queue = queue.Queue()
        self.trt_failure: str | None = None

    def close(self):
        process = self.process
        if process is None:
            return
        owned: dict[int, psutil.Process] = {}
        if process.poll() is None:
            try:
                root = psutil.Process(process.pid)
                owned[root.pid] = root
                owned.update((child.pid, child) for child in root.children(recursive=True))
            except psutil.NoSuchProcess:
                pass
        # Windows' venv executable is a redirector, not the actual interpreter.
        # Retain the identified worker too, even if its redirector exited first.
        worker = self._worker_process
        if worker is not None and worker.is_running():
            owned[worker.pid] = worker
            try:
                owned.update((child.pid, child) for child in worker.children(recursive=True))
            except psutil.NoSuchProcess:
                pass
        for target in reversed(list(owned.values())):
            try:
                target.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(list(owned.values()), timeout=3)
        for target in alive:
            try:
                target.kill()
            except psutil.NoSuchProcess:
                pass
        if alive:
            _, remaining = psutil.wait_procs(alive, timeout=3)
            if remaining:
                raise MediaError("Could not stop the owned image-worker process tree")
        process.wait(timeout=5)
        self.process, self._worker_process = None, None
        if process.stdin:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        # Do not inherit a video render's capture/profile or TensorRT selection.
        for key in list(environment):
            if key.startswith("SEEDVR2_TRT_") or key in {"SEEDVR2_LATENT_CAPTURE_DIR", "SEEDVR2_PROFILE_PATH"}:
                environment.pop(key)
        self.process = subprocess.Popen(
            [str(VENV_PYTHON), "-u", str(ROOT / "tools" / "run_image_inference.py")],
            cwd=ROOT, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.lines = queue.Queue()
        stream, messages = self.process.stdout, self.lines

        def read_output():
            try:
                for line in stream:
                    messages.put(line.rstrip())
            except (OSError, ValueError):
                pass  # Closing an owned worker may close its reader concurrently.
            finally:
                messages.put(None)

        Thread(target=read_output, daemon=True).start()

    def _run(self, source, output, settings, engine, cancelled, report):
        self._start()
        request = {"source": str(source), "output": str(output), "settings": asdict(settings), "engine": engine}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        started, tail = time.monotonic(), deque(maxlen=35)
        while True:
            if cancelled.is_set():
                self.close()
                raise RenderCancelledError("Image batch cancelled")
            if time.monotonic() - started > self.timeout:
                self.close()
                raise MediaError(f"Image inference exceeded {self.timeout:g} seconds; worker stopped")
            try:
                line = self.lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                self.close()
                raise MediaError("Image worker exited:\n" + "\n".join(tail))
            if line.startswith("STUDIO_IMAGE_WORKER "):
                identity = json.loads(line.removeprefix("STUDIO_IMAGE_WORKER "))
                try:
                    worker = psutil.Process(int(identity["pid"]))
                    if worker.pid != self.process.pid and worker.ppid() != self.process.pid:
                        raise MediaError("Image worker reported an unrelated process identity")
                    self._worker_process = worker
                except psutil.NoSuchProcess:
                    pass  # A failed import may already have ended this worker.
                report(0.01, "Image worker started")
                continue
            if line.startswith("STUDIO_IMAGE_RESULT "):
                result = json.loads(line.removeprefix("STUDIO_IMAGE_RESULT "))
                if not result.get("ok"):
                    self.close()
                    raise MediaError(result.get("error", "Image inference failed"))
                return result
            tail.append(line)
            if "Stop-before-VAE test mode:" in line:
                report(0.8, "AI restoration complete; decoding this image with TensorRT")
                continue
            progress = _progress_from_log(line)
            report(*(progress or (None, line)))

    def render(self, source: Path, output: Path, settings: ImageSettings, cancelled: Event,
               report: Callable) -> dict:
        for weight in (MODEL_FILES[settings.model_label], "ema_vae_fp16.safetensors"):
            if not (MODELS / weight).is_file():
                raise MediaError(f"Required installed model is missing: {weight}. Install it before starting; images never download models automatically.")
        image = decode_image(source.read_bytes())
        size = settings.target_size(*image.size)
        rgb = image.convert("RGB").resize(size, Image.Resampling.BICUBIC)
        # Pad an odd dimension by copying its edge, instead of letting the video
        # runtime crop away the original's last row/column. Strip this after AI.
        model_size = tuple(value + value % 2 for value in size)
        staged = Image.new("RGB", model_size)
        staged.paste(rgb)
        if model_size[0] != size[0]:
            staged.paste(rgb.crop((size[0] - 1, 0, size[0], size[1])), (size[0], 0))
        if model_size[1] != size[1]:
            staged.paste(staged.crop((0, size[1] - 1, model_size[0], size[1])), (0, size[1]))
        model_input, raw = source.parent / "model-input.png", source.parent / "model-output.png"
        staged.save(model_input, format="PNG")
        model_input.with_suffix(".size.json").write_text(json.dumps(model_size), encoding="utf-8")

        result = self._render_rgb(model_input, raw, settings, cancelled, report)
        if cancelled.is_set():
            raise RenderCancelledError("Image batch cancelled")
        with Image.open(raw) as decoded:
            decoded.load()
            if decoded.width < size[0] or decoded.height < size[1]:
                raise MediaError("Image inference returned an incomplete canvas")
            finished = decoded.convert("RGB").crop((0, 0, *size))
        if image.mode == "RGBA":
            finished.putalpha(image.getchannel("A").resize(size, Image.Resampling.LANCZOS))
        finished.save(output, format="PNG")
        return {**result, "width": size[0], "height": size[1], "alpha": image.mode == "RGBA"}

    def _render_rgb(self, source, output, settings, cancelled, report):
        requested = settings.engine
        reason = self.trt_failure
        if requested != "seedvr" and reason is None:
            missing = [path.name for path in (TRT_ENGINE_5F, TRT_ENCODER_5F) if not path.is_file()]
            reason = "Missing TensorRT image artifacts: " + ", ".join(missing) if missing else None
        if requested == "tensorrt" and reason:
            raise MediaError(reason)
        if requested != "seedvr" and not reason:
            try:
                report(0.01, "TensorRT: five copies of this image only; restoring one output image")
                result = self._run(source, output, settings, "tensorrt", cancelled, report)
                return {**result, "requested_engine": requested, "fallback_reason": None}
            except RenderCancelledError:
                raise
            except Exception as exc:
                self.close()
                if requested == "tensorrt" or fatal_gpu_error(exc):
                    raise
                reason = self.trt_failure = str(exc)
        if reason and requested == "auto":
            report(0.01, f"TensorRT unavailable; using SeedVR (one frame): {reason}")
        if cancelled.is_set():
            raise RenderCancelledError("Image batch cancelled")
        try:
            result = self._run(source, output, settings, "seedvr", cancelled, report)
        except Exception:
            self.close()
            raise
        return {**result, "requested_engine": requested, "fallback_reason": reason if requested == "auto" else None}
