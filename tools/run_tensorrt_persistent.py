"""Decode multiple SeedVR2 latent batches with one TensorRT engine/context."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seedvr_studio.profiling import PerformanceReport


def positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def feather(length: int, overlap: int, left: bool, right: bool, device: torch.device) -> torch.Tensor:
    weight = torch.ones(length, device=device, dtype=torch.float32)
    if left and overlap:
        weight[:overlap] = torch.linspace(0.0, 1.0, overlap + 1, device=device)[1:]
    if right and overlap:
        weight[-overlap:] = torch.minimum(
            weight[-overlap:], torch.linspace(1.0, 0.0, overlap + 1, device=device)[1:]
        )
    return weight


class PersistentDecoder:
    def __init__(self, engine_path: Path, channels: int, frames: int, tile: int) -> None:
        import tensorrt_rtx as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT-RTX engine: {engine_path}")
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        self.output_name = next(name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        expected = tuple(self.engine.get_tensor_shape(self.input_name))
        requested = (1, channels, frames, tile, tile)
        if expected != requested:
            raise ValueError(f"Engine input {expected} does not match tile {requested}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Could not create TensorRT execution context: {engine_path}")
        self.stream = torch.cuda.Stream()
        self.input_shape = requested
        self.output_shape = (1, 3, frames * 4 - 3, tile * 8, tile * 8)
        actual_output = tuple(self.engine.get_tensor_shape(self.output_name))
        if actual_output != self.output_shape:
            raise ValueError(f"Engine output {actual_output} does not match decoder output {self.output_shape}")
        with torch.cuda.stream(self.stream):
            self.tile_input = torch.empty(requested, device="cuda", dtype=torch.float16)
            self.tile_output = torch.empty(self.output_shape, device="cuda", dtype=torch.float16)
        # These addresses NEVER change while executions are queued. Reusing a
        # context with changing addresses previously caused gray/missing tiles.
        self.context.set_tensor_address(self.input_name, self.tile_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.tile_output.data_ptr())
        self._geometry = None
        self._weights = None
        self._window_overlap = None
        self._windows: dict[tuple[bool, bool, bool, bool], torch.Tensor] = {}

    @torch.inference_mode()
    def decode_tensor(self, source: torch.Tensor, overlap: int) -> torch.Tensor:
        """Return a GPU canvas, ordered before work on the caller's CUDA stream.

        Only one tile input/output pair is retained. Copies, TensorRT execution,
        and accumulation use one stream, so no per-tile host wait is necessary.
        The caller owns the returned canvas; the next call never overwrites it.
        Calls must be serialized: one decoder has one execution context.
        """
        tile = self.input_shape[-1]
        if source.ndim != 5 or tuple(source.shape[:3]) != self.input_shape[:3]:
            raise ValueError(f"Latent shape {tuple(source.shape)} is incompatible with {self.input_shape}")
        _, _, frames, height, width = source.shape
        if min(height, width) < 1:
            raise ValueError("Latent spatial dimensions must be positive")
        if overlap <= 0 or overlap >= tile:
            raise ValueError("overlap_latent must be between 1 and tile_latent-1")
        out_h, out_w = height * 8, width * 8
        out_tile, out_overlap = tile * 8, overlap * 8
        ys, xs = positions(height, tile, overlap), positions(width, tile, overlap)
        padded_h, padded_w = max(height, ys[-1] + tile), max(width, xs[-1] + tile)
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            source = source.to(device="cuda", dtype=torch.float16, non_blocking=True)
            source.record_stream(self.stream)
            latent = torch.nn.functional.pad(source, (0, padded_w - width, 0, padded_h - height))
            result = torch.zeros((1, 3, frames * 4 - 3, out_h, out_w), device="cuda", dtype=torch.float32)
            geometry = (height, width, overlap)
            if overlap != self._window_overlap:
                self._windows.clear()
                self._window_overlap = overlap
            rebuild_weights = geometry != self._geometry
            if rebuild_weights:
                # Weights do not vary across channels or frames: 63x less storage
                # for a 21-frame batch. Cache only the latest canvas geometry.
                self._weights = torch.zeros((1, 1, 1, out_h, out_w), device="cuda", dtype=torch.float32)
            for y in ys:
                for x in xs:
                    key = (y != ys[0], y != ys[-1], x != xs[0], x != xs[-1])
                    if key not in self._windows:
                        wy = feather(out_tile, out_overlap, key[0], key[1], source.device)
                        wx = feather(out_tile, out_overlap, key[2], key[3], source.device)
                        self._windows[key] = (wy[:, None] * wx[None, :]).view(1, 1, 1, out_tile, out_tile)
                    oy, ox = y * 8, x * 8
                    h, w = min(out_tile, out_h - oy), min(out_tile, out_w - ox)
                    window = self._windows[key][:, :, :, :h, :w]
                    self.tile_input.copy_(latent[:, :, :, y:y + tile, x:x + tile])
                    if not self.context.execute_async_v3(self.stream.cuda_stream):
                        raise RuntimeError(f"TensorRT execution failed at tile y={y}, x={x}")
                    result[:, :, :, oy:oy + h, ox:ox + w].add_(self.tile_output[:, :, :, :h, :w].float() * window)
                    if rebuild_weights:
                        self._weights[:, :, :, oy:oy + h, ox:ox + w].add_(window)
            if rebuild_weights:
                self._weights.clamp_min_(1e-6)
                self._geometry = geometry
            result.div_(self._weights).clamp_(-2.0, 2.0)
        caller.wait_stream(self.stream)
        result.record_stream(caller)
        return result

    def decode(self, latent_path: Path, output_path: Path, tile: int, overlap: int) -> float:
        if tile != self.input_shape[-1]:
            raise ValueError("tile_latent must match the persistent engine profile")
        started = time.perf_counter()
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        decoded = self.decode_tensor(payload["latent"], overlap).cpu()
        original_frames = payload.get("original_frames")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "video": decoded,
            "original_frames": int(original_frames) if original_frames is not None else None,
            "source_latent": str(latent_path),
        }, output_path)
        return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs") or []
    if not jobs:
        raise ValueError("Persistent decoder manifest contains no jobs")
    tile = int(manifest["tile_latent"])
    overlap = int(manifest["overlap_latent"])
    finishing = manifest.get("finishing")
    report = PerformanceReport(kind="decode_and_finish" if finishing else "decode", gpu_resident=bool(finishing),
                               gpu=torch.cuda.get_device_name(), torch_version=torch.__version__)
    profile = (Path(finishing["output"]).with_suffix(".gpu-finishing-profile.json") if finishing
               else args.manifest.with_suffix(".decode-profile.json"))
    try:
        with report.stage("engine_initialization"):
            first_payload = torch.load(Path(jobs[0]["latent"]), map_location="cpu", weights_only=False)
            first_shape = tuple(first_payload["latent"].shape)
            decoder = PersistentDecoder(Path(manifest["engine"]), first_shape[1], first_shape[2], tile)
            del first_payload
        print(f"Persistent TensorRT decoder ready: {len(jobs)} batches, fixed tile buffers", flush=True)
        if finishing:
            run_decode_and_finish(decoder, jobs, overlap, finishing, report)
        else:
            for index, job in enumerate(jobs, start=1):
                with report.stage("decode_and_save", cuda=True):
                    elapsed = decoder.decode(Path(job["latent"]), Path(job["output"]), tile, overlap)
                print(f"PERSISTENT_PROGRESS {index}/{len(jobs)} {elapsed:.3f}", flush=True)
    finally:
        if error := sys.exc_info()[1]:
            report.metadata["error"] = str(error)
        report.metadata["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report.write(profile)


def run_decode_and_finish(decoder, jobs, overlap, finishing, report):
    """GPU decode -> effects -> frame quantization, with raw recovery in parallel."""
    if __package__:
        from .assemble_tensor_video import assemble_video
        from .tensor_artifacts import AsyncTensorWriter
    else:
        from assemble_tensor_video import assemble_video
        from tensor_artifacts import AsyncTensorWriter
    from seedvr_studio.encoding import ExportSettings

    output = Path(finishing["output"])
    pending_output = output.with_name(output.stem + ".gpu-pending.mp4")
    try:
        with AsyncTensorWriter() as writer:
            def batches():
                for index, job in enumerate(jobs, start=1):
                    with report.stage("latent_load"):
                        payload = torch.load(Path(job["latent"]), map_location="cpu", weights_only=False)
                    with report.stage("vae_decode", cuda=True):
                        raw = decoder.decode_tensor(payload["latent"], overlap)
                    keep = payload.get("original_frames")
                    with report.stage("raw_checkpoint_submit"):
                        writer.submit(Path(job["output"]), {
                            "video": raw, "original_frames": int(keep) if keep is not None else None,
                            "source_latent": job["latent"],
                        })
                    print(f"PERSISTENT_PROGRESS {index}/{len(jobs)}", flush=True)
                    yield raw, int(keep) if keep is not None else None
                    del raw, payload

            with report.stage("decode_finish_total"):
                assemble_video(
                    [], pending_output, audio=Path(finishing["audio"]), fps=finishing["fps"],
                    seam_mode=finishing["seam_mode"], seam_frames=finishing["seam_frames"],
                    target_width=finishing["target_width"], target_height=finishing["target_height"],
                    effects=finishing.get("effects"), decoded_batches=batches(), batch_count=len(jobs),
                    export=ExportSettings(**finishing.get("export", {})), report=report,
                    scene_cut_detection=finishing.get("scene_cut_detection", False),
                    progress_callback=lambda current, total: print(f"ASSEMBLY_PROGRESS {current}/{total}", flush=True),
                )
        # Raw checkpoints must all be durable before the final output is published.
        pending_output.replace(output)
    finally:
        pending_output.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
