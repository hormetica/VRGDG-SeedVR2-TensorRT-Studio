"""Assemble decoded VAE tensors into a video, optionally preserving audio."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn.functional as F

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seedvr_studio.encoding import ExportSettings, ffmpeg_executable, select_encoder
from seedvr_studio.profiling import PerformanceReport


def load_video_tensor(path: Path) -> tuple[torch.Tensor, int | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "video" in payload:
        tensor = payload["video"]
        keep = payload.get("original_frames")
        return tensor, int(keep) if keep is not None else None
    return payload, None


def _spatial_lowpass(video: torch.Tensor) -> torch.Tensor:
    """Blur each frame independently while retaining the tensor layout."""
    b, c, t, h, w = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    blurred = F.avg_pool2d(frames, kernel_size=5, stride=1, padding=2)
    return blurred.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def smooth_boundary(previous: torch.Tensor, current: torch.Tensor, mode: str, frame_count: int) -> torch.Tensor:
    """Reduce decoded-batch statistic jumps without changing frame count."""
    count = min(max(1, frame_count), previous.shape[2], current.shape[2])
    if mode == "off" or count < 1:
        return current
    result = current.clone()
    reference = previous[:, :, -count:].float()
    head = result[:, :, :count].float()
    fade = torch.linspace(1.0, 0.0, count + 1, dtype=head.dtype, device=head.device)[0:count].view(1, 1, count, 1, 1)

    if mode in {"noise", "match", "blend"}:
        reference_high = reference - _spatial_lowpass(reference)
        head_low = _spatial_lowpass(head)
        head_high = head - head_low
        ref_noise = reference_high.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-5)
        head_noise = head_high.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-5)
        noise_matched = head_low + head_high * (ref_noise / head_noise).clamp(0.5, 2.0)
        head = head.lerp(noise_matched, fade)

    if mode in {"match", "blend"}:
        ref_mean = reference.mean(dim=(0, 2, 3, 4), keepdim=True)
        ref_std = reference.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-5)
        head_mean = head.mean(dim=(0, 2, 3, 4), keepdim=True)
        head_std = head.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-5)
        matched = (head - head_mean) * (ref_std / head_std).clamp(0.75, 1.33) + ref_mean
        head = head.lerp(matched, fade)

    if mode == "blend":
        anchor = previous[:, :, -1:].float().expand(-1, -1, count, -1, -1)
        blend = torch.linspace(0.35, 0.0, count + 1, dtype=head.dtype, device=head.device)[0:count].view(1, 1, count, 1, 1)
        head = head.lerp(anchor, blend)

    result[:, :, :count] = head.clamp(-1, 1)
    return result.to(dtype=current.dtype)


def is_scene_cut(previous: torch.Tensor, current: torch.Tensor, threshold: float = 0.45) -> bool:
    """Conservative hard-cut detector: color histograms AND spatial change.

    Use small thumbnails, not full-resolution optical flow. Similar histograms
    under a camera pan do not count as cuts. This is deliberately opt-in: flashes
    and similarly colored shots cannot always be distinguished from a hard cut.
    """
    def thumbnail(frame):
        size = (min(32, frame.shape[-2]), min(32, frame.shape[-1]))
        return F.adaptive_avg_pool2d(frame.float(), size).clamp(-1, 1).cpu()

    before = thumbnail(previous[:, :, -1])
    after = thumbnail(current[:, :, 0])
    if before.shape != after.shape:
        return True
    spatial_change = (before - after).abs().mean().item() * 0.5
    if spatial_change < 0.18:
        return False
    distance = 0.0
    for channel in range(3):
        left = torch.histc(before[:, channel], bins=16, min=-1, max=1)
        right = torch.histc(after[:, channel], bins=16, min=-1, max=1)
        distance += (left / left.sum() - right / right.sum()).abs().sum().item() / 6
    return distance >= threshold


def finish_batches(
    sources: Iterable[tuple[torch.Tensor, int | None]],
    *,
    seam_mode: str = "match",
    seam_frames: int = 2,
    effects: dict[str, Any] | None = None,
    scene_cut_detection: bool = False,
    report: PerformanceReport | None = None,
) -> Iterator[torch.Tensor]:
    """Load once per batch and retain only the preceding seam's pixels.

    Effects run before padding is trimmed, just as in the standalone workflow.
    Grain advances by the *kept* frame count rather than the padded batch size.
    No processed tensor files or whole-video concatenation are needed.
    """
    previous = None
    frame_start = 0
    report = report or PerformanceReport()
    for tensor, keep in sources:
        if tensor.ndim != 5 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise ValueError(f"Expected [1,3,T,H,W], got {tuple(tensor.shape)}")
        if keep is not None and not 1 <= keep <= tensor.shape[2]:
            raise ValueError(f"Invalid original_frames={keep}")
        if tensor.shape[2] < 1:
            raise ValueError("No decoded frames in batch")
        if effects is not None:
            if __package__:
                from .postprocess_tensor_video import apply_postprocessing
            else:
                from postprocess_tensor_video import apply_postprocessing
            gpu_effects = str(effects.get("device", "cuda")).startswith("cuda")
            with report.stage("effects", cuda=gpu_effects):
                tensor = apply_postprocessing(
                    tensor, frame_start=frame_start,
                    output_device=str(tensor.device), **effects,
                )
        if keep is not None:
            tensor = tensor[:, :, :keep]
        if previous is not None:
            with report.stage("seam_treatment", cuda=tensor.is_cuda):
                cut = scene_cut_detection and is_scene_cut(previous, tensor)
                if scene_cut_detection and not cut:
                    # A cut inside the affected head should not inherit the
                    # preceding shot's statistics either.
                    for offset in range(1, min(max(1, seam_frames), tensor.shape[2])):
                        if is_scene_cut(tensor[:, :, offset - 1:offset], tensor[:, :, offset:offset + 1]):
                            cut = True
                            break
                if cut:
                    report.metadata["seams_skipped_at_cuts"] = report.metadata.get("seams_skipped_at_cuts", 0) + 1
                else:
                    tensor = smooth_boundary(previous, tensor, seam_mode, seam_frames)
        if seam_mode != "off":
            # A slice alone would keep the entire preceding batch's storage live.
            previous = tensor[:, :, -max(1, seam_frames):].clone()
        frame_start += tensor.shape[2]
        yield tensor
        del tensor


def iter_finished_batches(inputs: list[Path], **options) -> Iterator[torch.Tensor]:
    yield from finish_batches(map(load_video_tensor, inputs), **options)


def quantize_rgb(frame: torch.Tensor, bit_depth: int = 8):
    """One RGB frame: 8-bit legacy rounding or full-precision RGB48 for 10-bit."""
    pixels = frame.float().clamp(-1, 1).add_(1)
    if bit_depth == 10:
        return pixels.mul_(32767.5).round_().to(torch.uint16).cpu().numpy().astype("<u2", copy=False)
    return pixels.mul_(127.5).byte().cpu().numpy()


def crop_decoder_padding(frames, reference: Path | None, target_width: int = 0, target_height: int = 0):
    """Remove TensorRT's bottom/right alignment padding using the source aspect ratio.

    TensorRT alignment rows are not guaranteed to remain exactly black after
    sharpening, grain, or other post effects, so geometry is the authoritative
    signal. The decoder pads on the bottom/right; retain the largest top-left
    rectangle matching the source aspect ratio.
    """
    frame_height, frame_width = frames.shape[1:3]
    if target_width > 0 and target_height > 0:
        if target_width > frame_width or target_height > frame_height:
            raise ValueError(
                f"Requested unpadded size {target_width}x{target_height} exceeds "
                f"decoded canvas {frame_width}x{frame_height}"
            )
        if (target_width, target_height) != (frame_width, frame_height):
            print(
                f"Removing known resize padding: {frame_width}x{frame_height} "
                f"-> {target_width}x{target_height}"
            )
        return frames[:, :target_height, :target_width, :], (target_width, target_height)
    if not reference:
        return frames, (frames.shape[2], frames.shape[1])
    capture = cv2.VideoCapture(str(reference))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if source_width <= 0 or source_height <= 0:
        return frames, (frames.shape[2], frames.shape[1])
    source_ratio = source_width / source_height
    if source_width >= source_height:
        target_width = frame_width
        target_height = max(2, round(target_width / source_ratio) // 2 * 2)
        if target_height <= frame_height:
            if target_height != frame_height:
                print(f"Removing TensorRT alignment padding: {frame_width}x{frame_height} -> {target_width}x{target_height}")
            frames = frames[:, :target_height, :target_width, :]
            return frames, (target_width, target_height)
    else:
        target_height = frame_height
        target_width = max(2, round(target_height * source_ratio) // 2 * 2)
        if target_width <= frame_width:
            if target_width != frame_width:
                print(f"Removing TensorRT alignment padding: {frame_width}x{frame_height} -> {target_width}x{target_height}")
            frames = frames[:, :target_height, :target_width, :]
            return frames, (target_width, target_height)
    if (target_width, target_height) != (frame_width, frame_height):
        print(f"Restoring source aspect: {frame_width}x{frame_height} -> {target_width}x{target_height}")
    return frames, (target_width, target_height)


@torch.inference_mode()
def assemble_video(
    inputs: list[Path],
    output: Path,
    *,
    audio: Path | None = None,
    fps: float = 24.0,
    seam_mode: str = "match",
    seam_frames: int = 2,
    target_width: int = 0,
    target_height: int = 0,
    effects: dict[str, Any] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    decoded_batches: Iterable[tuple[torch.Tensor, int | None]] | None = None,
    batch_count: int = 0,
    export: ExportSettings | None = None,
    scene_cut_detection: bool = False,
    report: PerformanceReport | None = None,
) -> int:
    """Stream finished RGB frames to one encoder; return the written frame count.

    Memory is bounded by the current decoded/effects batch, a seam tail, and one
    RGB frame, independent of video duration. Encoder backpressure bounds writes.
    Encoding settings intentionally match the original assembly tool.
    """
    if not inputs and decoded_batches is None:
        raise ValueError("At least one decoded tensor is required")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive")
    temp = output.with_name(output.stem + "_noaudio.mp4")
    encoder = None
    encode_command: list[str] = []
    frame_count = 0
    canvas = None
    output_size = None
    write_error = None
    report = report or PerformanceReport()
    export = export or ExportSettings()
    total_batches = len(inputs) if decoded_batches is None else batch_count
    if total_batches < 1:
        raise ValueError("Batch count must be positive")
    try:
        sources = map(load_video_tensor, inputs) if decoded_batches is None else decoded_batches
        batches = finish_batches(sources, seam_mode=seam_mode, seam_frames=seam_frames, effects=effects,
                                 scene_cut_detection=scene_cut_detection, report=report)
        for index, batch in enumerate(batches, start=1):
            if canvas is not None and tuple(batch.shape[-2:]) != canvas:
                raise ValueError("Decoded batches must have matching spatial dimensions")
            for offset in range(batch.shape[2]):
                frame = batch[0, :, offset].permute(1, 2, 0)
                # Clamp creates owned storage, so the remaining arithmetic can
                # reuse it without modifying the decoded batch or seam history.
                with report.stage("rgb_quantize_and_download"):
                    frame = quantize_rgb(frame, export.bit_depth)
                if encoder is None:
                    canvas = tuple(batch.shape[-2:])
                    cropped, output_size = crop_decoder_padding(
                        frame[None], audio, target_width, target_height
                    )
                    crop_height, crop_width = cropped.shape[1:3]
                    width, height = output_size
                    with report.stage("encoder_preflight"):
                        plan = select_encoder(export, width, height)
                    report.metadata["encoder"] = asdict(plan)
                    if plan.fallback_reason:
                        print(f"NVIDIA export unavailable; using {plan.codec} ({plan.bit_depth}-bit): {plan.fallback_reason}", flush=True)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    encode_command = [
                        ffmpeg_executable(), "-y", "-loglevel", "error",
                        "-f", "rawvideo", "-pix_fmt", plan.input_format,
                        "-s:v", f"{width}x{height}", "-r", f"{fps:.9g}", "-i", "pipe:0",
                        "-an", *plan.arguments(), "-movflags", "+faststart", str(temp),
                    ]
                    encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE)
                    del cropped
                frame = frame[:crop_height, :crop_width]
                if (frame.shape[1], frame.shape[0]) != output_size:
                    frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)
                assert encoder.stdin is not None
                with report.stage("rgb_pipe_write"):
                    encoder.stdin.write(frame.tobytes())
                frame_count += 1
            if progress_callback:
                progress_callback(index, total_batches)
            del batch
    except BaseException as exc:  # noqa: BLE001 -- re-raised below after closing/reaping the encoder
        write_error = exc
    finally:
        if encoder is not None and encoder.stdin is not None:
            try:
                encoder.stdin.close()
            except BrokenPipeError as exc:
                if write_error is None:
                    write_error = exc
    return_code = encoder.wait() if encoder is not None else 0
    if return_code:
        temp.unlink(missing_ok=True)
        raise subprocess.CalledProcessError(return_code, encode_command) from write_error
    if write_error is not None:
        temp.unlink(missing_ok=True)
        raise write_error
    if not frame_count:
        raise ValueError("Decoded batch stream contained no frames")
    if audio:
        video_duration = frame_count / fps
        try:
            subprocess.run([ffmpeg_executable(), "-y", "-loglevel", "error", "-i", str(temp), "-i", str(audio),
                            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
                            "-c:a", "aac", "-t", f"{video_duration:.9f}", str(output)], check=True)
        finally:
            temp.unlink(missing_ok=True)
    else:
        temp.replace(output)
    assert output_size is not None
    print(f"Frames: {frame_count}")
    print(f"Output: {output_size[0]}x{output_size[1]} @ {fps:g}fps")
    print(f"Saved: {output}")
    report.metadata.update({"frames": frame_count, "width": output_size[0], "height": output_size[1], "fps": fps})
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="*")
    parser.add_argument("--manifest", type=Path, help="JSON inputs/effects; avoids Windows command-line length limits")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seam-mode", choices=("off", "noise", "match", "blend"), default="match")
    parser.add_argument("--seam-frames", type=int, default=2)
    parser.add_argument("--target-width", type=int, default=0)
    parser.add_argument("--target-height", type=int, default=0)
    parser.add_argument("--export-preset", default="software_h264")
    parser.add_argument("--scene-cut-detection", action="store_true")
    args = parser.parse_args()
    effects = None
    export = ExportSettings(preset=args.export_preset)
    if args.manifest:
        if args.inputs:
            parser.error("Use either positional inputs or --manifest, not both")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        args.inputs = [Path(path) for path in manifest["inputs"]]
        effects = manifest.get("effects")
        export = ExportSettings(**manifest.get("export", {"preset": args.export_preset}))
        args.scene_cut_detection = manifest.get("scene_cut_detection", args.scene_cut_detection)
    report = PerformanceReport(kind="finishing", gpu_resident=False)
    try:
        with report.stage("finishing_total"):
            assemble_video(
                args.inputs, args.output, audio=args.audio, fps=args.fps,
                seam_mode=args.seam_mode, seam_frames=args.seam_frames,
                target_width=args.target_width, target_height=args.target_height, effects=effects,
                export=export, scene_cut_detection=args.scene_cut_detection, report=report,
                progress_callback=lambda current, total: print(f"ASSEMBLY_PROGRESS {current}/{total}", flush=True),
            )
    finally:
        if error := sys.exc_info()[1]:
            report.metadata["error"] = str(error)
        report.write(args.output.with_suffix(".finishing-profile.json"))


if __name__ == "__main__":
    main()
