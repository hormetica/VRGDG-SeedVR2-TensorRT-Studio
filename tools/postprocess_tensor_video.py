"""Apply optional GPU post-processing to a decoded SeedVR tensor batch."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def _box_blur(value: torch.Tensor, kernel: int) -> torch.Tensor:
    radius = kernel // 2
    return F.avg_pool2d(F.pad(value, (radius, radius, radius, radius), mode="replicate"), kernel, stride=1)


def _skin_likelihood(flat: torch.Tensor) -> torch.Tensor:
    red, green, blue = flat[:, 0:1], flat[:, 1:2], flat[:, 2:3]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 0.5 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 0.5 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    skin = (
        torch.sigmoid((cb - 0.26) * 32.0)
        * torch.sigmoid((0.55 - cb) * 32.0)
        * torch.sigmoid((cr - 0.50) * 36.0)
        * torch.sigmoid((0.73 - cr) * 32.0)
        * torch.sigmoid((cr - cb - 0.015) * 30.0)
        * torch.sigmoid((luma - 0.055) * 28.0)
        * torch.sigmoid((0.97 - luma) * 28.0)
    )
    return _box_blur(skin, 9)


def _stable_center_mask(skin: torch.Tensor, batch: int, ext_frames: int, local_start: int, local_end: int) -> torch.Tensor:
    shaped = skin.reshape(batch, ext_frames, 1, skin.shape[-2], skin.shape[-1]).permute(0, 2, 1, 3, 4)
    shaped = F.avg_pool3d(F.pad(shaped, (0, 0, 0, 0, 1, 1), mode="replicate"), (3, 1, 1), stride=1)
    return shaped[:, :, local_start:local_end].permute(0, 2, 1, 3, 4).reshape(-1, 1, skin.shape[-2], skin.shape[-1])


def apply_skin_finishing(
    video: torch.Tensor,
    evenness: float,
    smoothing: float,
    redness: float,
    shine: float,
    blemish_mode: str,
    preserve_marks: bool,
) -> torch.Tensor:
    """Apply conservative, non-generative complexion finishing."""
    evenness = max(0.0, min(1.0, float(evenness)))
    smoothing = max(0.0, min(1.0, float(smoothing)))
    redness = max(0.0, min(1.0, float(redness)))
    shine = max(0.0, min(1.0, float(shine)))
    blemish_mode = blemish_mode if blemish_mode in {"off", "subtle", "strong"} else "off"
    if max(evenness, smoothing, redness, shine) <= 0 and blemish_mode == "off":
        return video
    batch, _, frame_count, _, _ = video.shape
    for start in range(0, frame_count, 4):
        end = min(frame_count, start + 4)
        ext_start, ext_end = max(0, start - 1), min(frame_count, end + 1)
        frames = video[:, :, ext_start:ext_end].permute(0, 2, 1, 3, 4)
        flat = frames.reshape(-1, 3, frames.shape[-2], frames.shape[-1])
        ext_frames = ext_end - ext_start
        local_start, local_end = start - ext_start, start - ext_start + (end - start)
        skin = _stable_center_mask(_skin_likelihood(flat), batch, ext_frames, local_start, local_end)
        center = flat.reshape(batch, ext_frames, 3, flat.shape[-2], flat.shape[-1])[:, local_start:local_end]
        result = center.reshape(-1, 3, flat.shape[-2], flat.shape[-1])
        luma = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
        medium = luma - _box_blur(luma, 7)
        edge_guard = 1.0 - 0.88 * torch.sigmoid((medium.abs() - 0.065) * 45.0)
        mask = (skin * edge_guard).clamp(0, 1)

        if evenness > 0:
            # Remove blotchy mid-scale variation while returning the fine band.
            target = result + _box_blur(result, 21) - _box_blur(result, 5)
            result = result.lerp(target.clamp(0, 1), mask * evenness * 0.75)
        if smoothing > 0:
            result = result.lerp(_box_blur(result, 3), mask * smoothing * 0.72)
        if blemish_mode != "off":
            luma = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
            local_luma = _box_blur(luma, 5)
            threshold = 0.085 if preserve_marks else 0.052
            spot = torch.sigmoid(((luma - local_luma).abs() - threshold) * 65.0) * mask
            amount = 0.24 if blemish_mode == "subtle" else 0.48
            result = result.lerp(_box_blur(result, 5), spot * amount)
        if redness > 0:
            red, green, blue = result[:, 0:1], result[:, 1:2], result[:, 2:3]
            excess = (red - 0.5 * (green + blue) - 0.025).clamp_min(0)
            correction = excess * mask * redness * 0.55
            result = torch.cat((red - correction, green + correction * 0.18, blue + correction * 0.08), dim=1).clamp(0, 1)
        if shine > 0:
            luma = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
            local_luma = _box_blur(luma, 15)
            highlight_floor = torch.maximum(local_luma + 0.045, torch.full_like(luma, 0.70))
            reduction = (luma - highlight_floor).clamp_min(0) * mask * shine * 0.75
            result = (result - reduction).clamp(0, 1)
        video[:, :, start:end] = result.reshape(batch, end - start, 3, result.shape[-2], result.shape[-1]).permute(0, 2, 1, 3, 4)
    return video


def apply_skin_microtexture(video: torch.Tensor, strength: float) -> torch.Tensor:
    """Enhance existing skin-scale luma detail without sharpening facial edges.

    The soft mask uses broad YCbCr skin likelihood rather than an identity-changing
    face restoration model. Processing a few frames at a time limits 4K VRAM use,
    while a three-frame mask average keeps the strength stable through motion.
    """
    strength = max(0.0, min(3.0, float(strength)))
    if strength <= 0:
        return video
    batch, _, frame_count, _, _ = video.shape
    for start in range(0, frame_count, 4):
        end = min(frame_count, start + 4)
        ext_start = max(0, start - 1)
        ext_end = min(frame_count, end + 1)
        frames = video[:, :, ext_start:ext_end].permute(0, 2, 1, 3, 4)
        flat = frames.reshape(-1, 3, frames.shape[-2], frames.shape[-1])
        skin = _skin_likelihood(flat)

        # Stabilize only the mask, not the pixels, avoiding temporal ghosting.
        ext_frames = ext_end - ext_start
        local_start = start - ext_start
        local_end = local_start + (end - start)
        skin = _stable_center_mask(skin, batch, ext_frames, local_start, local_end)

        center = flat.reshape(batch, ext_frames, 3, flat.shape[-2], flat.shape[-1])[:, local_start:local_end]
        center = center.reshape(-1, 3, flat.shape[-2], flat.shape[-1])
        center_luma = 0.299 * center[:, 0:1] + 0.587 * center[:, 1:2] + 0.114 * center[:, 2:3]
        fine = center_luma - _box_blur(center_luma, 3)
        medium = center_luma - _box_blur(center_luma, 7)
        detail = (0.78 * fine + 0.22 * medium).clamp(-0.08, 0.08)

        # Back away from strong contours such as eyes, nostrils, lips, and hair.
        contour = medium.abs()
        edge_guard = 1.0 - 0.85 * torch.sigmoid((contour - 0.065) * 45.0)
        delta = detail * skin * edge_guard * strength
        enhanced = (center + delta).clamp(0, 1)
        video[:, :, start:end] = enhanced.reshape(batch, end - start, 3, enhanced.shape[-2], enhanced.shape[-1]).permute(0, 2, 1, 3, 4)
    return video


@torch.inference_mode()
def apply_postprocessing(
    video: torch.Tensor,
    *,
    sharpen_strength: float = 0.0,
    grain_intensity: float = 0.0,
    grain_saturation: float = 0.5,
    microtexture_strength: float = 0.0,
    skin_evenness: float = 0.0,
    skin_smoothing: float = 0.0,
    skin_redness: float = 0.0,
    skin_shine: float = 0.0,
    blemish_mode: str = "off",
    preserve_marks: bool = False,
    seed: int = 42,
    frame_start: int = 0,
    device: str = "cuda",
    output_device: str = "cpu",
) -> torch.Tensor:
    """Finish one batch, returning CPU FP16 pixels in decoder range.

    Keep the padded frames until after effects: temporal skin masks need that
    context. Both the standalone tool and the streaming assembler use this
    implementation, including the same per-frame grain seeds.
    """
    video = video.to(device=device, dtype=torch.float32, non_blocking=True)
    # Decoder output is [-1, 1], while the effects operate in [0, 1].
    x = ((video.clamp(-1, 1) + 1.0) * 0.5)
    x = apply_skin_finishing(x, skin_evenness, skin_smoothing, skin_redness,
                             skin_shine, blemish_mode, preserve_marks)
    x = apply_skin_microtexture(x, microtexture_strength)
    sharpen = max(0.0, min(10.0, float(sharpen_strength)))
    if sharpen > 0:
        b, c, t, h, w = x.shape
        spatial = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        blurred = F.avg_pool2d(spatial, kernel_size=3, stride=1, padding=1)
        spatial = (spatial + sharpen * (spatial - blurred)).clamp(0, 1)
        x = spatial.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    intensity = max(0.0, min(1.0, float(grain_intensity)))
    if intensity > 0:
        saturation = max(0.0, min(1.0, float(grain_saturation)))
        for offset in range(x.shape[2]):
            generator = torch.Generator(device=x.device)
            generator.manual_seed((int(seed) + int(frame_start) + offset) & 0x7FFFFFFF)
            noise = torch.randn((x.shape[0], 1, x.shape[3], x.shape[4]), generator=generator,
                                device=x.device, dtype=x.dtype)
            # Mild chroma variation, matching the film-grain behavior used by
            # the reference tool while remaining stable across batch boundaries.
            colored = noise.repeat(1, 3, 1, 1)
            colored[:, 0] *= 2.0
            colored[:, 2] *= 3.0
            gray = colored[:, 1:2]
            grain = saturation * colored + (1.0 - saturation) * gray
            # Keep only one frame's noise live. Separate multiply/add kernels
            # preserve the original stacked implementation's rounding order.
            grain.mul_(intensity)
            x[:, :, offset].add_(grain).clamp_(0, 1)
    return (x * 2.0 - 1.0).to(dtype=torch.float16).to(output_device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sharpen-strength", type=float, default=0.0)
    parser.add_argument("--grain-intensity", type=float, default=0.0)
    parser.add_argument("--grain-saturation", type=float, default=0.5)
    parser.add_argument("--microtexture-strength", type=float, default=0.0)
    parser.add_argument("--skin-evenness", type=float, default=0.0)
    parser.add_argument("--skin-smoothing", type=float, default=0.0)
    parser.add_argument("--skin-redness", type=float, default=0.0)
    parser.add_argument("--skin-shine", type=float, default=0.0)
    parser.add_argument("--blemish-mode", choices=("off", "subtle", "strong"), default="off")
    parser.add_argument("--preserve-marks", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-start", type=int, default=0)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        payload = {"video": payload}
    options = vars(args).copy()
    del options["input"], options["output"]
    payload["video"] = apply_postprocessing(payload["video"], **options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
