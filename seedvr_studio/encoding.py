"""SDR export policy, including real NVENC probes and precision-preserving fallbacks."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

EXPORT_PRESETS = {
    "software_h264": "H.264 / CPU — original quality, widest compatibility",
    "h264_nvenc": "H.264 / NVIDIA — fast, widest compatibility",
    "hevc_nvenc_10bit": "HEVC / NVIDIA 10-bit — smooth gradients",
    "av1_nvenc_10bit": "AV1 / NVIDIA 10-bit — efficient, newer players",
}


@dataclass(frozen=True)
class ExportSettings:
    preset: str = "software_h264"
    quality: int = 18
    allow_fallback: bool = True

    def __post_init__(self):
        if self.preset not in EXPORT_PRESETS:
            raise ValueError(f"Unknown export preset: {self.preset}")
        if not 0 <= self.quality <= 40:
            raise ValueError("Export quality must be between 0 and 40")

    @property
    def bit_depth(self) -> int:
        return 10 if self.preset.endswith("10bit") else 8


@dataclass(frozen=True)
class EncoderPlan:
    requested: str
    codec: str
    bit_depth: int
    quality: int
    fallback_reason: str | None = None

    @property
    def input_format(self) -> str:
        # Never feed RGB8 into a nominally 10-bit encoder.
        return "rgb48le" if self.bit_depth == 10 else "rgb24"

    def arguments(self, *, video_filter: str | None = None, raw_rgb: bool = True) -> list[str]:
        nvenc = self.codec.endswith("_nvenc")
        result = ["-c:v", self.codec]
        if nvenc:
            result += ["-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", str(self.quality),
                       "-b:v", "0", "-multipass", "fullres", "-spatial-aq", "1"]
        elif self.codec == "libaom-av1":
            result += ["-cpu-used", "4", "-crf", str(self.quality), "-b:v", "0"]
        else:
            result += ["-preset", "medium", "-crf", str(self.quality)]
        pixel_format = "p010le" if nvenc and self.bit_depth == 10 else (
            "yuv420p10le" if self.bit_depth == 10 else "yuv420p"
        )
        filters = [video_filter] if video_filter else []
        if self.bit_depth == 10 or nvenc:
            # These are SDR exports, NOT an HDR conversion. RGB tensor values
            # are full range; encode limited-range BT.709 and tag it explicitly.
            filters.append(f"scale=in_range={'full' if raw_rgb else 'auto'}:out_range=tv:out_color_matrix=bt709")
            result += ["-color_primaries", "bt709", "-color_trc", "bt709",
                       "-colorspace", "bt709", "-color_range", "tv"]
        if filters:
            result += ["-vf", ",".join(filters)]
        result += ["-pix_fmt", pixel_format]
        if self.codec in {"hevc_nvenc", "libx265"}:
            result += ["-tag:v", "hvc1"]
        return result


def ffmpeg_executable() -> str:
    explicit = os.environ.get("SEEDVR_FFMPEG")
    executable = explicit or shutil.which("ffmpeg")
    return executable or "ffmpeg"


@lru_cache(maxsize=16)
def _probe_nvenc(executable: str, plan: EncoderPlan, width: int, height: int) -> str | None:
    """An encoder listing is insufficient: DLLs, drivers or this shape may fail."""
    command = [executable, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
               f"color=c=black:s={width}x{height}:r=1", "-frames:v", "1", "-an",
               *plan.arguments(), "-f", "null", "-"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=20, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return (result.stderr[-2000:] or f"Encoder exited with {result.returncode}") if result.returncode else None


def select_encoder(settings: ExportSettings, width: int, height: int) -> EncoderPlan:
    if min(width, height) < 2 or width % 2 or height % 2:
        raise ValueError("4:2:0 export dimensions must be positive even numbers")
    codecs = {"software_h264": "libx264", "h264_nvenc": "h264_nvenc",
              "hevc_nvenc_10bit": "hevc_nvenc", "av1_nvenc_10bit": "av1_nvenc"}
    codec = codecs[settings.preset]
    plan = EncoderPlan(settings.preset, codec, settings.bit_depth, settings.quality)
    if codec == "libx264":
        return plan
    reason = _probe_nvenc(ffmpeg_executable(), plan, width, height)
    if reason is None:
        return plan
    # Driver/session exhaustion may be transient. Never cache a failure for the
    # lifetime of the Studio server (chunk joining also selects encoders here).
    _probe_nvenc.cache_clear()
    if not settings.allow_fallback:
        raise RuntimeError(f"Requested NVIDIA encoder is unavailable: {reason}")
    fallback = {"h264_nvenc": "libx264", "hevc_nvenc": "libx265", "av1_nvenc": "libaom-av1"}[codec]
    # Keep the requested codec family AND bit depth; don't silently turn a
    # 10-bit AV1/HEVC export into an 8-bit H.264 file.
    return EncoderPlan(settings.preset, fallback, settings.bit_depth, settings.quality, reason)
