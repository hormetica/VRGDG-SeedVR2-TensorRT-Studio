"""Read-only hardware inspection and conservative, workload-aware starting settings."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict, dataclass, replace

GIB = 1024**3
HARDWARE_PRESETS = {
    "Custom / manual": None,
    "8 GB VRAM — low memory": {"model": "3B FP8 — faster / less VRAM", "resolution": 480, "max_resolution": 1920, "batch_size": 5, "attention": "sageattn_2", "blocks": 36, "vae_tiling": True},
    "12 GB VRAM — mainstream": {"model": "3B FP8 — faster / less VRAM", "resolution": 720, "max_resolution": 1920, "batch_size": 5, "attention": "sageattn_2", "blocks": 24, "vae_tiling": True},
    "16 GB VRAM — high memory": {"model": "3B FP8 — faster / less VRAM", "resolution": 1080, "max_resolution": 2560, "batch_size": 5, "attention": "sageattn_2", "blocks": 12, "vae_tiling": True},
    "24 GB VRAM — enthusiast": {"model": "3B FP8 — faster / less VRAM", "resolution": 1080, "max_resolution": 3840, "batch_size": 21, "attention": "sageattn_2", "blocks": 0, "vae_tiling": False},
    "32 GB+ VRAM — workstation": {"model": "3B FP16 — best 3B quality", "resolution": 1440, "max_resolution": 3840, "batch_size": 21, "attention": "sageattn_2", "blocks": 0, "vae_tiling": False},
}
OUTPUT_BOUNDS = {"1K / 1080p": (1920, 1080), "2K / 1440p": (2560, 1440), "4K / 2160p": (3840, 2160)}
VRAM_TIERS = (
    (28, "32 GB+ VRAM — workstation"),
    (20, "24 GB VRAM — enthusiast"),
    (13, "16 GB VRAM — high memory"),
    (9, "12 GB VRAM — mainstream"),
    (5, "8 GB VRAM — low memory"),
)


@dataclass(frozen=True)
class HardwareInfo:
    gpu_name: str | None = None
    device: str = "cuda:0"
    total_vram_bytes: int = 0
    free_vram_bytes: int | None = None
    compute_capability: tuple[int, int] | None = None
    total_ram_bytes: int | None = None
    available_ram_bytes: int | None = None
    cpu_threads: int | None = None
    attention_backends: tuple[str, ...] = ("sdpa",)
    error: str | None = None


def _package_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, OSError, ValueError):
        return False


def detect_hardware() -> HardwareInfo:
    """Inspect logical cuda:0, just like the renderer; never select a different GPU."""
    hardware = HardwareInfo(cpu_threads=os.cpu_count())
    try:
        import psutil

        memory = psutil.virtual_memory()
        hardware = replace(hardware, total_ram_bytes=memory.total, available_ram_bytes=memory.available)
    except (ImportError, OSError, RuntimeError):
        pass  # System RAM is useful context, not a requirement for inspection.
    try:
        import torch

        if not torch.cuda.is_available():
            return replace(hardware, error="CUDA is unavailable in Studio's PyTorch runtime. Install or repair the NVIDIA/CUDA setup first.")
        properties = torch.cuda.get_device_properties(0)
        hardware = replace(hardware, gpu_name=properties.name, total_vram_bytes=properties.total_memory,
                           compute_capability=(properties.major, properties.minor))
        try:
            free, _ = torch.cuda.mem_get_info(0)
            hardware = replace(hardware, free_vram_bytes=max(0, min(int(free), properties.total_memory)))
        except Exception:  # noqa: BLE001 -- some drivers cannot report current free VRAM
            pass
        attention = ["sdpa"]
        if properties.major >= 8:
            if _package_present("sageattention"):
                attention.insert(0, "sageattn_2")
            if _package_present("flash_attn"):
                attention.insert(len(attention) - 1, "flash_attn_2")
        return replace(hardware, attention_backends=tuple(attention))
    except Exception as exc:  # noqa: BLE001 -- missing runtimes/drivers must not break the settings UI
        return replace(hardware, error=f"Could not inspect the rendering GPU: {exc}")


def recommend_settings(hardware: HardwareInfo, *, backend_name: str, output_preset: str,
                       source_width: int = 0, source_height: int = 0) -> dict[str, object]:
    """Return a balanced estimate without modifying settings or benchmarking a model.

    Tier thresholds are usable GiB, not advertised VRAM. Reserve at least 1 GiB
    of currently free memory and 10% of total memory. Large frames and SDPA get
    shorter batches and more offloading. This is deliberately not an OOM promise.
    """
    if backend_name not in {"SeedVR2 + TensorRT", "SeedVR2 (Legacy)"}:
        raise ValueError("Unknown render engine")
    if output_preset == "Original / enhancement only":
        width, height = source_width, source_height
    elif output_preset in OUTPUT_BOUNDS:
        width, height = OUTPUT_BOUNDS[output_preset]
    else:
        raise ValueError("Unknown output preset")
    result = {"available": False, "hardware": asdict(hardware), "settings": None, "notes": []}
    if hardware.error or not hardware.gpu_name or hardware.total_vram_bytes <= 0:
        return {**result, "message": hardware.error or "No supported CUDA GPU was detected."}
    if min(width, height) <= 0:
        return {**result, "message": "Load a source video first so Original / enhancement only can be sized correctly."}

    total_gb = hardware.total_vram_bytes / GIB
    free_gb = hardware.free_vram_bytes / GIB if hardware.free_vram_bytes is not None else total_gb * 0.85
    budget = min(total_gb * 0.9, free_gb - 1)
    if total_gb < 7.5 or budget < 5:
        return {**result, "message": "Too little VRAM for a reliable starting recommendation (about 8 GB total and 6 GB free are needed). Close GPU-heavy applications and try again; manual settings remain available."}
    tier = next(name for minimum, name in VRAM_TIERS if budget >= minimum)
    baseline = HARDWARE_PRESETS[tier]
    settings = {key: baseline[key] for key in ("model", "batch_size", "attention", "blocks", "vae_tiling")}
    settings["attention"] = hardware.attention_backends[0]
    notes = [f"Using the {tier} baseline, adjusted for current free VRAM and the selected output."]
    if hardware.free_vram_bytes is None:
        notes.append("Current free VRAM could not be read; an extra-conservative total-memory estimate was used.")

    # Account for the model's spatial padding; preset rectangles are conservative
    # upper bounds for either orientation, regardless of the chosen crop policy.
    pixels = ((width + 127) // 128 * 128) * ((height + 127) // 128 * 128)
    if pixels > 1920 * 1152:
        settings.update(batch_size=5, blocks=min(36, settings["blocks"] + 12), vae_tiling=True)
        notes.append("Larger output uses batch 5, extra block swapping and VAE tiling to reduce memory pressure.")
    if pixels > 2560 * 1536:
        settings.update(model="3B FP8 — faster / less VRAM", blocks=min(36, settings["blocks"] + 12))
        notes.append("4K-class or larger output favors FP8. It may still exceed VRAM; lower the output size if the preview fails.")
    if backend_name == "SeedVR2 (Legacy)":
        settings["vae_tiling"] = True
        notes.append("VAE tiling is enabled for the Legacy PyTorch decoder's larger memory demand.")
    if settings["attention"] == "sdpa":
        settings.update(batch_size=5, blocks=min(36, settings["blocks"] + 12), vae_tiling=True)
        notes.append("No compatible accelerated attention package was detected; SDPA uses a more conservative memory profile.")
    else:
        notes.append("Attention selection uses GPU generation and installed-package hints; rendering validates the kernels and may fall back.")
    if settings["blocks"] and hardware.available_ram_bytes is not None and hardware.available_ram_bytes < 8 * GIB:
        notes.append("Less than 8 GiB of system RAM is free. Block swapping may be slow or fail; close memory-heavy applications.")
    notes.append("Hardware-based estimate, not a benchmark or a guarantee against out-of-memory errors. Test a short preview before a full render.")
    return {**result, "available": True, "settings": settings, "notes": notes,
            "message": f"Balanced starting settings for {backend_name} at {output_preset}."}
