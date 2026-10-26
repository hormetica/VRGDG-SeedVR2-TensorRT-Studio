"""Private JSONL image worker: independent frames, cached models, no video export."""

from __future__ import annotations

import gc
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "seedvr2"))


def main():
    print("STUDIO_IMAGE_WORKER " + json.dumps({"pid": os.getpid()}), flush=True)
    # Import the runtime before torch so its allocator setup happens in a fresh
    # process, never in the Studio host which is also serving browser requests.
    import inference_cli as cli
    import numpy as np
    import psutil
    import torch
    from PIL import Image

    from seedvr_studio.backend import MODEL_FILES, TRT_ENGINE_5F
    from seedvr_studio.paths import MODELS
    from src.optimization.compatibility import ATTENTION_BACKENDS_USED

    cache, decoder = {}, None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            settings, engine = request["settings"], request["engine"]
            source, output = Path(request["source"]), Path(request["output"])
            with Image.open(source) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            frames = torch.from_numpy(rgb[None]).to(torch.float16)
            # The host stages source.png with a sibling request-size record.
            target = json.loads(source.with_suffix(".size.json").read_text(encoding="utf-8"))
            width, height = target
            model = MODEL_FILES[settings["model_label"]]
            for name in (model, "ema_vae_fp16.safetensors"):
                if not (MODELS / name).is_file():
                    raise FileNotFoundError(f"Install model before rendering: {name}")
            arguments = [str(source), "--model_dir", str(MODELS), "--dit_model", model,
                         "--resolution", str(min(width, height)), "--max_resolution", str(max(width, height)),
                         "--batch_size", "5" if engine == "tensorrt" else "1",
                         "--seed", str(settings["seed"]), "--attention_mode", settings["attention_mode"],
                         "--color_correction", "none", "--dit_offload_device", "cpu", "--vae_offload_device", "cpu"]
            if settings["blocks_to_swap"]:
                arguments += ["--blocks_to_swap", str(settings["blocks_to_swap"]), "--swap_io_components"]
            if settings["vae_tiling"]:
                arguments += ["--vae_encode_tiled", "--vae_decode_tiled"]
            capture = source.parent / "latents"
            for key in list(os.environ):
                if key.startswith("SEEDVR2_TRT_") or key in {"SEEDVR2_LATENT_CAPTURE_DIR", "SEEDVR2_PROFILE_PATH"}:
                    os.environ.pop(key)
            if engine == "tensorrt":
                arguments += ["--uniform_batch_size", "--stop_before_vae"]
                os.environ.update(SEEDVR2_TRT_ENCODER="1", SEEDVR2_LATENT_CAPTURE_DIR=str(capture))
            args = cli.parse_arguments(arguments)
            # Retain models in CPU RAM only when there is adequate headroom.
            cache_ok = bool(cache) or psutil.virtual_memory().available > (MODELS / model).stat().st_size * 2 + 2 * 1024**3
            args.cache_dit = args.cache_vae = cache_ok
            ATTENTION_BACKENDS_USED.clear()
            torch.cuda.reset_peak_memory_stats()
            result = cli._process_frames_core(frames, args, "0", cli.debug, cache if cache_ok else None)
            if engine == "tensorrt":
                # Bypass the video/image CLI save branch entirely. Its latent-only
                # result is empty by design; decoding is explicitly image-owned.
                del result
                ctx = cache.get("ctx", {})
                for key in list(ctx):
                    if key not in {"dit_device", "vae_device", "dit_offload_device", "vae_offload_device", "tensor_offload_device", "compute_dtype"}:
                        del ctx[key]
                gc.collect()
                torch.cuda.empty_cache()
                files = sorted(capture.glob("vae_latent_*.pt"))
                if len(files) != 1:
                    raise ValueError("A still image must produce exactly one captured latent")
                payload = torch.load(files[0], map_location="cpu", weights_only=True)
                latent = payload["latent"]
                if latent.ndim != 5 or tuple(latent.shape[:3]) != (1, 16, 2):
                    raise ValueError(f"Unsupported still-image latent shape: {tuple(latent.shape)}")
                if not torch.isfinite(latent).all() or latent.abs().max() > 128:
                    raise ValueError("Image restoration produced invalid latent values")
                if decoder is None:
                    from tools.run_tensorrt_persistent import PersistentDecoder
                    decoder = PersistentDecoder(TRT_ENGINE_5F, 16, 2, 64)
                decoded = decoder.decode_tensor(latent, 24)
                # Ordinary transforms trim each resized axis down to even. Crop
                # that actual canvas, then the host restores an odd requested axis.
                actual_w, actual_h = width // 2 * 2, height // 2 * 2
                result = decoded[0, :, 0, :actual_h, :actual_w].permute(1, 2, 0).float().cpu()
                result = (result + 1.0) / 2.0
                del decoded, payload, latent
            else:
                result = result[0].float().cpu()
            if not torch.isfinite(result).all():
                raise ValueError("Image decoder produced non-finite pixels")
            pixels = (result.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            Image.fromarray(pixels).save(output, format="PNG")
            peak = torch.cuda.max_memory_allocated()
            del result, pixels, frames
            ctx = cache.get("ctx", {})
            for key in list(ctx):
                if key not in {"dit_device", "vae_device", "dit_offload_device", "vae_offload_device", "tensor_offload_device", "compute_dtype"}:
                    del ctx[key]
            gc.collect()
            torch.cuda.empty_cache()
            print("STUDIO_IMAGE_RESULT " + json.dumps({"ok": True, "engine": engine,
                  "temporal_size": 5 if engine == "tensorrt" else 1, "peak_cuda_bytes": peak,
                  "attention_requested": settings["attention_mode"],
                  "attention_used": dict(ATTENTION_BACKENDS_USED) or None}), flush=True)
        except Exception as exc:
            traceback.print_exc()
            print("STUDIO_IMAGE_RESULT " + json.dumps({"ok": False, "error": str(exc)}), flush=True)
            return  # Any failed generation retires every model/engine/context.


if __name__ == "__main__":
    main()
