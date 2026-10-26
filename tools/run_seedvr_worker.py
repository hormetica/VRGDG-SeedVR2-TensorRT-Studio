"""JSONL warm SeedVR worker. One cached model selection, parked in CPU RAM.

This worker only captures latents. TensorRT decoding/export remain separate so
the 24 GB GPU budget is not consumed by an idle resident DiT model.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "seedvr2"))
sys.path.insert(0, str(ROOT))


def serve(process_request):
    """Protocol seam also exercised with an in-memory request handler in tests."""
    print('STUDIO_WORKER {"ready": true}', flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            metadata = process_request(request)
            response = {"id": request["id"], "ok": True, **metadata}
        except Exception as exc:  # noqa: BLE001 -- protocol reports the failure and retires this process
            traceback.print_exc()
            print("STUDIO_WORKER " + json.dumps({"id": request["id"], "ok": False, "error": str(exc)}), flush=True)
            # Any failed generation retires the cache, avoiding poisoned state.
            return
        print("STUDIO_WORKER " + json.dumps(response), flush=True)


def main():
    import inference_cli as cli
    import psutil
    import torch
    from src.core.model_cache import get_global_cache

    cache = {}
    previous_key = None

    def process(request):
        nonlocal previous_key
        args = cli.parse_arguments(request["arguments"])
        if not args.stop_before_vae:
            raise ValueError("Warm worker is restricted to latent capture")
        for name in ("SEEDVR2_LATENT_CAPTURE_DIR", "SEEDVR2_PROFILE_PATH"):
            os.environ.pop(name, None)
        os.environ.update(request["environment"])
        args.cache_dit = args.cache_vae = True
        args.dit_offload_device = args.vae_offload_device = "cpu"
        args.output_format = "mp4"
        key = (args.model_dir, args.dit_model, args.attention_mode, args.blocks_to_swap, args.swap_io_components,
               args.vae_encode_tiled, args.vae_encode_tile_size, args.vae_encode_tile_overlap,
               args.compile_dit, args.compile_mode, args.compile_backend)
        if key != previous_key:
            global_cache = get_global_cache()
            global_cache.remove_dit({"node_id": "cli_dit"})
            global_cache.remove_vae({"node_id": "cli_vae"})
            cache.clear()
            gc.collect()
            model = Path(args.model_dir) / args.dit_model
            if not model.is_file():
                raise FileNotFoundError(f"Warm caching requires installed model weights: {model}")
            required = model.stat().st_size * 2 + 2 * 1024**3
            if psutil.virtual_memory().available < required:
                raise MemoryError("Not enough free system RAM to retain this model; use cold inference")
        reused = key == previous_key and bool(cache)
        torch.cuda.reset_peak_memory_stats()
        cli.process_single_file(args.input, args, ["0"], args.output, runner_cache=cache)
        # The normal CLI phase cleanup already parks cached models on the CPU.
        # Do not clean BlockSwap twice or invalidate its retained cache state.
        # Release every generation tensor immediately, not only on the next job.
        ctx = cache.get("ctx", {})
        keep = {"dit_device", "vae_device", "dit_offload_device", "vae_offload_device", "tensor_offload_device", "compute_dtype"}
        for name in list(ctx):
            if name not in keep:
                del ctx[name]
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # hand VRAM back to the external decoder, once per generation
        retained = torch.cuda.memory_allocated()
        if retained > 512 * 1024**2:
            raise MemoryError(f"Warm cache retained {retained / 1024**2:.0f} MiB on GPU; retiring it to protect decoder VRAM")
        previous_key = key
        return {"cache_reused": reused, "idle_cuda_bytes": retained}

    serve(process)


if __name__ == "__main__":
    main()
