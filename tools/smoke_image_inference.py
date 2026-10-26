"""Opt-in, installed-model GPU proof for native and five-frame TensorRT images.

No downloads/builds. Run only when the GPU is idle. All fixtures/results stay in
the explicitly supplied disposable directory. Not part of unittest discovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from threading import Event

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from seedvr_studio.gpu_jobs import reserve_gpu
from seedvr_studio.image_inference import ImageRenderer, ImageSettings


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--engine", choices=("seedvr", "tensorrt", "both"), default="both")
    parser.add_argument("--timeout", type=float, default=300)
    options = parser.parse_args()
    root = options.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    y, x = np.mgrid[:256, :384]
    pixels = np.stack((x * 255 // 383, y * 255 // 255, ((x // 24 + y // 24) % 2) * 150 + 50), axis=-1).astype(np.uint8)
    a = root / "rgb.png"
    Image.fromarray(pixels).save(a)
    b = root / "alpha.png"
    image = Image.fromarray(pixels).resize((191, 257))
    alpha = Image.fromarray(np.tile(np.linspace(0, 255, 191, dtype=np.uint8), (257, 1)))
    image.putalpha(alpha); image.save(b)
    fingerprints = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (a, b)}
    records = []
    cancelled = Event()
    engines = ("seedvr", "tensorrt") if options.engine == "both" else (options.engine,)
    try:
        with reserve_gpu("image-smoke", cancelled):
            for engine in engines:
                renderer = ImageRenderer(timeout=options.timeout)
                results = []
                try:
                    for index, source in enumerate((a, b, a)):
                        run = root / f"{engine}-{index}"
                        run.mkdir()
                        snapshot = run / "original.png"
                        snapshot.write_bytes(source.read_bytes())
                        output = run / "restored.png"
                        started = time.perf_counter()
                        metadata = renderer.render(snapshot, output, ImageSettings(engine=engine, scale=2, max_edge=768), cancelled,
                                                   lambda progress, message: print(f"{engine} {index}: {message}", flush=True))
                        with Image.open(output) as result:
                            result.load()
                            expected = (768, 512) if source == a else (382, 514)
                            assert result.size == expected, (result.size, expected)
                            array = np.asarray(result.convert("RGB"))
                            assert float(array.std()) > 2, "Output is gray/empty"
                            if source == b:
                                assert result.mode == "RGBA"
                                assert np.array_equal(np.asarray(result.getchannel("A")), np.asarray(alpha.resize(expected, Image.Resampling.LANCZOS)))
                            results.append(array)
                        records.append({"engine": engine, "image": source.name, "seconds": time.perf_counter() - started, **metadata})
                        print("GPU_IMAGE_CHECK " + json.dumps(records[-1]), flush=True)
                    records.append({"engine": engine, "repeat_mean_absolute_error": float(np.abs(results[0].astype(float) - results[2].astype(float)).mean())})
                finally:
                    renderer.close()
        for path, digest in fingerprints.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, "Original changed"
    finally:
        (root / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
