"""Compare whole-video vs streaming CPU tensor preparation, excluding encoding.

Run from the repository root with:
    python -m tools.benchmark_video_finishing --batches 24 --threads 4

Requires torch, OpenCV, and psutil on Windows (resource is used on Unix). Each
measurement uses a fresh process and the same saved FP32 decoder batches. A
discarding encoder isolates loading, seam treatment, RGB conversion, and copies;
these numbers are NOT end-to-end render or GPU inference benchmarks.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch

if __package__:
    from . import assemble_tensor_video as assembler
else:
    import assemble_tensor_video as assembler


def peak_memory_mib() -> float:
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset / 1024**2
    import resource
    scale = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def measure(mode: str, directory: Path, threads: int) -> None:
    torch.set_num_threads(threads)
    paths = sorted(directory.glob("batch-*.pt"))
    output = directory / f"{mode}.mp4"

    class Encoder:
        stdin = None
        byte_count = 0

        def __init__(self):
            self.stdin = self

        def write(self, data):
            self.byte_count += len(data)
            return len(data)

        def close(self):
            pass

        def wait(self):
            output.with_name(output.stem + "_noaudio.mp4").touch()
            return 0

    sink = Encoder()
    started = time.perf_counter()
    with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
        if mode == "streaming":
            with patch.object(assembler.subprocess, "Popen", return_value=sink):
                frame_count = assembler.assemble_video(paths, output)
        else:
            tensors = []
            for path in paths:
                tensor, keep = assembler.load_video_tensor(path)
                if keep is not None:
                    tensor = tensor[:, :, :keep]
                if tensors:
                    dtype = tensor.dtype
                    # Original seam treatment cloned the entire batch in FP32.
                    tensor = assembler.smooth_boundary(tensors[-1], tensor.float(), "match", 2).to(dtype)
                tensors.append(tensor)
            frames = torch.cat(tensors, dim=2)[0].permute(1, 2, 3, 0)
            frames = ((frames.float().clamp(-1, 1) + 1) * 127.5).byte().numpy()
            frame_count = len(frames)
            for frame in frames:
                sink.write(frame.tobytes())
    print(json.dumps({
        "mode": mode, "seconds": round(time.perf_counter() - started, 3),
        "peak_process_mib": round(peak_memory_mib(), 1),
        "frames": frame_count, "rgb_bytes": sink.byte_count, "threads": threads,
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=("legacy", "streaming"), help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.batches, args.frames, args.height, args.width, args.threads, args.repeats) < 1:
        parser.error("Sizes, thread count, and repeats must be positive")
    if args.mode:
        measure(args.mode, args.directory, args.threads)
        return
    with tempfile.TemporaryDirectory(prefix="seedvr-finishing-benchmark-") as temporary:
        directory = Path(temporary)
        torch.manual_seed(42)
        for index in range(args.batches):
            tensor = torch.randn(1, 3, args.frames, args.height, args.width)
            torch.save({"video": tensor, "original_frames": args.frames}, directory / f"batch-{index:05d}.pt")
        print(f"{args.batches} batches x {args.frames} frames x {args.width}x{args.height}, FP32, {args.threads} CPU threads")
        results = {"legacy": [], "streaming": []}
        for _ in range(args.repeats):
            for mode, runs in results.items():
                result = subprocess.run([
                    sys.executable, str(Path(__file__).resolve()), "--mode", mode,
                    "--directory", str(directory), "--threads", str(args.threads),
                ], check=True, text=True, capture_output=True)
                runs.append(json.loads(result.stdout))
        for mode, runs in results.items():
            assert all(run["frames"] == args.batches * args.frames for run in runs)
            assert all(run["rgb_bytes"] == args.batches * args.frames * args.height * args.width * 3 for run in runs)
            print(json.dumps({
                "mode": mode, "runs": args.repeats,
                "median_seconds": statistics.median(run["seconds"] for run in runs),
                "median_peak_process_mib": statistics.median(run["peak_process_mib"] for run in runs),
            }))


if __name__ == "__main__":
    main()
