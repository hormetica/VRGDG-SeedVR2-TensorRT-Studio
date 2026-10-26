# Performance architecture

## Streaming TensorRT finishing

`seedvr_studio/finishing.py` owns the shared finishing interface for both normal
renders and post-only reprocessing. Callers supply ordered raw decoded batches,
effect settings, output geometry, and source timing; they do not load tensors or
initialize CUDA. A small JSON manifest avoids Windows command-line length limits.

One `tools/assemble_tensor_video.py` worker performs the entire finishing pass:

1. Load one raw decoded batch.
2. Apply optional effects in the same process/CUDA context, using
   `apply_postprocessing` from `tools/postprocess_tensor_video.py`.
3. Trim temporal padding **after** effects, preserving skin-mask context.
4. Match the seam against a copied tail of the preceding finished batch.
5. Quantize and write one RGB frame at a time to FFmpeg. Pipe backpressure bounds
   queued writes while encoding runs concurrently in FFmpeg.
6. Release the batch and advance the grain seed offset by its kept frame count.

Raw `tensorrt_decoded/decoded_*.pt` files remain available for recovery and post-only
reprocessing. New renders no longer write or reload `processed_*.pt` intermediates.
Existing raw and processed tensor files and the standalone tools remain supported.

### Costs removed

For **N** decoded batches, with effects enabled:

| Cost | Previous path | Streaming path |
|---|---|---|
| Python finishing workers | N effects workers + 1 assembler | 1 worker |
| CUDA initialization during finishing | Once per effects worker | Once per finishing worker |
| Decoded batch loads for a render | N | N |
| Extra raw loads for grain during post-only reprocessing | N | 0 |
| Processed tensor writes + reloads | N + N | 0 |
| Tensor memory growth with video length | Whole-video tensors and concatenation | Current batch/effects workspace and a seam tail |
| RGB conversion workspace | Whole video | One frame |

Effects can still require substantial VRAM for an individual high-resolution batch.
This is not a constant-memory decoder or a zero-copy GPU-to-encoder pipeline: raw
decoded tensors are intentionally retained on disk, and encoding still consumes
CPU RGB frames. No thread pool or unbounded prefetch queue is added.

### Quality and behavior

- The effect algorithms, order, FP16 effect output, and per-frame grain seeds are
  shared with the standalone post-processing tool.
- Grain is accumulated one frame at a time rather than stacking a batch of noise
  tensors; its random sequence and arithmetic order are unchanged.
- Seam matching uses FP32 arithmetic on only the affected head/tail frames.
- Cropping, source FPS, audio duration, and `original_frames` trimming are preserved.
- H.264 remains `libx264`, preset `medium`, CRF `18`, with `yuv420p` output. Faster
  presets or hardware encoding are not silently substituted for image quality.
- Cancellation continues to target the existing assembler worker and its FFmpeg
  children through `seedvr_studio/cancellation.py`.

## Verification

From the repository root in an environment with PyTorch and OpenCV:

```console
python -m unittest tools.test_video_finishing tools.test_frame_rate
```

The tests cover exact seam/RGB parity, FP16/FP32 batches, shortened final batches,
deterministic grain, padded temporal context, bounded batch lifetimes, worker
progress/failures, and encoder cleanup. FFmpeg integration tests exercise real
encoding, audio, fractional FPS, cropping, manifest effects, and the existing
positional CLI. They use FFmpeg from PATH or the optional `imageio-ffmpeg` package.
The CUDA standalone-versus-streaming effects test skips when CUDA is unavailable.

## Bounded TensorRT source reading

The normal single-GPU TensorRT latent-capture path reads source pixels on demand,
one temporal batch at a time. Previously, the CLI read the entire source as FP32
before encoding: 17,413 frames at 1280x720 required about 179 GiB just for those
pixels, before stacking and FP16 conversion. Both warm and cold inference could
fail in NumPy despite ample free GPU memory.

`_VideoFrameSource` in `vendor/seedvr2/inference_cli.py` exposes frame metadata and
integer/unit-step slices. The existing encoder requests its usual temporal
batches, including overlap and final-batch padding. RGB normalization and FP16
input precision are unchanged. Encoding still completes before DiT restoration;
the core is not restarted per batch, so latent numbering, seeds, and model loading
follow the existing pipeline. The capture is released even if generation fails.

This bounds **source-pixel buffering**, not total inference RAM: encoded/upscaled
latents and any warm model cache still consume memory. Explicit CLI chunking,
frame prepending, MPS, multi-GPU, and Legacy rendering retain their existing paths.

```console
python -X utf8 -m unittest tools.test_inference_input_memory
```

These tests inject the original NumPy allocation failure with a one-batch budget,
exercise the original frame count without model downloads, and check exact input
and latent-metadata parity through the real encoding/capture orchestration with
stubbed model calls. They do not establish end-to-end SeedVR model parity.

## NVIDIA acceleration and quality options

The browser's **Performance** and **Export quality** sections expose the new
options. GPU-resident finishing, warm models, and scene-cut protection are opt-in;
the original CPU H.264 export and existing restoration settings remain defaults.
These options currently apply to the TensorRT workflow, not the Legacy renderer.

### Fixed-buffer TensorRT decoding

`PersistentDecoder.decode_tensor` in `tools/run_tensorrt_persistent.py` owns one
fixed tile input/output pair. TensorRT addresses are set once. Input copies,
inference, blending, and normalization are ordered on one CUDA stream, then the
caller stream waits on it without a host wait per tile. Returned canvases belong
to the caller and are never overwritten by the next batch.

Blend weights have shape `[1,1,1,H,W]`, not `[1,3,T,H,W]`. This makes **that
allocation** 63 times smaller for 21-frame batches. Feather windows and the latest
canvas's weights are cached; the cache does not grow with video length. The CUDA
allocator is not flushed between decoder batches. Canvases smaller than a tile
are padded for inference and clipped during accumulation.

### GPU-resident decode and effects (Beta)

When enabled, the same persistent worker decodes each batch, applies effects on
the GPU, treats seams, and quantizes one frame at a time before downloading it
for FFmpeg. It does not reload raw tensors or upload decoded pixels for effects.
Encoding still uses a CPU pipe: this is not zero-copy GPU/NVENC surface sharing.

`tools/tensor_artifacts.py` separately copies unmodified raw GPU output into a
pinned host buffer and writes it with a single background thread. There is at
most **one pending raw checkpoint** by default. Copy events preserve producer
ordering and allocation lifetime; disk errors propagate. A checkpoint is renamed
from `.tmp` only after saving succeeds. The completed video is published only
after all raw checkpoints are saved. Intermediate videos are hidden from history.

The existing Stable decoder remains the fallback. A failed combined worker does
not rerun DiT: the host retries saved latents, then finishes the saved raw batches
through the normal streaming path, preserving the selected export settings.

### Stage timings and real attention dispatch

Per-output JSON reports record:

- `*.inference-host-profile.json`: worker startup/inference wall time, warm-cache
  reuse, and any cold-process fallback reason.
- `*.inference-profile.json`: model setup, VAE encoding, DiT restoration, latent
  capture or PyTorch decoding, requested attention, actual dispatched attention
  call counts, and peak PyTorch CUDA allocation.
- `*.warm-inference-profile.json`: the same inference measurements for a warm
  request, kept separately so a cold retry does not overwrite failure diagnostics.
- `*.finishing-profile.json`: effects, seams, RGB quantization/download, pipe
  writes, encoder preflight and the selected encoder.
- `*.gpu-finishing-profile.json`: the combined GPU path, also recording TensorRT
  decoding and raw-checkpoint submission. Kept separately from a CPU fallback.
- `persistent-decode-manifest.decode-profile.json`: standalone persistent decoding.

CUDA events synchronize at measured stage boundaries, **not every spatial tile**.
`cuda_ms` is the elapsed stream span, not GPU utilization or a sum of individual
kernel durations. Wall time includes CPU work and I/O. Overlapping stages should
not be added together to estimate total render time.

Attention counts are recorded at dispatch, including SageAttention 3's runtime
SageAttention 2 fallback for variable-length inputs. Importing a package alone is
not evidence that a kernel was used. Selection also checks the GPU generation:
Ada/RTX 4090 must not select Blackwell-only SageAttention 3 or Hopper-only FA3 just
because a package imports. No attention precision is silently changed for speed.

After a browser render (including a failure), **Performance report** opens
`/api/jobs/<job-id>/profile`, which aggregates that job's reports, including chunk
reports. JSON files remain in the project folder for later inspection.
The aggregate includes decoder selection/fallback manifests; failed reports retain
their error and stage failure counts.

### Warm model worker (Beta)

`seedvr_studio/inference_worker.py` manages one reusable latent-capture process.
The worker uses SeedVR's existing model cache, with explicit CPU offload between
generations. It retains one model/configuration selection in system RAM, not an
idle model consuming the decoder's VRAM. It checks available RAM before caching,
prunes generation tensors, and rejects caches retaining more than 512 MiB of
PyTorch GPU allocations after parking. TensorRT/driver allocations are not included
in that PyTorch counter.

Changing the model/configuration invalidates the cache. Changing TensorRT encoder
mode restarts the process. Errors retire it and trigger a fresh-process retry;
cancellation does not retry. The worker expires after five idle minutes, is closed
when a subsequent render disables warm mode, and is included in workspace cancellation/shutdown.
GPU render/reprocess jobs in the same server are serialized to avoid competing
over the same VRAM budget. Other applications using the GPU remain outside this
process's control.

The reusable CLI import preserves an already loaded PyTorch allocator. Allocator
environment defaults are set only before PyTorch is imported: changing them
afterward caused a reproducible Windows native crash during the BF16 probe on
the installed nightly build. A subprocess regression test covers that import
order, alongside normal cold CLI startup.

### NVENC and true 10-bit SDR

`seedvr_studio/encoding.py` defines these export policies:

| Browser preset | Preferred encoder | Software fallback | Input precision |
|---|---|---|---|
| H.264 / CPU (default) | `libx264`, medium, CRF 18 | — | RGB8 |
| H.264 / NVIDIA | `h264_nvenc` | `libx264` | RGB8 |
| HEVC / NVIDIA 10-bit | `hevc_nvenc` | `libx265` | RGB48 |
| AV1 / NVIDIA 10-bit | `av1_nvenc` | `libaom-av1` | RGB48 |

NVENC presets use P7/HQ, VBR/CQ 18, full-resolution multipass, and spatial AQ.
These quality values are **not equivalent** to x264/x265 CRF 18. The selected
hardware encoder is tested with an actual encode at the output dimensions before
consuming the frame stream; an encoder listing does not prove runtime support.
Unsupported shapes, drivers, DLLs, or codecs fall back before encoding begins.
Runtime failures after encoding begins are surfaced, not silently swallowed.

The 10-bit path quantizes floating-point pixels directly to little-endian RGB48,
then converts to 10-bit YUV. It never passes through RGB8 first. It encodes and
tags limited-range BT.709 **SDR**, not HDR. This can preserve gradients introduced
by restoration/finishing but cannot recover tonal information lost from a source.
HEVC/AV1 browser playback depends on the browser/OS; use an external player or
H.264 when compatibility matters. Software AV1 fallback can be much slower.

Chunk trimming and final assembly preserve the selected codec family and bit
depth. They do not interpret already limited-range YUV as full-range RGB. Each
chunk has its own latent/raw/profile directory, preventing stale batches from a
longer previous chunk. Batch files are ordered numerically beyond batch 999.

### Scene-cut protection

Optional scene-cut protection compares small color/spatial thumbnails at each
seam, including cuts inside the affected head frames. On a detected hard cut, it
skips color/noise matching and blending rather than carrying statistics from the
previous shot. It does not drop frames or alter grain seed offsets. The detector
is conservative, not perfect: flashes and similar-color cuts remain ambiguous.
This is seam protection, not scene-aware DiT generation or optical-flow synthesis.

### NVIDIA verification

```console
python -X utf8 -m unittest tools.test_video_finishing tools.test_gpu_decoder tools.test_nvidia_pipeline tools.test_render_contracts tools.test_frame_rate
```

The GPU tests build a tiny fixed-profile TensorRT engine locally; no SeedVR model
download is required. They exercise actual TensorRT streams/buffers, GPU effects
parity, pinned checkpoint copies, combined decoding/finishing, and available
NVENC encoders. CPU tests cover fallback/recovery, warm-worker process reuse and
retirement, browser settings, precision, scene cuts, and chunk resume/export.

These checks have run on an RTX 4090 with CUDA PyTorch and TensorRT RTX. They do
**not** establish full SeedVR VAE image parity, real model-cache memory use, or
end-to-end speedups: production SeedVR weights/engines are required for those.
Keep the Beta options opt-in until representative real clips have been compared.

## Reproducible CPU preparation benchmark

```console
python -m tools.benchmark_video_finishing --batches 24 --threads 4 --repeats 3
```

The benchmark uses the same saved synthetic FP32 batches in fresh child processes
and reports median elapsed time and peak process memory. On Windows it requires
`psutil`; Unix uses `resource`. Increase `--batches` to check memory scaling or
adjust `--frames`, `--width`, and `--height` to match a target workload.

An encoder that discards RGB bytes isolates tensor loading, seam treatment,
quantization, and copies. These are **not** end-to-end render, video-encoding, or
GPU inference measurements. Compare full rendering separately on the target GPU
using identical source material, restoration settings, and output codec settings.

Example local results (Windows, Python 3.14, CPU PyTorch 2.14.1; median of three
fresh processes per path, five 512x288 frames per batch):

| Batches / CPU threads | Whole-video time | Streaming time | Whole-video peak | Streaming peak |
|---|---:|---:|---:|---:|
| 24 / 4 | 0.718 s | 0.652 s | 1,048.3 MiB | 266.6 MiB |
| 48 / 8 | 1.376 s | 1.293 s | 1,886.1 MiB | 267.8 MiB |

Peak memory includes the Python/PyTorch process, not just tensors. Results vary
with CPU, thread count, filesystem caching, resolution, and batch size; the
important architectural change is that tensor memory no longer scales with the
number of batches. These measurements do not include the additional savings from
eliminating per-batch CUDA initialization or processed-tensor disk round trips.
