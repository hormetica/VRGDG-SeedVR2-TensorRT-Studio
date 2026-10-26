# VRGDG SeedVR2 TensorRT Studio

Local, GPU-accelerated video and bulk still-image restoration and upscaling with SeedVR2, TensorRT, and a purpose-built browser interface.

VRGDG SeedVR2 TensorRT Studio turns the SeedVR2 pipeline into a practical Windows workflow: restore videos with previews and resumable checkpoints, or upscale collections of still images with custom output locations. Processing stays on your machine.

![VRGDG SeedVR2 TensorRT Studio full interface](images/seedvr2-studio-screenshot.png)

## Highlights

- **Bulk still images** — combine local image/folder selections, preview a still, and upscale JPEG/PNG collections into your chosen output location with transparency, progress, cancellation, and failed/missing-image retry.
- **Fast local restoration** — SeedVR2 inference with TensorRT-accelerated VAE decoding on supported NVIDIA RTX GPUs.
- **Fast 2K upscaling** — As a real-world example, an 8-second clip took approximately 8 minutes to upscale to 2K on an NVIDIA RTX 5090 using the largest **7B Sharp FP16** model. Render times vary with source resolution, frame rate, settings, and available VRAM.
- **Preview before committing** — render a short segment, then inspect Original, Restored, Compare, or Side by side views.
- **Long-render recovery** — save completed chunks and continue from the first unfinished chunk after an interruption.
- **Practical output controls** — choose resolution, aspect policy, model precision, temporal batch, seed, and color correction.
- **Non-destructive finishing** — reprocess sharpening, grain, seam smoothing, and optional skin finishing without rerunning restoration.
- **Optional NVIDIA tuning** — GPU-resident finishing, warm model caching, per-stage reports, scene-cut protection, and H.264/HEVC/AV1 NVENC export, including true 10-bit SDR output.
- **Project-based video history** — reopen previous videos and keep media, manifests, and logs together under `outputs\`. Image history also tracks custom output locations.

## Before and after examples

Each recording demonstrates the Studio's interactive comparison view, with the original footage on one side of the wipe and the restored result on the other.

- [Watch comparison sample 1 in your browser (MP4, 6.6 MB)](https://cdn.jsdelivr.net/gh/vrgamegirl19/VRGDG-SeedVR2-TensorRT-Studio@eebba66e6c73cb420f2f24a581961802b6177f65/videos/comparisons/comparison-sample-01.mp4)
- [Watch comparison sample 2 in your browser (MP4, 13 MB)](https://cdn.jsdelivr.net/gh/vrgamegirl19/VRGDG-SeedVR2-TensorRT-Studio@eebba66e6c73cb420f2f24a581961802b6177f65/videos/comparisons/comparison-sample-02.mp4)

## Requirements

| | Recommended setup |
|---|---|
| Operating system | Windows 11 |
| GPU | NVIDIA RTX with a current driver |
| Storage | At least 35 GB free |
| Network | Internet connection for the first installation |
| Python | Installed automatically when needed; the project requires Python 3.12+ |

TensorRT engines are built locally for the installed GPU and runtime. Do not copy `.rtxplan` files between different GPU models or TensorRT versions.

## Quick start

1. Download or clone this repository.
2. Double-click **Install SeedVR Studio.bat** and leave the window open until setup completes.
3. Double-click **Launch SeedVR Studio Pro.bat**.
4. Drop a source video into the Media panel.
5. Render a short preview, inspect it, and then render the full video.

The installer prepares the private Python environment, FFmpeg, CUDA PyTorch, SeedVR2, SageAttention 2, TensorRT RTX, model weights, and GPU-specific engines. Downloads and engine preparation are resumable. The Pro launcher also detects an incomplete setup and opens the installer automatically.

For repair options and troubleshooting, see the [installation guide](docs/INSTALLATION.md).

## How to update

### Update from inside Studio

If the **update button (↻)** is visible beside **Exit Studio**:

1. Finish or stop any active render.
2. Click **↻** to check the `main` update channel.
3. If an update is available, click **Update and restart**.

Studio closes, downloads only the changed application files, refreshes Python dependencies only when required, and then restarts automatically. Models, GPU-specific TensorRT engines, `.venv`, outputs, manifests, logs, and saved settings remain in place.

The safe updater requires a clean Git-backed installation on the `main` branch. It refuses to overwrite tracked local edits or diverged Git history. If dependency repair fails after an update, it restores the previous application revision automatically.

### One-time update for existing installations

Users who do not yet see the **↻ button** can install the update system with one final manual update:

1. Download [`Update SeedVR Studio.bat`](bootstrap/Update%20SeedVR%20Studio.bat).
2. Place the downloaded BAT beside `Launch SeedVR Studio Pro.bat` in the existing Studio folder.
3. Completely close SeedVR Studio.
4. Double-click `Update SeedVR Studio.bat`.

The one-time updater downloads the current application code, creates an application-code backup under `outputs\`, and restarts Studio. It preserves the existing `.venv`, downloaded models, TensorRT engines, rendered videos, browser-saved settings, and other untracked user files.

For installations originally downloaded as a ZIP, this step also converts the existing folder into a clean Git-backed `main` installation. After it finishes, future updates use the **↻ button** inside Studio.

## Workflow

### 1. Load and inspect media

Drop a video onto the Media panel or browse for a file. After a preview or full render completes, use the viewer tabs to switch between the source and restored result.

| Load a source | Choose a viewer mode |
|---|---|
| ![Media panel with the original-video drop zone](images/ui-guide/controls/media-input.png) | ![Original, Restored, Compare, and Side by side viewer tabs](images/ui-guide/controls/viewer-modes.png) |

- **Original** shows the uploaded source.
- **Restored** shows the latest rendered result.
- **Compare** overlays the restored video with a draggable wipe divider.
- **Side by side** places both videos next to each other.

The shared viewer also supports timeline scrubbing, frame stepping, synchronized playback, fullscreen, zoom, and pan.

### 2. Choose the render path and target

Start with **SeedVR2 + TensorRT** for the accelerated path, then select a hardware preset and output target. Presets provide sensible starting values; **Custom / manual** leaves every setting under your control.

For automatic help, choose your output size and click **Recommend settings**.
Studio inspects the rendering GPU, total/free VRAM, system RAM, and installed
attention packages, then shows a balanced recommendation to review and apply.
It adjusts the model, temporal batch, attention, block swapping, and VAE tiling
for the selected engine and output size. Other settings, including output size,
finishing, export, and Beta options, stay unchanged. Recommendations are conservative
estimates, not benchmarks or guaranteed optimal settings: test a short preview,
then use **Save current settings** if you want to retain them. Original-size
recommendations require a loaded source video. No models are loaded or rendered
during detection. If CUDA is unavailable or too little VRAM is free, your
settings remain untouched.

| Render engine | Hardware preset | Output and aspect |
|---|---|---|
| ![SeedVR2 and TensorRT render-engine selector](images/ui-guide/controls/render-engine.png) | ![Hardware preset selector](images/ui-guide/controls/hardware-preset.png) | ![Output-size and crop-policy selectors](images/ui-guide/controls/output-and-crop.png) |

Output presets include original-size enhancement, 1K/1080p, 2K/1440p, and 4K/2160p. Preserve the source aspect ratio or center-crop it to 16:9.

### 3. Tune restoration

Choose the model, temporal batch, seed, and color-matching strategy. FP8 models use less VRAM; FP16 models prioritize quality and require more memory. Temporal batches follow SeedVR2's `4n+1` rule, while the TensorRT path currently supports batches **5** and **21**.

| Model, batch, and seed | Color correction | Resumable rendering |
|---|---|---|
| ![Model, temporal-batch, and seed controls](images/ui-guide/controls/model-batch-seed.png) | ![Color-correction selector and available modes](images/ui-guide/controls/color-correction.png) | ![Resumable long-video render and chunk-length controls](images/ui-guide/controls/resumable-render.png) |

Color correction is applied during the main restoration. Use `none` to retain source colors; `lab` is a useful general-purpose matching mode. The other available modes are `wavelet`, `wavelet_adaptive`, `hsv`, and `adain`.

For long videos, enable resumable rendering. **Auto** selects a chunk length from the duration, frame count, frame rate, and temporal batch; manual choices range from 30 seconds to 30 minutes. Completed chunks remain available if a job stops, and **Retry from last chunk** continues at the first unfinished section.

### 4. Balance speed, memory, and finishing

TensorRT uses **Optimized Fast** decoding automatically. If it fails, the Studio retries the saved latents with its internal Stable decoder, avoiding another restoration pass.

| Performance | Post-processing | Skin finishing |
|---|---|---|
| ![Attention, block swapping, and VAE-tiling controls](images/ui-guide/controls/performance.png) | ![Sharpening, film grain, and seam-smoothing controls](images/ui-guide/controls/post-processing.png) | ![Optional skin-finishing controls](images/ui-guide/controls/skin-finishing.png) |

- **Attention:** SageAttention 2 is the preferred default. Missing accelerated backends fall through to another installed option and finally SDPA.
- **Blocks to swap:** higher values reduce VRAM demand but cost speed.
- **VAE tiling:** reduces memory use at a speed cost; leave it off unless VRAM is insufficient.
- **Smooth TensorRT batch seams:** matches color and noise around decoded batch boundaries and is enabled by default.
- **Skin finishing:** uses a stabilized skin mask to refine existing pixels. It is non-generative and cannot recreate missing facial identity detail; preview stronger settings first.

The **Performance** section also offers opt-in GPU-resident finishing and warm
models cached in system RAM. **Export quality** adds optional NVENC and 10-bit
HEVC/AV1 presets while keeping CPU H.264 as the default. Use **Performance report**
after a render to inspect stage timings, actual attention dispatch, and encoder
fallbacks. See [performance architecture and validation](docs/PERFORMANCE.md) for
requirements, recovery behavior, and the limits of the GPU tests.

Post-processing can be changed later with **Reprocess post only**, without rerunning SeedVR2.

### 5. Preview, render, and reopen results

Set a short preview range, render it, and compare the result before starting the complete video. The status bar reports readiness, elapsed time, and ETA while a job runs.

| Preview range | Render actions |
|---|---|
| ![Preview start and length fields](images/ui-guide/controls/preview-range.png) | ![Reprocess, preview, and full-render buttons](images/ui-guide/controls/render-actions.png) |

| Render status | Previous results |
|---|---|
| ![Ready state, elapsed time, ETA, power, and fullscreen controls](images/ui-guide/controls/render-status.png) | ![Previous-results selector and load button](images/ui-guide/controls/previous-results.png) |

Every job receives its own folder under `outputs\` with rendered media, manifests, and logs. Use **Load selected output** to reopen a result from the current workspace.

## Bulk still images

Choose **Images** at the top of Studio. Use **Add images**, **Add folder**, or an
absolute folder path to collect images from Desktop, Downloads, or different
folders. Folder additions are non-recursive. Select your **output location**,
review the filenames and dimensions, preview a selected image, then click
**Start image batch**. Every batch gets a new folder under your chosen location;
originals and existing results are never overwritten.

Image settings are independent of video settings. **Auto** tries the existing
five-frame TensorRT VAE engines for each still independently, then visibly falls
back to one-frame SeedVR if needed. Both routes have passed bounded RTX 4090 image
checks, including alpha and mixed dimensions. TensorRT is not necessarily faster
for stills. See [the image workflow guide](docs/BULK_IMAGES.md) for recovery,
format limits, custom-output history, and validation details.

## Render engines

| Engine | Best for | Temporal batches | Notes |
|---|---|---:|---|
| **SeedVR2 + TensorRT** | Recommended accelerated workflow | 5, 21 | Uses TensorRT VAE decoding and automatic fallback from saved latents. |
| **SeedVR2 (Legacy)** | Compatibility and additional batch choices | SeedVR2-compatible values | Standard PyTorch path; can be substantially slower on long videos. |

The temporal-batch menu updates automatically when the render engine changes.

## Settings that matter most

- Use **Recommend settings**, or begin with the hardware preset closest to your VRAM capacity.
- Use a short representative preview before every long render.
- Increase temporal batch only when the selected engine and available VRAM support it.
- Use block swapping or VAE tiling when memory is the limiting factor, accepting the speed tradeoff.
- Keep skin finishing subtle and confirm faces, freckles, moles, and other permanent marks in the preview.
- Use post-only reprocessing for finishing changes; it is much faster than repeating restoration.

## Files and local data

Model weights live in `models\` and are not committed to GitHub. Outputs, virtual environments, TensorRT engines and caches, temporary ONNX exports, and optional runtime files also remain local.

| Path | Purpose |
|---|---|
| `web/` | JavaScript Studio interface |
| `api_server.py` | FastAPI service |
| `seedvr_studio/` | Rendering pipeline and legacy Gradio integration |
| `tools/` | TensorRT, assembly, post-processing, and diagnostic utilities |
| `tensorrt_backend/` | Optional native TensorRT sources |
| `scripts/` | Setup, launcher, download, verification, and engine-preparation scripts |
| `vendor/seedvr2/` | Compatible Apache-2.0 SeedVR2 integration required by the Studio |

## Documentation

- [Installation and troubleshooting](docs/INSTALLATION.md)
- [SageAttention setup and verification](docs/SAGEATTENTION.md)
- [Performance architecture and benchmarks](docs/PERFORMANCE.md)
- [Bulk still-image workflow](docs/BULK_IMAGES.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## License

The Studio source is released under the [MIT License](LICENSE). SeedVR2 and bundled runtime components retain their upstream licenses; see the [third-party notices](THIRD_PARTY_NOTICES.md).
