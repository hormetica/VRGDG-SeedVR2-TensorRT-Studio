# Installation guide

## Supported setup

The packaged installer currently targets:

- Windows 11
- An NVIDIA RTX GPU with a current driver
- At least 35 GB of free disk space for the environment, models, temporary ONNX files, and TensorRT engines
- An internet connection for the first installation

TensorRT RTX engines are built for the GPU in the computer running the installer. Do not copy RTX plan files between different GPU models or TensorRT versions.

## One-click installation

1. Download or clone the repository.
2. Double-click **Install SeedVR Studio.bat**.
3. Leave the installer window open. It installs or checks Python 3.12, FFmpeg, CUDA PyTorch, the included compatible SeedVR2 integration, SageAttention 2, Triton for Windows, TensorRT RTX, ONNX tools, the default model and VAE, and four TensorRT engines.
4. When it reports success, double-click **Launch SeedVR Studio Pro.bat**.

The Pro launcher also detects an incomplete setup and starts the installer. Setup is resumable: completed model downloads and valid engines are skipped.

## Optional installer switches

Run these from PowerShell only if needed:

~~~powershell
.\scripts\install.ps1 -Repair
.\scripts\install.ps1 -SkipTensorRT
.\scripts\install.ps1 -SkipModels
~~~

**SkipTensorRT** leaves only SeedVR2 Legacy usable. **SkipModels** delays model downloads until a model is first used. A complete normal installation uses neither switch.

## Why engines are built locally

TensorRT plan files are tied to the GPU architecture and runtime version. The repository includes the exporter and builder, not another computer's plans. The installer creates:

- vae_encoder_5f_tile512.rtxplan
- vae_encoder_21f_tile512.rtxplan
- vae_decoder_tile_512_5f.rtxplan
- vae_decoder_tile_256_21f.rtxplan

If setup stops during this stage, run it again. Existing valid plans are retained.

## Logs and troubleshooting

The installer log is saved to **outputs\install.log**. Render logs are saved inside project directories under **outputs**.

Common failures:

- **No NVIDIA driver detected:** install or update the NVIDIA driver, restart Windows, and rerun setup.
- **Not enough disk space:** clear space and rerun. Partial model downloads resume.
- **Launch reports that the app window did not open:** the server often did start. Older launchers required an Edge window handle that Windows 11 does not always expose, then shut the server down. Use **Launch SeedVR Studio Pro.bat** (not the installer). If the UI still vanishes, open http://127.0.0.1:7870/ and check **outputs\js_server_error.log**.
- **TensorRT build failure:** close GPU-heavy programs, confirm the NVIDIA driver is current, then rerun.
- **SageAttention verification failure:** follow the [SageAttention guide](SAGEATTENTION.md).
- **FFmpeg still missing after winget:** restart Windows so the system PATH refreshes, then rerun.

The installer does not require a separate full CUDA Toolkit. The Python CUDA and TensorRT packages provide the runtime used by the Studio.


### If TensorRT export appears hung

Each TensorRT profile performs a fixed-shape ONNX export followed by an engine
build. Both steps print elapsed-time heartbeats; these indicate that the process
is active, not a percentage or guaranteed completion time. Engine preparation is
resumable, so completed `.rtxplan` files are skipped on the next run.

`TracerWarning: Converting a tensor to a Python boolean` is expected for the
fixed-shape VAE profiles. These engines are intentionally specialized to the
printed dimensions and temporal frame count; this warning alone is not failure.

GPU export keeps cuDNN enabled. Only the VAE's direct-cuDNN override is bypassed
during tracing, allowing standard ONNX `Conv` operators without forcing enormous
native convolution workspaces. If CUDA still runs out of memory, the exporter
exits completely before a fresh CPU process retries. CPU computation uses FP32
instead of slow CPU FP16 convolution, while explicit casts preserve the engines'
FP16 input/output contract. Other export errors are reported rather than being
mislabelled as OOM and blindly retried.

Older exporters printed **"CUDA ONNX export ran out of memory; falling back to
CPU export"** and could spend a very long time on large CPU FP16 traces. They
also ignored `--device cpu`. If using that code, update the exporter before
retrying rather than repeatedly launching another copy. ONNX files and engine
plans are now published atomically so an interrupted new export does not replace
a completed artifact.

To resume only missing engine preparation after fixing an export problem, without
reinstalling packages or downloading models again:

```powershell
.\.venv\Scripts\python.exe -u .\scripts\prepare_tensorrt.py
```

After it succeeds, rerun the normal installer for the final readiness check, or
run `scripts\verify_install.py` in a shell where FFmpeg and FFprobe are on PATH.

If you need to finish setup without TensorRT, run:

```powershell
.\scripts\install.ps1 -SkipTensorRT
```

This leaves SeedVR2 Legacy available. Run the installer again later to build missing GPU-specific engines.
