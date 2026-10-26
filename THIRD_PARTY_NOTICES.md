# Third-party notices

## SeedVR2 integration

The compatible source under **vendor/seedvr2** is derived from ComfyUI-SeedVR2_VideoUpscaler by NumZ and contributors, based on SeedVR by ByteDance Seed. It is distributed under the Apache License 2.0 included at **vendor/seedvr2/LICENSE**.

Studio-specific changes add TensorRT encoder handoff, latent capture, bounded source-frame reading, and stop-before-VAE integration used by the fast engine, plus stage profiling, attention-dispatch reporting, GPU-generation checks, and reusable CLI argument parsing for the optional warm worker.

Model weights, NVIDIA TensorRT RTX packages, FFmpeg, PyTorch, SageAttention, Triton, and other dependencies retain their own licenses and are downloaded or installed separately.
