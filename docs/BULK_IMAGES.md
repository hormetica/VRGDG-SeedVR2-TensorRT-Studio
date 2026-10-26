# Bulk still-image upscaling

The **Images** tab is a same-machine Windows workflow beside the existing
**Videos** tab. It uses the already installed SeedVR models; image rendering
does not download weights or build TensorRT engines automatically.

## Use it

1. Restart Studio after installing this code, then choose **Images**.
2. **Add images** can select multiple files. Use it again to add files from
   another folder. **Add folder** includes that folder's immediate JPG/JPEG/PNG
   files, not subfolders. A folder-path field is available if the native picker
   cannot open. Desktop, Downloads, and other accessible folders work directly;
   there is no requirement to move originals into the repository.
3. Choose an **output location** with Browse or a full path. To create a missing
   parent directory, explicitly enable **Create output folder if missing**.
   Existing directories can also be created in the Windows folder picker.
4. Set shared model, scale (1× enhancement / 2× / 4×), maximum edge and engine.
   The maximum edge can reduce the requested scale, including a large 1× source.
   Advanced controls include seed, attention, block swapping and VAE tiling.
5. Review the exact dimensions, source locations, output names and complete batch
   destination. Remove unwanted items before Start. The table shows 50 items per
   page rather than constructing thousands of image thumbnails.
6. Select a row and **Preview selected** to test one image. Previews are separate
   runs under Studio's default image outputs and do not consume the full review.
   Compare, Original, Restored, Side by side, wipe, zoom, pan and fullscreen work
   with still images, without video playback controls.
7. **Start image batch** processes images serially. Status and each completed
   row identify the actual route; a fallback reason remains visible on the item.
   **Open output folder** works for custom destinations as well as the default.

Changing settings or destinations requires a fresh review. Later folder additions
are not watched automatically. Adding the same canonical source path twice keeps
one item; different paths remain separate even if their names or bytes match.

## Output and source safety

- The chosen **output root** receives a new, uniquely named **batch output
  folder**. Selecting Desktop as the root creates a new batch folder on Desktop,
  not replacement files beside the originals.
- Normal names retain the complete original name: `photo.jpg_upscaled.png` and
  `photo.png_upscaled.png`. Same-named inputs from different folders, Windows
  reserved names and long names are safely disambiguated/shortened with stable
  item identifiers where necessary. The review shows the final mapping.
- Sources are read-only. A content fingerprint ties processing to reviewed bytes;
  changed/missing sources fail that item rather than silently substituting new
  content. Temporary per-image copies are isolated in the batch folder and removed
  after use. The whole collection is not uploaded or copied in advance.
- PNGs are verified before no-clobber publication. Existing targets are never
  overwritten, and partial writes are not marked complete. A filesystem without
  hard-link support uses exclusive-create publication instead.
- The small history/index files live in `outputs/image-batch-history/`, even when
  image results are elsewhere. Keep that directory to retain custom-root history.
  Moved/disconnected result folders are reported unavailable, not silently replaced.
- New local-filesystem routes require loopback, a localhost Host and same-origin
  browser requests. Images are served by registered batch/item IDs, not an
  arbitrary disk-path URL. Remote-browser folder upload is not supported.

## Inference and fidelity

**Auto** prefers TensorRT. Each independent image is padded temporally with copies
of itself to use the existing five-frame VAE encoder/decoder profiles; only its
first real frame is retained. Unrelated images never share temporal context.
DiT restoration remains PyTorch. If TensorRT fails, its worker is retired and
Auto retries through native one-frame SeedVR. That reason is recorded; subsequent
images in the same batch avoid repeatedly trying the known-failed accelerated
route. **SeedVR** selects native one-frame processing directly. **TensorRT — no
fallback** makes failures explicit without silently changing routes.

Models are cached within a batch when system RAM permits. Caches live in a
dedicated worker, are parked in CPU RAM between generations, and are retired on
failure or completion. A shared GPU reservation serializes images and video
renders in the same Studio server. Image cancellation targets its own worker's
process tree (on Windows the venv `python.exe` is a launcher with a separate
interpreter child; both are stopped). It never pattern-matches other processes.
Cancelling a queued video does not stop an image batch or a different video.

Inputs are static 8-bit JPEG/PNG, including normal grayscale and palette PNGs
converted to RGB/RGBA. Visible EXIF orientation is applied. Odd output dimensions
are preserved with edge padding around the model, not silent even-size cropping.
An original PNG alpha channel is resized deterministically with Lanczos and
reattached independently of the model; it does not use guided/generated matte
changes. Only RGB goes through diffusion.

High-bit-depth PNG, animated images, other formats, metadata/profile preservation,
professional color-management guarantees, per-image settings overrides and video
finishing controls are not included. Sources and requested outputs are limited to
64 megapixels, source compressed size to 512 MiB, maximum edge to 64–16384 pixels,
and at least two pixels on each edge. Generative restoration can change details;
inspect faces, text, colors and transparency edges before a large run.

## Stop and retry

**Stop batch** stops only that batch and retains completed PNGs. Individual file
errors normally do not prevent other images from completing. Failed inference
retires the worker before another image can reuse it. Unsafe driver/device errors
stop the batch instead of repeatedly submitting work to an unhealthy GPU.

**Retry failed / missing** keeps the original settings, reviewed source identities
and output mapping. It skips unchanged completed outputs and retries failed,
cancelled or missing outputs. It will not overwrite an unexpected or changed
existing output; move that file aside or start a new batch. A source that changed
needs a new review. Invalid files rejected during review also need a fresh review
after fixing them.

History remains available after reopening Studio. Interrupted runs never start
automatically; explicit retry is available. There is no durable automatic queue
recovery or recursive source-folder watch. **Inference log** opens the saved
worker messages and diagnostic context for the selected run.

## Verification

CPU/HTTP contracts and existing video routing/recovery tests:

```powershell
.venv\Scripts\python.exe -X utf8 -B -m unittest tools.test_image_batches tools.test_render_contracts -v
```

The full suite is `python -m unittest discover -s tools -p "test_*.py" -v`
using the same virtual environment. On 2026-10-25, all **122 tests passed**,
including CUDA/TensorRT, FFmpeg and NVENC checks, with no skips. Run from a normal
Windows environment with FFmpeg on PATH; a WSL-launched process may not inherit
the registered Windows PATH.

An opt-in installed-model GPU check is provided; run it only with an idle GPU and
a **new disposable output directory**:

```powershell
.venv\Scripts\python.exe -X utf8 -B tools\smoke_image_inference.py "$env:TEMP\opencode\my-new-image-proof" --timeout 300
```

On 2026-10-25, an RTX 4090 with the installed 3B FP16 weights and five-frame
engines completed RGB → transparent/mixed-size PNG → RGB sequences through both
routes. Outputs were 768×512 and 382×514 (including a canvas smaller than one
TensorRT tile); alpha matched the deterministic resample, originals remained
unchanged, and repeated RGB outputs had mean absolute pixel error 0 within each
route. This proves those bounded paths, not cross-route quality parity, arbitrary
high-resolution capacity, every attention/model option, or thousand-image
throughput. The warmed native path was faster than TensorRT on these small
fixtures; no general TensorRT speedup is claimed.

The live browser workflow also completed a preview and a two-image TensorRT batch
into a custom output root. Reloading recovered that batch's history and displayed
both original/restored images; comparison, zoom and switching back to Videos were
checked without browser errors. Stopping a batch during real GPU restoration
terminated both the venv launcher and its interpreter child, and no incomplete
PNG was published; `OwnedWorkerProcessTests` guards that Windows behavior. Windows Tk root
creation was checked; native dialog choice/cancellation still uses the standard
Windows picker. No large-scale throughput or subjective model-quality claim is
implied by these checks.
