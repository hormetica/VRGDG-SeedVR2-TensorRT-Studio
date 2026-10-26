"""Portable, observable ONNX export helpers for TensorRT engine preparation."""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from threading import Event, Thread
from types import MethodType

import torch

CUDA_OOM_EXIT_CODE = 86


def export_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export was requested, but CUDA is unavailable")
    return torch.device(name)


@contextmanager
def _portable_convolutions(module):
    """Bypass vendor-only operators, NOT cuDNN itself.

    The VAE's direct torch.cudnn_convolution workaround is not exportable. The
    standard Conv3d forward exports to ONNX Conv while still using fast cuDNN
    execution during tracing. Disabling cuDNN creates huge native workspaces.
    Override only this model's instances and always restore them.
    """
    previous = []
    for layer in module.modules():
        if isinstance(layer, torch.nn.Conv3d) and type(layer)._conv_forward is not torch.nn.Conv3d._conv_forward:
            previous.append((layer, layer.__dict__.get("_conv_forward")))
            layer._conv_forward = MethodType(torch.nn.Conv3d._conv_forward, layer)
    try:
        yield
    finally:
        for layer, original in previous:
            if original is None:
                del layer._conv_forward
            else:
                layer._conv_forward = original


class _CPUFloatExport(torch.nn.Module):
    """FP32 CPU convolution with the fixed engines' FP16 input/output contract."""
    def __init__(self, module):
        super().__init__()
        self.module = module.float()

    def forward(self, value):
        return self.module(value.float()).half()


@contextmanager
def export_progress(label: str, interval: float = 15):
    started = time.perf_counter()
    stopped = Event()

    def heartbeat():
        while not stopped.wait(interval):
            import psutil
            memory = psutil.Process().memory_info()
            print(f"{label}: still working ({time.perf_counter() - started:.0f}s elapsed, "
                  f"RAM {memory.rss / 1024**3:.1f} GiB resident)", flush=True)

    thread = Thread(target=heartbeat, name="export-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()

def _math_attention_context():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except (ImportError, AttributeError):
        return nullcontext()


def _portable_export(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: Path, *, legacy: bool) -> None:
    is_encoder = args[0].ndim == 5 and args[0].shape[1] == 3
    if args[0].device.type == "cpu" and args[0].dtype == torch.float16:
        module = _CPUFloatExport(module)
    with torch.inference_mode(), _portable_convolutions(module), _math_attention_context():
        torch.onnx.export(
            module,
            args,
            str(output),
            input_names=["video"] if is_encoder else ["latent"],
            output_names=["latent_raw"] if is_encoder else ["sample"],
            opset_version=20,
            dynamo=not legacy,
            optimize=False,
            do_constant_folding=False,
            external_data=False,
        )


def export_portable_onnx(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output: Path,
    *,
    legacy: bool,
) -> None:
    """Export atomically; the parent retries CUDA OOM in a fresh CPU process."""
    tracer = "legacy tracer" if legacy else "dynamo"
    print(f"Exporting ONNX ({tracer}, {args[0].device}, fast portable convs) -> {output}", flush=True)
    started = time.perf_counter()
    temporary = output.with_name(output.stem + ".exporting.onnx")
    try:
        with export_progress("ONNX export"):
            _portable_export(module, args, temporary, legacy=legacy)
        import onnx
        onnx.checker.check_model(str(temporary))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"ONNX export finished in {time.perf_counter() - started:.1f}s", flush=True)


def run_export(main) -> None:
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        # Leave this process before CPU fallback so exception frames, weights,
        # and the failed trace cannot retain GPU/native convolution workspaces.
        print("CUDA ONNX export ran out of memory. Releasing this process before CPU FP32 retry.", flush=True)
        raise SystemExit(CUDA_OOM_EXIT_CODE) from None
