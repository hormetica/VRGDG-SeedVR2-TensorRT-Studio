"""Portable ONNX tracing must not disable fast convolution or hide CUDA OOM."""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import prepare_tensorrt
from tools import onnx_export_utils as exporter


class VendorLikeConv3d(torch.nn.Conv3d):
    """The VAE's direct-cuDNN workaround is not a portable ONNX operator."""
    def _conv_forward(self, value, weight, bias):
        raise AssertionError("Export must use standard Conv3d dispatch, not the vendor-only workaround")


class PortableExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "model.onnx"

    def test_trace_keeps_cudnn_enabled_and_bypasses_only_vendor_conv_dispatch(self):
        module = VendorLikeConv3d(3, 4, kernel_size=3, padding=1).eval()
        value = torch.randn(1, 3, 5, 8, 8)
        original = module._conv_forward

        def export(model, args, path, **options):
            self.assertTrue(torch.backends.cudnn.enabled, "Disabling cuDNN forces expensive native Conv3d workspaces")
            result = model(*args)
            self.assertEqual(result.shape, (1, 4, 5, 8, 8))
            Path(path).write_bytes(b"fixture")

        with patch.object(torch.onnx, "export", side_effect=export):
            exporter._portable_export(module, (value,), self.output, legacy=True)
        self.assertEqual(module._conv_forward, original)

    def test_cpu_retry_must_start_in_a_fresh_process_not_inside_the_oom_traceback(self):
        module = torch.nn.Identity()
        oom = torch.cuda.OutOfMemoryError("CUDA out of memory")
        with (
            patch.object(exporter, "_portable_export", side_effect=oom) as export,
            self.assertRaises(torch.cuda.OutOfMemoryError),
        ):
            exporter.export_portable_onnx(module, (torch.zeros(1, 3, 5, 8, 8),), self.output, legacy=True)
        self.assertEqual(export.call_count, 1)

    def test_cpu_device_does_not_require_cuda(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(exporter.export_device("cpu"), torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError, "CUDA.*unavailable"):
                exporter.export_device("cuda")

    def test_cpu_half_interface_runs_float32_convolution(self):
        module = torch.nn.Conv3d(3, 4, 3, padding=1).half()
        wrapper = exporter._CPUFloatExport(module)
        seen = []
        hook = module.register_forward_pre_hook(lambda model, inputs: seen.append(inputs[0].dtype))
        try:
            output = wrapper(torch.zeros(1, 3, 5, 8, 8).half())
        finally:
            hook.remove()
        self.assertEqual(seen, [torch.float32])
        self.assertEqual(module.weight.dtype, torch.float32)
        self.assertEqual(output.dtype, torch.float16)

    def test_failed_export_preserves_completed_graph_and_removes_partial_file(self):
        self.output.write_bytes(b"previous validated graph")

        def fail(module, args, output, *, legacy):
            output.write_bytes(b"partial new graph")
            raise RuntimeError("unsupported fixture operator")

        with patch.object(exporter, "_portable_export", side_effect=fail), self.assertRaisesRegex(RuntimeError, "unsupported"):
            exporter.export_portable_onnx(torch.nn.Identity(), (torch.zeros(1),), self.output, legacy=True)
        self.assertEqual(self.output.read_bytes(), b"previous validated graph")
        self.assertFalse(self.output.with_name("model.exporting.onnx").exists())

    def test_preparation_retries_only_explicit_cuda_out_of_memory(self):
        failure = subprocess.CalledProcessError(86, ["exporter"])
        with patch.object(prepare_tensorrt, "run", side_effect=[failure, None]) as run:
            prepare_tensorrt.export_with_fallback("export_vae_encoder.py", ("--frames", "21"), self.output)
        self.assertEqual(run.call_count, 2)
        self.assertIn("cuda", run.call_args_list[0].args[0])
        self.assertIn("cpu", run.call_args_list[1].args[0])
        with (
            patch.object(prepare_tensorrt, "run", side_effect=subprocess.CalledProcessError(1, ["exporter"])) as run,
            self.assertRaises(subprocess.CalledProcessError),
        ):
            prepare_tensorrt.export_with_fallback("export_vae_encoder.py", (), self.output)
        self.assertEqual(run.call_count, 1)

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("tensorrt_rtx"), "CUDA and TensorRT required")
    def test_portable_gpu_and_cpu_exports_are_parseable_and_match_tensor_rt(self):
        import onnx
        import tensorrt_rtx as trt

        for device in ("cuda", "cpu"):
            with self.subTest(device=device):
                torch.manual_seed(42)
                module = VendorLikeConv3d(3, 4, 3, padding=1).eval().to(device).half()
                value = torch.randn(1, 3, 5, 8, 8, device=device).half()
                exporter.export_portable_onnx(module, (value,), self.output, legacy=True)
                graph = onnx.load(str(self.output))
                self.assertIn("Conv", [node.op_type for node in graph.graph.node])
                self.assertEqual(graph.graph.input[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT16)
                self.assertEqual(graph.graph.output[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT16)
                logger = trt.Logger(trt.Logger.ERROR)
                builder = trt.Builder(logger)
                network = builder.create_network()
                parser = trt.OnnxParser(network, logger)
                self.assertTrue(parser.parse_from_file(str(self.output)), [str(parser.get_error(i)) for i in range(parser.num_errors)])
                blob = builder.build_serialized_network(network, builder.create_builder_config())
                self.assertIsNotNone(blob)
                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(blob)
                context = engine.create_execution_context()
                source = value.cuda()
                actual = torch.empty(1, 4, 5, 8, 8, device="cuda", dtype=torch.float16)
                context.set_tensor_address(engine.get_tensor_name(0), source.data_ptr())
                context.set_tensor_address(engine.get_tensor_name(1), actual.data_ptr())
                stream = torch.cuda.current_stream()
                self.assertTrue(context.execute_async_v3(stream.cuda_stream))
                stream.synchronize()
                # Both traces represent ordinary convolution, even though the
                # runtime VAE subclass has a vendor-specific forward override.
                with exporter._portable_convolutions(module), torch.inference_mode():
                    expected = module(value.float() if device == "cpu" else value).half().cuda()
                torch.testing.assert_close(actual, expected, rtol=0.003, atol=0.003)


if __name__ == "__main__":
    unittest.main()
