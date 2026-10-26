"""Export precision, scene cuts, checkpoints, profiling, and warm-worker contracts."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from seedvr_studio.encoding import EncoderPlan, ExportSettings, select_encoder
from seedvr_studio.inference_worker import WarmInferenceProcess
from seedvr_studio.media import RenderCancelledError
from seedvr_studio.profiling import PerformanceReport
from tools.assemble_tensor_video import finish_batches, is_scene_cut, quantize_rgb
from tools.tensor_artifacts import AsyncTensorWriter


class ExportAndCutTests(unittest.TestCase):
    def test_default_export_keeps_original_codec_and_quality(self):
        plan = select_encoder(ExportSettings(), 1920, 1080)
        self.assertEqual(plan.codec, "libx264")
        self.assertEqual(plan.input_format, "rgb24")
        self.assertEqual(plan.arguments(), ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"])

    def test_hardware_failure_preserves_codec_family_and_ten_bit_depth(self):
        for preset, codec in (("h264_nvenc", "libx264"), ("hevc_nvenc_10bit", "libx265"), ("av1_nvenc_10bit", "libaom-av1")):
            with self.subTest(preset=preset), patch("seedvr_studio.encoding._probe_nvenc", return_value="driver unavailable"):
                plan = select_encoder(ExportSettings(preset=preset), 1920, 1080)
                self.assertEqual(plan.codec, codec)
                self.assertEqual(plan.bit_depth, 8 if preset == "h264_nvenc" else 10)
                self.assertEqual(plan.fallback_reason, "driver unavailable")
        with (
            patch("seedvr_studio.encoding._probe_nvenc", return_value="missing"),
            self.assertRaisesRegex(RuntimeError, "unavailable"),
        ):
            select_encoder(ExportSettings("hevc_nvenc_10bit", allow_fallback=False), 128, 128)

    def test_ten_bit_path_contains_precision_that_rgb8_would_destroy(self):
        frame = torch.linspace(-1, 1, 1024).view(1, 1024, 1).expand(1, 1024, 3)
        high = quantize_rgb(frame, 10)
        low = quantize_rgb(frame, 8)
        self.assertEqual(len(set(high[:, :, 0].ravel().tolist())), 1024)
        self.assertEqual(len(set(low[:, :, 0].ravel().tolist())), 256)
        self.assertTrue((high % 257 != 0).any())
        self.assertEqual(high.dtype.str, "<u2")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_rgb48_gpu_quantization_matches_cpu(self):
        frame = torch.linspace(-1, 1, 1024).view(1, 1024, 1).expand(1, 1024, 3)
        cpu = quantize_rgb(frame, 10).astype("int32")
        gpu = quantize_rgb(frame.cuda(), 10).astype("int32")
        self.assertLessEqual(abs(cpu - gpu).max(), 1)

    def test_chunk_filters_are_composed_and_do_not_reinterpret_yuv_as_full_range(self):
        plan = EncoderPlan("hevc_nvenc_10bit", "hevc_nvenc", 10, 18)
        args = plan.arguments(video_filter="trim=start=1,setpts=PTS-STARTPTS", raw_rgb=False)
        self.assertEqual(args.count("-vf"), 1)
        self.assertIn("trim=start=1,setpts=PTS-STARTPTS,scale=in_range=auto", args[args.index("-vf") + 1])

    def test_hard_cuts_are_skipped_but_motion_with_similar_colors_is_not(self):
        dark = torch.full((1, 3, 3, 16, 16), -0.75)
        light = -dark
        self.assertTrue(is_scene_cut(dark, light))
        torch.manual_seed(6)
        texture = torch.rand(1, 3, 3, 16, 16) * 2 - 1
        self.assertFalse(is_scene_cut(texture[:, :, :1], texture[:, :, :1].roll(5, -1)))
        report = PerformanceReport()
        actual = list(finish_batches([(dark, None), (light, None)], seam_mode="blend", scene_cut_detection=True, report=report))
        torch.testing.assert_close(actual[1], light, rtol=0, atol=0)
        self.assertEqual(report.metadata["seams_skipped_at_cuts"], 1)

    def test_a_cut_inside_the_seam_head_also_skips_matching(self):
        previous = torch.full((1, 3, 3, 8, 8), -0.75)
        current = previous.clone()
        current[:, :, 1:] = 0.75
        report = PerformanceReport()
        actual = list(
            finish_batches([(previous, None), (current, None)], seam_mode="blend", seam_frames=2,
                           scene_cut_detection=True, report=report)
        )
        torch.testing.assert_close(actual[1], current, rtol=0, atol=0)
        self.assertEqual(report.metadata["seams_skipped_at_cuts"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gpu_finishing_matches_cpu_seams_within_one_rgb8_level(self):
        torch.manual_seed(17)
        videos = [torch.rand(1, 3, 5, 16, 16) * 2 - 1 for _ in range(3)]
        effects = {"grain_intensity": 0.02, "sharpen_strength": 0.2, "seed": 31}
        cpu = list(finish_batches([(video, 5) for video in videos], effects=effects))
        gpu = list(finish_batches([(video.cuda(), 5) for video in videos], effects=effects))
        for left, right in zip(cpu, gpu):
            for frame in range(5):
                a = quantize_rgb(left[0, :, frame].permute(1, 2, 0)).astype("int16")
                b = quantize_rgb(right[0, :, frame].permute(1, 2, 0)).astype("int16")
                self.assertLessEqual(abs(a - b).max(), 1)


class CheckpointAndProfileTests(unittest.TestCase):
    def test_raw_cpu_checkpoint_is_an_immutable_snapshot_and_is_atomically_published(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.pt"
            raw = torch.full((1, 3, 2, 4, 4), 0.25)
            with AsyncTensorWriter() as writer:
                writer.submit(path, {"video": raw, "original_frames": 1})
                raw.fill_(0.75)
            payload = torch.load(path, weights_only=False)
            torch.testing.assert_close(payload["video"], torch.full_like(raw, 0.25))
            self.assertEqual(payload["original_frames"], 1)
            self.assertFalse(path.with_name("raw.pt.tmp").exists())

    def test_disk_failure_is_propagated_not_reported_as_render_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.pt"
            writer = AsyncTensorWriter()
            with patch("tools.tensor_artifacts.torch.save", side_effect=OSError("disk full")):
                writer.submit(path, {"video": torch.zeros(1)})
                with self.assertRaisesRegex(OSError, "disk full"):
                    writer.close()
            self.assertFalse(path.exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_pinned_checkpoint_copy_waits_for_its_gpu_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.pt"
            with AsyncTensorWriter() as writer:
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    raw = torch.ones((1, 3, 5, 16, 16), device="cuda") * 0.625
                    writer.submit(path, {"video": raw})
                del raw
            actual = torch.load(path, weights_only=False)["video"]
            torch.testing.assert_close(actual, torch.full_like(actual, 0.625))

    def test_reports_keep_stage_counts_and_record_failures_without_hiding_them(self):
        report = PerformanceReport(kind="test")
        with report.stage("phase"):
            pass
        with self.assertRaisesRegex(ValueError, "failed"), report.stage("phase"):
            raise ValueError("failed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            report.write(path)
            data = json.loads(path.read_text())
        self.assertEqual(data["stages"]["phase"]["calls"], 2)
        self.assertEqual(data["stages"]["phase"]["failed_calls"], 1)
        self.assertGreaterEqual(data["stages"]["phase"]["wall_seconds"], 0)


class WarmWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.script = Path(self.temporary.name) / "worker.py"
        self.script.write_text(
            "import json, os, sys, time\n"
            "print('STUDIO_WORKER {\"ready\": true}', flush=True)\n"
            "count = 0\n"
            "for line in sys.stdin:\n"
            "    request = json.loads(line)\n"
            "    count += 1\n"
            "    if request['arguments'] == ['hang']:\n"
            "        print('fixture waiting',flush=True)\n"
            "        time.sleep(60)\n"
            "    if request['arguments'] == ['fail']:\n"
            "        print('STUDIO_WORKER ' + json.dumps({'id':request['id'],'ok':False,'error':'fixture failed'}),flush=True)\n"
            "        break\n"
            "    print('processing fixture',flush=True)\n"
            "    print('STUDIO_WORKER ' + json.dumps({'id':request['id'],'ok':True,'pid':os.getpid(),'count':count}),flush=True)\n",
            encoding="utf-8",
        )
        self.worker = WarmInferenceProcess(idle_seconds=60)
        self.addCleanup(self.worker.close)

    def test_reuses_process_for_successive_requests_and_restarts_for_runtime_changes(self):
        logs = []
        env = os.environ.copy()
        with patch("seedvr_studio.inference_worker.RUNNER", self.script), patch("seedvr_studio.inference_worker.VENV_PYTHON", Path(sys.executable)):
            first = self.worker.run(["one"], env, logs.append)
            second = self.worker.run(["two"], env, logs.append)
            third = self.worker.run(["three"], {**env, "SEEDVR2_TRT_ENCODER_LEGACY": "1"}, logs.append)
        self.assertEqual(first["pid"], second["pid"])
        self.assertEqual(second["count"], 2)
        self.assertNotEqual(second["pid"], third["pid"])
        self.assertEqual(third["count"], 1)
        self.assertIn("processing fixture", logs)

    def test_failed_generation_retires_worker_before_next_request(self):
        with patch("seedvr_studio.inference_worker.RUNNER", self.script), patch("seedvr_studio.inference_worker.VENV_PYTHON", Path(sys.executable)):
            first = self.worker.run(["one"], os.environ.copy(), lambda line: None)
            with self.assertRaisesRegex(RuntimeError, "fixture failed"):
                self.worker.run(["fail"], os.environ.copy(), lambda line: None)
            recovered = self.worker.run(["again"], os.environ.copy(), lambda line: None)
        self.assertNotEqual(first["pid"], recovered["pid"])
        self.assertEqual(recovered["count"], 1)

    def test_idle_cache_is_released_and_the_next_request_starts_fresh(self):
        import psutil

        self.worker._idle_seconds = 0.1
        with patch("seedvr_studio.inference_worker.RUNNER", self.script), patch("seedvr_studio.inference_worker.VENV_PYTHON", Path(sys.executable)):
            first = self.worker.run(["one"], os.environ.copy(), lambda line: None)
            try:
                psutil.Process(first["pid"]).wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
            second = self.worker.run(["two"], os.environ.copy(), lambda line: None)
        self.assertNotEqual(first["pid"], second["pid"])
        self.assertEqual(second["count"], 1)

    def test_close_interrupts_active_work_instead_of_waiting_for_generation(self):
        ready = threading.Event()
        failures = []

        def run():
            try:
                self.worker.run(["hang"], os.environ.copy(), lambda line: ready.set() if line == "fixture waiting" else None)
            except Exception as exc:  # noqa: BLE001 -- capture the worker-thread exception for assertion below
                failures.append(exc)

        with patch("seedvr_studio.inference_worker.RUNNER", self.script), patch("seedvr_studio.inference_worker.VENV_PYTHON", Path(sys.executable)):
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            self.assertTrue(ready.wait(timeout=5))
            self.worker.close()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], RenderCancelledError)


class AttentionReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vendor = str(Path(__file__).resolve().parents[1] / "vendor" / "seedvr2")
        with patch.object(sys, "path", [vendor, *sys.path]):
            from src.optimization import compatibility
        cls.compatibility = compatibility

    def test_imported_blackwell_and_hopper_backends_are_not_selected_on_ada(self):
        compatibility = self.compatibility
        with (
            patch.object(compatibility.torch.cuda, "is_available", return_value=True),
            patch.object(compatibility.torch.cuda, "get_device_capability", return_value=(8, 9)),
            patch.object(compatibility, "SAGE_ATTN_3_AVAILABLE", True),
            patch.object(compatibility, "SAGE_ATTN_2_AVAILABLE", False),
            patch.object(compatibility, "FLASH_ATTN_3_AVAILABLE", True),
            patch.object(compatibility, "FLASH_ATTN_2_AVAILABLE", True),
        ):
            self.assertEqual(compatibility.validate_attention_mode("sageattn_3"), "flash_attn_2")
            self.assertEqual(compatibility.validate_attention_mode("flash_attn_3"), "flash_attn_2")

    def test_runtime_varlen_fallback_counts_sage2_not_sage3(self):
        compatibility = self.compatibility
        query = torch.zeros(4, 2, 8, dtype=torch.float16)
        lengths = torch.tensor([0, 1, 4], dtype=torch.int32)
        with (
            patch.dict(os.environ, {"SEEDVR2_PROFILE_PATH": "fixture"}),
            patch.dict(compatibility.ATTENTION_BACKENDS_USED, {}, clear=True),
            patch.object(compatibility, "SAGE_ATTN_3_AVAILABLE", True),
            patch.object(compatibility, "SAGE_ATTN_2_AVAILABLE", True),
            patch.object(compatibility, "sageattn_varlen", side_effect=lambda q, *args: q.clone()),
        ):
            compatibility.call_sage_attn_3_varlen(query, query, query, lengths, lengths, 3, 3)
            self.assertEqual(compatibility.ATTENTION_BACKENDS_USED, {"sageattn_2": 1})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_installed_sage2_kernel_dispatch_is_recorded_on_gpu(self):
        compatibility = self.compatibility
        if not compatibility.SAGE_ATTN_2_AVAILABLE:
            self.skipTest("SageAttention 2 is not installed")
        query = torch.randn(32, 4, 64, device="cuda", dtype=torch.float16)
        lengths = torch.tensor([0, 16, 32], device="cuda", dtype=torch.int32)
        with patch.dict(os.environ, {"SEEDVR2_PROFILE_PATH": "fixture"}), patch.dict(compatibility.ATTENTION_BACKENDS_USED, {}, clear=True):
            actual = compatibility.call_sage_attn_2_varlen(query, query, query, lengths, lengths, 16, 16)
            self.assertTrue(torch.isfinite(actual).all().item())
            self.assertEqual(actual.shape, query.shape)
            self.assertEqual(compatibility.ATTENTION_BACKENDS_USED, {"sageattn_2": 1})


if __name__ == "__main__":
    unittest.main()
