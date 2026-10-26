"""Host routing/recovery and browser-facing settings, without model downloads."""

from __future__ import annotations

import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from seedvr_studio import backend, persistent_decoder, tensorrt_pipeline
from seedvr_studio.media import MediaError, RenderCancelledError, VideoInfo


def settings(**overrides):
    values = {"resolution": 8, "max_resolution": 16, "batch_size": 5, "seed": 42,
              "model_label": "3B FP16 — best 3B quality", "color_correction": "none", "attention_mode": "sdpa",
              "blocks_to_swap": 0, "vae_tiling": False, "stop_before_vae": False,
              "sharpen_enabled": False, "sharpen_strength": 0.25, "grain_enabled": False,
              "grain_intensity": 0.02, "grain_saturation": 0.5, "decoder_mode": "optimized_fast"}
    return backend.RenderSettings(**{**values, **overrides})


class RenderRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.capture = self.root / "vae_latents"
        self.capture.mkdir()
        torch.save({"latent": torch.zeros(1, 16, 2, 2, 2)}, self.capture / "vae_latent_001.pt")
        self.output = self.root / "restored.mp4"

    def test_gpu_pipeline_failure_uses_stable_decoder_and_same_export_policy(self):
        def stable(profiles, env, progress):
            outputs = []
            for profile in profiles:
                path = Path(profile["output"])
                torch.save({"video": torch.zeros(1, 3, 5, 8, 16)}, path)
                outputs.append(path)
            return outputs

        with (
            patch.object(tensorrt_pipeline, "probe", return_value=VideoInfo(1, 16, 8, 24, 24)),
            patch.object(tensorrt_pipeline, "_optimized_decode", side_effect=RuntimeError("GPU fixture failed")),
            patch.object(tensorrt_pipeline, "_stable_decode", side_effect=stable) as fallback,
            patch.object(tensorrt_pipeline, "cancellation_requested", return_value=False),
            patch.object(tensorrt_pipeline, "finish_decoded_video", side_effect=lambda *args, **kwargs: self.output.touch()) as finish,
        ):
            result = tensorrt_pipeline.decode_postprocess_and_assemble(
                self.output, self.root / "source.mp4", self.capture,
                settings(gpu_finishing=True, scene_cut_detection=True, export_preset="hevc_nvenc_10bit"), {}, None,
            )
        self.assertEqual(result, self.output)
        fallback.assert_called_once()
        self.assertEqual(finish.call_args.kwargs["export"].bit_depth, 10)
        self.assertTrue(finish.call_args.kwargs["scene_cut_detection"])
        record = json.loads((self.root / "decoder-manifest.json").read_text())
        self.assertEqual(record["used"], "stable-fallback")
        self.assertEqual(record["fallback_reason"], "GPU fixture failed")

    def test_successful_gpu_pipeline_does_not_launch_a_second_finishing_process(self):
        def optimized(*args):
            self.output.touch()
            return [self.root / "raw.pt"]

        with (
            patch.object(tensorrt_pipeline, "probe", return_value=VideoInfo(1, 16, 8, 24, 24)),
            patch.object(tensorrt_pipeline, "_optimized_decode", side_effect=optimized) as decode,
            patch.object(tensorrt_pipeline, "finish_decoded_video") as finish,
        ):
            tensorrt_pipeline.decode_postprocess_and_assemble(
                self.output, self.root / "source.mp4", self.capture, settings(gpu_finishing=True), {}, None,
            )
        finish.assert_not_called()
        self.assertEqual(decode.call_args.args[4]["output"], str(self.output))

    def test_cancelled_gpu_pipeline_does_not_retry_saved_latents(self):
        with (
            patch.object(tensorrt_pipeline, "probe", return_value=VideoInfo(1, 16, 8, 24, 24)),
            patch.object(tensorrt_pipeline, "_optimized_decode", side_effect=RuntimeError("stopped")),
            patch.object(tensorrt_pipeline, "cancellation_requested", return_value=True),
            patch.object(tensorrt_pipeline, "_stable_decode") as fallback,
            self.assertRaises(RenderCancelledError),
        ):
            tensorrt_pipeline.decode_postprocess_and_assemble(
                self.output, self.root / "source.mp4", self.capture, settings(gpu_finishing=True), {}, None,
            )
        fallback.assert_not_called()

    def test_cancelled_warm_inference_does_not_start_a_cold_retry(self):
        for flag, error in ((True, MediaError("stopped")), (False, RenderCancelledError("stopped"))):
            with (
                self.subTest(flag=flag),
                patch.object(backend, "backend_status", return_value=(True, "ready")),
                patch.object(backend, "tensorrt_status", return_value=(True, "ready")),
                patch("seedvr_studio.inference_worker.WARM_INFERENCE.run", side_effect=error),
                patch("seedvr_studio.cancellation.cancellation_requested", return_value=flag),
                patch.object(backend.subprocess, "Popen") as launch,
                self.assertRaises(RenderCancelledError),
            ):
                backend.render("SeedVR2 + TensorRT", self.root / "source.mp4", self.output, settings(warm_worker=True))
            launch.assert_not_called()

    def test_combined_progress_is_monotonic(self):
        class Worker:
            stdout = io.StringIO("\n".join(
                line for index in range(1, 4)
                for line in (f"PERSISTENT_PROGRESS {index}/3", f"ASSEMBLY_PROGRESS {index}/3")
            ))

            def wait(self):
                return 0

        updates = []
        with patch.object(persistent_decoder.subprocess, "Popen", return_value=Worker()):
            persistent_decoder.run_persistent_decoder(self.root / "manifest.json", 3, {},
                                                      lambda value, message: updates.append(value), combined=True)
        self.assertEqual(updates, sorted(updates))
        self.assertLessEqual(updates[-1], 1)

    def test_post_only_reprocessing_keeps_numeric_order_past_batch_999(self):
        decoded = self.root / "tensorrt_decoded"
        decoded.mkdir()
        for index in (1, 999, 1000, 1001):
            (decoded / f"decoded_{index:03d}.pt").touch()
        configuration = settings()
        arguments = {name: getattr(configuration, name) for name in inspect.signature(backend.reprocess_tensorrt).parameters
                     if hasattr(configuration, name)}
        with (
            patch.object(backend, "probe", return_value=VideoInfo(1, 16, 8, 24, 24)),
            patch.object(backend, "finish_decoded_video") as finish,
        ):
            backend.reprocess_tensorrt(self.root, self.root / "source.mp4", self.output, **arguments)
        self.assertEqual([path.name for path in finish.call_args.args[0]],
                         ["decoded_001.pt", "decoded_999.pt", "decoded_1000.pt", "decoded_1001.pt"])


class BrowserAndChunkContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient

            import api_server
        except ImportError as exc:
            raise unittest.SkipTest(f"Studio browser dependencies required: {exc}") from exc
        cls.api = api_server
        cls.TestClient = TestClient

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_new_browser_settings_reach_render_settings_and_invalid_exports_are_rejected(self):
        api = self.api
        captured = []

        def render(engine, source, output, configuration, report):
            captured.append(configuration)
            output.touch()

        class DeferredThread:
            def __init__(self, **kwargs):
                pass

            def start(self):
                pass

        with (
            patch.object(api, "OUTPUTS", self.root), patch.object(api, "ensure_workspace"),
            patch.object(api, "Thread", DeferredThread), patch.dict(api.JOBS, {}, clear=True),
            patch.object(api, "probe", return_value=VideoInfo(1, 16, 8, 24, 24)),
            patch.object(api, "render", side_effect=render), self.TestClient(api.app) as client,
        ):
            response = client.post("/api/jobs", files={"file": ("source.mp4", b"fixture", "video/mp4")}, data={
                "job_type": "full", "output_preset": "Original / enhancement only",
                "gpu_finishing": "true", "warm_worker": "true", "scene_cut_detection": "true",
                "export_preset": "hevc_nvenc_10bit",
            })
            self.assertEqual(response.status_code, 200, response.text)
            job_id = response.json()["id"]
            job = api.JOBS[job_id]
            api._render_job(job_id, Path(job["_source"]), Path(job["_job_dir"]), job["_values"])
            self.assertEqual(api.JOBS[job_id]["status"], "complete", api.JOBS[job_id])
            self.assertTrue(captured[0].gpu_finishing)
            self.assertTrue(captured[0].warm_worker)
            self.assertTrue(captured[0].scene_cut_detection)
            self.assertEqual(captured[0].export_preset, "hevc_nvenc_10bit")
            bad = client.post("/api/jobs", files={"file": ("source.mp4", b"fixture")}, data={"export_preset": "invalid"})
            self.assertEqual(bad.status_code, 400)

    def test_profile_endpoint_returns_only_this_jobs_reports(self):
        api = self.api
        report_dir = self.root / "job"
        report_dir.mkdir()
        (report_dir / "restored.finishing-profile.json").write_text('{"version":1}')
        (report_dir / "broken.inference-profile.json").write_text("incomplete")
        (self.root / "other.finishing-profile.json").write_text('{"private":true}')
        with patch.dict(api.JOBS, {"fixture": {"_job_dir": str(report_dir)}}, clear=True), self.TestClient(api.app) as client:
            response = client.get("/api/jobs/fixture/profile")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["reports"]), 1)
            self.assertEqual(client.get("/api/jobs/missing/profile").status_code, 404)

    def test_cancelled_job_remains_cancelled_after_global_flag_has_been_cleared(self):
        api = self.api
        source = self.root / "source.mp4"
        source.touch()
        values = {**settings().__dict__, "backend_name": "SeedVR2 + TensorRT", "job_type": "full",
                  "output_preset": "Original / enhancement only"}
        with (
            patch.dict(api.JOBS, {}, clear=True), patch.object(api, "OUTPUTS", self.root),
            patch.object(api, "probe", return_value=VideoInfo(1, 16, 8, 24, 24)),
            patch.object(api, "render", side_effect=RenderCancelledError("Render cancelled")),
            patch.object(api, "cancellation_requested", return_value=False),
        ):
            api._render_job("fixture", source, self.root, values)
            self.assertEqual(api.JOBS["fixture"]["status"], "cancelled")

    def test_chunk_artifacts_are_isolated_and_resume_keeps_ten_bit_export(self):
        api = self.api
        configuration = settings(export_preset="hevc_nvenc_10bit")
        info = VideoInfo(61, 16, 8, 24, 1464)
        values = {"backend_name": "SeedVR2 + TensorRT", "chunk_seconds": 30}
        output = self.root / "result.mp4"
        rendered_to = []

        def write_clip(source, target, *args):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"clip")
            return target

        def render(engine, source, target, configuration, report):
            rendered_to.append(target)
            target.write_bytes(b"restored")

        def trim(source, target, start, *, export):
            self.assertEqual(export.bit_depth, 10)
            shutil.copy2(source, target)
            return target

        def concat(chunks, target, *, export):
            self.assertEqual(export.bit_depth, 10)
            self.assertEqual(len(chunks), 3)
            target.touch()
            return target

        with (
            patch.object(api, "make_clip", side_effect=write_clip), patch.object(backend, "render", side_effect=render) as worker,
            patch.object(api, "trim_video_start", side_effect=trim), patch.object(api, "concat_videos", side_effect=concat),
            patch.object(api, "_update"),
        ):
            api._render_chunked("fixture", self.root / "source.mp4", self.root, output, configuration, info, values, lambda *args: None)
            self.assertEqual(len({path.parent for path in rendered_to}), 3)
            worker.reset_mock()
            api._render_chunked("fixture", self.root / "source.mp4", self.root, output, configuration, info, values, lambda *args: None)
            worker.assert_not_called()


class InferenceStageReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vendor = str(Path(__file__).resolve().parents[1] / "vendor" / "seedvr2")
        try:
            with patch.object(sys, "path", [vendor, *sys.path]):
                import inference_cli
        except ImportError as exc:
            raise unittest.SkipTest(f"SeedVR runtime dependencies required: {exc}") from exc
        cls.cli = inference_cli

    def run_core(self, profile, failure=False):
        cli = self.cli
        args = cli.parse_arguments(["fixture.mp4", "--stop_before_vae", "--attention_mode", "sageattn_2"])
        ctx = {"dit_device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "compute_dtype": torch.float16}

        def upscale(*args, **kwargs):
            from src.optimization.compatibility import record_attention_backend
            record_attention_backend("sdpa")
            if failure:
                raise ValueError("fixture inference failed")
            return ctx

        with (
            patch.dict("os.environ", {"SEEDVR2_PROFILE_PATH": str(profile)}),
            patch.object(cli, "setup_generation_context", return_value=ctx),
            patch.object(cli, "prepare_runner", return_value=(object(), {})),
            patch.object(cli, "load_text_embeddings", return_value={}),
            patch.object(cli, "compute_generation_info", side_effect=lambda **kwargs: (kwargs["images"], {})),
            patch.object(cli, "log_generation_start"), patch.object(cli, "encode_all_batches", return_value=ctx),
            patch.object(cli, "upscale_all_batches", side_effect=upscale), patch.object(cli, "_capture_pending_vae_latents"),
        ):
            cli._process_frames_core(torch.zeros(5, 8, 8, 3), args, "0", cli.debug)

    def test_latent_capture_reports_each_inference_stage_and_actual_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference-profile.json"
            self.run_core(path)
            report = json.loads(path.read_text())
        self.assertEqual(report["metadata"]["attention_requested"], "sageattn_2")
        self.assertEqual(report["metadata"]["attention_calls"], {"sdpa": 1})
        for stage in ("model_setup", "vae_encoding", "dit_restoration", "latent_capture", "inference_total"):
            self.assertEqual(report["stages"][stage]["calls"], 1)
        self.assertNotIn("vae_decoding", report["stages"])

    def test_failed_inference_still_writes_a_diagnostic_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference-profile.json"
            with self.assertRaisesRegex(ValueError, "fixture inference failed"):
                self.run_core(path, failure=True)
            report = json.loads(path.read_text())
        self.assertEqual(report["metadata"]["error"], "fixture inference failed")
        self.assertEqual(report["stages"]["dit_restoration"]["failed_calls"], 1)


class InferenceImportSafetyTests(unittest.TestCase):
    def test_import_after_torch_preserves_the_already_selected_cuda_allocator(self):
        environment = os.environ.copy()
        environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        environment.pop("PYTORCH_ALLOC_CONF", None)
        script = (
            "import os, sys, torch; sys.path.insert(0, 'vendor/seedvr2'); "
            "import inference_cli; "
            "assert 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ; print('ALLOCATOR_IMPORT_OK')"
        )
        result = subprocess.run([sys.executable, "-X", "utf8", "-X", "faulthandler", "-c", script],
                                cwd=Path(__file__).resolve().parents[1], env=environment,
                                capture_output=True, text=True, check=False, timeout=60)
        if "ModuleNotFoundError" in result.stderr:
            self.skipTest("SeedVR runtime dependencies required")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ALLOCATOR_IMPORT_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
