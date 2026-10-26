"""Image workflow contracts without model downloads, GUI dialogs or GPU inference."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import psutil
from PIL import Image

from seedvr_studio.gpu_jobs import active_gpu_owner, cancel_gpu_owner, reserve_gpu
from seedvr_studio.image_batches import ImageBatches, _output_name
from seedvr_studio.image_inference import ImageRenderer, ImageSettings, decode_image
from seedvr_studio.media import MediaError, RenderCancelledError
from seedvr_studio.paths import VENV_PYTHON


class FakeRenderer:
    def __init__(self):
        self.calls = []
        self.closed = 0
        self.fail_next = False

    def render(self, source, output, settings, cancelled, report):
        self.calls.append(source.read_bytes())
        if self.fail_next:
            self.fail_next = False
            raise MediaError("Deliberate fixture failure")
        image = decode_image(source.read_bytes())
        image.resize(settings.target_size(*image.size)).save(output, format="PNG")
        report(0.9, "Fixture image finished")
        return {"engine": "seedvr", "temporal_size": 1}

    def close(self):
        self.closed += 1


class ImageTestCase(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(tempfile.gettempdir()) / "opencode"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="image-tests-", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.renderer = FakeRenderer()
        self.manager = ImageBatches(self.root / "state", self.root / "default", renderer_factory=lambda: self.renderer)
        self.addCleanup(self.manager.shutdown)

    def image(self, name="source.png", *, mode="RGB", size=(17, 11), color=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new(mode, size, color if color is not None else (30, 90, 150, 110) if mode == "RGBA" else 100 if mode == "L" else (30, 90, 150))
        image.save(path)
        return path

    def review(self, paths=(), **kwargs):
        return self.manager.review(paths=[str(path) for path in paths], folders=[], settings={"scale": 1.0, "max_edge": 64}, **kwargs)

    def finish(self, record):
        self.manager.threads[record["id"]].join(timeout=10)
        self.assertFalse(self.manager.threads[record["id"]].is_alive())
        return self.manager.get(record["id"])


class SelectionTests(ImageTestCase):
    def test_mixed_sources_deduplicate_paths_not_names_or_contents(self):
        first, second = self.image("Desktop/a/photo.png"), self.image("Downloads/b/photo.png")
        record = self.review([first, second, first])
        self.assertEqual(len(record["items"]), 2)
        self.assertEqual(record["duplicates"], 1)
        names = [item["output_name"] for item in record["items"]]
        self.assertEqual(len(set(name.casefold() for name in names)), 2)
        self.assertIn("photo.png_upscaled.png", names)
        self.assertNotIn("fingerprint", record["items"][0])

    def test_folder_is_flat_naturally_sorted_and_skips_other_files(self):
        self.image("folder/image10.jpg")
        self.image("folder/image2.png")
        self.image("folder/nested/hidden.png")
        (self.root / "folder/readme.txt").write_text("not an image")
        record = self.manager.review(paths=[], folders=[str(self.root / "folder")])
        self.assertEqual([item["name"] for item in record["items"]], ["image2.png", "image10.jpg"])
        self.assertEqual({entry["reason"] for entry in record["skipped"]}, {"subfolder", "unsupported file"})

    def test_same_stem_different_extensions_are_distinct(self):
        record = self.review([self.image("photo.jpg"), self.image("photo.png")])
        self.assertEqual({item["output_name"] for item in record["items"]}, {"photo.jpg_upscaled.png", "photo.png_upscaled.png"})

    def test_bad_image_does_not_discard_healthy_files(self):
        bad = self.root / "broken.jpg"
        bad.write_bytes(b"not an image")
        record = self.review([bad, self.image()])
        result = self.finish(self.manager.start(record["id"]))
        self.assertEqual(result["status"], "partial")
        self.assertEqual({item["status"] for item in result["items"]}, {"complete", "error"})

    def test_snapshot_does_not_add_new_files_and_detects_changed_bytes(self):
        path = self.image()
        record = self.review([path])
        self.image("new.png")
        self.image(color=(90, 10, 20))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.manager.source_preview(record["id"], record["items"][0]["id"], review=True)
        result = self.finish(self.manager.start(record["id"]))
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["status"], "error")
        self.assertIn("changed since review", result["items"][0]["error"])
        self.assertFalse(self.renderer.calls)

    def test_long_reserved_and_case_equivalent_output_names_are_safe(self):
        used = set()
        first = _output_name("Photo.JPG", "123456789012", self.root, used)
        second = _output_name("photo.jpg", "abcdefghijkl", self.root, used)
        self.assertNotEqual(first.casefold(), second.casefold())
        long = _output_name("x" * 300 + ".png", "123456789012", self.root, used)
        self.assertLessEqual(len(str(self.root / long).encode("utf-16-le")) // 2, 246)
        self.assertTrue(_output_name("CON.jpg", "123456789012", self.root, used).startswith("_CON"))

    def test_relative_and_missing_custom_output_roots_require_explicit_creation(self):
        path = self.image()
        with self.assertRaisesRegex(ValueError, "absolute"):
            self.review([path], output_root="relative")
        missing = self.root / "custom"
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.review([path], output_root=str(missing))
        record = self.review([path], output_root=str(missing), create_output_root=True)
        self.assertFalse(missing.exists())  # Review itself does not write there.
        result = self.finish(self.manager.start(record["id"]))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(Path(result["output_dir"]).parent, missing)

    def test_quoted_windows_output_path_from_explorer_is_accepted(self):
        self.assertEqual(self.review([self.image()], output_root=f'"{self.root}"')["output_root"], str(self.root))

    def test_thousand_image_review_is_metadata_not_loaded_canvases(self):
        source = self.image()
        data = source.read_bytes()
        folder = self.root / "many"; folder.mkdir()
        for index in range(1001):
            (folder / f"image{index}.png").write_bytes(data)
        record = self.manager.review(paths=[], folders=[str(folder)])
        self.assertEqual(len(record["items"]), 1001)
        self.assertEqual(record["items"][2]["name"], "image2.png")
        self.assertFalse(self.renderer.calls)
        self.assertFalse(Path(record["output_dir"]).exists())


class ProcessingTests(ImageTestCase):
    def test_custom_outputs_preserve_original_and_survive_new_manager(self):
        source = self.image("Desktop/photo.png", mode="RGBA")
        before = source.read_bytes()
        output_root = self.root / "Downloads/results"
        output_root.mkdir(parents=True)
        record = self.review([source], output_root=str(output_root))
        result = self.finish(self.manager.start(record["id"]))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(before, source.read_bytes())
        item = result["items"][0]
        output = self.manager.file(result["id"], item["id"])
        self.assertEqual(output.parent.parent, output_root)
        with Image.open(output) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.size, (17, 11))
            self.assertEqual(image.getchannel("A").getextrema(), (110, 110))
        (self.root / "state/nonobject.json").write_text("[]", encoding="utf-8")
        (self.root / "state/invaliditems.json").write_text(json.dumps({"version": 1, "id": "invaliditems", "items": None}), encoding="utf-8")
        restored = ImageBatches(self.root / "state", self.root / "default")
        self.assertEqual(restored.file(result["id"], item["id"]), output)
        self.assertEqual(restored.history()[0]["status"], "complete")
        self.assertFalse(list(output.parent.glob(".image-work-*")))

    def test_output_root_can_be_source_parent_without_reprocessing_results(self):
        source = self.image("Desktop/source.png")
        record = self.review([source], output_root=str(source.parent))
        result = self.finish(self.manager.start(record["id"]))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(self.renderer.calls), 1)
        self.assertTrue(source.is_file())

    def test_conflicting_target_is_never_overwritten(self):
        record = self.review([self.image()])
        with patch.object(self.manager, "_launch"):
            batch = self.manager.start(record["id"])
        target = Path(batch["output_dir"]) / batch["items"][0]["output_name"]
        target.write_bytes(b"keep this existing file")
        self.manager._run(batch["id"])
        self.assertEqual(target.read_bytes(), b"keep this existing file")
        self.assertEqual(self.manager.get(batch["id"])["items"][0]["status"], "error")

    def test_failure_continues_and_retry_skips_completed_outputs(self):
        self.renderer.fail_next = True
        record = self.review([self.image("a.png"), self.image("b.png")])
        result = self.finish(self.manager.start(record["id"]))
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(self.renderer.calls), 2)
        completed = next(item for item in result["items"] if item["status"] == "complete")
        output = self.manager.file(result["id"], completed["id"])
        before = output.stat().st_mtime_ns
        result = self.finish(self.manager.retry(result["id"]))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(self.renderer.calls), 3)
        self.assertEqual(output.stat().st_mtime_ns, before)

    def test_missing_completed_output_can_be_explicitly_retried(self):
        result = self.finish(self.manager.start(self.review([self.image()])["id"]))
        self.manager.file(result["id"], result["items"][0]["id"]).unlink()
        result = self.finish(self.manager.retry(result["id"]))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(self.renderer.calls), 2)

    def test_changed_completed_output_and_unavailable_root_are_not_replaced(self):
        result = self.finish(self.manager.start(self.review([self.image()])["id"]))
        output = self.manager.file(result["id"], result["items"][0]["id"])
        output.write_bytes(b"user changed this result")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.manager.retry(result["id"])
        self.assertEqual(output.read_bytes(), b"user changed this result")
        root = output.parent; moved = root.with_name(root.name + "-moved")
        root.rename(moved)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.manager.retry(result["id"])
        self.assertFalse(root.exists())

    def test_driver_failure_stops_batch_instead_of_reusing_unsafe_gpu(self):
        self.renderer.render = lambda *args: (_ for _ in ()).throw(MediaError("CUDA illegal memory access"))
        result = self.finish(self.manager.start(self.review([self.image("a.png"), self.image("b.png")])["id"]))
        self.assertEqual(result["status"], "error")
        self.assertIn("Unsafe GPU", result["message"])
        self.assertEqual(result["items"][1]["status"], "pending")

    def test_cancelling_queued_image_does_not_cancel_other_gpu_owner(self):
        record = self.review([self.image()])
        with reserve_gpu("video:other", threading.Event()):
            self.assertEqual(active_gpu_owner(), "video:other")
            queued = self.manager.start(record["id"])
            self.manager.cancel(queued["id"])
            result = self.finish(queued)
            self.assertEqual(result["status"], "cancelled")
            self.assertEqual(active_gpu_owner(), "video:other")
            self.assertFalse(self.renderer.calls)
            with patch("seedvr_studio.cancellation.cancel_current_render") as stop:
                self.assertIsNone(cancel_gpu_owner("video:different", stop))
                stop.assert_not_called()

    def test_cancel_in_flight_keeps_completed_and_retries_remaining(self):
        started = threading.Event()
        normal = self.renderer.render
        def render(source, output, settings, cancelled, report):
            if self.renderer.calls:
                started.set()
                if not cancelled.wait(timeout=5):
                    raise AssertionError("Cancellation was not delivered")
                raise RenderCancelledError("fixture cancelled")
            return normal(source, output, settings, cancelled, report)
        self.renderer.render = render
        queued = self.manager.start(self.review([self.image("a.png"), self.image("b.png")])["id"])
        self.assertTrue(started.wait(timeout=5))
        self.manager.cancel(queued["id"])
        result = self.finish(queued)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual([item["status"] for item in result["items"]], ["complete", "cancelled"])
        self.renderer.render = normal
        result = self.finish(self.manager.retry(result["id"]))
        self.assertEqual(result["status"], "complete")

    def test_preview_does_not_consume_full_review_or_custom_destination(self):
        custom = self.root / "custom"
        custom.mkdir()
        review = self.review([self.image("a.png"), self.image("b.png")], output_root=str(custom))
        preview = self.finish(self.manager.start(review["id"], preview_item=review["items"][0]["id"]))
        self.assertEqual(preview["kind"], "preview")
        self.assertEqual(Path(preview["output_dir"]).parent, self.manager.default_output_root)
        self.assertFalse(list(custom.iterdir()))
        full = self.finish(self.manager.start(review["id"]))
        self.assertEqual(len(full["items"]), 2)
        self.assertEqual(Path(full["output_dir"]).parent, custom)

    def test_interrupted_history_never_auto_starts(self):
        review = self.review([self.image()])
        with patch.object(self.manager, "_launch"):
            queued = self.manager.start(review["id"])
        restored = ImageBatches(self.root / "state", self.root / "default", renderer_factory=lambda: self.renderer)
        self.assertEqual(restored.get(queued["id"])["status"], "interrupted")
        self.assertFalse(restored.threads)


class ImageMediaTests(ImageTestCase):
    def setUp(self):
        super().setUp()
        models = self.root / "models"
        models.mkdir()
        for name in ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors", "encoder.plan", "decoder.plan"):
            (models / name).touch()
        for name, value in (("MODELS", models), ("TRT_ENCODER_5F", models / "encoder.plan"), ("TRT_ENGINE_5F", models / "decoder.plan")):
            change = patch(f"seedvr_studio.image_inference.{name}", value)
            change.start()
            self.addCleanup(change.stop)

    def test_orientation_grayscale_and_palette_alpha_normalization(self):
        source = self.root / "oriented.jpg"
        exif = Image.Exif(); exif[274] = 6
        Image.new("RGB", (17, 11)).save(source, exif=exif)
        self.assertEqual(decode_image(source.read_bytes()).size, (11, 17))
        gray = self.image("gray.png", mode="L")
        self.assertEqual(decode_image(gray.read_bytes()).mode, "RGB")
        palette = Image.new("P", (8, 8)); palette.info["transparency"] = 0
        data = io.BytesIO(); palette.save(data, format="PNG")
        self.assertEqual(decode_image(data.getvalue()).mode, "RGBA")

    def test_high_bit_depth_is_rejected_without_silent_quantization(self):
        data = io.BytesIO(); Image.new("I;16", (8, 8)).save(data, format="PNG")
        with self.assertRaisesRegex(ValueError, "bit-depth"):
            decode_image(data.getvalue())

    def test_settings_validate_and_size_preserves_aspect_and_odd_axes(self):
        self.assertEqual(ImageSettings(scale=2, max_edge=64).target_size(17, 11), (34, 22))
        self.assertEqual(ImageSettings(scale=4, max_edge=64).target_size(40, 20), (64, 32))
        for values in ({"engine": "unknown"}, {"max_edge": -1}, {"seed": True}, {"scale": 3}, {"vae_tiling": "false"}, {"blocks_to_swap": 33}):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                ImageSettings(**values)

    def test_finishing_preserves_odd_dimensions_and_original_alpha(self):
        source = self.image(mode="RGBA")
        renderer = ImageRenderer()
        def rgb(stage, raw, settings, cancelled, report):
            with Image.open(stage) as image:
                self.assertEqual(image.size, (18, 12))
                image.save(raw, format="PNG")
            return {"engine": "seedvr"}
        output = self.root / "finished.png"
        with patch.object(renderer, "_render_rgb", side_effect=rgb):
            renderer.render(source, output, ImageSettings(scale=1), threading.Event(), lambda *args: None)
        with Image.open(output) as image:
            self.assertEqual(image.size, (17, 11))
            self.assertEqual(image.getchannel("A").getextrema(), (110, 110))

    def test_automatic_fallback_is_visible_and_not_retried_per_file(self):
        renderer = ImageRenderer()
        calls = []
        def run(source, output, settings, engine, cancelled, report):
            calls.append(engine)
            if engine == "tensorrt":
                raise MediaError("engine fixture failed")
            return {"engine": engine}
        with patch.object(renderer, "_run", side_effect=run):
            first = renderer._render_rgb(None, None, ImageSettings(), threading.Event(), lambda *args: None)
            second = renderer._render_rgb(None, None, ImageSettings(), threading.Event(), lambda *args: None)
        self.assertEqual(calls, ["tensorrt", "seedvr", "seedvr"])
        self.assertIn("fixture failed", first["fallback_reason"])
        self.assertEqual(first["fallback_reason"], second["fallback_reason"])

    def test_cancellation_never_launches_a_fallback(self):
        renderer = ImageRenderer()
        with patch.object(renderer, "_run", side_effect=RenderCancelledError("stopped")) as run:
            with self.assertRaises(RenderCancelledError):
                renderer._render_rgb(None, None, ImageSettings(), threading.Event(), lambda *args: None)
        run.assert_called_once()

    def test_unsafe_driver_error_never_launches_a_fallback(self):
        renderer = ImageRenderer()
        with patch.object(renderer, "_run", side_effect=MediaError("CUDA illegal memory access")) as run:
            with self.assertRaises(MediaError):
                renderer._render_rgb(None, None, ImageSettings(), threading.Event(), lambda *args: None)
        run.assert_called_once()


class ImageApiTests(ImageTestCase):
    def setUp(self):
        super().setUp()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from seedvr_studio import image_api
        self.api = image_api
        self.addCleanup(patch.stopall)
        patch.object(image_api, "BATCHES", self.manager).start()
        app = FastAPI(); app.include_router(image_api.router)
        self.client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))
        self.addCleanup(self.client.close)

    def test_full_registered_custom_root_workflow(self):
        source = self.image("Desktop/photo.png", mode="RGBA")
        custom = self.root / "external"; custom.mkdir()
        response = self.client.post("/api/images/review", json={"paths": [str(source)], "output_root": str(custom), "settings": {"scale": 1}})
        self.assertEqual(response.status_code, 200, response.text)
        review = response.json()
        self.assertEqual(self.client.get(review["items"][0]["source_url"]).status_code, 200)
        response = self.client.post("/api/images/batches", json={"review_id": review["id"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.finish(response.json())
        result = self.client.get(f"/api/images/batches/{review['id']}").json()
        output = self.client.get(result["items"][0]["output_url"])
        self.assertEqual(output.status_code, 200)
        self.assertEqual(Image.open(io.BytesIO(output.content)).mode, "RGBA")
        history = self.client.get("/api/images/batches").json()
        self.assertEqual(history[0]["completed"], 1)
        self.assertEqual(Path(history[0]["output_dir"]).parent, custom)
        self.assertEqual(self.client.get("/api/images/batches/missing/items/private/output").status_code, 404)

    def test_foreign_origin_host_and_network_clients_cannot_access_local_files(self):
        for headers in ({"Origin": "https://evil.example"}, {"Origin": "null"}, {"Origin": "http://["}, {"Sec-Fetch-Site": "cross-site"}, {"Host": "evil.example"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.client.post("/api/images/pick", json={"kind": "files"}, headers=headers).status_code, 403)
        from fastapi.testclient import TestClient
        with TestClient(self.client.app, base_url="http://127.0.0.1", client=("192.0.2.10", 50000)) as remote:
            self.assertEqual(remote.get("/api/images/config").status_code, 403)

    def test_invalid_settings_and_unknown_registered_ids_are_rejected(self):
        response = self.client.post("/api/images/review", json={"paths": [str(self.image())], "settings": {"scale": 3}})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.post("/api/images/batches", json={"review_id": "missing"}).status_code, 404)
        self.assertEqual(self.client.post("/api/images/batches/missing/retry", json={}).status_code, 404)


class VideoOwnershipTests(ImageTestCase):
    def test_video_cancellation_is_still_checked_after_global_flag_is_reset(self):
        import api_server as api
        cancelled = threading.Event()
        cancelled.set()
        with patch.dict(api.JOBS, {"fixture": {"_cancel_event": cancelled}}, clear=True):
            with self.assertRaises(RenderCancelledError):
                api._check_video_cancelled("fixture")

    def test_cancelling_queued_video_does_not_stop_running_images(self):
        import api_server as api
        from fastapi.testclient import TestClient
        with patch.dict(api.JOBS, {"queued": {"status": "queued"}}, clear=True), patch.object(api, "cancel_current_render") as stop:
            with reserve_gpu("images:other", threading.Event()):
                action = unittest.mock.Mock()
                thread = threading.Thread(target=api._owned_video_job, args=("queued", action))
                thread.start()
                with TestClient(api.app) as client:
                    response = client.post("/api/jobs/queued/cancel")
                self.assertEqual(response.status_code, 200)
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
                stop.assert_not_called()
                action.assert_not_called()
                self.assertEqual(active_gpu_owner(), "images:other")
            self.assertEqual(api.JOBS["queued"]["status"], "cancelled")

    def test_only_active_video_owner_invokes_process_cancellation(self):
        import api_server as api
        from fastapi.testclient import TestClient
        with patch.dict(api.JOBS, {"active": {"status": "running"}, "other": {"status": "queued"}}, clear=True):
            with reserve_gpu("video:active", threading.Event()), patch.object(api, "cancel_current_render", return_value="Stopped owned render") as stop:
                with TestClient(api.app) as client:
                    self.assertEqual(client.post("/api/jobs/active/cancel").status_code, 200)
                stop.assert_called_once()
                self.assertEqual(api.JOBS["other"]["status"], "queued")


@unittest.skipUnless(sys.platform == "win32", "Windows venv redirector regression")
class OwnedWorkerProcessTests(unittest.TestCase):
    def test_close_terminates_real_python_child_not_only_venv_launcher(self):
        renderer = ImageRenderer()
        renderer.process = subprocess.Popen(
            [str(VENV_PYTHON), "-u", "-c", "import os,time; print(os.getpid(),flush=True); time.sleep(30)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        worker = psutil.Process(int(renderer.process.stdout.readline().strip()))
        try:
            started = time.monotonic()
            renderer.close()
            self.assertLess(time.monotonic() - started, 3, "Stop waited for inference to finish instead of terminating it")
            self.assertFalse(worker.is_running(), "The real Python worker survived cancellation of its venv launcher")
        finally:
            if worker.is_running():
                worker.kill()
                worker.wait(timeout=5)
            renderer.close()


if __name__ == "__main__":
    unittest.main()
