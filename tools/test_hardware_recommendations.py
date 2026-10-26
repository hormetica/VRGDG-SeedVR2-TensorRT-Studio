"""Hardware recommendations without model loading, CUDA allocations or downloads."""

from __future__ import annotations

import copy
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from seedvr_studio.hardware import GIB, HARDWARE_PRESETS, HardwareInfo, detect_hardware, recommend_settings


def hardware(total=24, free=None, **overrides):
    return HardwareInfo(**{
        "gpu_name": "Fixture RTX", "total_vram_bytes": int(total * GIB),
        "free_vram_bytes": int((total - 0.5 if free is None else free) * GIB),
        "compute_capability": (8, 9), "attention_backends": ("sageattn_2", "sdpa"),
        "total_ram_bytes": 32 * GIB, "available_ram_bytes": 16 * GIB, **overrides,
    })


def recommend(info=None, **overrides):
    return recommend_settings(info or hardware(), **{
        "backend_name": "SeedVR2 + TensorRT", "output_preset": "1K / 1080p", **overrides,
    })


class RecommendationTests(unittest.TestCase):
    def test_memory_tiers_follow_existing_balanced_presets(self):
        for total, batch, blocks, fp16 in ((8, 5, 36, False), (12, 5, 24, False), (16, 5, 12, False),
                                           (24, 21, 0, False), (32, 21, 0, True)):
            with self.subTest(total=total):
                result = recommend(hardware(total))
                self.assertTrue(result["available"])
                settings = result["settings"]
                self.assertEqual(settings["batch_size"], batch)
                self.assertEqual(settings["blocks"], blocks)
                self.assertEqual("FP16" in settings["model"], fp16)

    def test_driver_reported_24_gb_is_not_mistaken_for_a_smaller_card(self):
        result = recommend(hardware(total=24563 / 1024, free=22.4))
        self.assertEqual(result["settings"]["batch_size"], 21)

    def test_busy_gpu_gets_a_smaller_profile_not_the_total_memory_profile(self):
        busy = recommend(hardware(32, free=11))
        self.assertEqual(busy["settings"]["batch_size"], 5)
        self.assertEqual(busy["settings"]["blocks"], 24)
        self.assertIn("FP8", busy["settings"]["model"])

    def test_unreadable_free_memory_uses_a_conservative_estimate(self):
        result = recommend(replace(hardware(32), free_vram_bytes=None))
        self.assertTrue(result["available"])
        self.assertIn("FP8", result["settings"]["model"])
        self.assertTrue(any("could not be read" in note for note in result["notes"]))

    def test_unsupported_or_exhausted_gpu_does_not_propose_settings(self):
        cases = (HardwareInfo(error="CUDA unavailable"), hardware(6), hardware(24, free=4))
        for info in cases:
            with self.subTest(info=info):
                result = recommend(info)
                self.assertFalse(result["available"])
                self.assertIsNone(result["settings"])
                self.assertTrue(result["message"])

    def test_larger_outputs_reduce_memory_pressure_without_changing_output(self):
        standard = recommend(hardware(32))
        large = recommend(hardware(32), output_preset="2K / 1440p")
        uhd = recommend(hardware(32), output_preset="4K / 2160p")
        self.assertEqual(standard["settings"]["batch_size"], 21)
        self.assertEqual(large["settings"]["batch_size"], 5)
        self.assertGreater(large["settings"]["blocks"], standard["settings"]["blocks"])
        self.assertIn("FP8", uhd["settings"]["model"])
        self.assertTrue(uhd["settings"]["vae_tiling"])
        self.assertGreater(uhd["settings"]["blocks"], large["settings"]["blocks"])

    def test_original_size_requires_source_geometry_and_uses_its_pixel_count(self):
        unloaded = recommend(output_preset="Original / enhancement only")
        self.assertFalse(unloaded["available"])
        self.assertIn("Load a source", unloaded["message"])
        for width, height in ((3840, 2160), (2160, 3840)):
            with self.subTest(width=width):
                result = recommend(output_preset="Original / enhancement only", source_width=width, source_height=height)
                self.assertEqual(result["settings"]["batch_size"], 5)
                self.assertTrue(result["settings"]["vae_tiling"])

    def test_legacy_uses_vae_tiling_and_supported_temporal_batches(self):
        for total in (8, 12, 16, 24, 32, 80):
            with self.subTest(total=total):
                settings = recommend(hardware(total), backend_name="SeedVR2 (Legacy)")["settings"]
                self.assertTrue(settings["vae_tiling"])
                self.assertIn(settings["batch_size"], (5, 21))

    def test_sdpa_and_low_system_ram_have_explicit_memory_cautions(self):
        result = recommend(hardware(attention_backends=("sdpa",), available_ram_bytes=2 * GIB))
        self.assertEqual(result["settings"]["attention"], "sdpa")
        self.assertEqual(result["settings"]["batch_size"], 5)
        self.assertTrue(result["settings"]["vae_tiling"])
        self.assertTrue(any("system RAM" in note for note in result["notes"]))

    def test_recommendations_do_not_change_creative_export_or_beta_settings(self):
        original = copy.deepcopy(HARDWARE_PRESETS)
        for output in ("1K / 1080p", "2K / 1440p", "4K / 2160p"):
            for total in (8, 12, 16, 24, 32, 80):
                result = recommend(hardware(total), output_preset=output)
                settings = result["settings"]
                self.assertEqual(set(settings), {"model", "batch_size", "attention", "blocks", "vae_tiling"})
                self.assertIn(settings["batch_size"], (5, 21))
                self.assertTrue(0 <= settings["blocks"] <= 36)
        self.assertEqual(HARDWARE_PRESETS, original)

    def test_unknown_engine_or_output_is_rejected(self):
        for arguments in ({"backend_name": "unknown"}, {"output_preset": "unknown"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                recommend(**arguments)


class DetectionTests(unittest.TestCase):
    def probe(self, *, major=8, cuda_available=True, packages=("sageattention",), free_error=False):
        cuda = SimpleNamespace(
            is_available=Mock(return_value=cuda_available),
            get_device_properties=Mock(return_value=SimpleNamespace(name="Visible GPU zero", total_memory=24 * GIB, major=major, minor=0)),
            mem_get_info=Mock(side_effect=RuntimeError("driver cannot report memory") if free_error else None,
                              return_value=(22 * GIB, 24 * GIB)),
        )
        with (
            patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda),
                                     "psutil": SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=32 * GIB, available=16 * GIB))}),
            patch("seedvr_studio.hardware._package_present", side_effect=lambda name: name in packages),
        ):
            return detect_hardware(), cuda

    def test_inspects_only_logical_gpu_zero_not_largest_physical_gpu(self):
        result, cuda = self.probe()
        cuda.get_device_properties.assert_called_once_with(0)
        cuda.mem_get_info.assert_called_once_with(0)
        self.assertEqual(result.device, "cuda:0")
        self.assertEqual(result.free_vram_bytes, 22 * GIB)
        self.assertEqual(result.available_ram_bytes, 16 * GIB)

    def test_sage2_is_preferred_over_flash2_but_not_assumed_installed(self):
        for packages, expected in (((), "sdpa"), (("flash_attn",), "flash_attn_2"),
                                    (("sageattention", "flash_attn"), "sageattn_2")):
            with self.subTest(packages=packages):
                result, _ = self.probe(packages=packages)
                self.assertEqual(result.attention_backends[0], expected)

    def test_older_gpu_does_not_get_ampere_only_attention(self):
        result, _ = self.probe(major=7, packages=("sageattention", "flash_attn"))
        self.assertEqual(result.attention_backends, ("sdpa",))

    def test_unavailable_cuda_and_driver_failure_are_reported_without_throwing(self):
        result, cuda = self.probe(cuda_available=False)
        self.assertIn("CUDA is unavailable", result.error)
        cuda.get_device_properties.assert_not_called()
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=Mock(side_effect=RuntimeError("driver failed"))))}):
            self.assertIn("driver failed", detect_hardware().error)

    def test_free_memory_failure_does_not_discard_gpu_information(self):
        result, _ = self.probe(free_error=True)
        self.assertIsNone(result.free_vram_bytes)
        self.assertIsNone(result.error)
        self.assertEqual(result.gpu_name, "Visible GPU zero")


class HardwareApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient

            import api_server
        except ImportError as exc:
            raise unittest.SkipTest(f"Studio browser dependencies required: {exc}") from exc
        cls.api = api_server
        cls.TestClient = TestClient

    def test_inspection_is_read_only_and_advertised_by_config(self):
        with (
            patch.object(self.api, "detect_hardware", return_value=hardware()),
            patch.object(self.api, "backend_status", return_value=(True, "ready")),
            patch.object(self.api, "tensorrt_status", return_value=(True, "ready")),
            patch.object(self.api, "render") as render,
            patch.dict(self.api.JOBS, {}, clear=True), self.TestClient(self.api.app) as client,
        ):
            response = client.get("/api/hardware/recommendation")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["available"])
            self.assertTrue(client.get("/api/config").json()["features"]["hardware_recommendations"])
            self.assertEqual(self.api.JOBS, {})
            render.assert_not_called()

    def test_api_validation_and_missing_cuda_leave_settings_unavailable(self):
        with (
            patch.object(self.api, "detect_hardware", return_value=HardwareInfo(error="CUDA unavailable")),
            patch.object(self.api, "tensorrt_status", return_value=(True, "ready")),
            self.TestClient(self.api.app) as client,
        ):
            self.assertFalse(client.get("/api/hardware/recommendation").json()["available"])
            for arguments in ({"backend_name": "invalid"}, {"output_preset": "invalid"}):
                self.assertEqual(client.get("/api/hardware/recommendation", params=arguments).status_code, 400)
            self.assertEqual(client.get("/api/hardware/recommendation", params={"source_width": -1}).status_code, 422)

    def test_original_geometry_and_missing_tensorrt_are_reflected_in_response(self):
        with (
            patch.object(self.api, "detect_hardware", return_value=hardware()),
            patch.object(self.api, "tensorrt_status", return_value=(False, "not installed")),
            self.TestClient(self.api.app) as client,
        ):
            result = client.get("/api/hardware/recommendation", params={
                "output_preset": "Original / enhancement only", "source_width": 3840, "source_height": 2160,
            }).json()
            self.assertEqual(result["settings"]["batch_size"], 5)
            self.assertTrue(any("Complete installation" in note for note in result["notes"]))


if __name__ == "__main__":
    unittest.main()
