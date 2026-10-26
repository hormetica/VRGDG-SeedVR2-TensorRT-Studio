"""Exercise real TensorRT/CUDA scheduling using a tiny engine, without model weights."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.run_tensorrt_persistent import PersistentDecoder


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("tensorrt_rtx"), "CUDA and TensorRT RTX required")
class PersistentDecoderGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.engine_path = Path(cls.temporary.name) / "tiny.rtxplan"
        cls.build_engine(cls.engine_path)

    @staticmethod
    def build_engine(path, frames=2, tile=2):
        import tensorrt_rtx as trt

        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        source = network.add_input("latent", trt.float16, (1, 3, frames, tile, tile))
        resize = network.add_resize(source)
        resize.shape = (1, 3, frames * 4 - 3, tile * 8, tile * 8)
        resize.resize_mode = trt.InterpolationMode.NEAREST
        resize.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
        output = resize.get_output(0)
        output.name = "video"
        network.mark_output(output)
        serialized = builder.build_serialized_network(network, builder.create_builder_config())
        if serialized is None:
            raise RuntimeError("Could not build tiny TensorRT decoder test engine")
        path.write_bytes(bytes(serialized))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_tiling_matches_full_image_for_small_and_rectangular_canvases(self):
        decoder = PersistentDecoder(self.engine_path, 3, 2, 2)
        for height, width in ((1, 1), (1, 3), (2, 2), (3, 5), (5, 3)):
            with self.subTest(height=height, width=width):
                source = torch.linspace(-1.5, 1.5, 3 * 2 * height * width).reshape(1, 3, 2, height, width).half()
                actual = decoder.decode_tensor(source, 1).cpu()
                expected = F.interpolate(source.float(), size=(5, height * 8, width * 8), mode="nearest")
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                self.assertEqual(decoder._weights.numel(), height * width * 64)

    def test_fixed_addresses_and_independently_owned_results_across_queued_batches(self):
        decoder = PersistentDecoder(self.engine_path, 3, 2, 2)
        addresses = (decoder.tile_input.data_ptr(), decoder.tile_output.data_ptr())
        first = decoder.decode_tensor(torch.full((1, 3, 2, 5, 5), 0.25), 1)
        second = decoder.decode_tensor(torch.full((1, 3, 2, 5, 5), 0.75), 1)
        self.assertEqual(addresses, (decoder.tile_input.data_ptr(), decoder.tile_output.data_ptr()))
        torch.testing.assert_close(first.cpu(), torch.full_like(first.cpu(), 0.25), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(second.cpu(), torch.full_like(second.cpu(), 0.75), rtol=1e-6, atol=1e-6)

    def test_gpu_inputs_produced_on_nondefault_stream_are_ordered(self):
        decoder = PersistentDecoder(self.engine_path, 3, 2, 2)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            source = torch.ones((1, 3, 2, 3, 3), device="cuda", dtype=torch.float16) * 0.625
            result = decoder.decode_tensor(source, 1)
            actual = result.cpu()
        torch.testing.assert_close(actual, torch.full_like(actual, 0.625), rtol=1e-6, atol=1e-6)

    def test_twenty_one_frame_profile_uses_one_spatial_weight_plane(self):
        path = Path(self.temporary.name) / "tiny-21f.rtxplan"
        self.build_engine(path, frames=6, tile=32)
        decoder = PersistentDecoder(path, 3, 6, 32)
        result = decoder.decode_tensor(torch.full((1, 3, 6, 64, 96), 0.25).half(), 12)
        self.assertEqual(result.shape, (1, 3, 21, 512, 768))
        self.assertEqual(result.numel() // decoder._weights.numel(), 63)
        low, high = result.aminmax()
        self.assertAlmostEqual(low.item(), 0.25, places=6)
        self.assertAlmostEqual(high.item(), 0.25, places=6)

    def test_combined_worker_keeps_raw_checkpoints_and_finishes_on_gpu(self):
        from tools.test_video_finishing import find_ffmpeg
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            self.skipTest("FFmpeg required")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                            "sine=frequency=440:duration=1", str(source)], check=True)
            jobs = []
            for index, (value, keep) in enumerate(((0.25, 5), (0.75, 2))):
                latent = root / f"latent-{index}.pt"
                torch.save({"latent": torch.full((1, 3, 2, 3, 5), value).half(), "original_frames": keep}, latent)
                jobs.append({"latent": str(latent), "output": str(root / f"raw-{index}.pt")})
            output = root / "restored.mp4"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "engine": str(self.engine_path), "tile_latent": 2, "overlap_latent": 1, "jobs": jobs,
                "finishing": {"output": str(output), "audio": str(source), "fps": 29.97,
                              "seam_mode": "match", "seam_frames": 2, "target_width": 40, "target_height": 24,
                              "effects": {"sharpen_strength": 0.2, "grain_intensity": 0.02},
                              "export": {"preset": "software_h264"}},
            }))
            result = subprocess.run([
                sys.executable, str(Path(__file__).with_name("run_tensorrt_persistent.py")), str(manifest),
            ], env={**os.environ, "SEEDVR_FFMPEG": ffmpeg, "PYTHONUTF8": "1"}, text=True,
                capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(output.exists())
            for job, value, keep in zip(jobs, (0.25, 0.75), (5, 2)):
                raw = torch.load(job["output"], weights_only=False)
                torch.testing.assert_close(raw["video"], torch.full((1, 3, 5, 24, 40), value), rtol=1e-6, atol=1e-6)
                self.assertEqual(raw["original_frames"], keep)
            profile = json.loads(output.with_suffix(".gpu-finishing-profile.json").read_text())
            self.assertTrue(profile["metadata"]["gpu_resident"])
            self.assertEqual(profile["metadata"]["frames"], 7)
            self.assertEqual(profile["stages"]["vae_decode"]["calls"], 2)
            self.assertEqual(profile["stages"]["effects"]["calls"], 2)
            self.assertFalse(list(root.glob("*.tmp")))
            self.assertFalse(list(root.glob("*.gpu-pending.mp4")))


if __name__ == "__main__":
    unittest.main()
