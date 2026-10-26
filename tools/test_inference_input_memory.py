"""Exercise the real video reader under deterministic, simulated RAM pressure."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


class FrameCapture:
    def __init__(self, cli, frames):
        self.cli = cli
        self.frames = frames
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {self.cli.cv2.CAP_PROP_FPS: 60, self.cli.cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
                self.cli.cv2.CAP_PROP_FRAME_WIDTH: self.frames.shape[2],
                self.cli.cv2.CAP_PROP_FRAME_HEIGHT: self.frames.shape[1]}[prop]

    def set(self, prop, value):
        self.position = int(value)
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


class InferenceInputMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vendor = str(Path(__file__).resolve().parents[1] / "vendor" / "seedvr2")
        with patch.object(sys, "path", [vendor, *sys.path]):
            import inference_cli
        cls.cli = inference_cli

    def check_input_budget(self, batch_size, frame_count, runner_cache=None):
        cli = self.cli
        buffers = []
        processed = []
        cvt_color = cli.cv2.cvtColor

        class BudgetedRGB(np.ndarray):
            def astype(self, dtype, *args, **kwargs):
                if np.dtype(dtype) == np.dtype("float32"):
                    buffers[:] = [buffer for buffer in buffers if buffer() is not None]
                    if len(buffers) >= batch_size:
                        raise np._core._exceptions._ArrayMemoryError(self.shape, np.dtype(dtype))
                return super().astype(dtype, *args, **kwargs)

            def __truediv__(self, value):
                result = super().__truediv__(value).view(np.ndarray)
                buffers.append(weakref.ref(result))
                return result

        class Capture:
            position = 0

            def isOpened(self):
                return True

            def get(self, prop):
                return {cli.cv2.CAP_PROP_FPS: 60, cli.cv2.CAP_PROP_FRAME_COUNT: frame_count,
                        cli.cv2.CAP_PROP_FRAME_WIDTH: 4, cli.cv2.CAP_PROP_FRAME_HEIGHT: 2}[prop]

            def read(self):
                if self.position >= frame_count:
                    return False, None
                frame = np.full((2, 4, 3), self.position % 256, dtype=np.uint8)
                self.position += 1
                return True, frame

            def set(self, prop, value):
                self.position = int(value)
                return True

            def release(self):
                pass

        def process(frames_tensor, **kwargs):
            from src.core.generation_phases import _prepare_video_batch

            for start in range(0, len(frames_tensor), batch_size):
                batch = _prepare_video_batch(frames_tensor, start, min(start + batch_size, len(frames_tensor)))
                processed.extend((batch[:, 0, 0, 0] * 255).round().int().tolist())
            return torch.empty((0, 0, 0, 0))

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.touch()
            args = cli.parse_arguments([str(source), "--stop_before_vae", "--batch_size", str(batch_size),
                                        "--output_format", "mp4"])
            with (
                patch.object(cli.cv2, "VideoCapture", return_value=Capture()),
                patch.object(cli.cv2, "cvtColor", side_effect=lambda *a: cvt_color(*a).view(BudgetedRGB)),
                patch.object(cli, "_process_frames_core", side_effect=process),
                patch.object(cli, "clear_memory"),
            ):
                cli.process_single_file(str(source), args, ["0"], str(Path(directory) / "output.mp4"),
                                        runner_cache=runner_cache)
        self.assertEqual(processed, [index % 256 for index in range(frame_count)],
                         "Latent capture must process every frame, not just the first chunk")

    def test_latent_capture_does_not_buffer_the_entire_video_as_float32(self):
        self.check_input_budget(5, 6)

    def test_original_frame_count_fits_one_batch_budget_in_cold_and_warm_paths(self):
        for batch_size in (5, 21):
            for warm in (False, True):
                with self.subTest(batch_size=batch_size, warm=warm):
                    self.check_input_budget(batch_size, 17413, runner_cache={} if warm else None)

    def test_source_slices_match_eager_rgb_fp16_conversion(self):
        cli = self.cli
        frames = np.arange(9 * 2 * 4 * 3, dtype=np.uint8).reshape(9, 2, 4, 3)
        cap = FrameCapture(cli, frames)
        cap.set(cli.cv2.CAP_PROP_POS_FRAMES, 2)
        source = cli._VideoFrameSource(cap, 6, 2, 4, start_frame=2)
        expected = torch.from_numpy(frames[2:8, :, :, ::-1].astype(np.float32) / 255.0).to(torch.float16)
        self.assertEqual(source.shape, tuple(expected.shape))
        for index in (0, slice(0, 5), slice(3, 6), -1, slice(None, 2), slice(4, 4)):
            with self.subTest(index=index):
                torch.testing.assert_close(source[index], expected[index], rtol=0, atol=0)
        with self.assertRaises(IndexError):
            source[6]
        with self.assertRaises(ValueError):
            source[::2]

    def test_source_reports_early_eof_and_seek_failures(self):
        cli = self.cli
        cap = FrameCapture(cli, np.zeros((2, 2, 4, 3), dtype=np.uint8))
        source = cli._VideoFrameSource(cap, 4, 2, 4)
        with self.assertRaisesRegex(ValueError, "ended before frame 4"):
            source[:4]
        with patch.object(cap, "set", return_value=False), self.assertRaisesRegex(ValueError, "Cannot seek"):
            source[1]

    def test_generation_failure_releases_the_capture(self):
        cli = self.cli
        cap = FrameCapture(cli, np.zeros((6, 2, 4, 3), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.touch()
            args = cli.parse_arguments([str(source), "--stop_before_vae", "--output_format", "mp4"])
            with (
                patch.object(cli.cv2, "VideoCapture", return_value=cap),
                patch.object(cli, "_process_frames_core", side_effect=RuntimeError("fixture failed")),
                self.assertRaisesRegex(RuntimeError, "fixture failed"),
            ):
                cli.process_single_file(str(source), args, ["0"], str(Path(directory) / "output.mp4"))
        self.assertTrue(cap.released)

    def test_real_encoding_and_latent_capture_match_eager_input_with_overlap_and_padding(self):
        cli = self.cli
        from src.core import generation_phases as phases

        frames = np.arange(15 * 16 * 16 * 3, dtype=np.uint8).reshape(15, 16, 16, 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.touch()
            for overlap in (0, 1):
                reference = None
                for warm, chunk_size in ((False, 11), (False, 0), (True, 0)):
                    with self.subTest(overlap=overlap, warm=warm, chunk_size=chunk_size):
                        capture_dir = root / f"latents-{overlap}-{warm}-{chunk_size}"
                        ctx = cli.setup_generation_context(dit_device="cpu", vae_device="cpu")
                        encoded = []

                        def encode(videos):
                            encoded.append(videos[0].clone())
                            return [torch.full((2, 2, 2, 16), len(encoded) / 10, dtype=torch.float16)]

                        def upscale(runner, ctx, **kwargs):
                            ctx["all_upscaled_latents"] = ctx["all_latents"]
                            return ctx

                        vae_config = SimpleNamespace(scaling_factor=0.5, get=lambda key, default: default)
                        runner = SimpleNamespace(vae=torch.nn.Linear(1, 1), tile_debug=None, encode_tiled=False,
                                                 vae_encode=encode, config=SimpleNamespace(vae=vae_config))
                        cap = FrameCapture(cli, frames)
                        args = cli.parse_arguments([
                            str(source), "--stop_before_vae", "--output_format", "mp4", "--batch_size", "5",
                            "--uniform_batch_size", "--skip_first_frames", "2", "--load_cap", "11",
                            "--resolution", "16", "--max_resolution", "16", "--color_correction", "none",
                            "--chunk_size", str(chunk_size), "--temporal_overlap", str(overlap),
                        ])
                        with (
                            patch.dict(os.environ, {"SEEDVR2_PROFILE_PATH": "", "SEEDVR2_TRT_ENCODER": "0",
                                                    "SEEDVR2_LATENT_CAPTURE_DIR": str(capture_dir)}),
                            patch.object(cli.cv2, "VideoCapture", return_value=cap),
                            patch.object(cli, "setup_generation_context", return_value=ctx),
                            patch.object(cli, "prepare_runner", return_value=(runner, {"vae_cache": False})),
                            patch.object(cli, "load_text_embeddings", return_value={}),
                            patch.object(cli, "upscale_all_batches", side_effect=upscale),
                            patch.object(phases, "ensure_precision_initialized"),
                            patch.object(phases, "manage_model_device"),
                            patch.object(cli, "_read_frames_from_cap", wraps=cli._read_frames_from_cap) as read,
                        ):
                            cli.process_single_file(str(source), args, ["0"], str(root / "output.mp4"),
                                                    runner_cache={} if warm else None)
                        self.assertTrue(cap.released)
                        payloads = [torch.load(path, weights_only=False)
                                    for path in sorted(capture_dir.glob("vae_latent_*.pt"))]
                        self.assertEqual([item["batch_index"] for item in payloads], [1, 2, 3])
                        self.assertEqual([item["original_frames"] for item in payloads],
                                         [5, 5, 1] if overlap == 0 else [5, 5, 3])
                        if chunk_size == 0:
                            self.assertLessEqual(max(call.args[1] for call in read.call_args_list), args.batch_size)
                        if reference is None:
                            reference = encoded, payloads
                        else:
                            for actual, expected in zip(encoded, reference[0], strict=True):
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            for actual, expected in zip(payloads, reference[1], strict=True):
                                torch.testing.assert_close(actual["latent"], expected["latent"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
