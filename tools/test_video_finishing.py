"""CPU regression tests; FFmpeg integration also works without CUDA or models."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from seedvr_studio import finishing
from seedvr_studio.encoding import ExportSettings, select_encoder
from seedvr_studio.media import MediaError
from seedvr_studio.profiling import PerformanceReport
from tools import assemble_tensor_video as assembler
from tools.postprocess_tensor_video import apply_postprocessing


def reference_boundary(previous, current, mode, frame_count):
    """Original whole-batch algorithm, retained as a numerical parity oracle."""
    count = min(max(1, frame_count), previous.shape[2], current.shape[2])
    if mode == "off" or count < 1:
        return current
    result = current.float().clone()
    reference = previous[:, :, -count:].float()
    head = result[:, :, :count]
    fade = torch.linspace(1.0, 0.0, count + 1)[:count].view(1, 1, count, 1, 1)
    if mode in {"noise", "match", "blend"}:
        reference_high = reference - assembler._spatial_lowpass(reference)
        head_low = assembler._spatial_lowpass(head)
        head_high = head - head_low
        ref_noise = reference_high.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-5)
        head_noise = head_high.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-5)
        head = head.lerp(head_low + head_high * (ref_noise / head_noise).clamp(0.5, 2.0), fade)
    if mode in {"match", "blend"}:
        ref_mean = reference.mean(dim=(0, 2, 3, 4), keepdim=True)
        ref_std = reference.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-5)
        head_mean = head.mean(dim=(0, 2, 3, 4), keepdim=True)
        head_std = head.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-5)
        head = head.lerp((head - head_mean) * (ref_std / head_std).clamp(0.75, 1.33) + ref_mean, fade)
    if mode == "blend":
        anchor = previous[:, :, -1:].float().expand(-1, -1, count, -1, -1)
        blend = torch.linspace(0.35, 0.0, count + 1)[:count].view(1, 1, count, 1, 1)
        head = head.lerp(anchor, blend)
    result[:, :, :count] = head.clamp(-1, 1)
    return result.to(dtype=current.dtype)


class TensorBatchFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save_batch(self, name, tensor, keep=None, bare=False):
        path = self.root / name
        torch.save(tensor if bare else {"video": tensor, "original_frames": keep}, path)
        return path


class StreamingFinishingTests(TensorBatchFixture, unittest.TestCase):
    def test_all_seam_modes_match_original_whole_video_algorithm(self):
        for dtype in (torch.float16, torch.float32):
            generator = torch.Generator().manual_seed(38)
            tensors = [torch.randn(1, 3, count, 8, 10, generator=generator).to(dtype) for count in (5, 5, 5)]
            inputs = [self.save_batch(f"batch-{i}.pt", tensor, keep) for i, (tensor, keep) in enumerate(zip(tensors, (5, 3, 1)))]
            for mode in ("off", "noise", "match", "blend"):
                for seam_frames in (1, 2, 12):
                    with self.subTest(dtype=dtype, mode=mode, seam_frames=seam_frames):
                        expected = []
                        for tensor, keep in zip(tensors, (5, 3, 1)):
                            tensor = tensor[:, :, :keep]
                            if expected:
                                tensor = reference_boundary(expected[-1], tensor, mode, seam_frames)
                            expected.append(tensor)
                        actual = list(assembler.iter_finished_batches(inputs, seam_mode=mode, seam_frames=seam_frames))
                        for actual_batch, expected_batch in zip(actual, expected):
                            torch.testing.assert_close(actual_batch, expected_batch, rtol=0, atol=0)

    def test_effects_keep_padded_temporal_context_and_use_kept_frame_offsets(self):
        torch.manual_seed(51)
        tensors = [torch.rand(1, 3, 5, 12, 12) * 2 - 1 for _ in range(3)]
        inputs = [self.save_batch(f"effects-{i}.pt", tensor, keep) for i, (tensor, keep) in enumerate(zip(tensors, (2, 3, 1)))]
        effects = {
            "sharpen_strength": 0.2, "grain_intensity": 0.02, "grain_saturation": 0.4,
            "microtexture_strength": 0.3, "skin_evenness": 0.2, "skin_smoothing": 0.2,
            "skin_redness": 0.1, "skin_shine": 0.1, "blemish_mode": "subtle",
            "preserve_marks": True, "seed": 100, "device": "cpu",
        }
        actual = list(assembler.iter_finished_batches(inputs, seam_mode="off", effects=effects))
        for tensor, keep, start, result in zip(tensors, (2, 3, 1), (0, 2, 5), actual):
            expected = apply_postprocessing(tensor, frame_start=start, **effects)[:, :, :keep]
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_grain_is_deterministic_across_batch_partitions(self):
        video = torch.zeros(1, 3, 7, 8, 8)
        options = {"grain_intensity": 0.02, "grain_saturation": 0.5, "seed": 123, "device": "cpu"}
        whole = apply_postprocessing(video, **options)
        split = torch.cat([
            apply_postprocessing(video[:, :, :3], **options),
            apply_postprocessing(video[:, :, 3:], frame_start=3, **options),
        ], dim=2)
        torch.testing.assert_close(split, whole, rtol=0, atol=0)

    def test_grain_matches_original_stacked_noise_algorithm(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=device):
                video = torch.linspace(-1, 1, 1 * 3 * 5 * 8 * 8).reshape(1, 3, 5, 8, 8).to(device)
                x = (video.clamp(-1, 1) + 1.0) * 0.5
                frames = []
                for offset in range(5):
                    generator = torch.Generator(device=device).manual_seed(100 + 3 + offset)
                    noise = torch.randn((1, 1, 8, 8), generator=generator, device=device)
                    colored = noise.repeat(1, 3, 1, 1)
                    colored[:, 0] *= 2.0
                    colored[:, 2] *= 3.0
                    gray = colored[:, 1:2].repeat(1, 3, 1, 1)
                    frames.append(0.4 * colored + 0.6 * gray)
                expected = (x + torch.stack(frames, dim=2) * 0.02).clamp(0, 1)
                expected = (expected * 2.0 - 1.0).half().cpu()
                actual = apply_postprocessing(
                    video, grain_intensity=0.02, grain_saturation=0.4, seed=100, frame_start=3, device=device,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_stream_does_not_retain_prior_batch_storage_or_load_ahead(self):
        references = []

        def load(path):
            self.assertTrue(all(reference() is None for reference in references))
            tensor = torch.zeros(1, 3, 21, 8, 8)
            references.append(weakref.ref(tensor))
            return tensor, 5

        with patch.object(assembler, "load_video_tensor", side_effect=load) as loader:
            batches = assembler.iter_finished_batches([Path(str(i)) for i in range(20)])
            for index in range(20):
                batch = next(batches)
                self.assertEqual(loader.call_count, index + 1)
                self.assertEqual(batch.shape[2], 5)
                del batch
            with self.assertRaises(StopIteration):
                next(batches)
        self.assertTrue(all(reference() is None for reference in references))

    def test_raw_rgb_bytes_match_original_concat_and_crop(self):
        torch.manual_seed(19)
        first = torch.randn(1, 3, 3, 8, 10)
        second = torch.randn(1, 3, 3, 8, 10).half()
        inputs = [self.save_batch("one.pt", first, bare=True), self.save_batch("two.pt", second, 2)]
        matched = reference_boundary(first, second[:, :, :2], "blend", 2)
        frames = torch.cat([first, matched], dim=2)[0].permute(1, 2, 3, 0)
        expected = ((frames.float().clamp(-1, 1) + 1) * 127.5).byte().numpy()[:, :6, :8].tobytes()
        chunks = []
        output = self.root / "result.mp4"
        temp = self.root / "result_noaudio.mp4"

        class Encoder:
            stdin = io.BytesIO()

            def wait(self):
                temp.write_bytes(b"encoded")
                return 0

        encoder = Encoder()
        encoder.stdin.write = lambda value: chunks.append(value)
        with (
            patch.object(assembler.subprocess, "Popen", return_value=encoder) as launch,
            patch.object(torch, "cat", side_effect=AssertionError("Whole-video concatenation is forbidden")),
        ):
            count = assembler.assemble_video(inputs, output, seam_mode="blend", target_width=8, target_height=6)
        self.assertEqual(count, 5)
        self.assertEqual(b"".join(chunks), expected)
        self.assertTrue(encoder.stdin.closed)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(output.read_bytes(), b"encoded")

    def test_invalid_frame_counts_fail_before_encoding(self):
        for keep in (0, -1, 4):
            with self.subTest(keep=keep):
                path = self.save_batch("invalid.pt", torch.zeros(1, 3, 3, 4, 4), keep)
                with patch.object(assembler.subprocess, "Popen") as launch:
                    with self.assertRaisesRegex(ValueError, "original_frames"):
                        assembler.assemble_video([path], self.root / "invalid.mp4")
                    launch.assert_not_called()

    def test_failed_encoder_closes_pipe_waits_and_removes_partial_video(self):
        path = self.save_batch("input.pt", torch.zeros(1, 3, 1, 4, 4))
        output = self.root / "failed.mp4"
        temp = self.root / "failed_noaudio.mp4"
        temp.write_bytes(b"partial")

        class Encoder:
            stdin = io.BytesIO()

            def wait(self):
                self.waited = True
                return 9

        encoder = Encoder()
        encoder.stdin.write = lambda value: (_ for _ in ()).throw(BrokenPipeError())
        with (
            patch.object(assembler.subprocess, "Popen", return_value=encoder),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            assembler.assemble_video([path], output)
        self.assertTrue(encoder.stdin.closed)
        self.assertTrue(encoder.waited)
        self.assertFalse(temp.exists())
        self.assertFalse(output.exists())

    def test_host_uses_one_worker_and_a_manifest_instead_of_long_file_arguments(self):
        output = self.root / "result.mp4"
        inputs = [self.root / f"decoded-{i:05d}.pt" for i in range(2000)]
        updates = []

        class Worker:
            stdout = io.StringIO("ASSEMBLY_PROGRESS 1/2000\nASSEMBLY_PROGRESS 2000/2000\n")

            def wait(self):
                output.write_bytes(b"video")
                return 0

        with patch.object(finishing.subprocess, "Popen", return_value=Worker()) as launch:
            finishing.finish_decoded_video(
                inputs, output, self.root / "source.mp4", {}, fps=29.97,
                seam_mode="match", seam_frames=2,
                effects=finishing.PostprocessSettings(grain_intensity=0.02),
                progress_callback=lambda value, message: updates.append(value),
            )
        launch.assert_called_once()
        command = launch.call_args.args[0]
        self.assertLess(sum(map(len, command)), 2000)
        manifest = json.loads(output.with_suffix(".finishing.json").read_text())
        self.assertEqual(len(manifest["inputs"]), 2000)
        self.assertEqual(manifest["effects"]["grain_intensity"], 0.02)
        self.assertEqual(updates, sorted(updates))
        self.assertEqual(updates[0], 0.0)
        self.assertEqual(updates[-1], 1.0)

    def test_host_surfaces_worker_failure_with_log_tail(self):
        class Worker:
            stdout = io.StringIO("decoder fixture failed\n")

            def wait(self):
                return 3

        with (
            patch.object(finishing.subprocess, "Popen", return_value=Worker()),
            self.assertRaisesRegex(MediaError, "decoder fixture failed"),
        ):
            finishing.finish_decoded_video(
                [self.root / "batch.pt"], self.root / "output.mp4", self.root / "source.mp4", {},
                fps=24, seam_mode="off", seam_frames=2,
            )


def find_ffmpeg():
    executable = os.environ.get("SEEDVR_FFMPEG") or shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


@unittest.skipUnless(find_ffmpeg(), "FFmpeg or imageio-ffmpeg is required")
class FFmpegFinishingTests(TensorBatchFixture, unittest.TestCase):
    def prepare_ffmpeg(self):
        self.ffmpeg = find_ffmpeg()
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        alias = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        try:
            os.link(self.ffmpeg, alias)
        except OSError:
            shutil.copy2(self.ffmpeg, alias)
        self.env = os.environ.copy()
        self.env["PATH"] = str(bin_dir) + os.pathsep + self.env.get("PATH", "")

    def decode_rgb(self, path):
        return subprocess.run([
            self.ffmpeg, "-loglevel", "error", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ], check=True, capture_output=True).stdout

    def test_worker_encodes_trimmed_batches_at_source_fps_with_audio(self):
        self.prepare_ffmpeg()
        source = self.root / "source.wav"
        subprocess.run([
            self.ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=48000:duration=1", str(source),
        ], check=True)
        inputs = [
            self.save_batch("one.pt", torch.zeros(1, 3, 5, 12, 16), 5),
            self.save_batch("two.pt", torch.ones(1, 3, 5, 12, 16) * 0.5, 2),
        ]
        output = self.root / "result.mp4"
        with patch.object(finishing, "VENV_PYTHON", Path(sys.executable)):
            finishing.finish_decoded_video(
                inputs, output, source, self.env, fps=29.97, seam_mode="match", seam_frames=2,
                target_size=(14, 10),
            )
        capture = cv2.VideoCapture(str(output))
        try:
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 7)
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 29.97, places=3)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 14)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 10)
        finally:
            capture.release()
        metadata = subprocess.run([self.ffmpeg, "-i", str(output)], capture_output=True, text=True, check=False).stderr
        self.assertIn("Audio: aac", metadata)
        self.assertFalse(list(self.root.glob("processed_*.pt")))
        self.assertFalse((self.root / "result_noaudio.mp4").exists())

    def test_manifest_effects_match_saved_effects_with_legacy_positional_cli(self):
        self.prepare_ffmpeg()
        torch.manual_seed(73)
        tensors = [torch.rand(1, 3, 5, 12, 16) * 2 - 1 for _ in range(2)]
        inputs = [self.save_batch(f"raw-{i}.pt", tensor, keep) for i, (tensor, keep) in enumerate(zip(tensors, (5, 2)))]
        effects = {"sharpen_strength": 0.2, "grain_intensity": 0.02, "seed": 17, "device": "cpu"}
        processed = [
            self.save_batch(f"processed-{i}.pt", apply_postprocessing(tensor, frame_start=i * 5, **effects), keep)
            for i, (tensor, keep) in enumerate(zip(tensors, (5, 2)))
        ]
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"inputs": list(map(str, inputs)), "effects": effects}))
        streamed = self.root / "streamed.mp4"
        saved = self.root / "saved.mp4"
        for input_arguments, output in ((["--manifest", str(manifest)], streamed), (list(map(str, processed)), saved)):
            subprocess.run([
                sys.executable, str(Path(assembler.__file__)), *input_arguments,
                "--output", str(output), "--fps", "24", "--seam-mode", "blend",
            ], check=True, env=self.env, capture_output=True)
        self.assertEqual(self.decode_rgb(streamed), self.decode_rgb(saved))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU effects parity")
    def test_cuda_manifest_effects_match_standalone_postprocess_cli(self):
        self.prepare_ffmpeg()
        inputs = [self.save_batch(f"raw-{i}.pt", torch.zeros(1, 3, 5, 12, 16), keep) for i, keep in enumerate((5, 2))]
        processed = []
        for index, path in enumerate(inputs):
            output = self.root / f"processed-{index}.pt"
            subprocess.run([
                sys.executable, str(Path(assembler.__file__).with_name("postprocess_tensor_video.py")),
                str(path), "--output", str(output), "--frame-start", str(index * 5),
                "--sharpen-strength", "0.2", "--grain-intensity", "0.02", "--seed", "17",
            ], check=True, env=self.env, capture_output=True)
            processed.append(output)
        manifest = self.root / "gpu-manifest.json"
        manifest.write_text(json.dumps({
            "inputs": list(map(str, inputs)),
            "effects": {"sharpen_strength": 0.2, "grain_intensity": 0.02, "seed": 17},
        }))
        streamed, saved = self.root / "gpu-streamed.mp4", self.root / "gpu-saved.mp4"
        for input_arguments, output in ((["--manifest", str(manifest)], streamed), (list(map(str, processed)), saved)):
            subprocess.run([
                sys.executable, str(Path(assembler.__file__)), *input_arguments, "--output", str(output),
            ], check=True, env=self.env, capture_output=True)
        self.assertEqual(self.decode_rgb(streamed), self.decode_rgb(saved))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for NVENC validation")
    def test_real_nvenc_presets_and_ten_bit_stream_metadata(self):
        self.prepare_ffmpeg()
        ramp = torch.linspace(-1, 1, 1024).view(1, 1, 1, 1, 1024).expand(1, 3, 3, 256, 1024).clone()
        source = self.save_batch("gradient.pt", ramp)
        tested = 0
        for preset, codec, depth in (("h264_nvenc", "h264_nvenc", 8),
                                     ("hevc_nvenc_10bit", "hevc_nvenc", 10),
                                     ("av1_nvenc_10bit", "av1_nvenc", 10)):
            with self.subTest(preset=preset), patch.dict(os.environ, self.env):
                if select_encoder(ExportSettings(preset), 1024, 256).codec != codec:
                    continue  # e.g. AV1 is not available on an RTX 30-series GPU
                tested += 1
                output = self.root / f"{preset}.mp4"
                report = PerformanceReport()
                assembler.assemble_video([source], output, export=ExportSettings(preset, allow_fallback=False), report=report)
                self.assertEqual(report.metadata["encoder"]["codec"], codec)
                metadata = subprocess.run([self.ffmpeg, "-i", str(output)], capture_output=True, text=True, check=False).stderr
                self.assertIn("yuv420p10le" if depth == 10 else "yuv420p", metadata)
                self.assertIn("bt709", metadata)
                if depth == 10:
                    luma = subprocess.run([
                        self.ffmpeg, "-loglevel", "error", "-i", str(output), "-frames:v", "1",
                        "-vf", "extractplanes=y", "-pix_fmt", "gray10le", "-f", "rawvideo", "pipe:1",
                    ], capture_output=True, check=True).stdout
                    self.assertGreater(np.unique(np.frombuffer(luma, dtype="<u2")).size, 256)
        if not tested:
            self.skipTest("No usable NVENC encoder in this FFmpeg/driver environment")


if __name__ == "__main__":
    unittest.main()
