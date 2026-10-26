"""Single-owner, opt-in warm inference process; models park in CPU RAM when idle."""

from __future__ import annotations

import atexit
import json
import queue
import subprocess
import threading
import time
from uuid import uuid4

from .media import MediaError, RenderCancelledError
from .paths import ROOT, VENV_PYTHON

RUNNER = ROOT / "tools" / "run_seedvr_worker.py"
PROTOCOL = "STUDIO_WORKER "


class WarmInferenceProcess:
    def __init__(self, idle_seconds: float = 300):
        self._lock = threading.RLock()
        self._process = None
        self._lines = None
        self._reader_stop = threading.Event()
        self._closing = threading.Event()
        self._timer = None
        self._generation = 0
        self._idle_seconds = idle_seconds
        self._environment_key = None

    def _start(self, env):
        self._reader_stop = threading.Event()
        self._lines = queue.Queue(maxsize=256)
        self._process = subprocess.Popen(
            [str(VENV_PYTHON), "-u", str(RUNNER)], cwd=ROOT, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        process, lines, stop = self._process, self._lines, self._reader_stop

        def enqueue(value):
            while not stop.is_set():
                try:
                    lines.put(value, timeout=0.25)
                    return
                except queue.Full:
                    continue

        def read():
            try:
                for line in process.stdout:
                    enqueue(line.rstrip())
            finally:
                process.stdout.close()
                enqueue(None)

        threading.Thread(target=read, name="warm-inference-log", daemon=True).start()

    def _next(self, deadline=None):
        while True:
            if self._closing.is_set():
                raise RenderCancelledError("Render cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                raise MediaError("Warm inference worker startup timed out")
            try:
                line = self._lines.get(timeout=0.25)
            except queue.Empty:
                if self._process.poll() is not None:
                    raise MediaError("Warm inference worker exited unexpectedly")
                continue
            if line is None:
                raise MediaError("Warm inference worker closed its output")
            return line

    def _close_locked(self):
        self._generation += 1
        if self._timer:
            self._timer.cancel()
            self._timer = None
        process, self._process = self._process, None
        self._reader_stop.set()
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdin is not None:
                process.stdin.close()

    def close(self):
        # Do not wait for a generation while holding an application shutdown
        # hostage. Interrupt our owned process first, then acquire its run lock.
        self._closing.set()
        try:
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()
            with self._lock:
                self._close_locked()
        finally:
            self._closing.clear()

    def _expire(self, generation):
        with self._lock:
            if self._closing.is_set():
                return
            if generation == self._generation:
                self._close_locked()

    def run(self, arguments: list[str], env: dict[str, str], on_log) -> dict:
        with self._lock:
            if self._closing.is_set():
                raise RenderCancelledError("Render cancelled")
            self._generation += 1
            if self._timer:
                self._timer.cancel()
            key = tuple(env.get(name) for name in (
                "SEEDVR2_TRT_ENCODER", "SEEDVR2_TRT_ENCODER_FAST", "SEEDVR2_TRT_ENCODER_LEGACY", "CUDA_VISIBLE_DEVICES",
            ))
            try:
                if self._process is None or self._process.poll() is not None or key != self._environment_key:
                    self._close_locked()
                    self._start(env)
                    self._environment_key = key
                    deadline = time.monotonic() + 120
                    while True:
                        line = self._next(deadline)
                        if line == PROTOCOL + '{"ready": true}':
                            break
                        on_log(line)
                request_id = uuid4().hex
                request = {"id": request_id, "arguments": arguments,
                           "environment": {name: env[name] for name in (
                               "SEEDVR2_LATENT_CAPTURE_DIR", "SEEDVR2_PROFILE_PATH",
                           ) if name in env}}
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
                while True:
                    line = self._next()
                    if line.startswith(PROTOCOL):
                        response = json.loads(line[len(PROTOCOL):])
                        if response.get("id") != request_id:
                            raise MediaError("Warm inference worker returned a mismatched request ID")
                        if not response.get("ok"):
                            raise MediaError(response.get("error", "Warm inference failed"))
                        self._timer = threading.Timer(self._idle_seconds, self._expire, args=(self._generation,))
                        self._timer.daemon = True
                        self._timer.start()
                        return response
                    on_log(line)
            except BaseException as exc:
                self._close_locked()
                if self._closing.is_set() and isinstance(exc, Exception):
                    raise RenderCancelledError("Render cancelled") from exc
                raise


WARM_INFERENCE = WarmInferenceProcess()
atexit.register(WARM_INFERENCE.close)
