"""Isolate Python 3.11/NumPy 1 inference from Python 3.12/NumPy 2 Reactor."""
import atexit
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time

import numpy as np

from worldcrafter_model import WorldCrafterInput, WorldCrafterResult


class WorldCrafterBackend:
    def __init__(self, settings):
        self._settings = settings
        self._lock = threading.Lock()
        self._process = None
        self._connection = None
        self._stream = None
        self._log = None
        self._temporary = tempfile.TemporaryDirectory(prefix="wc-", dir=settings.scratch)
        self._root = Path(self._temporary.name)
        self._sequence = 0
        self._closed = False
        self._listener = socket.socket(socket.AF_UNIX)
        endpoint = str(self._root / "rpc.sock")
        if len(endpoint.encode()) >= 104:
            raise ValueError("WORLDCRAFTER_SCRATCH path is too long for a Unix socket")
        self._listener.bind(endpoint)
        self._listener.listen(1)
        self._listener.settimeout(1)
        environment = os.environ.copy()
        environment.update(
            PYTHONPATH=str(settings.source), PYTHONUNBUFFERED="1",
            WORLDCRAFTER_DEMO_MODEL=str(settings.weights),
            WORLDCRAFTER_DEMO_DATA=str(self._root / "engine"),
            WORLDCRAFTER_DEMO_BACKEND="context" if settings.gpu_count == 8 else "single",
            WORLDCRAFTER_DEMO_GPUS=str(settings.gpu_count),
            WORLDCRAFTER_DEMO_COMPILE=str(int(settings.compile)),
            WORLDCRAFTER_DEMO_OPTIMIZED=str(int(settings.gpu_count == 8)),
            WORLDCRAFTER_VAE_MODE="pipeline8" if settings.gpu_count == 8 else "single",
            WORLDCRAFTER_KV_FUSION="rope",
            WORLDCRAFTER_CONTROL_SOCKET=str(self._root / "control.sock"),
        )
        worker = str(Path(__file__).with_name("worker.py"))
        command = [str(settings.python)]
        if settings.gpu_count == 8:
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8"]
        command += [worker, "--socket", endpoint, "--source", str(settings.source),
                    "--root", str(self._root), "--max-chunks", str(settings.max_chunks)]
        self._log = (settings.scratch / f"worker-{self._root.name}.log").open("w")
        try:
            self._process = subprocess.Popen(command, cwd=settings.source, env=environment,
                                             stdout=self._log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
            deadline = time.monotonic() + settings.timeout
            while True:
                if self._process.poll() is not None:
                    raise RuntimeError(f"Inference worker exited; inspect {self._log.name}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Inference worker startup timed out; inspect {self._log.name}")
                try:
                    self._connection, _ = self._listener.accept()
                    break
                except socket.timeout:
                    continue
            self._connection.settimeout(settings.timeout)
            self._stream = self._connection.makefile("rwb")
            ready = self._receive()
            if ready.get("gpu_count") != settings.gpu_count:
                raise RuntimeError("Loaded GPU count does not match configuration")
            atexit.register(self.close)
        except BaseException:
            self.close()
            raise

    def _receive(self):
        line = self._stream.readline(4 * 1024 * 1024)
        if not line or not line.endswith(b"\n"):
            raise RuntimeError("Inference worker disconnected or sent an invalid response")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def _request(self, command, **arguments):
        if self._closed:
            raise RuntimeError("Inference worker is closed")
        self._sequence += 1
        request_id = self._sequence
        try:
            self._stream.write((json.dumps(dict(id=request_id, command=command, **arguments)) + "\n").encode())
            self._stream.flush()
            reply = self._receive()
            if reply.get("id") != request_id:
                raise RuntimeError("Inference worker response identity mismatch")
            return reply
        except BaseException:
            self.close()
            raise

    def generate(self, input: WorldCrafterInput) -> WorldCrafterResult:
        with self._lock:
            if input.image is not None:
                (self._root / "anchor.image").write_bytes(input.image)
            np.savez(self._root / "poses.npz", local=input.local_poses, world=input.world_poses)
            reply = self._request("generate", world_id=input.world_id,
                                  has_image=input.image is not None, prompt=input.prompt,
                                  seed=input.seed, gravity_align=input.gravity_align)
            output = self._root / "frames.npy"
            try:
                frames = np.load(output, allow_pickle=False)
                if frames.dtype != np.uint8 or frames.shape != (33, 384, 640, 3):
                    raise ValueError(f"Unexpected frames: {frames.shape}, {frames.dtype}")
                frames = np.ascontiguousarray(frames)
            finally:
                output.unlink(missing_ok=True)
            return WorldCrafterResult(frames=frames, **{key: reply[key] for key in
                ("world_id", "chunk_index", "camera_end", "gravity_aligned", "gpu_count", "world_up")})

    def reset(self):
        with self._lock:
            self._request("reset")
            (self._root / "anchor.image").unlink(missing_ok=True)
            (self._root / "poses.npz").unlink(missing_ok=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._stream is not None:
            self._stream.close()
        if self._connection is not None:
            self._connection.close()
        self._listener.close()
        if self._process is not None and self._process.poll() is None:
            os.killpg(self._process.pid, signal.SIGTERM)
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self._process.pid, signal.SIGKILL)
                self._process.wait(timeout=10)
        if self._log is not None:
            self._log.close()
        self._temporary.cleanup()
