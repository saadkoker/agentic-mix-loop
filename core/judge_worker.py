"""Audiobox in a separate process, restartable.

A CUDA fault ("illegal memory access") or a hung kernel leaves the process's
GPU context unusable, so the model lives in a child process: the parent sends
windows, gets scores back, and on an error, a crash or a timeout kills the
child and starts a fresh one.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np

MODEL_SR = 16000


class WorkerFailed(Exception):
    pass


class JudgeError(Exception):
    pass


def _serve(conn, batch_size: int):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from audiobox_aesthetics.infer import initialize_predictor
    pred = initialize_predictor()
    conn.send(("ready", None))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "stop":
            return
        _, segs, lens = msg
        try:
            out = []
            for i in range(0, len(segs), batch_size):
                items = [{"path": torch.from_numpy(np.ascontiguousarray(segs[j, : lens[j]]))[None, :],
                          "sample_rate": MODEL_SR} for j in range(i, min(i + batch_size, len(segs)))]
                out.extend(pred.forward(items))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            conn.send(("ok", out))
        except Exception as e:  # noqa: BLE001
            try:
                conn.send(("err", f"{type(e).__name__}: {str(e)[:300]}"))
            finally:
                os._exit(1)        # CUDA context may be poisoned: die, the parent restarts us


class JudgeWorker:
    def __init__(self, batch_size: int = 4, start_timeout_s: float = 600.0, call_timeout_s: float = 300.0):
        self.batch_size = batch_size
        self.start_timeout = start_timeout_s
        self.call_timeout = call_timeout_s
        self.restarts = 0
        self._proc = None
        self._conn = None
        import atexit
        atexit.register(self.close)
        self.start()

    def start(self):
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, args=(child, self.batch_size), daemon=True)
        self._proc.start()
        self._conn = parent
        if not parent.poll(self.start_timeout):
            self.kill()
            raise WorkerFailed("worker did not start")
        kind, _ = parent.recv()
        if kind != "ready":
            raise WorkerFailed(f"unexpected start message {kind}")

    def kill(self):
        if self._proc is not None and self._proc.is_alive():
            self._proc.kill()
            self._proc.join(10)
        self._proc = None

    def restart(self):
        self.kill()
        self.restarts += 1
        time.sleep(2)              # let the driver finish recovering
        self.start()

    def score(self, segs: np.ndarray, lens: list[int]) -> list[dict]:
        if self._proc is None or not self._proc.is_alive():
            raise WorkerFailed("worker not running")
        try:
            self._conn.send(("score", segs, lens))
            if not self._conn.poll(self.call_timeout):
                self.kill()
                raise WorkerFailed(f"no reply in {self.call_timeout:.0f} s (hung kernel)")
            kind, payload = self._conn.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as e:
            self.kill()
            raise WorkerFailed(f"worker died: {type(e).__name__}") from e
        if kind != "ok":
            self.kill()
            raise WorkerFailed(payload)
        return payload

    def close(self):
        self.kill()
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
