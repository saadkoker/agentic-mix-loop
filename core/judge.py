"""Audiobox Aesthetics judge: windows, composite score.

The model loads once and stays in memory. Audio in is the loudness-matched
final render; it is downmixed to mono and resampled to 16 kHz (what the model
uses anyway), then cut into overlapping windows that are scored in batches.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal

AXES = ("PQ", "CE", "PC", "CU")
MODEL_SR = 16000


@dataclass
class Score:
    score: float
    PQ: float
    CE: float
    PC: float
    CU: float
    worst: float               # min over windows of (PQ_w + CE_w) / 2
    worst_at_s: float          # start time of the worst window
    windows: list[dict]

    def to_dict(self, windows: bool = False) -> dict:
        d = asdict(self)
        if not windows:
            d.pop("windows")
        return d

    def short(self) -> str:
        return (f"score {self.score:.4f} | PQ {self.PQ:.3f} CE {self.CE:.3f} "
                f"worst {self.worst:.3f}@{self.worst_at_s:.0f}s | PC {self.PC:.3f} CU {self.CU:.3f}")


def axis_weights(cfg: dict) -> dict:
    s = cfg["score"]
    return {a: float(s.get(f"w_{a.lower()}", 0.0)) for a in AXES}


def window_quality(w: dict, weights: dict) -> float:
    """Per-window quality: weighted mean of the axes in use (default (PQ+CE)/2)."""
    tot = sum(weights.values()) or 1.0
    return sum(weights[a] * w[a] for a in AXES) / tot


def composite(windows: list[dict], cfg: dict) -> Score:
    """score = sum_a w_a * mean(a) + w_worst * min_window(weighted mean of the axes in use).
    With the defaults (w_pq = w_ce = 0.5, w_pc = w_cu = 0) this is the plan's formula."""
    s = cfg["score"]
    wts = axis_weights(cfg)
    m = {a: float(np.mean([w[a] for w in windows])) for a in AXES}
    q = [window_quality(w, wts) for w in windows]
    i = int(np.argmin(q))
    total = sum(wts[a] * m[a] for a in AXES) + s["w_worst"] * q[i]
    return Score(score=float(total), PQ=m["PQ"], CE=m["CE"], PC=m["PC"], CU=m["CU"],
                 worst=float(q[i]), worst_at_s=float(windows[i]["t"]), windows=windows)


def to_model_input(x: np.ndarray, sr: int) -> np.ndarray:
    mono = x.mean(axis=0).astype(np.float64)
    if sr != MODEL_SR:
        g = np.gcd(sr, MODEL_SR)
        mono = signal.resample_poly(mono, MODEL_SR // g, sr // g)
    return mono.astype(np.float32)


def window_starts(n: int, sr: int, win_s: float, hop_s: float) -> list[int]:
    win, hop = int(win_s * sr), int(hop_s * sr)
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n - hop // 2:     # cover the tail
        starts.append(n - win)
    return starts


class AudioboxJudge:
    name = "audiobox"

    """The model runs in its own worker process. On this machine the GPU
    intermittently faults inside the model ("illegal memory access" or a hung
    kernel); a CUDA fault poisons the whole process, so the worker is killed,
    restarted and the same windows re-scored. A fault costs seconds, not the run."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._worker = None

    def _get_worker(self):
        if self._worker is None:
            from .judge_worker import JudgeWorker
            self._worker = JudgeWorker(int(self.cfg["score"].get("batch_size", 4)))
        return self._worker

    def raw_windows(self, x: np.ndarray, sr: int) -> list[dict]:
        from .judge_worker import JudgeError, WorkerFailed
        s = self.cfg["score"]
        mono = to_model_input(x, sr)
        win = int(s["window_s"] * MODEL_SR)
        starts = window_starts(len(mono), MODEL_SR, s["window_s"], s["hop_s"])
        segs = np.zeros((len(starts), win), np.float32)
        for i, st in enumerate(starts):
            seg = mono[st: st + win]
            segs[i, : len(seg)] = seg
        lens = [min(win, len(mono) - st) for st in starts]
        w = self._get_worker()
        last = None
        for attempt in range(int(s.get("judge_retries", 3)) + 1):
            try:
                out = w.score(segs, lens)
                return [{"t": st / MODEL_SR, **{a: float(o[a]) for a in AXES}} for st, o in zip(starts, out)]
            except WorkerFailed as e:
                last = e
                print(f"  judge worker failed ({e}); restarting it (attempt {attempt + 1})", flush=True)
                w.restart()
        raise JudgeError(f"judge failed after retries: {last}")

    def score(self, x: np.ndarray, sr: int) -> Score:
        return composite(self.raw_windows(x, sr), self.cfg)


class DummyJudge(AudioboxJudge):
    """Deterministic stand-in for tests without the model: rewards a balanced
    spectrum and moderate crest factor. NOT a quality measure."""
    name = "dummy"

    def raw_windows(self, x, sr):
        s = self.cfg["score"]
        mono = to_model_input(x, sr)
        win = int(s["window_s"] * MODEL_SR)
        out = []
        for st in window_starts(len(mono), MODEL_SR, s["window_s"], s["hop_s"]):
            seg = mono[st: st + win].astype(np.float64)
            f, p = signal.welch(seg, MODEL_SR, nperseg=2048)
            bands = [p[(f >= lo) & (f < hi)].sum() + 1e-12 for lo, hi in
                     ((20, 250), (250, 2000), (2000, 8000))]
            tilt = np.std(10 * np.log10(bands)) / 10
            crest = 20 * np.log10(np.max(np.abs(seg)) / (np.sqrt(np.mean(seg ** 2)) + 1e-9) + 1e-9)
            pq = 7.5 - tilt - abs(crest - 12) / 10
            ce = 6.5 - tilt * 0.5
            out.append({"t": st / MODEL_SR, "PQ": pq, "CE": ce, "PC": 5.0, "CU": 6.0})
        return out


def make_judge(cfg: dict, kind: str = "audiobox"):
    return DummyJudge(cfg) if kind == "dummy" else AudioboxJudge(cfg)
