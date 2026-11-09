"""Your blind listening check.

Two loudness-matched clips of the same 20 s section, A and B in random order.
Press a / b / s(ame) / r(eplay). You only learn which was which afterwards.
Votes go to votes.jsonl.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from . import audio as A


def _play(x: np.ndarray, sr: int, label: str, out_dir: Path):
    try:
        import sounddevice as sd
        print(f"  playing {label} ...", flush=True)
        sd.play(x.T, sr)
        sd.wait()
    except Exception as e:  # noqa: BLE001
        p = out_dir / f"{label}.wav"
        A.write(p, x, sr)
        print(f"  (no audio device: {e}) open {p}")
        if hasattr(os, "startfile"):
            os.startfile(str(p))  # noqa: S606


def pick_section(old: np.ndarray, new: np.ndarray, sr: int, dur_s: float,
                 rng: random.Random, hint_s: float | None = None) -> float:
    """Start time of a section where the two mixes differ most (or near the hint)."""
    n = min(old.shape[1], new.shape[1])
    win = int(dur_s * sr)
    if n <= win:
        return 0.0
    if hint_s is not None:
        return float(min(max(0.0, hint_s - dur_s / 4), (n - win) / sr))
    hop = sr
    best, best_t = -1.0, 0.0
    d = (old[:, :n] - new[:, :n]).mean(axis=0)
    e = np.cumsum(np.concatenate([[0.0], d.astype(np.float64) ** 2]))
    for s in range(0, n - win, hop):
        v = e[s + win] - e[s]
        if v > best:
            best, best_t = v, s / sr
    # small random jitter so you don't always hear the exact same bar
    return float(max(0.0, min((n - win) / sr, best_t + rng.uniform(-2, 2))))


def blind_ab(old: np.ndarray, new: np.ndarray, sr: int, out_dir: Path, votes_path: Path,
             dur_s: float = 20.0, context: dict | None = None, seed: int | None = None) -> str:
    """Returns 'old', 'new' or 'same'. Requires an interactive terminal."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pick_section(old, new, sr, dur_s, rng)
    a_is_new = rng.random() < 0.5
    clips = {"A": new if a_is_new else old, "B": old if a_is_new else new}
    clips = {k: A.normalize_lufs(A.section(v, sr, start, dur_s), sr, -14.0)[0] for k, v in clips.items()}
    print(f"\n=== Blind listening check: section {start:.0f}-{start + dur_s:.0f} s ===")
    while True:
        for k in ("A", "B"):
            _play(clips[k], sr, k, out_dir)
        ans = input("  Which is better? [a/b/s(ame)/r(eplay)]: ").strip().lower()[:1]
        if ans in ("a", "b", "s"):
            break
    if ans == "s":
        pick = "same"
    else:
        pick = "new" if (ans == "a") == a_is_new else "old"
    print(f"  A was the {'NEW' if a_is_new else 'OLD'} mix -> you preferred: {pick}")
    with open(votes_path, "a") as f:
        f.write(json.dumps({"t": time.time(), "start_s": start, "dur_s": dur_s,
                            "a_is_new": a_is_new, "answer": ans, "pick": pick,
                            **(context or {})}) + "\n")
    return pick


def interactive() -> bool:
    return sys.stdin is not None and sys.stdin.isatty()
