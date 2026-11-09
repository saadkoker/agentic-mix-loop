"""Audio I/O, loudness, true peak and the fixed final limiter.

Convention everywhere: audio is float32, shape (channels, samples).
"""
from __future__ import annotations

import functools
import warnings
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import ndimage, signal

EPS = 1e-12


def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), EPS))


def undb(d):
    return 10.0 ** (np.asarray(d, dtype=np.float64) / 20.0)


def read(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), sr


def write(path: Path, audio: np.ndarray, sr: int, subtype: str = "PCM_24"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32).T, sr, subtype=subtype)


def load_stems(stem_dir: Path, remove_dc: bool = True) -> tuple[dict[str, np.ndarray], int]:
    """Load every WAV/FLAC/AIFF in a folder. Tracks must share a sample rate; shorter
    tracks are zero-padded at the end in memory (multitrack exports all start at 0).
    remove_dc=False (raw sessions) uses the samples exactly as they are in the files."""
    files = sorted([p for p in Path(stem_dir).iterdir()
                    if p.suffix.lower() in (".wav", ".flac", ".aif", ".aiff")])
    if not files:
        raise FileNotFoundError(f"no stems in {stem_dir}")
    stems, srs = {}, set()
    for p in files:
        a, sr = read(p)
        if a.shape[0] > 2:
            raise ValueError(f"{p.name}: {a.shape[0]} channels, only mono/stereo supported")
        if remove_dc:
            # remove DC from the source recordings so the DC guardrail only catches
            # DC that processing adds (subtracting the mean is exact and linear)
            a = a - a.mean(axis=1, keepdims=True)
        stems[p.stem] = a.astype(np.float32)
        srs.add(sr)
    if len(srs) != 1:
        raise ValueError(f"stems have mixed sample rates {srs}; resample them first")
    n = max(a.shape[1] for a in stems.values())
    for k, a in stems.items():
        if a.shape[1] != n:
            warnings.warn(f"stem {k} is {n - a.shape[1]} samples short, zero-padding")
            stems[k] = np.pad(a, ((0, 0), (0, n - a.shape[1])))
    return stems, srs.pop()


def to_stereo(a: np.ndarray) -> np.ndarray:
    if a.shape[0] == 2:
        return a
    if a.shape[0] == 1:
        return np.repeat(a, 2, axis=0)
    raise ValueError(f"bad channel count {a.shape[0]}")


def to_mono(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=0, keepdims=True)


# ---------------------------------------------------------------- loudness

@functools.lru_cache(maxsize=8)
def _meter(sr: int) -> pyln.Meter:
    return pyln.Meter(sr)


def lufs(a: np.ndarray, sr: int) -> float:
    """Integrated loudness (BS.1770). Returns -inf for silence."""
    if a.shape[1] < int(0.4 * sr):
        return float("-inf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(_meter(sr).integrated_loudness(a.T.astype(np.float64)))


def normalize_lufs(a: np.ndarray, sr: int, target: float) -> tuple[np.ndarray, float]:
    """Static gain to hit `target` LUFS. Returns (audio, gain_db)."""
    cur = lufs(a, sr)
    if not np.isfinite(cur):
        return a, 0.0
    g = target - cur
    return (a * undb(g)).astype(np.float32), float(g)


def true_peak(a: np.ndarray, oversample: int = 4) -> float:
    """True peak in dBTP via 4x oversampling (BS.1770-4 style)."""
    up = signal.resample_poly(a, oversample, 1, axis=1)
    return float(db(np.max(np.abs(up)) if up.size else 0.0))


def sample_peak(a: np.ndarray) -> float:
    return float(db(np.max(np.abs(a)) if a.size else 0.0))


def plr(a: np.ndarray, sr: int) -> float:
    """Peak-to-loudness ratio: true peak minus integrated loudness."""
    return true_peak(a) - lufs(a, sr)


# ----------------------------------------------------------------- limiter

def _true_peak_envelope(a: np.ndarray, oversample: int = 4) -> np.ndarray:
    """Per-sample peak of the oversampled signal, max across channels."""
    up = np.abs(signal.resample_poly(a, oversample, 1, axis=1)).max(axis=0)
    n = a.shape[1]
    up = up[: n * oversample]
    if up.size < n * oversample:
        up = np.pad(up, (0, n * oversample - up.size))
    return up.reshape(n, oversample).max(axis=1)


def _smooth_min_gain(g: np.ndarray, half: int) -> np.ndarray:
    """Smoothed gain curve that never exceeds g: centred min filter followed by a
    centred box filter of the same width (every box sample is <= g at centre)."""
    if half < 1:
        return g
    size = 2 * half + 1
    m = ndimage.minimum_filter1d(g, size=size, mode="nearest")
    return ndimage.uniform_filter1d(m, size=size, mode="nearest")


def limit(a: np.ndarray, sr: int, ceiling_dbtp: float = -1.0,
          attack_ms: float = 1.5, release_ms: float = 60.0) -> np.ndarray:
    """The fixed final brickwall limiter. Offline and zero-latency (it looks
    both ways), true-peak aware, deterministic. No agent can edit this."""
    a = np.asarray(a, dtype=np.float32)
    ceil = float(undb(ceiling_dbtp - 0.05))
    env = _true_peak_envelope(a)
    g = np.minimum(1.0, ceil / np.maximum(env, EPS))
    if g.min() >= 1.0:
        return a
    g = _smooth_min_gain(g, int(sr * attack_ms / 1000))
    g = _smooth_min_gain(g, int(sr * release_ms / 1000 / 2))
    out = (a * g[None, :]).astype(np.float32)
    # interpolation between samples can still overshoot a hair: static trim
    tp = true_peak(out)
    if tp > ceiling_dbtp - 0.01:
        out = (out * undb(ceiling_dbtp - 0.02 - tp)).astype(np.float32)
    return out


def limit_to_lufs(a: np.ndarray, sr: int, target_lufs: float, ceiling_dbtp: float = -1.0,
                  iters: int = 12) -> np.ndarray:
    """Push gain into the final limiter until the output hits target_lufs."""
    out = limit(a, sr, ceiling_dbtp)
    gain = 0.0
    for _ in range(iters):
        cur = lufs(out, sr)
        if not np.isfinite(cur) or abs(cur - target_lufs) < 0.1:
            break
        gain += target_lufs - cur
        out = limit((a * undb(gain)).astype(np.float32), sr, ceiling_dbtp)
    return out


def section(a: np.ndarray, sr: int, start_s: float, dur_s: float) -> np.ndarray:
    s = int(start_s * sr)
    return a[:, s: s + int(dur_s * sr)]
