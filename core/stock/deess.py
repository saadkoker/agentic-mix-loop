"""Stock de-esser: compresses only the top band when sibilance ("s", "t", "sh") gets loud.

The signal is split at freq_hz into low + high (high = zero-phase high-pass, low = the rest, so
the bands sum back to the input exactly). The high band's level drives a gain reduction on the
high band only: threshold_db / ratio, at most range_db, with attack / release smoothing.
TYPE is comp (it is a frequency-selective compressor).
"""
import numpy as np
from scipy import signal

TYPE = "comp"
PARAMS = {
    "freq_hz":      {"default": 6000.0, "min": 2500.0, "max": 12000.0},
    "threshold_db": {"default": -30.0,  "min": -60.0,  "max": 0.0},
    "ratio":        {"default": 4.0,    "min": 1.0,    "max": 10.0},
    "range_db":     {"default": 10.0,   "min": 0.0,    "max": 24.0},
    "attack_ms":    {"default": 1.0,    "min": 0.1,    "max": 10.0},
    "release_ms":   {"default": 60.0,   "min": 10.0,   "max": 300.0},
}


def _onepole(x, ms, sr):
    a = 1.0 - np.exp(-1.0 / max(ms / 1000.0 * sr, 1.0))
    return signal.lfilter([a], [1.0, -(1.0 - a)], x)


def process(audio, sr, params):
    p = params
    x = np.asarray(audio, dtype=np.float64)
    sos = signal.butter(4, min(float(p["freq_hz"]), sr * 0.45), "highpass", fs=sr, output="sos")
    hi = signal.sosfiltfilt(sos, x, axis=1)
    lo = x - hi
    det = np.sqrt(_onepole(np.mean(hi * hi, axis=0), 5.0, sr) + 1e-20)
    lvl = 20.0 * np.log10(det + 1e-12)
    over = np.maximum(lvl - float(p["threshold_db"]), 0.0)
    gr = np.minimum(over * (1.0 - 1.0 / float(p["ratio"])), float(p["range_db"]))
    # fast rise (attack), slow fall (release)
    gr_s = np.maximum(_onepole(gr, float(p["attack_ms"]), sr), _onepole(gr, float(p["release_ms"]), sr))
    gr_s = np.minimum(gr_s, gr.max() if gr.size else 0.0)
    g = np.power(10.0, -gr_s / 20.0)
    return (lo + hi * g[None, :]).astype(np.float32)
