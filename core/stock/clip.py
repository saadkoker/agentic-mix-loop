"""Stock clipper: hard-to-soft knee at a ceiling, 4x oversampled.
With input_db = 0 and ceiling_db = 0, anything under full scale is untouched."""
import numpy as np
from scipy import signal

TYPE = "clip"
OVERSAMPLE = 4
PARAMS = {
    "input_db":   {"default": 0.0, "min": 0.0,   "max": 12.0},
    "ceiling_db": {"default": 0.0, "min": -24.0, "max": 0.0},
    "softness":   {"default": 0.3, "min": 0.0,   "max": 1.0},
    "output_db":  {"default": 0.0, "min": -12.0, "max": 6.0},
}


def _clip(x, c, k):
    t = c * (1.0 - 0.9 * k)            # knee start
    ax = np.abs(x)
    over = ax > t
    if not np.any(over):
        return x
    y = x.copy()
    if c - t < 1e-9:
        y[over] = np.sign(x[over]) * c
    else:
        y[over] = np.sign(x[over]) * (t + (c - t) * np.tanh((ax[over] - t) / (c - t)))
    return y


def process(audio, sr, params):
    p = params
    x = np.asarray(audio, dtype=np.float32) * np.float32(10 ** (p["input_db"] / 20))
    c = 10 ** (p["ceiling_db"] / 20)
    knee = c * (1.0 - 0.9 * p["softness"])
    if np.max(np.abs(x), initial=0.0) <= knee * 0.7:
        y = x                           # nothing can reach the knee, even between samples
    else:
        up = signal.resample_poly(x, OVERSAMPLE, 1, axis=1)
        up = _clip(up, c, p["softness"])
        y = signal.resample_poly(up, 1, OVERSAMPLE, axis=1)[:, : x.shape[1]]
    return np.asarray(y * np.float32(10 ** (p["output_db"] / 20)), dtype=np.float32)
