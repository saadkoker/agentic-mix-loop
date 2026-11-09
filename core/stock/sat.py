"""Stock saturation: tanh waveshaper at 4x oversampling, unity small-signal
gain, parallel mix. drive_db = 0 is an exact bypass."""
import numpy as np
from scipy import signal

TYPE = "sat"
OVERSAMPLE = 4
PARAMS = {
    "drive_db":  {"default": 0.0, "min": 0.0,   "max": 24.0},
    "mix":       {"default": 1.0, "min": 0.0,   "max": 1.0},
    "output_db": {"default": 0.0, "min": -12.0, "max": 6.0},
}


def process(audio, sr, params):
    p = params
    x = np.asarray(audio, dtype=np.float32)
    out_g = np.float32(10 ** (p["output_db"] / 20))
    if p["drive_db"] < 1e-3 or p["mix"] < 1e-4:
        return x * out_g
    d = 10 ** (p["drive_db"] / 20)
    up = signal.resample_poly(x, OVERSAMPLE, 1, axis=1)
    up = np.tanh(d * up) / d
    wet = signal.resample_poly(up, 1, OVERSAMPLE, axis=1)[:, : x.shape[1]]
    m = p["mix"]
    return np.asarray((m * wet + (1 - m) * x) * out_g, dtype=np.float32)
