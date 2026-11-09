"""Stock compressor: Pedalboard Compressor with makeup gain and parallel mix.
ratio = 1 is an exact bypass (makeup still applies)."""
import numpy as np
import pedalboard as pb

TYPE = "comp"
PARAMS = {
    "threshold_db": {"default": -12.0, "min": -40.0, "max": 0.0},
    "ratio":        {"default": 1.0,   "min": 1.0,   "max": 10.0},
    "attack_ms":    {"default": 10.0,  "min": 0.1,   "max": 100.0},
    "release_ms":   {"default": 120.0, "min": 10.0,  "max": 1000.0},
    "makeup_db":    {"default": 0.0,   "min": 0.0,   "max": 18.0},
    "mix":          {"default": 1.0,   "min": 0.0,   "max": 1.0},
}


def process(audio, sr, params):
    p = params
    x = np.asarray(audio, dtype=np.float32)
    makeup = np.float32(10 ** (p["makeup_db"] / 20))
    if p["ratio"] <= 1.0001:
        m = np.float32(p["mix"])
        return np.asarray(m * x * makeup + (1 - m) * x, dtype=np.float32)
    board = pb.Pedalboard([
        pb.Compressor(threshold_db=p["threshold_db"], ratio=p["ratio"],
                      attack_ms=p["attack_ms"], release_ms=p["release_ms"]),
    ])
    wet = np.asarray(board(x, sr), dtype=np.float32) * makeup
    m = np.float32(p["mix"])
    return np.asarray(m * wet + (1 - m) * x, dtype=np.float32)
