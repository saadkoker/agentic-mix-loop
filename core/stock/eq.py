"""Stock EQ: 12 dB/oct high/low-pass, two shelves, two peaks. Linear and
time-invariant. Neutral values bypass their band exactly."""
import numpy as np
import pedalboard as pb
from scipy import signal

TYPE = "eq"
PARAMS = {
    "hp_hz":         {"default": 20.0,    "min": 20.0,   "max": 400.0},
    "low_shelf_hz":  {"default": 120.0,   "min": 40.0,   "max": 500.0},
    "low_shelf_db":  {"default": 0.0,     "min": -12.0,  "max": 6.0},
    "peak1_hz":      {"default": 300.0,   "min": 60.0,   "max": 16000.0},
    "peak1_db":      {"default": 0.0,     "min": -12.0,  "max": 6.0},
    "peak1_q":       {"default": 1.0,     "min": 0.3,    "max": 8.0},
    "peak2_hz":      {"default": 3000.0,  "min": 60.0,   "max": 16000.0},
    "peak2_db":      {"default": 0.0,     "min": -12.0,  "max": 6.0},
    "peak2_q":       {"default": 1.0,     "min": 0.3,    "max": 8.0},
    "high_shelf_hz": {"default": 8000.0,  "min": 1500.0, "max": 16000.0},
    "high_shelf_db": {"default": 0.0,     "min": -12.0,  "max": 6.0},
    "lp_hz":         {"default": 20000.0, "min": 2000.0, "max": 20000.0},
}


def process(audio, sr, params):
    p = params
    x = np.asarray(audio, dtype=np.float32)
    nyq = sr / 2.0
    if p["hp_hz"] > 20.5:
        sos = signal.butter(2, min(p["hp_hz"], nyq * 0.95), "highpass", fs=sr, output="sos")
        x = signal.sosfilt(sos, x, axis=1).astype(np.float32)
    if p["lp_hz"] < 19999.0 and p["lp_hz"] < nyq * 0.95:
        sos = signal.butter(2, p["lp_hz"], "lowpass", fs=sr, output="sos")
        x = signal.sosfilt(sos, x, axis=1).astype(np.float32)
    board = []
    if abs(p["low_shelf_db"]) > 1e-3:
        board.append(pb.LowShelfFilter(cutoff_frequency_hz=p["low_shelf_hz"],
                                       gain_db=p["low_shelf_db"], q=0.707))
    for k in ("peak1", "peak2"):
        if abs(p[k + "_db"]) > 1e-3:
            board.append(pb.PeakFilter(cutoff_frequency_hz=min(p[k + "_hz"], nyq * 0.9),
                                       gain_db=p[k + "_db"], q=p[k + "_q"]))
    if abs(p["high_shelf_db"]) > 1e-3:
        board.append(pb.HighShelfFilter(cutoff_frequency_hz=min(p["high_shelf_hz"], nyq * 0.9),
                                        gain_db=p["high_shelf_db"], q=0.707))
    if board:
        x = pb.Pedalboard(board)(x, sr)
    return np.asarray(x, dtype=np.float32)
