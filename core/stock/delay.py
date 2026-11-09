"""Stock delay (echo). For a send bus set mix = 1 and feed it with track sends.

- time_ms: delay time. For tempo sync: quarter note = 60000 / BPM, eighth = half that.
- feedback: how many repeats (0 = one echo).
- lowcut_hz / highcut_hz: tone of the repeats only (darker repeats sit behind the vocal).
- mix: 0 dry .. 1 echoes only.
"""
import numpy as np
import pedalboard as pb

TYPE = "space"
PARAMS = {
    "time_ms":    {"default": 375.0,  "min": 1.0,    "max": 2000.0},
    "feedback":   {"default": 0.3,    "min": 0.0,    "max": 0.9},
    "lowcut_hz":  {"default": 150.0,  "min": 20.0,   "max": 2000.0},
    "highcut_hz": {"default": 6000.0, "min": 1000.0, "max": 20000.0},
    "mix":        {"default": 0.25,   "min": 0.0,    "max": 1.0},
}


def process(audio, sr, params):
    p = params
    x = np.asarray(audio, dtype=np.float32)
    n = x.shape[1]
    echo = pb.Pedalboard([pb.Delay(delay_seconds=float(p["time_ms"]) / 1000.0,
                                   feedback=float(p["feedback"]), mix=1.0),
                          pb.HighpassFilter(cutoff_frequency_hz=float(p["lowcut_hz"])),
                          pb.LowpassFilter(cutoff_frequency_hz=float(min(p["highcut_hz"], sr * 0.45)))])
    wet = echo(x, sr)[:, :n]
    m = float(p["mix"])
    return ((1.0 - m) * x + m * wet).astype(np.float32)
