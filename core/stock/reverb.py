"""Stock reverb (algorithmic, pedalboard Freeverb). For a shared send bus set mix = 1 (wet only)
and feed it with track sends; as an insert keep mix low.

- room_size, damping, width: the space.
- predelay_ms: gap before the reverb starts (keeps vocals clear of their own tail).
- lowcut_hz / highcut_hz: tone of the reverb return only (cut mud and fizz).
- mix: 0 dry .. 1 wet only.
"""
import numpy as np
import pedalboard as pb

TYPE = "space"
PARAMS = {
    "room_size":   {"default": 0.5,    "min": 0.0,    "max": 1.0},
    "damping":     {"default": 0.5,    "min": 0.0,    "max": 1.0},
    "width":       {"default": 1.0,    "min": 0.0,    "max": 1.0},
    "predelay_ms": {"default": 20.0,   "min": 0.0,    "max": 120.0},
    "lowcut_hz":   {"default": 200.0,  "min": 20.0,   "max": 1000.0},
    "highcut_hz":  {"default": 8000.0, "min": 2000.0, "max": 20000.0},
    "mix":         {"default": 0.25,   "min": 0.0,    "max": 1.0},
}


def process(audio, sr, params):
    p = params
    x = np.asarray(audio, dtype=np.float32)
    n = x.shape[1]
    verb = pb.Reverb(room_size=float(p["room_size"]), damping=float(p["damping"]), wet_level=1.0,
                     dry_level=0.0, width=float(p["width"]), freeze_mode=0.0)
    wet = verb(x, sr)
    tone = pb.Pedalboard([pb.HighpassFilter(cutoff_frequency_hz=float(p["lowcut_hz"])),
                          pb.LowpassFilter(cutoff_frequency_hz=float(min(p["highcut_hz"], sr * 0.45)))])
    wet = tone(np.ascontiguousarray(wet, dtype=np.float32), sr)
    d = int(round(p["predelay_ms"] / 1000.0 * sr))
    if d > 0:
        wet = np.concatenate([np.zeros((wet.shape[0], d), np.float32), wet[:, : n - d]], axis=1)
    m = float(p["mix"])
    return ((1.0 - m) * x + m * wet[:, :n]).astype(np.float32)
