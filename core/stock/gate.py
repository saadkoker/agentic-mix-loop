"""Stock noise gate / downward expander (pedalboard NoiseGate): turns a track down when it
drops below the threshold - bleed, hiss and room noise between notes or phrases.

- threshold_db: level below which the gate closes.
- ratio: expansion ratio below threshold (higher = closer to a hard gate).
- attack_ms / release_ms: how fast it opens / closes.
For known gaps (e.g. between vocal phrases) volume automation on gain_pan is more exact.
"""
import numpy as np
import pedalboard as pb

TYPE = "gate"
PARAMS = {
    "threshold_db": {"default": -50.0, "min": -90.0, "max": -10.0},
    "ratio":        {"default": 4.0,   "min": 1.0,   "max": 20.0},
    "attack_ms":    {"default": 1.0,   "min": 0.1,   "max": 50.0},
    "release_ms":   {"default": 150.0, "min": 10.0,  "max": 1000.0},
}


def process(audio, sr, params):
    p = params
    x = np.asarray(audio, dtype=np.float32)
    g = pb.NoiseGate(threshold_db=float(p["threshold_db"]), ratio=float(p["ratio"]),
                     attack_ms=float(p["attack_ms"]), release_ms=float(p["release_ms"]))
    return np.asarray(g(x, sr), dtype=np.float32)[:, : x.shape[1]]
