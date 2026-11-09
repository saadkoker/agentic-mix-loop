"""Stock gain / pan / width: the track fader. Always the last plugin in a track chain; always
outputs stereo.

- gain_db: level, -60 (effectively muted) .. +12 dB. Automatable per sample: volume rides,
  clip-gain regions, mutes of noise between phrases.
- pan: -100 (hard L) .. +100 (hard R). Mono input: constant-power pan law (centre = -3 dB per
  side). Stereo input: constant-power balance. Automatable per sample.
- width: mid/side. 0 folds stereo to mono, 1 is unchanged, 1.5 is wider. Folding compensates
  for the lost side energy (capped at +3 dB) so the level doesn't jump. Ignored for mono input.
"""
import math

import numpy as np

TYPE = "gain_pan"
PARAMS = {
    "gain_db": {"default": 0.0, "min": -60.0, "max": 12.0},
    "pan":     {"default": 0.0, "min": -100.0, "max": 100.0},
    "width":   {"default": 1.0, "min": 0.0,   "max": 1.5},
}
SAMPLE_AUTOMATION = ("gain_db", "pan")   # the renderer may pass these as per-sample arrays


def process(audio, sr, params):
    p = params
    x = np.asarray(audio, dtype=np.float32)
    g = np.power(10.0, np.asarray(p["gain_db"], dtype=np.float64) / 20.0)
    theta = (np.asarray(p["pan"], dtype=np.float64) + 100.0) / 200.0 * (math.pi / 2)
    if x.shape[0] == 1:
        m = x[0] * g
        return np.stack([m * np.cos(theta), m * np.sin(theta)]).astype(np.float32)
    L, R = x[0], x[1]
    mid, side = (L + R) * 0.5, (L - R) * 0.5
    w = float(p["width"])
    if abs(w - 1.0) > 1e-6:
        em = float(np.dot(mid, mid))
        es = float(np.dot(side, side))
        comp = math.sqrt((em + es) / max(em + w * w * es, 1e-20)) if em + es > 0 else 1.0
        comp = min(comp, math.sqrt(2.0))
        mid, side = mid * comp, side * (w * comp)
    L, R = mid + side, mid - side
    gl = np.minimum(1.0, math.sqrt(2.0) * np.cos(theta))
    gr = np.minimum(1.0, math.sqrt(2.0) * np.sin(theta))
    return np.stack([L * (g * gl), R * (g * gr)]).astype(np.float32)
