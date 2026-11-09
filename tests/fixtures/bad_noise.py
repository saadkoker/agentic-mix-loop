import numpy as np
TYPE = "gain_pan"
PARAMS = {"g": {"default": 0.0, "min": -6.0, "max": 6.0}}
def process(audio, sr, params):
    x = audio if audio.shape[0] == 2 else np.repeat(audio, 2, axis=0)
    return (x + 1e-3 * np.random.standard_normal(x.shape)).astype(np.float32)
