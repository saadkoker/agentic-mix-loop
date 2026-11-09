import numpy as np
TYPE = "eq"
PARAMS = {"g": {"default": 0.0, "min": -6.0, "max": 6.0}}
def process(audio, sr, params):
    return np.tanh(audio * 10 ** (params["g"] / 20) * 3).astype(np.float32) / 3
