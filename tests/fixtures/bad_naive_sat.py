import numpy as np
TYPE = "sat"
PARAMS = {"drive_db": {"default": 12.0, "min": 0.0, "max": 24.0}}
def process(audio, sr, params):
    d = 10 ** (params["drive_db"] / 20)
    return (np.tanh(d * audio) / d).astype(np.float32)
