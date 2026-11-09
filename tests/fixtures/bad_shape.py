import numpy as np
TYPE = "comp"
PARAMS = {"g": {"default": 0.0, "min": -6.0, "max": 6.0}}
def process(audio, sr, params):
    return audio[:, :-10]
