"""Kill the judge worker mid-run and check the next score still comes back,
identical to before. Needs the GPU model.   python tests/test_judge_worker.py"""
import sys, time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np
from core.config import load_config
from core.judge import AudioboxJudge


def main():
    cfg = load_config()
    j = AudioboxJudge(cfg)
    rng = np.random.default_rng(0)
    t = np.arange(44100 * 30) / 44100
    x = np.stack([0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))] * 2).astype(np.float32)
    t0 = time.time(); a = j.raw_windows(x, 44100); print(f"first score ({time.time() - t0:.1f}s incl. model load): {len(a)} windows")
    j._get_worker()._proc.kill()          # simulate a CUDA fault killing the worker
    t0 = time.time(); b = j.raw_windows(x, 44100); print(f"after kill: {time.time() - t0:.1f}s, restarts={j._worker.restarts}")
    same = all(abs(p[k] - q[k]) < 1e-4 for p, q in zip(a, b) for k in ("PQ", "CE", "PC", "CU"))
    ok = same and j._worker.restarts == 1
    print("ALL OK" if ok else "FAILED"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
