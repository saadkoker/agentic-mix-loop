"""Which Audiobox axes should the score use? Re-weights every song's stored
Phase 0 windows (no GPU needed) and asks, for each weighting:

- how many good-vs-broken checks pass (good beats broken by > 2x its own noise);
- how far apart they are in units of that weighting's own noise (separation);
- which kinds of breakage it catches.

A weighting is only better if it separates good from broken by more noise
units, not just by bigger raw numbers.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .evaluate import noise_from_windows
from .judge import composite

CANDIDATES = {
    "PQ+CE (plan)": {"pq": 0.5, "ce": 0.5},
    "PQ": {"pq": 1.0},
    "CE": {"ce": 1.0},
    "CU": {"cu": 1.0},
    "PC": {"pc": 1.0},
    "PQ+CU": {"pq": 0.5, "cu": 0.5},
    "PQ+CE+CU": {"pq": 1 / 3, "ce": 1 / 3, "cu": 1 / 3},
    "all four": {"pq": 0.25, "ce": 0.25, "pc": 0.25, "cu": 0.25},
}


def _cfg_with(cfg: dict, w: dict) -> dict:
    c = copy.deepcopy(cfg)
    for a in ("pq", "ce", "pc", "cu"):
        c["score"][f"w_{a}"] = float(w.get(a, 0.0))
    return c


def evaluate_weighting(results: list[dict], cfg: dict) -> dict:
    passes, total, seps, caught = 0, 0, [], {}
    per_song = {}
    for r in results:
        noise = noise_from_windows(r["noise"]["windows"], cfg)
        spread = max(noise["spread"], 1e-6)
        sc = {k: composite(v["windows"], cfg).score for k, v in r["scores"].items()}
        ok_song = True
        for c in r["checks"]:
            g, b = c["good"], c["broken"]
            if g not in sc or b not in sc:
                continue
            d = sc[g] - sc[b]
            ok = d > noise["margin"]
            passes += ok
            total += 1
            seps.append(d / spread)
            caught.setdefault(b, []).append(ok)
            ok_song &= ok
        per_song[r["song"]] = ok_song
    return {"passes": passes, "total": total, "songs_pass": sum(per_song.values()),
            "n_songs": len(per_song), "median_sep": float(np.median(seps)) if seps else 0.0,
            "worst_sep": float(np.min(seps)) if seps else 0.0,
            "caught": {k: f"{sum(v)}/{len(v)}" for k, v in caught.items()}}


def study(song_dirs: list[Path], cfg: dict) -> dict:
    results = []
    for d in song_dirs:
        p = Path(d) / "phase0.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        if not r.get("noise", {}).get("windows") or not all("windows" in v for v in r["scores"].values()):
            print(f"  {Path(d).name}: phase0.json has no stored windows - rerun phase0 for it")
            continue
        results.append(r)
    if not results:
        raise SystemExit("no usable phase0.json files")
    out = {name: evaluate_weighting(results, _cfg_with(cfg, w)) for name, w in CANDIDATES.items()}
    kinds = sorted({k for o in out.values() for k in o["caught"]})
    print(f"\nWeightings on {len(results)} songs ({', '.join(r['song'] for r in results)})")
    print("separation = (good - broken) / that weighting's own noise spread; acceptance needs > 2\n")
    head = f"{'weighting':<14} {'checks':>7} {'songs':>6} {'median sep':>11} {'worst sep':>10}  " + \
        "  ".join(f"{k[:14]:>14}" for k in kinds)
    print(head)
    print("-" * len(head))
    for name, o in sorted(out.items(), key=lambda kv: (-kv[1]["passes"], -kv[1]["median_sep"])):
        print(f"{name:<14} {o['passes']:>3}/{o['total']:<3} {o['songs_pass']:>2}/{o['n_songs']:<3} "
              f"{o['median_sep']:>11.2f} {o['worst_sep']:>10.2f}  " +
              "  ".join(f"{o['caught'].get(k, '-'):>14}" for k in kinds))
    return out
