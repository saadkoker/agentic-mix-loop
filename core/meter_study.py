"""Does the Airwindows Meter port tell good mixes from broken ones? Same test as `axes`:
flat sum vs deliberately broken versions, in units of the metric's own noise (tiny inaudible
changes). Also scores each song's final mix against its flat sum. No GPU, no API calls."""
from __future__ import annotations

import json
import time

import numpy as np

from . import audio as A
from .awmeter import meter
from .config import Project, load_config
from .evaluate import Evaluator
from .phase0 import variants
from .plugins import flat_chain, with_stem

METRICS = ["hit", "balance_mean", "power_mean", "detail_mean", "authority_mean",
           "power_grade", "detail_grade", "authority_grade"]
BROKEN = ["vocal_-10dB", "bass_muted", "over_compressed", "harsh_top"]


def song_study(project: Project) -> dict:
    cfg = load_config(project.root)
    ev = Evaluator(project, cfg, "dummy")
    sr = ev.sr
    v = variants(ev, cfg)
    flat = v["flat"]
    noise = {"orig": flat, "+0.1dB": flat * np.float32(A.undb(0.1)), "-0.1dB": flat * np.float32(A.undb(-0.1)),
             "shift+1": np.pad(flat, ((0, 0), (1, 0)))[:, : flat.shape[1]],
             "shift-1": np.pad(flat, ((0, 0), (0, 1)))[:, 1:]}
    base = flat_chain(ev.stem_names)
    stem = max(ev.stem_names, key=lambda s: float(np.mean(ev.stems[s] ** 2)))
    for d in (0.1, -0.1):
        items = [dict(it, params=dict(it["params"])) for it in base["stems"][stem]]
        items[-1]["params"]["gain_db"] = d
        noise[f"fader{d:+.1f}"] = A.normalize_lufs(ev.matched(with_stem(base, stem, items)), sr,
                                                   cfg["audio"]["target_lufs"])[0]
    t0 = time.time()
    m_noise = {k: meter(x, sr) for k, x in noise.items()}
    m_var = {k: meter(x, sr) for k, x in v.items() if k != "flat"}
    out = {"song": project.root.name, "noise": m_noise, "variants": m_var, "flat": m_noise["orig"]}
    if project.state.exists():
        st = json.loads(project.state.read_text())
        best = A.normalize_lufs(ev.matched(st["best_chain"]), sr, cfg["audio"]["target_lufs"])[0]
        out["best"] = meter(best, sr)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(results: list[dict]) -> str:
    L = []
    for r in results:
        L.append(f"\n=== {r['song']} ===")
        f = r["flat"]
        L.append(f"  flat: grade {f['grade']!r} hit {f['hit']:.2f}  directions {f['directions']!r}")
        for k, m in r["variants"].items():
            L.append(f"  {k:16s} grade {m['grade']!r:10s} hit {m['hit']:6.2f}  directions {m['directions']!r}")
        if "best" in r:
            b = r["best"]
            L.append(f"  BEST MIX         grade {b['grade']!r:10s} hit {b['hit']:6.2f}  directions {b['directions']!r}")
    # separation table: (flat - broken) / noise spread, per metric, per breakage
    L.append("\nSeparation of flat vs broken, in noise units (positive = metric prefers the flat mix;"
             " a pass needs > 2):")
    L.append(f"  {'metric':16s} " + " ".join(f"{b[:12]:>12s}" for b in BROKEN) + "   passes")
    for met in METRICS:
        cells, passes, total = [], 0, 0
        for b in BROKEN:
            seps = []
            for r in results:
                if b not in r["variants"]:
                    continue
                vals = [m[met] for m in r["noise"].values()]
                spread = max(max(vals) - min(vals), 1e-9)
                seps.append((r["flat"][met] - r["variants"][b][met]) / spread)
            total += len(seps)
            passes += sum(s > 2 for s in seps)
            cells.append(f"{np.median(seps):+12.1f}" if seps else f"{'-':>12s}")
        L.append(f"  {met:16s} " + " ".join(cells) + f"   {passes}/{total}")
    L.append("  (cells are the median over songs)")
    L.append("\nFinal mix vs flat (best - flat, in noise units):")
    for r in results:
        if "best" not in r:
            continue
        row = []
        for met in ("hit", "balance_mean", "power_mean", "detail_mean", "authority_mean"):
            vals = [m[met] for m in r["noise"].values()]
            spread = max(max(vals) - min(vals), 1e-9)
            row.append(f"{met} {(r['best'][met] - r['flat'][met]) / spread:+.1f}")
        L.append(f"  {r['song']:10s} " + "  ".join(row))
    return "\n".join(L)
