"""The critic is the judge itself. Every brief comes from the model that
scores the result, so agents never spend rounds pleasing a second opinion.

For the current best mix it reports:
- the weakest windows (lowest (PQ+CE)/2), with times;
- per-stem sensitivity: score change with each stem at -3 dB and +3 dB,
  overall and inside the weak windows (a measured gradient for the hill-climb);
- per-stem solo quality: each stem's own PQ/CE over the windows where it plays.

Variants re-mix cached stem outputs through the master chain and the fixed
final step, so this is diagnostic only: guardrails are not applied.
"""
from __future__ import annotations

import numpy as np

from . import audio as A
from .judge import AXES, axis_weights, composite, window_quality


def mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60}:{t % 60:02d}"


def _active_windows(x: np.ndarray, sr: int, windows: list[dict], win_s: float, thresh_db: float = -45.0):
    """Indices of windows where the stem is actually playing."""
    m = x.mean(axis=0)
    out = []
    for i, w in enumerate(windows):
        s = int(w["t"] * sr)
        seg = m[s: s + int(win_s * sr)]
        if seg.size and 20 * np.log10(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-12) > thresh_db:
            out.append(i)
    return out


def diagnose(ev, chain: dict, cfg: dict, step_db: float = 3.0, n_weak: int = 3,
             min_gain: float | None = None) -> dict:
    sr = ev.sr
    target = cfg["audio"]["target_lufs"]
    win_s = cfg["score"]["window_s"]
    r = ev.render(chain)
    master = chain.get("master", [])

    def score_mix(mix):
        _, final, _, _ = ev.renderer.finish(mix, master)
        x, _ = A.normalize_lufs(final, sr, target)
        return ev.judge.raw_windows(x, sr)

    x0, _ = A.normalize_lufs(r.final, sr, target)
    base_w = ev.judge.raw_windows(x0, sr)
    base = composite(base_w, cfg)
    wts = axis_weights(cfg)
    pc = np.array([window_quality(w, wts) for w in base_w])
    weak = list(np.argsort(pc)[:n_weak])
    has_buses = bool(chain.get("buses"))
    mix = None if has_buses else sum(r.stems.values())

    def variant_mix(name, y, g):
        if not has_buses:
            return mix + y * np.float32(A.undb(g) - 1.0)
        outs = dict(r.stems)
        outs[name] = y * np.float32(A.undb(g))
        return ev.renderer.route(outs, chain)[0]

    stems = {}
    for name, y in r.stems.items():
        d = {}
        for label, g in (("minus", -step_db), ("plus", step_db)):
            w = score_mix(variant_mix(name, y, g))
            s = composite(w, cfg)
            wpc = np.array([window_quality(v, wts) for v in w])
            d[f"{label}{step_db:g}dB"] = round(s.score - base.score, 4)
            d[f"{label}{step_db:g}dB_in_weak"] = round(float(np.mean(wpc[weak] - pc[weak])), 4)
        # solo quality over the windows where it plays
        solo, _ = A.normalize_lufs(y, sr, target)
        sw = ev.judge.raw_windows(solo, sr)
        act = _active_windows(y, sr, sw, win_s)
        if act:
            d["solo_PQ"] = round(float(np.mean([sw[i]["PQ"] for i in act])), 3)
            d["solo_CE"] = round(float(np.mean([sw[i]["CE"] for i in act])), 3)
        d["plays_in_pct_of_windows"] = round(100 * len(act) / max(1, len(sw)))
        stems[name] = d

    weak_list = [{"time": f"{mmss(base_w[i]['t'])}-{mmss(base_w[i]['t'] + win_s)}",
                  **{a: round(base_w[i][a], 3) for a in AXES},
                  "quality": round(float(pc[i]), 3)} for i in weak]
    timeline = [{"t": mmss(w["t"]), "quality": round(window_quality(w, wts), 3)} for w in base_w]

    hints = []
    noise = cfg["score"]["min_margin"] if min_gain is None else min_gain
    moves = []
    for name, d in stems.items():
        for label in (f"minus{step_db:g}dB", f"plus{step_db:g}dB"):
            moves.append((d[label], name, label, d[f"{label}_in_weak"]))
    for gain, name, label, wk in sorted(moves, reverse=True)[:4]:
        if gain > noise:
            hints.append(f"{name} {'down' if label.startswith('minus') else 'up'} {step_db:g} dB: "
                         f"{gain:+.4f} overall, {wk:+.3f} in the weak windows")
    solos = sorted([(d["solo_PQ"], n) for n, d in stems.items() if "solo_PQ" in d])
    if solos:
        hints.append(f"lowest solo production quality: {solos[0][1]} (PQ {solos[0][0]:.2f})"
                     + (f", then {solos[1][1]} (PQ {solos[1][0]:.2f})" if len(solos) > 1 else ""))
    if weak_list:
        hints.append("weakest sections: " + ", ".join(f"{w['time']} ({w['quality']:.2f})" for w in weak_list))
    return {"score": round(base.score, 4), "weakest_windows": weak_list, "stems": stems,
            "hints": hints, "timeline": timeline, "step_db": step_db}


def short(diag: dict) -> list[str]:
    return diag.get("hints", [])
