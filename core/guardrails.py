"""Hard technical limits. A fail throws the candidate out before scoring.

Three checks (stereo correlation, PLR, clicks) are relative to the flat
baseline when the raw stems themselves already break the absolute limit, so
a song whose stems are e.g. pre-crushed doesn't lock the loop forever. The
report says when that relief applied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage, signal

from . import audio as A
from .render import Render


@dataclass
class Check:
    name: str
    passed: bool
    value: float | str
    limit: str
    note: str = ""


@dataclass
class GuardReport:
    passed: bool
    checks: list[Check] = field(default_factory=list)

    def failures(self) -> list[str]:
        return [f"{c.name}: {c.value} (limit {c.limit}){' - ' + c.note if c.note else ''}"
                for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}


# ------------------------------------------------------------------ metrics

def stereo_correlation(x: np.ndarray, sr: int, win_s: float = 1.0) -> float:
    """Average L/R correlation over 1 s windows, skipping near-silent windows."""
    L, R = x[0].astype(np.float64), x[1].astype(np.float64)
    w = int(win_s * sr)
    n = (len(L) // w) * w
    if n == 0:
        return 1.0
    L, R = L[:n].reshape(-1, w), R[:n].reshape(-1, w)
    el, er = (L * L).sum(1), (R * R).sum(1)
    keep = (el > 1e-7 * w) & (er > 1e-7 * w)
    if not keep.any():
        return 1.0
    c = (L * R).sum(1)[keep] / np.sqrt(el[keep] * er[keep])
    return float(np.mean(c))


def low_band_correlation(x: np.ndarray, sr: int, hz: float) -> float:
    """Energy-weighted correlation of the band below `hz`."""
    sos = signal.butter(4, hz, "lowpass", fs=sr, output="sos")
    lo = signal.sosfiltfilt(sos, x.astype(np.float64), axis=1)
    el, er = np.dot(lo[0], lo[0]), np.dot(lo[1], lo[1])
    if el < 1e-9 or er < 1e-9:
        return 1.0
    return float(np.dot(lo[0], lo[1]) / np.sqrt(el * er))


def count_clicks(x: np.ndarray, sr: int, ratio: float, floor: float, win_ms: float = 10.0) -> int:
    """Sample-to-sample jumps larger than `ratio` x local RMS (and > floor)."""
    total = 0
    w = max(3, int(sr * win_ms / 1000))
    for ch in x:
        ch = ch.astype(np.float64)
        d = np.abs(np.diff(ch))
        rms = np.sqrt(np.maximum(ndimage.uniform_filter1d(ch * ch, size=w, mode="nearest"), 0.0))[1:]
        hits = (d > ratio * rms) & (d > floor)
        # count events, not samples
        total += int(np.count_nonzero(hits[1:] & ~hits[:-1]) + (1 if hits[:1].any() else 0))
    return total


def staged_dc(r: Render, target_lufs: float) -> float:
    """DC of the pre-limiter bus at the level the renderer stages it to. The
    fixed limiter itself can create a little DC on asymmetric material
    (kicks), which no agent controls, so DC is checked before it."""
    l = _pre_lufs(r)
    g = float(A.undb(target_lufs - l)) if np.isfinite(l) else 1.0
    return float(np.max(np.abs(r.pre.astype(np.float64).mean(axis=1)))) * g


@dataclass
class Baseline:
    """Stats from the flat baseline render that guardrails compare against."""
    stem_rel_lufs: dict[str, float]
    corr: float
    low_corr: float
    plr: float
    clicks: int
    n: int
    dc: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


_stem_lufs_cache: dict[str, float] = {}


def _pre_lufs(r: Render) -> float:
    if not np.isfinite(r.pre_lufs):
        r.pre_lufs = A.lufs(r.pre, r.sr)
    return r.pre_lufs


def _plr(r: Render) -> float:
    return A.true_peak(r.pre) - _pre_lufs(r)


def stem_rel_levels(r: Render) -> dict[str, float]:
    mix_l = _pre_lufs(r)
    out = {}
    for name, y in r.stems.items():
        key = r.stem_hashes.get(name, "")
        k = f"{name}:{key}"
        if key and k in _stem_lufs_cache:
            l = _stem_lufs_cache[k]
        else:
            l = A.lufs(y, r.sr)
            if key:
                _stem_lufs_cache[k] = l
        out[name] = l - mix_l if np.isfinite(l) and np.isfinite(mix_l) else float("-inf")
    return out


def baseline_stats(r: Render, cfg: dict) -> Baseline:
    g = cfg["guardrails"]
    return Baseline(stem_rel_lufs=stem_rel_levels(r),
                    corr=stereo_correlation(r.final, r.sr),
                    low_corr=low_band_correlation(r.final, r.sr, g["low_corr_hz"]),
                    plr=_plr(r),
                    clicks=count_clicks(r.pre, r.sr, g["click_ratio"], g["click_floor"]),
                    n=r.final.shape[1],
                    dc=staged_dc(r, cfg["audio"]["target_lufs"]))


# -------------------------------------------------------------------- check

def check(r: Render, base: Baseline, cfg: dict, bus_ref=None, chain: dict | None = None) -> GuardReport:
    """bus_ref(inputs) -> the flat baseline level (vs mix, LU) of those tracks summed. With it,
    each bus that has routed inputs must stay within stem_presence_db of that level too, so a
    bus fader can't bury a group that the per-track check would protect."""
    g = cfg["guardrails"]
    sr = r.sr
    checks: list[Check] = []

    def add(name, ok, value, limit, note=""):
        checks.append(Check(name, bool(ok), value if isinstance(value, str) else round(float(value), 4),
                            limit, note))

    # NaN / Inf / silence first: other metrics are meaningless otherwise
    bad = [k for k, y in r.stems.items() if not np.all(np.isfinite(y))]
    finite = np.all(np.isfinite(r.pre)) and np.all(np.isfinite(r.final)) and not bad
    silent = finite and A.sample_peak(r.final) < -80.0
    add("nan_inf_silence", finite and not silent,
        "ok" if finite and not silent else ("silent" if finite else f"non-finite {bad or 'mix'}"),
        "finite, not silent")
    if not finite or silent:
        return GuardReport(False, checks)

    add("length", not r.length_errors and abs(r.final.shape[1] - base.n) <= g["length_tol"],
        "; ".join(r.length_errors) or r.final.shape[1], f"{base.n} +/-{g['length_tol']}")

    tp = A.true_peak(r.final)
    add("true_peak", tp <= g["true_peak_dbtp"] + 0.01, tp, f"<= {g['true_peak_dbtp']} dBTP")

    dc = staged_dc(r, cfg["audio"]["target_lufs"])
    lim = max(g["dc_offset"], base.dc * 1.5)
    add("dc_offset", dc < lim, dc, f"< {lim:.4f}", "relaxed to baseline" if lim > g["dc_offset"] else "")

    corr = stereo_correlation(r.final, sr)
    lim = min(g["corr_min"], base.corr - 0.05)
    add("stereo_corr", corr > lim, corr, f"> {lim:.3f}",
        "relaxed to baseline" if lim < g["corr_min"] else "")
    lc = low_band_correlation(r.final, sr, g["low_corr_hz"])
    lim = min(g["low_corr_min"], base.low_corr - 0.05)
    add("low_band_corr", lc > lim, lc, f"> {lim:.3f} below {g['low_corr_hz']:g} Hz",
        "relaxed to baseline" if lim < g["low_corr_min"] else "")

    p = _plr(r)
    lim = min(g["plr_min_db"], base.plr - 0.5)
    add("plr", p >= lim, p, f">= {lim:.2f} dB (pre-limiter)",
        "relaxed to baseline" if lim < g["plr_min_db"] else "")

    rel = stem_rel_levels(r)
    worst, worst_name = 0.0, ""
    for name, b in base.stem_rel_lufs.items():
        if not np.isfinite(b):
            continue
        d = abs(rel.get(name, float("-inf")) - b) if np.isfinite(rel.get(name, float("-inf"))) else 999.0
        if d > worst:
            worst, worst_name = d, name
    mix_l = _pre_lufs(r)
    if bus_ref is not None and chain is not None and np.isfinite(mix_l):
        for bname, bus in (chain.get("buses") or {}).items():
            if not bus.get("inputs") or bname not in r.buses:
                continue
            b = bus_ref(tuple(bus["inputs"]))
            l = A.lufs(r.buses[bname], sr)
            if not np.isfinite(b):
                continue
            d = abs(l - mix_l - b) if np.isfinite(l) else 999.0
            if d > worst:
                worst, worst_name = d, f"bus {bname}"
    add("stem_presence", worst <= g["stem_presence_db"], worst,
        f"<= {g['stem_presence_db']} dB from baseline", f"worst: {worst_name}" if worst_name else "")

    clicks = count_clicks(r.pre, sr, g["click_ratio"], g["click_floor"])
    lim = base.clicks + g["click_extra_allowed"]
    add("clicks", clicks <= lim, clicks, f"<= {lim} (baseline {base.clicks})")

    return GuardReport(all(c.passed for c in checks), checks)
