"""Outside ear: a periodic, neutral listen by a second model (opt-in, [audit] enabled = true).

Every `every` rounds, and once per stall of `plateau` rounds without an accept, a different model
(default Gemini 4 Argon) listens to the current best mix `runs` times with the neutral prompt from
core/audit.py: no reference, no system prompt, nothing about what changed. Only MIX problems heard
in every listen are kept, each with measurements for its time range, and handed to the orchestrator
as leads on where to look.

It never scores anything. Audiobox + the re-test still decide every accept, so agents can't win
by pleasing the listener; they can only be pointed at a section the judge wasn't flagging.
"""
from __future__ import annotations

import copy
import json
import re

import numpy as np
from scipy import signal

from . import audio as A
from .audit import OPEN_PROMPT
from .gemini import Listener
from .llm import OpenRouter

KINDS = ["mix", "performance", "arrangement", "recording"]

AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "time": {"type": "string"}, "issue": {"type": "string"},
                "severity": {"type": "string", "enum": ["minor", "moderate", "major"]},
                "kind": {"type": "string", "enum": KINDS,
                         "description": "mix: fixable with levels, automation, EQ, dynamics, effects or "
                                        "panning. The others are in the song or the takes themselves."}},
            "required": ["time", "issue", "severity", "kind"]}},
        "overall": {"type": "string"}},
    "required": ["issues", "overall"],
}

MERGE_SCHEMA = {
    "type": "object",
    "properties": {"issues": {"type": "array", "items": {
        "type": "object",
        "properties": {"time": {"type": "string", "description": "m:ss-m:ss"},
                       "issue": {"type": "string"},
                       "severity": {"type": "string", "enum": ["minor", "moderate", "major"]},
                       "kind": {"type": "string", "enum": KINDS}},
        "required": ["time", "issue", "severity", "kind"]}}},
    "required": ["issues"],
}

MERGE_SYSTEM = ("You merge independent listening reports on the SAME mix. Keep only problems that EVERY "
                "report describes: the same problem (same element, same kind of fault) at overlapping "
                "times. Drop anything that only some reports mention. For each kept problem give the "
                "overlapping time range, one plain sentence combining the reports' wording (add nothing "
                "they did not say), the lower severity, and its kind (mix only if levels, automation, EQ, "
                "dynamics, effects or panning could fix it).")


def audit_listener(cfg: dict, client: OpenRouter) -> Listener:
    """Same Listener plumbing, but the audit model, full-band and stereo."""
    a = cfg.get("audit", {})
    c = copy.deepcopy(cfg)
    c["models"]["listener"] = a.get("model", "google/gemini-4-argon")
    c["models"]["audio_sr"] = int(a.get("audio_sr", 44100))
    c["models"]["audio_channels"] = int(a.get("channels", 2))
    return Listener(c, client, None)


def listen_once(lis: Listener, x: np.ndarray, sr: int, tag: str) -> dict:
    parts = lis._parts(x, sr)
    content = [{"type": "text", "text": OPEN_PROMPT}] + lis._audio_content("Song", parts, x.shape[1] / sr)
    out, _ = lis.client.chat(lis.model, [{"role": "user", "content": content}], schema=AUDIT_SCHEMA,
                             max_tokens=4000, tag=tag)
    return out


def parse_range(text: str, total: float) -> tuple[float, float]:
    """'1:56-2:43' -> (116, 163); one stamp -> +/-3 s; none -> the whole song."""
    ts = [int(m) * 60 + int(s) for m, s in re.findall(r"(\d+):(\d{2})", text or "")][:2]
    if not ts:
        return 0.0, total
    if len(ts) == 1 or ts[0] == ts[-1]:
        return max(0.0, ts[0] - 3.0), min(total, ts[0] + 3.0)
    a, b = min(ts), max(ts)
    return float(a), float(min(total, max(b, a + 3.0)))


def mmss(t: float) -> str:
    t = int(round(t))
    return f"{t // 60}:{t % 60:02d}"


def _bands(x: np.ndarray, sr: int) -> np.ndarray:
    m = x.mean(axis=0).astype(np.float64)
    f, p = signal.welch(m, sr, nperseg=min(4096, len(m)))
    e = np.array([p[(f >= lo) & (f < hi)].sum() for lo, hi in ((20, 250), (250, 4000), (4000, 20000))])
    return e / max(e.sum(), 1e-20)


def _r(v) -> float | None:
    return round(float(v), 1) if np.isfinite(v) else None


def evidence(ev, best: dict, flat: dict, t0: float, t1: float) -> dict:
    """Measurements for one time range, best vs flat, each relative to its own mix in that range.
    Track levels are post-track-chain, pre-bus; bus levels are listed separately."""
    sr = ev.sr
    rb, rf = ev.render(best), ev.render(flat)
    s, e = int(t0 * sr), int(t1 * sr)
    lb, lf = A.lufs(rb.pre, sr), A.lufs(rf.pre, sr)
    rlb, rlf = A.lufs(rb.pre[:, s:e], sr), A.lufs(rf.pre[:, s:e], sr)
    tracks = []
    for name in rb.stems:
        b = A.lufs(rb.stems[name][:, s:e], sr) - rlb
        f = A.lufs(rf.stems[name][:, s:e], sr) - rlf
        if np.isfinite(b) and b > -40:
            tracks.append({"track": name, "level_vs_mix_here_dB": _r(b),
                           "change_vs_flat_mix_dB": _r(b - f) if np.isfinite(f) else None})
    loud = sorted(tracks, key=lambda d: -d["level_vs_mix_here_dB"])[:4]
    moved = sorted([d for d in tracks if d["change_vs_flat_mix_dB"] is not None],
                   key=lambda d: -abs(d["change_vs_flat_mix_dB"]))[:4]
    out = {"range": f"{mmss(t0)}-{mmss(t1)}",
           "range_loudness_vs_whole_song_LU": {"best": _r(rlb - lb), "flat": _r(rlf - lf)},
           "loudest_tracks_here": loud, "tracks_moved_most_vs_flat_here": moved}
    bb, bs = _bands(rb.pre[:, s:e], sr), _bands(rb.pre, sr)
    out["band_energy_vs_whole_song_dB"] = {
        k: _r(10 * np.log10(max(x, 1e-12) / max(y, 1e-12)))
        for k, x, y in zip(("<250Hz", "250Hz-4k", ">4k"), bb, bs)}
    buses = {n: _r(A.lufs(y[:, s:e], sr) - rlb) for n, y in (rb.buses or {}).items()}
    buses = {k: v for k, v in buses.items() if v is not None}
    if buses:
        out["bus_level_vs_mix_here_dB"] = buses
    return out


def merge(client: OpenRouter, cfg: dict, runs: list[dict], tag: str) -> list[dict]:
    reports = {f"report_{i + 1}": r.get("issues", []) for i, r in enumerate(runs)}
    out, _ = client.chat(cfg["models"]["orchestrator"],
                         [{"role": "system", "content": MERGE_SYSTEM},
                          {"role": "user", "content": json.dumps(reports, indent=1)}],
                         schema=MERGE_SCHEMA, max_tokens=3000, tag=tag)
    return out.get("issues", [])


def outside_ear(ev, client: OpenRouter, cfg: dict, best: dict, flat: dict, rnd: int) -> dict:
    """Listen `runs` times to the loudness-matched best mix; keep repeated mix problems + numbers."""
    a = cfg.get("audit", {})
    lis = audit_listener(cfg, client)
    sr = ev.sr
    x = ev.matched(best)                      # final render at -14 LUFS, as the judge hears it
    total = x.shape[1] / sr
    runs = []
    for i in range(int(a.get("runs", 2))):
        try:
            runs.append(listen_once(lis, x, sr, tag=f"audit:r{rnd}:{i + 1}"))
        except Exception as e:  # noqa: BLE001 - an audit must never stop the loop
            print(f"  outside ear: listen {i + 1} failed ({e})")
    base = {"round": rnd, "model": lis.model,
            "runs": [{"overall": r.get("overall"), "issues": r.get("issues", [])} for r in runs]}
    if len(runs) < 2:
        return {**base, "error": "fewer than 2 listens succeeded", "issues": [], "dropped": []}
    try:
        repeated = merge(client, cfg, runs, tag=f"audit_merge:r{rnd}")
    except Exception as e:  # noqa: BLE001
        print(f"  outside ear: merge failed ({e})")
        return {**base, "error": str(e), "issues": [], "dropped": []}
    kept, dropped = [], []
    for it in repeated:
        if it.get("kind") != "mix":
            dropped.append({**it, "why_dropped": f"not a mix problem ({it.get('kind')})"})
            continue
        t0, t1 = parse_range(it.get("time", ""), total)
        kept.append({**it, "measured": evidence(ev, best, flat, t0, t1)})
    return {**base, "issues": kept, "dropped": dropped}
