"""Measurement tools the agents call (the "meters" of the agentic DAW). Plain DSP, no LLM.

    song_info()                      length, tempo estimate, every track: channels, level, where it plays
    levels(track, start, end)        phrase-by-phrase loudness of the raw track, and how loud the track
                                     sits against the current mix in each phrase
    pitch(track, start, end)         what is sung/played: notes with times and cents off, pitch-class
                                     time histogram, key candidates (the agent decides the key)
    spectrum(target, start, end)     octave-band levels of a track / bus / the mix, and the target's
                                     share of the mix per band (masking)
    judge_view(track)                the Audiobox diagnosis entries that concern this target

All times are seconds from the start of the song.
"""
from __future__ import annotations

import functools

import numpy as np
import pyloudnorm as pyln
from scipy import signal

from . import audio as A
from .config import STOCK_DIR
from .plugins import load_module

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Kessler key profiles
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
OCTAVES = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def mmss(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t) // 60}:{t % 60:04.1f}"


def _span(x: np.ndarray, sr: int, start, end) -> tuple[np.ndarray, float]:
    n = x.shape[1]
    s = 0 if start is None else int(max(0.0, float(start)) * sr)
    e = n if end is None else int(min(n / sr, float(end)) * sr)
    s = min(max(0, s), n)
    e = min(max(s, e), n)
    return x[:, s:e], s / sr


@functools.lru_cache(maxsize=4)
def _kfilters(sr: int):
    return list(pyln.Meter(sr)._filters.values())


def k_weighted_ms(x: np.ndarray, sr: int, block_s: float) -> np.ndarray:
    """Mean-square of the K-weighted signal (summed over channels) per block -> LUFS-like levels."""
    y = np.asarray(x, np.float64)
    out = []
    for ch in y:
        c = ch
        for f in _kfilters(sr):
            c = f.apply_filter(c)
        out.append(c)
    y = np.array(out)
    b = max(1, int(block_s * sr))
    nb = y.shape[1] // b
    if nb == 0:
        return np.zeros(0)
    ms = (y[:, : nb * b].reshape(y.shape[0], nb, b) ** 2).mean(axis=2).sum(axis=0)
    return ms


def _lufs_blocks(x, sr, block_s):
    ms = k_weighted_ms(x, sr, block_s)
    return -0.691 + 10 * np.log10(ms + 1e-20)


def _phrases(active: np.ndarray, block_s: float, gap_blocks: int = 1) -> list[tuple[int, int]]:
    """Runs of active blocks, bridging gaps of <= gap_blocks."""
    runs, s, last = [], None, None
    for i, a in enumerate(active):
        if a:
            if s is None:
                s = i
            elif last is not None and i - last > gap_blocks + 1:
                runs.append((s, last + 1))
                s = i
            last = i
    if s is not None:
        runs.append((s, last + 1))
    return runs


# ------------------------------------------------------------------ tools

def song_info(ev, cards: dict | None = None) -> dict:
    sr = ev.sr
    n = ev.renderer.n
    tracks = {}
    for name, x in ev.stems.items():
        lv = _lufs_blocks(x, sr, 1.0)
        act = lv > -60
        on = np.nonzero(act)[0]
        tracks[name] = {
            "channels": int(x.shape[0]),
            "plays_pct": round(100 * float(act.mean()), 1) if len(act) else 0.0,
            "first_last": f"{mmss(on[0])}-{mmss(on[-1] + 1)}" if len(on) else "silent",
            "level_lufs_when_playing": round(float(np.median(lv[act])), 1) if act.any() else None,
            "description": (cards or {}).get(name, {}).get("role") if cards else None,
        }
    return {"length_s": round(n / sr, 2), "sample_rate": sr, "tempo_bpm_estimate": tempo_estimate(ev),
            "tracks": tracks}


def tempo_estimate(ev) -> float | None:
    """Onset-envelope autocorrelation on the flat sum of everything (rough, +/- octave errors)."""
    try:
        sr = ev.sr
        mix = sum(A.to_stereo(x) for x in ev.stems.values()).mean(axis=0)
        hop = 512
        nb = len(mix) // hop
        e = np.sqrt((mix[: nb * hop].reshape(nb, hop) ** 2).mean(axis=1) + 1e-12)
        onset = np.maximum(np.diff(np.log(e)), 0)
        onset -= onset.mean()
        fps = sr / hop
        ac = np.correlate(onset, onset, "full")[len(onset) - 1:]
        lags = np.arange(len(ac)) / fps
        ok = (lags >= 60 / 180) & (lags <= 60 / 70)
        if not ok.any():
            return None
        lag = lags[ok][int(np.argmax(ac[ok]))]
        return round(60.0 / lag, 1)
    except Exception:  # noqa: BLE001
        return None


def levels(ev, render, track: str, start=None, end=None, block_s: float = 0.5) -> dict:
    """Phrase-by-phrase loudness of the raw track and of the track inside the current mix."""
    sr = ev.sr
    raw, t0 = _span(ev.stems[track], sr, start, end)
    if track in render.stems:
        out, _ = _span(render.stems[track], sr, start, end)
    else:
        out = raw
    mix, _ = _span(render.pre, sr, start, end)
    lr = _lufs_blocks(raw, sr, block_s)
    lo = _lufs_blocks(out, sr, block_s)
    lm = _lufs_blocks(mix, sr, block_s)
    act = lr > -60
    runs = _phrases(act, block_s)
    phrases = []
    for a, b in runs:
        seg_r = lr[a:b][act[a:b]]
        seg_o = lo[a:b][act[a:b]]
        seg_m = lm[a:b][act[a:b]]
        if not len(seg_r):
            continue
        phrases.append({
            "time": f"{mmss(t0 + a * block_s)}-{mmss(t0 + b * block_s)}",
            "raw_lufs": round(float(np.median(seg_r)), 1),
            "raw_range": f"{float(np.min(seg_r)):.1f}..{float(np.max(seg_r)):.1f}",
            "in_mix_vs_mix_db": round(float(np.median(seg_o - seg_m)), 1),
        })
    meds = [p["raw_lufs"] for p in phrases]
    res = {"track": track, "block_s": block_s, "phrases": phrases[:60],
           "phrase_level_spread_db": round(max(meds) - min(meds), 1) if meds else 0.0,
           "note": "raw_lufs = the file as recorded; in_mix_vs_mix_db = this track's level after its "
                   "chain minus the whole mix's level over the same blocks (0 = as loud as the mix)."}
    if len(phrases) > 60:
        res["truncated"] = f"{len(phrases)} phrases; ask for a shorter start/end range for the rest"
    return res


@functools.lru_cache(maxsize=1)
def _tuner():
    return load_module(STOCK_DIR / "tune.py")


def pitch(ev, track: str, start=None, end=None, min_hz: float = 60.0, max_hz: float = 1100.0) -> dict:
    """Notes (with cents off equal temperament), pitch-class histogram and key candidates."""
    sr = ev.sr
    x, t0 = _span(ev.stems[track], sr, start, end)
    mono = x.mean(axis=0)
    T = _tuner()
    hop = max(1, int(round(sr * 0.0058)))
    f0, ap, centres = T.yin(mono, sr, min_hz, max_hz, 0.15, hop)
    v = f0 > 0
    if len(f0) == 0 or v.mean() < 0.02:
        return {"track": track, "voiced_pct": round(100 * float(v.mean()) if len(v) else 0.0, 1),
                "note": "almost no stable pitch: not a monophonic pitched source (or silent here)"}
    midi = np.where(v, 69 + 12 * np.log2(np.where(v, f0, 440.0) / 440.0), np.nan)
    # segment into notes: voiced runs, split where the pitch moves > 60 cents from the running note
    notes = []
    cur = []
    for i in range(len(midi)):
        m = midi[i]
        if np.isnan(m):
            if len(cur) >= 4:
                notes.append(cur)
            cur = []
            continue
        if cur and abs(m - np.median([midi[j] for j in cur[-8:]])) > 0.6:
            if len(cur) >= 4:
                notes.append(cur)
            cur = []
        cur.append(i)
    if len(cur) >= 4:
        notes.append(cur)
    hist = np.zeros(12)
    out_notes = []
    offs = []
    for nt in notes:
        ms = midi[nt]
        med = float(np.median(ms))
        q = int(round(med))
        cents = (med - q) * 100
        dur = len(nt) * hop / sr
        hist[q % 12] += dur
        offs.append((abs(cents), dur))
        out_notes.append({"t": round(t0 + centres[nt[0]] / sr, 2), "dur": round(dur, 2),
                          "note": f"{NOTE_NAMES[q % 12]}{q // 12 - 1}", "cents": round(cents, 1),
                          "wobble_cents": round(float(np.std(ms) * 100), 1)})
    keys = []
    if hist.sum() > 0:
        h = hist / hist.sum()
        for tonic in range(12):
            for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
                r = float(np.corrcoef(h, np.roll(prof, tonic))[0, 1])
                keys.append((r, f"{NOTE_NAMES[tonic]} {mode}"))
        keys.sort(reverse=True)
    w = sum(d for _, d in offs) or 1.0
    res = {
        "track": track,
        "voiced_pct": round(100 * float(v.mean()), 1),
        "range": f"{NOTE_NAMES[int(round(np.nanmin(midi))) % 12]}{int(round(np.nanmin(midi))) // 12 - 1} - "
                 f"{NOTE_NAMES[int(round(np.nanmax(midi))) % 12]}{int(round(np.nanmax(midi))) // 12 - 1}",
        "pitch_class_seconds": {NOTE_NAMES[i]: round(float(hist[i]), 2) for i in range(12) if hist[i] > 0},
        "key_candidates": [{"key": k, "r": round(r, 3)} for r, k in keys[:5]],
        "mean_abs_cents_off": round(sum(c * d for c, d in offs) / w, 1),
        "pct_time_over_20_cents_off": round(100 * sum(d for c, d in offs if c > 20) / w, 1),
        "notes": out_notes[:80],
        "note": "cents = how far each note's median pitch is from the nearest equal-tempered note "
                "(+ sharp, - flat). wobble = pitch movement inside the note (vibrato/scoops). "
                "Key candidates are a statistical guess from note durations; you decide the key.",
    }
    if len(out_notes) > 80:
        res["truncated"] = f"{len(out_notes)} notes; ask for a shorter start/end range for the rest"
    return res


def _band_levels(x: np.ndarray, sr: int) -> list[float]:
    m = np.asarray(x, np.float64).mean(axis=0)
    if len(m) < 2048:
        return [float("-inf")] * len(OCTAVES)
    f, p = signal.welch(m, sr, nperseg=8192)
    out = []
    for c in OCTAVES:
        lo, hi = c / np.sqrt(2), min(c * np.sqrt(2), sr / 2)
        band = (f >= lo) & (f < hi)
        out.append(10 * np.log10(p[band].sum() + 1e-20) if band.any() else float("-inf"))
    return out


def spectrum(ev, render, target: str, start=None, end=None) -> dict:
    sr = ev.sr
    mix, _ = _span(render.pre, sr, start, end)
    if target == "mix":
        x = mix
    elif target in render.stems:
        x, _ = _span(render.stems[target], sr, start, end)
    elif target.startswith("bus:") and target[4:] in render.buses:
        x, _ = _span(render.buses[target[4:]], sr, start, end)
    else:
        return {"error": f"unknown target {target!r}: use a track name, 'bus:<name>' or 'mix'"}
    bt, bm = _band_levels(x, sr), _band_levels(mix, sr)
    ref = max(bt) if np.isfinite(max(bt)) else 0.0
    return {"target": target, "bands_hz": OCTAVES,
            "target_db_rel_its_loudest_band": [round(v - ref, 1) if np.isfinite(v) else None for v in bt],
            "share_of_mix_db": [round(a - b, 1) if np.isfinite(a) and np.isfinite(b) else None
                                for a, b in zip(bt, bm)],
            "note": "share_of_mix_db near 0 = this target dominates that band; -20 = minor part of it."}


def judge_view(diagnosis: dict | None, target: str) -> dict:
    if not diagnosis:
        return {"note": "no diagnosis yet"}
    out = {"score": diagnosis.get("score"), "weakest_windows": diagnosis.get("weakest_windows"),
           "timeline": diagnosis.get("timeline"), "hints": diagnosis.get("hints"),
           "step_db": diagnosis.get("step_db")}
    st = (diagnosis.get("stems") or {}).get(target)
    if st:
        out["this_track"] = st
    out["note"] = ("Measured by the judge on the current best mix: score change with this track "
                   f"-/+{diagnosis.get('step_db', 3)} dB overall and in the weakest windows, and its solo quality.")
    return out
