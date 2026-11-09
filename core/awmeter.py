"""Offline port of Airwindows Meter (Hit Record Meter) v0.1.9.

Original: Chris Johnson / Airwindows, MIT License, github.com/airwindows/Meter
(source/PluginProcessor.cpp: the audio-thread analysis; source/AirwindowsUI.cpp + PluginEditor.cpp:
the scoring, which in the plugin lives inside the display code). Ported to Python/numba for
offline use in mixloop; the arithmetic follows the C++ line by line, including C fmin/fmax NaN
semantics, unsigned wrap-around in the bin index, and the bin (index 14) that is filled but never
scored.

What it measures, on the full stereo mix:
  power      ("peaks")        how widely and evenly peak levels spread below 0 dB
  detail     ("slews")        the same for slew rate (sample-to-sample change), i.e. brightness
  authority  ("zero crosses") the same for the longest gap between zero crossings, i.e. bass / low mids
  hit        (letter grade)   running, loudness-weighted balance between those three (0..27.6, A = 27)

Choices the plugin leaves to the host / screen, fixed here:
  - display width 1258 px (the plugin's default 1280x720 window minus its margins); one redraw per
    data point (the plugin redraws on each ~24 Hz data point).
  - the meter is reset at the start of the song and the whole song is played once.

The plugin's power/detail/authority letters are max-hold ("only goes up"), which is useless for
comparing mixes, so we also report their loudness-weighted means over the song (marked *_mean:
our addition, not in the plugin). The `hit` value is the plugin's own running score.
"""
from __future__ import annotations

import math

import numba
import numpy as np

PHI = 1.618033988749894
TOTAL_BINS = 14
DISPLAY_WIDTH = 1258
LETTERS_TOTAL = {i: c for i, c in enumerate("  ZYXWVUTSRQPONMLKJIHGFEDCBA")}
LETTERS_POWER = "qponmlkjihgfedcbaABCDEFGHIJKL"        # power: grade 0..28
LETTERS_AUTHORITY = "ponmlkjihgfedcbaABCDEFGHIJKLM"    # authority: grade 0..28
LETTERS_DETAIL = "tsrqponmlkjihgfedcbaABCDEFGHIJKLM"   # detail: grade 0..32


# --------------------------------------------------------------- C semantics
@numba.njit(cache=True)
def _fmin(a, b):
    if a != a:
        return b
    if b != b:
        return a
    return a if a < b else b


@numba.njit(cache=True)
def _fmax(a, b):
    if a != a:
        return b
    if b != b:
        return a
    return a if a > b else b


@numba.njit(cache=True)
def _sqrt(x):
    return math.sqrt(x) if x >= 0.0 else (x if x == 0.0 else np.nan)


@numba.njit(cache=True)
def _bin(x):
    """(unsigned long)x, then totalBins - fabs((int)bt - totalBins), then the 0 < bt <= 14 test.
    Returns the bin or -1. Negative / NaN x wraps to a huge unsigned value and is skipped."""
    if x != x or x <= -1.0:
        return -1
    bt = int(x) if x >= 0.0 else 0
    m = TOTAL_BINS - abs(bt - TOTAL_BINS)
    if m > 0 and m <= TOTAL_BINS:
        return m
    return -1


# ------------------------------------------------------- audio thread (DSP)
@numba.njit(cache=True)
def _analyze(xl, xr, sr):
    """PluginProcessor::processBlock. Returns (n_points, 18) of the values pushed to the UI:
    PL PR SL SR ZL ZR BSL BSR BRL BRR BZL BZR DSL DSR DRL DRR DZL DZR."""
    n = xl.shape[0]
    window = (1881.0 / 44100.0) * sr
    zcs = (1.0 / sr) * 44100.0
    pos = 1.0 / sr
    wt = 1.618 / sr
    out = np.zeros((int(n / window) + 2, 18))
    k = 0
    # state per channel: [peak, slew, prev, zero, holdZero, holdPeak, longest, bSlew, bZero, bRMS,
    #                     dSlew, dZero, dRMS, wasPositive]
    st = np.zeros((2, 14))
    wc = 0
    for i in range(n):
        for c in range(2):
            s = xl[i] if c == 0 else xr[i]
            S = st[c]
            cs = (abs(s - S[2]) / 32000.0) * sr
            if cs > S[1]:
                S[1] = cs
            S[2] = s
            if abs(S[7] - cs) < 0.2:
                S[7] = S[7] * (1.0 - pos) + cs * pos
            else:
                S[7] = S[7] * (1.0 - pos)
            S[10] = S[10] * (1.0 - wt) + abs(S[7] - cs) * wt
            rect = abs(s)
            if rect > S[0]:
                S[0] = rect
            S[9] = S[9] * (1.0 - pos) + rect * pos
            S[12] = S[12] * (1.0 - wt) + abs(abs(S[9]) - S[5]) * wt
            S[3] += zcs
            if S[6] < S[3]:
                S[6] = S[3]
            if S[13] > 0.5 and s < 0.0:
                S[13] = 0.0
                S[4] = S[3]
                S[3] = 0.0
            elif S[13] < 0.5 and s > 0.0:
                S[13] = 1.0
                S[4] = S[3]
                S[3] = 0.0
            S[8] = S[8] * (1.0 - pos) + S[4] * pos
            S[11] = S[11] * (1.0 - wt) + abs(S[8] - S[4]) * wt
        wc += 1
        if wc > window:
            L, R = st[0], st[1]
            out[k, 0] = math.sqrt(L[0]); out[k, 1] = math.sqrt(R[0])
            out[k, 2] = L[1]; out[k, 3] = R[1]
            out[k, 4] = L[6]; out[k, 5] = R[6]
            out[k, 6] = L[7]; out[k, 7] = R[7]
            out[k, 8] = L[9]; out[k, 9] = R[9]
            out[k, 10] = L[8]; out[k, 11] = R[8]
            out[k, 12] = L[10]; out[k, 13] = R[10]
            out[k, 14] = L[12]; out[k, 15] = R[12]
            out[k, 16] = L[11]; out[k, 17] = R[11]
            k += 1
            for c in range(2):
                S = st[c]
                S[5] = S[0]          # holdPeak = peak
                S[0] = 0.0; S[1] = 0.0; S[3] = 0.0
                S[10] = 0.0; S[12] = 0.0; S[11] = 0.0
                S[6] = 0.0
            wc = 0
    return out[:k]


# ------------------------------------------------------------- UI (scoring)
@numba.njit(cache=True)
def _score(pts, W):
    """PluginEditor::idle (INCREMENT) + AirwindowsMeter::paint, one redraw per data point.
    Returns (summary[12], per-point trace (n, 6): peakScore slewScore bassScore applyCurve vol hue)."""
    n = pts.shape[0]
    D = np.zeros((18, W + 2))
    D[10, :] = 1000.0
    D[11, :] = 1000.0                  # dataBZL/R reset to 1000
    pk = np.zeros(TOTAL_BINS + 2); sl = np.zeros(TOTAL_BINS + 2); bs = np.zeros(TOTAL_BINS + 2)
    p = 0
    peakPrev = 0.0; slewPrev = 0.0; bassPrev = 0.0
    cumulative = 0.0000001; duration = 0.00001
    cumPlus = 0.0000001; durPlus = 0.00001
    oR = 10.0; oG = 10.0; oB = 10.0
    oVol = 0.0
    clip = 180.0
    hue = 0.0
    trace = np.zeros((n, 6))
    for k in range(n):
        # ---- idle(): push the new values, INCREMENT
        for j in range(18):
            D[j, p] = pts[k, j]
            D[j, p + 1] = pts[k, j]
        p += 1
        if p >= W:
            p = 0
        oMax = _fmax(_fmax(oR + 0.000001, oG), oB)
        oMin = _fmin(_fmin(oR, oG), oB) / oMax
        oMin = 1.0 - ((1.0 - oMin) * 0.125 * oVol * oVol)
        oMin = _fmax(_fmin(oMin, 1.0), 0.0)
        sR = (oR / oMax) ** PHI; sG = (oG / oMax) ** PHI; sB = (oB / oMax) ** PHI
        cyan = ((1.0 - sR) * (1.0 - sR) * (1.0 - ((sG - sB) * (sG - sB)))) ** PHI
        sG -= cyan; sB -= cyan
        # backdrop colour -> 8-bit -> JUCE getHue()
        r8 = 0 if sR <= 0 else (255 if sR >= 1 else int(round(sR * 255.0)))
        g8 = 0 if sG <= 0 else (255 if sG >= 1 else int(round(sG * 255.0)))
        b8 = 0 if sB <= 0 else (255 if sB >= 1 else int(round(sB * 255.0)))
        hi = max(r8, g8, b8); lo = min(r8, g8, b8)
        hue = 0.0
        if hi > 0 and hi != lo:
            inv = 1.0 / (hi - lo)
            red = (hi - r8) * inv; green = (hi - g8) * inv; blue = (hi - b8) * inv
            if r8 == hi:
                hue = blue - green
            elif g8 == hi:
                hue = 2.0 + red - blue
            else:
                hue = 4.0 + green - red
            hue /= 6.0
            if hue < 0.0:
                hue += 1.0
        oR *= oMin; oG *= oMin; oB *= oMin
        # ---- paint(): walk the whole display
        cur = p - 1                                   # unsigned: p == 0 -> never matches
        for c in range(W):
            PL = D[0, c]; PR = D[1, c]; SL = D[2, c]; SR = D[3, c]
            peakL = PL * 200.0; peakR = PR * 200.0
            slewL = _sqrt(SL) * 300.0; slewR = _sqrt(SR) * 300.0
            mzL = (_sqrt(D[4, c]) * 6.0) - 6.0
            if mzL > 192.0:
                mzL = 192.0
            bassL = _fmin((_sqrt(mzL) * 16.2) - 20.0, 199.0)
            mzR = (_sqrt(D[5, c]) * 6.0) - 6.0
            if mzR > 192.0:
                mzR = 192.0
            bassR = _fmin((_sqrt(mzR) * 16.2) - 20.0, 199.0)
            beamZ = (_sqrt((D[10, c] + D[11, c]) * 0.5) * 6.0) - 6.0
            beamZero = _fmax(_fmin((_sqrt(beamZ) * 16.2) - 20.0, 199.0), 0.5)
            depZero = _fmin(6.18033988749894, (D[16, c] * D[17, c]) * (23000000.0 / beamZero ** 4))
            depRMS = _fmin(6.18033988749894, (D[14, c] * D[15, c]) * 12000.0)
            beamSlew = _fmin(199.0, _sqrt(D[6, c] + D[7, c]) * 618.033988749894)
            depSlew = _fmin(6.18033988749894 / (1.0 + (_fmax(beamSlew - 186.0, 0.0) * 0.2)),
                            _fmax(0.0, (5.0 - (D[12, c] + D[13, c])) * _sqrt(_fmax(D[6, c], D[7, c])))
                            * 6.18033988749894)
            if beamSlew > 198.0:
                depSlew = 199.0 - beamSlew
            if peakL > 197.0:
                clip *= 0.99
                if peakR > 197.0:
                    clip *= 0.975
            else:
                clip = 180.0
            psL = (PL * 64.0) / (abs((peakL * 0.945433426957143) - slewL) + 6.18033988749894)
            slewDotL = (math.sin(0.1618 / psL) * 6.18 if psL != 0.0 else np.nan) + (_sqrt(slewL) * 0.1618)
            bassDotL = _sqrt(mzL * 0.1 * PL)
            if peakR > 197.0:
                clip *= 0.99
                if peakL > 197.0:
                    clip *= 0.975
            else:
                clip = 180.0
            psR = (PR * 64.0) / (abs((peakR * 0.945433426957143) - slewR) + 6.18033988749894)
            slewDotR = (math.sin(0.1618 / psR) * 6.18 if psR != 0.0 else np.nan) + (_sqrt(slewR) * 0.1618)
            bassDotR = _sqrt(mzR * 0.1 * PR)
            b = _bin((peakL - ((180.0 - clip) * PHI)) * (0.005 * TOTAL_BINS))
            if b >= 0:
                pk[b] += psL
            b = _bin((peakR - ((180.0 - clip) * PHI)) * (0.005 * TOTAL_BINS))
            if b >= 0:
                pk[b] += psR
            b = _bin(slewL * (0.005 * TOTAL_BINS))
            if b >= 0:
                sl[b] += slewDotL
            b = _bin(slewR * (0.005 * TOTAL_BINS))
            if b >= 0:
                sl[b] += slewDotR
            b = _bin(bassL * (0.005 * TOTAL_BINS))
            if b >= 0:
                bs[b] += bassDotL
            b = _bin(bassR * (0.005 * TOTAL_BINS))
            if b >= 0:
                bs[b] += bassDotR
            peakScore = 0.0; slewScore = 0.0; bassScore = 0.0
            for q in range(TOTAL_BINS):
                pk[q] = _fmax(_fmin(pk[q] - 0.1618033988749894, PHI), 0.0)
                s1 = _fmax(math.sin((q / TOTAL_BINS) * (math.pi / 2)), 0.0)
                peakScore += math.sin(pk[q]) * s1
                sl[q] = _fmax(_fmin(sl[q] - 0.1618033988749894, PHI), 0.0)
                slewScore += math.sin(sl[q]) * s1
                bs[q] = _fmax(_fmin(bs[q] - 0.1618033988749894, PHI), 0.0)
                s2 = _fmax(math.sin((q / TOTAL_BINS) * math.pi), 0.0)
                bassScore += math.sin(bs[q]) * s2
            if c == cur:
                oB += peakScore; oR += slewScore; oG += bassScore
                oVol = _sqrt(_fmax(_fmax(PL, PR), _fmax(SL, SR)))
            if c == p:
                ap = _fmin(_fmin(oR, oG), oB) / _fmax(_fmax(oR + 0.000001, oG), oB)
                ap = 1.0 - (1.0 - ap) ** PHI
                cumulative += ap * oVol
                duration += oVol
                trace[k, 3] = ap
                trace[k, 4] = oVol
            if c == cur:
                if peakScore ** PHI > peakPrev:
                    peakPrev = peakPrev * 0.9 + (peakScore ** PHI) * 0.1
                if slewScore ** PHI > slewPrev:
                    slewPrev = slewPrev * 0.9 + (slewScore ** PHI) * 0.1
                if bassScore ** PHI > bassPrev:
                    bassPrev = bassPrev * 0.9 + (bassScore ** PHI) * 0.1
                durPlus += 1.0
                black = math.sqrt(depZero * depRMS * depSlew) * 0.3 if depZero * depRMS * depSlew >= 0 else 0.0
                dA = abs(peakPrev - (slewPrev - 3.0)) + 1.0
                dB = abs(bassPrev - (slewPrev - 4.0)) + 1.0
                dC = abs((peakPrev - 1.0) - bassPrev) + 1.0
                cumPlus += black / math.sqrt(dA * dB * dC)
                if cumPlus > durPlus * PHI:
                    cumPlus = durPlus * PHI
                trace[k, 0] = peakScore; trace[k, 1] = slewScore; trace[k, 2] = bassScore
                trace[k, 5] = hue
    summary = np.array([cumulative, duration, peakPrev, slewPrev, bassPrev, cumPlus, durPlus, hue,
                        0.0, 0.0, 0.0, 0.0])
    return summary, trace


DIRECTIONS = {0: "less bright", **{i: "too much air" for i in range(1, 5)},
              **{i: "allow fullness" for i in range(5, 10)}, **{i: "allow detail" for i in range(10, 14)},
              **{i: "allow activity" for i in range(14, 19)}, **{i: "less boring" for i in range(19, 23)},
              **{i: "less harrowing" for i in range(23, 26)}, **{i: "more calm" for i in range(26, 29)},
              29: "less bright", 30: "less bright"}


def meter(x: np.ndarray, sr: int, width: int = DISPLAY_WIDTH) -> dict:
    """Run the whole song through the meter. x: (channels, n) float, real level (not normalised)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    pts = _analyze(np.ascontiguousarray(x[0]), np.ascontiguousarray(x[1]), float(sr))
    summ, tr = _score(pts, int(min(width, 5150)))
    cumulative, duration, peakPrev, slewPrev, bassPrev, cumPlus, durPlus, hue = summ[:8]
    hit = (1.0 - (1.0 - cumulative / duration) ** PHI) * 27.618033988749894
    hit = float(min(max(hit, 0.0), 27.0)) if np.isfinite(hit) else 0.0
    bonus = 100 - int(min((cumPlus / durPlus) * 100.0, 99.0))
    in_number_zone = not (100 - int((cumPlus / durPlus) * 100.0) > 99 or int(hit) < 27)
    vol = tr[:, 4]
    w = vol / vol.sum() if vol.sum() > 0 else np.full(len(vol), 1 / max(len(vol), 1))
    pg, sg, bg = int(min(max(peakPrev, 0), 28)), int(min(max(slewPrev, 0), 32)), int(min(max(bassPrev, 0), 28))
    # display order: total, authority, power, detail
    letters = (LETTERS_TOTAL.get(int(hit), " ") + LETTERS_AUTHORITY[bg] + LETTERS_POWER[pg]
               + LETTERS_DETAIL[sg]).strip()
    return {
        "hit": hit,                                   # plugin's running balance score, 0..27 (A = 27)
        "grade": (f"{bonus}-" if in_number_zone and 10 < bonus < 100 else "") + letters,
        "power_grade": float(peakPrev), "detail_grade": float(slewPrev), "authority_grade": float(bassPrev),
        "power_mean": float(np.sum(w * tr[:, 0])),    # ours: loudness-weighted mean, not max-hold
        "detail_mean": float(np.sum(w * tr[:, 1])),
        "authority_mean": float(np.sum(w * tr[:, 2])),
        "balance_mean": float(np.sum(w * tr[:, 3])),  # = cumulative/duration before the letter curve
        "directions": "" if in_number_zone else DIRECTIONS.get(int(hue * 30.0), ""),
        "points": int(len(pts)),
    }
