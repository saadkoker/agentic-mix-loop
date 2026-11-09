"""Stock vocal tuner (pitch correction) for monophonic sources: lead vocals, doubles, bass lines.

How it works:
1. Pitch tracking (YIN): f0 every ~6 ms, plus a voiced / unvoiced decision. Unvoiced sound
   (breaths, consonants, noise) is never touched.
2. Target: each voiced frame's nearest note in the chosen key + scale. The correction is
   strength x (target - sung pitch), smoothed with a one-pole filter of retune_ms (0 = hard
   "auto-tune" snap, 50-150 ms = natural, keeps scoops and vibrato).
3. Resynthesis (TD-PSOLA): pitch-synchronous grains are re-spaced to the corrected period.
   Formants are preserved (no chipmunk effect for small shifts). Timing is unchanged.
A note that is already in tune passes through bit-exact (grains are put back where they came
from), so the tuner only changes what it corrects.

Params:
  key        0..11  tonic: 0=C 1=C# 2=D 3=D# 4=E 5=F 6=F# 7=G 8=G# 9=A 10=A# 11=B
  scale      0..7   0 chromatic, 1 major, 2 natural minor, 3 harmonic minor, 4 major pentatonic,
                    5 minor pentatonic, 6 dorian, 7 mixolydian
  strength   0..1   fraction of the way to the target note
  retune_ms  0..400 how fast the correction follows (lower = harder)
  min_hz / max_hz   expected pitch range of the source
  threshold  0.05..0.5  YIN voicing threshold (lower = stricter about what counts as pitched)
  tolerance_cents 0..50  deviations smaller than this are left alone
"""
import math

import numpy as np
from scipy import signal

TYPE = "pitch"
PARAMS = {
    "key":             {"default": 0.0,   "min": 0.0,  "max": 11.0},
    "scale":           {"default": 0.0,   "min": 0.0,  "max": 7.0},
    "strength":        {"default": 0.8,   "min": 0.0,  "max": 1.0},
    "retune_ms":       {"default": 60.0,  "min": 0.0,  "max": 400.0},
    "min_hz":          {"default": 70.0,  "min": 40.0, "max": 400.0},
    "max_hz":          {"default": 1000.0, "min": 200.0, "max": 1600.0},
    "threshold":       {"default": 0.15,  "min": 0.05, "max": 0.5},
    "tolerance_cents": {"default": 5.0,   "min": 0.0,  "max": 50.0},
}

SCALES = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [0, 2, 4, 5, 7, 9, 11],
    [0, 2, 3, 5, 7, 8, 10],
    [0, 2, 3, 5, 7, 8, 11],
    [0, 2, 4, 7, 9],
    [0, 3, 5, 7, 10],
    [0, 2, 3, 5, 7, 9, 10],
    [0, 2, 4, 5, 7, 9, 10],
]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def yin(mono, sr, min_hz=70.0, max_hz=1000.0, threshold=0.15, hop=None):
    """f0 (Hz, 0 = unvoiced), aperiodicity and frame centres (samples) for a mono signal."""
    x = np.asarray(mono, dtype=np.float64)
    hop = hop or max(1, int(round(sr * 0.0058)))
    tau_max = min(int(sr / min_hz) + 2, 4096)
    tau_min = max(2, int(sr / max_hz))
    W = tau_max                               # integration window
    L = W + tau_max                           # frame length
    n = len(x)
    if n < L:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
    starts = np.arange(0, n - L + 1, hop)
    f0 = np.zeros(len(starts))
    ap = np.ones(len(starts))
    nfft = 1
    while nfft < 2 * L:
        nfft *= 2
    gate = 10 ** (-55 / 20)
    taus = np.arange(tau_max + 1)
    chunk = 512
    for c0 in range(0, len(starts), chunk):
        st = starts[c0:c0 + chunk]
        idx = st[:, None] + np.arange(L)[None, :]
        fr = x[idx]
        a = fr[:, :W]
        r = np.fft.irfft(np.fft.rfft(fr, nfft, axis=1) * np.conj(np.fft.rfft(a, nfft, axis=1)),
                         nfft, axis=1)[:, : tau_max + 1]
        e1 = np.sum(a * a, axis=1)
        cs = np.concatenate([np.zeros((len(st), 1)), np.cumsum(fr * fr, axis=1)], axis=1)
        e2 = cs[:, taus + W] - cs[:, taus]
        d = e1[:, None] + e2 - 2.0 * r
        d[:, 0] = 0.0
        cum = np.cumsum(d[:, 1:], axis=1)
        dn = np.ones_like(d)
        dn[:, 1:] = d[:, 1:] * np.arange(1, tau_max + 1)[None, :] / np.maximum(cum, 1e-20)
        rms = np.sqrt(e1 / W)
        for i in range(len(st)):
            if rms[i] < gate:
                continue
            row = dn[i]
            seg = row[tau_min: tau_max]
            below = np.nonzero(seg < threshold)[0]
            if len(below):
                t = below[0] + tau_min
                while t + 1 < tau_max and row[t + 1] < row[t]:
                    t += 1
            else:
                continue
            if 1 <= t < tau_max:
                y0, y1, y2 = row[t - 1], row[t], row[t + 1]
                den = y0 - 2 * y1 + y2
                off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
                tt = t + max(-1.0, min(1.0, off))
            else:
                tt = float(t)
            f0[c0 + i] = sr / tt
            ap[c0 + i] = row[t]
    centres = starts + W // 2
    return f0, ap, centres


def _target_cents(f0, key, scale):
    """Cents correction (target - sung) for each voiced frame."""
    pcs = SCALES[scale]
    out = np.zeros(len(f0))
    v = f0 > 0
    midi = 69.0 + 12.0 * np.log2(np.where(v, f0, 440.0) / 440.0)
    best = np.full(len(f0), np.inf)
    base = np.floor(midi)
    for off in range(-7, 8):
        cand = base + off
        ok = np.isin(((cand - key) % 12).astype(int), pcs)
        dist = cand - midi
        better = ok & (np.abs(dist) < np.abs(best))
        best = np.where(better, dist, best)
    out[v] = 100.0 * best[v]
    return out


def correction_curve(f0, sr, hop, key, scale, strength, retune_ms, tolerance):
    corr = _target_cents(f0, key, scale) * strength
    corr[np.abs(corr) < tolerance * strength] = 0.0
    alpha = 1.0 if retune_ms <= 0 else 1.0 - math.exp(-hop / (retune_ms / 1000.0 * sr))
    out = np.zeros(len(f0))
    prev_v = False
    state = 0.0
    for i in range(len(f0)):
        if f0[i] <= 0:
            prev_v = False
            continue
        if not prev_v:
            state = corr[i]                  # start each phrase on its target: no swoop in
        else:
            state += alpha * (corr[i] - state)
        out[i] = state
        prev_v = True
    return out


def _marks(mono, sr, f0_s, unv_step):
    """Analysis marks: pitch-synchronous where voiced (refined to waveform peaks), fixed
    spacing elsewhere. Returns marks, half-window lengths, exact periods, voiced flags."""
    n = len(mono)
    sos = signal.butter(2, 900.0, "lowpass", fs=sr, output="sos")
    lp = signal.sosfiltfilt(sos, mono)
    marks, halfs, periods, voiced = [], [], [], []
    m = 0
    while m < n:
        f = f0_s[m]
        if f > 0:
            T = int(round(sr / f))
            marks.append(m); halfs.append(T); periods.append(sr / f); voiced.append(True)
            nxt = m + T
            lo, hi = nxt - T // 3, nxt + T // 3
            if hi < n and lo > m:
                seg = np.abs(lp[lo:hi])
                nxt = lo + int(np.argmax(seg))
            m = max(nxt, m + 1)
        else:
            marks.append(m); halfs.append(unv_step); periods.append(float(unv_step)); voiced.append(False)
            m += unv_step
    return np.array(marks), np.array(halfs), np.array(periods), np.array(voiced)


def process(audio, sr, params):
    p = params
    x = np.asarray(audio, dtype=np.float32)
    n = x.shape[1]
    mono = x.mean(axis=0).astype(np.float64)
    hop = max(1, int(round(sr * 0.0058)))
    f0, ap, centres = yin(mono, sr, p["min_hz"], max(p["max_hz"], p["min_hz"] + 50), p["threshold"], hop)
    if len(f0) == 0 or not np.any(f0 > 0) or p["strength"] <= 0:
        return x.copy()
    key = int(round(p["key"])) % 12
    scale = int(round(min(max(p["scale"], 0), 7)))
    corr = correction_curve(f0, sr, hop, key, scale, p["strength"], p["retune_ms"], p["tolerance_cents"])
    # per-sample f0 (0 where unvoiced) and ratio
    voiced_fr = f0 > 0
    samp = np.arange(n)
    vf = np.interp(samp, centres, voiced_fr.astype(np.float64)) > 0.5
    f0_s = np.where(vf, np.interp(samp, centres, np.where(voiced_fr, f0, 0.0)), 0.0)
    f0_s = np.where(f0_s > 0, f0_s, 0.0)
    ratio_s = np.where(vf, np.power(2.0, np.interp(samp, centres, corr) / 1200.0), 1.0)
    if np.max(np.abs(ratio_s - 1.0)) < 1e-4:
        return x.copy()
    unv = max(16, int(round(sr * 0.005)))
    marks, halfs, periods, voiced = _marks(mono, sr, f0_s, unv)
    out = np.zeros((x.shape[0], n), np.float64)
    wsum = np.zeros(n, np.float64)
    ts = float(marks[0])
    k = 0
    K = len(marks)
    win_cache = {}
    while ts < n and k < K:
        # nearest analysis mark at or around ts
        k = int(np.searchsorted(marks, ts))
        if k >= K:
            k = K - 1
        if k > 0 and abs(marks[k - 1] - ts) < abs(marks[k] - ts):
            k -= 1
        r = ratio_s[min(int(ts), n - 1)]
        if not voiced[k] or abs(r - 1.0) < 1e-4:
            ts = float(marks[k])              # in tune / unvoiced: put the grain back in place
        P = int(halfs[k])
        m = int(marks[k])
        a0, a1 = max(0, m - P), min(n, m + P + 1)
        c = int(round(ts))
        b0 = c - (m - a0)
        b1 = b0 + (a1 - a0)
        if b0 < 0:
            a0 -= b0; b0 = 0
        if b1 > n:
            a1 -= b1 - n; b1 = n
        if a1 > a0:
            w = win_cache.get(P)
            if w is None:
                w = np.hanning(2 * P + 3)[1:-1]
                win_cache[P] = w
            ws = w[(a0 - (m - P)):(a1 - (m - P))]
            out[:, b0:b1] += x[:, a0:a1] * ws[None, :]
            wsum[b0:b1] += ws
        step = periods[k] / r if (voiced[k] and abs(r - 1.0) >= 1e-4) else None
        if step is None:
            if k + 1 >= K:
                break
            ts = float(marks[k + 1])
        else:
            ts += max(step, 1.0)
    good = wsum > 1e-3
    res = x.astype(np.float64).copy()
    res[:, good] = out[:, good] / wsum[good]
    return res.astype(np.float32)
