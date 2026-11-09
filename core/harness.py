"""Plugin test harness. Runs before any plugin touches a real stem.

1. Static check: AST scan. Only numpy, scipy.signal, pedalboard, math may be
   imported; no file / network / subprocess / plugin-loading / dunder access.
2. Signal tests: silence, impulse, 20 Hz-20 kHz sweep, pink noise, at default
   and extreme params, mono and stereo.
3. Must pass: same shape; finite; silence stays < -90 dBFS; gain within
   +/-24 dB; latency < 5 ms; >= 2x faster than real time.
4. Type check: eq / gain_pan must be linear and add no harmonics; comp must
   never expand; sat / clip must not alias (i.e. must oversample); sat / clip
   must not add DC; pitch must leave an in-tune note alone (and, if it has the
   stock tuner's knobs, pull a detuned note to pitch); gate must never boost;
   space (reverb / delay) is exempt from the latency rule (pre-delay is the point).

Tests 2-4 run in a separate `python -I` process with a timeout, so a plugin
that hangs or crashes can't take the loop down with it. The static check is a
guard against accidents and lazy shortcuts, not a security sandbox: only run
agent code you're comfortable running on this machine.
"""
from __future__ import annotations

import ast
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
    "memoryview", "__builtins__", "print",
}
BANNED_ATTRS = {
    # numpy file / ctypes access
    "load", "save", "savez", "savez_compressed", "fromfile", "tofile", "loadtxt",
    "savetxt", "genfromtxt", "memmap", "ctypeslib", "DataSource", "lib", "f2py",
    "distutils", "testing", "ctypes", "fromregex", "dump", "dumps",
    # pedalboard file / external plugin access
    "io", "load_plugin", "VST3Plugin", "AudioUnitPlugin", "ExternalPlugin",
    "AudioFile", "AudioStream", "ReadableAudioFile", "WriteableAudioFile", "Convolution",
    "system", "popen",
}


def static_check(source: str, allowed: list[str]) -> list[str]:
    errs = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error line {e.lineno}: {e.msg}"]

    def mod_ok(name: str) -> bool:
        if name == "numpy" or name.startswith("numpy."):
            return "numpy" in allowed
        return name in allowed

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if not mod_ok(a.name):
                    errs.append(f"line {node.lineno}: import {a.name} not allowed")
        elif isinstance(node, ast.ImportFrom):
            m = node.module or ""
            if node.level:
                errs.append(f"line {node.lineno}: relative imports not allowed")
            for a in node.names:
                full = f"{m}.{a.name}"
                if not (mod_ok(m) or mod_ok(full)):
                    errs.append(f"line {node.lineno}: from {m} import {a.name} not allowed")
                if a.name in BANNED_ATTRS or a.name == "*":
                    errs.append(f"line {node.lineno}: importing {a.name} not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errs.append(f"line {node.lineno}: use of {node.id} not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in BANNED_ATTRS:
                errs.append(f"line {node.lineno}: attribute .{node.attr} not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errs.append(f"line {node.lineno}: global/nonlocal not allowed")
    names = {n.targets[0].id for n in tree.body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for need in ("TYPE", "PARAMS"):
        if need not in names:
            errs.append(f"missing top-level {need} = ...")
    if "process" not in funcs:
        errs.append("missing top-level def process(audio, sr, params)")
    return errs


# ----------------------------------------------------------------- signals

SR = 44100


def _pink(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = np.fft.rfft(w) / np.sqrt(np.maximum(f, 20.0))
    x = np.fft.irfft(spec, n)
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (-18 / 20)).astype(np.float32)


def signals(channels: int) -> dict[str, np.ndarray]:
    n = 2 * SR
    t = np.arange(n) / SR
    imp = np.zeros(n, np.float32)
    imp[SR // 2] = 0.5
    k = math.log(20000 / 20)
    sweep = 0.5 * np.sin(2 * np.pi * 20 * 2.0 / k * (np.exp(t / 2.0 * k) - 1))
    out = {"silence": np.zeros(n, np.float32), "impulse": imp,
           "sweep": sweep.astype(np.float32), "pink": _pink(n)}
    res = {}
    for k_, v in out.items():
        if channels == 1:
            res[k_] = v[None, :].copy()
        else:
            other = _pink(n, seed=7) if k_ == "pink" else v
            res[k_] = np.stack([v, other]).astype(np.float32)
    return res


def param_sets(mod, max_sets: int = 24, seed: int = 0) -> list[tuple[str, dict]]:
    P = mod.PARAMS
    d = {k: float(v["default"]) for k, v in P.items()}
    sets = [("default", d), ("all_min", {k: float(v["min"]) for k, v in P.items()}),
            ("all_max", {k: float(v["max"]) for k, v in P.items()})]
    extremes = []
    for k, v in P.items():
        for side in ("min", "max"):
            extremes.append((f"{k}={side}", {**d, k: float(v[side])}))
    if len(extremes) > max_sets - 3:
        random.Random(seed).shuffle(extremes)
        extremes = extremes[: max_sets - 3]
    return sets + extremes


def _rms_db(x):
    return 10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-30)


def _energy_db(x):
    """Total power summed over channels (pan-law neutral)."""
    x = np.asarray(x, np.float64)
    return 10 * np.log10(np.sum(x ** 2) / x.shape[-1] + 1e-30)


def _harmonic_split(y: np.ndarray, f0: float):
    """Energy at the fundamental, at its in-band harmonics, and elsewhere."""
    y = y[: , SR // 2:].mean(axis=0).astype(np.float64)
    w = np.hanning(len(y))
    Y = np.abs(np.fft.rfft(y * w)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    near = lambda c: np.abs(f - c) < 40  # noqa: E731
    fund = Y[near(f0)].sum()
    harm_mask = np.zeros_like(f, bool)
    for k in range(2, int(SR / 2 / f0) + 1):
        harm_mask |= near(k * f0)
    harm = Y[harm_mask].sum()
    other = Y[~harm_mask & ~near(f0) & (f > 30)].sum()
    return fund, harm, other


def _lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> int:
    a = x.mean(axis=0).astype(np.float64)
    b = y.mean(axis=0).astype(np.float64)
    n = len(a)
    F = np.fft.rfft(a, 2 * n)
    G = np.fft.rfft(b, 2 * n)
    c = np.fft.irfft(G * np.conj(F), 2 * n)
    c = np.concatenate([c[-max_lag:], c[: max_lag + 1]])
    return int(np.argmax(np.abs(c)) - max_lag)


def run_tests(path: Path, cfg: dict) -> dict:
    """Signal + type tests (call inside the worker process)."""
    sys.path.insert(0, str(ROOT))
    from core.plugins import load_module, PluginError  # noqa: WPS433

    h = cfg["harness"]
    fails, info = [], {}
    try:
        mod = load_module(path)
    except PluginError as e:
        return {"passed": False, "failures": [str(e)], "info": {}}
    t = mod.TYPE
    info["type"] = t
    sets = param_sets(mod)

    def call(x, p):
        y = mod.process(x.copy(), SR, dict(p))
        return np.asarray(y)

    for ch in (1, 2):
        sig = signals(ch)
        for pname, p in sets:
            for sname, x in sig.items():
                where = f"{sname}/{'mono' if ch == 1 else 'stereo'}/{pname}"
                try:
                    y = call(x, p)
                except Exception as e:  # noqa: BLE001
                    fails.append(f"{where}: raised {type(e).__name__}: {e}")
                    continue
                exp_ch = 2 if t == "gain_pan" else ch
                if y.ndim != 2 or y.shape != (exp_ch, x.shape[1]):
                    fails.append(f"{where}: shape {y.shape}, expected {(exp_ch, x.shape[1])}")
                    continue
                if not np.all(np.isfinite(y)):
                    fails.append(f"{where}: NaN/Inf in output")
                    continue
                if sname == "silence":
                    pk = 20 * np.log10(np.max(np.abs(y)) + 1e-30)
                    if pk > h["silence_dbfs"]:
                        fails.append(f"{where}: silence came out at {pk:.1f} dBFS")
                if sname == "pink":
                    g = _energy_db(y) - _energy_db(x)
                    # boosts are capped for every type; big cuts are fine for faders, gates and
                    # wet-only effects, not for an EQ / compressor / saturator / tuner
                    cut_ok = t in ("gain_pan", "gate", "space")
                    if g > h["max_gain_db"] + 0.1 or (not cut_ok and g < -h["max_gain_db"] - 0.1):
                        fails.append(f"{where}: gain change {g:+.1f} dB exceeds +/-{h['max_gain_db']}")
        if len(fails) > 12:
            break

    # latency (pink noise at default and all_max)
    max_lag = int(0.05 * SR)
    lat = 0
    x = signals(2)["pink"]
    for pname, p in sets[:3]:
        try:
            y = call(x, p)
            if y.shape[1] == x.shape[1] and np.all(np.isfinite(y)) and np.any(y):
                lat = max(lat, abs(_lag(x, y, max_lag)))
        except Exception:  # noqa: BLE001
            pass
    info["latency_samples"] = lat
    declared = getattr(mod, "LATENCY_SAMPLES", 0)
    info["declared_latency"] = declared
    if t != "space" and lat / SR * 1000 > h["max_latency_ms"]:
        fails.append(f"latency {lat / SR * 1000:.2f} ms exceeds {h['max_latency_ms']} ms")

    # speed
    n = int(h["speed_test_s"] * SR)
    rng = np.random.default_rng(3)
    big = (rng.standard_normal((2, n)) * 0.1).astype(np.float32)
    worst = 0.0
    for pname, p in sets[:3]:
        t0 = time.perf_counter()
        try:
            call(big, p)
        except Exception:  # noqa: BLE001
            continue
        worst = max(worst, time.perf_counter() - t0)
    rt = h["speed_test_s"] / max(worst, 1e-9)
    info["realtime_x"] = round(rt, 1)
    if rt < h["realtime_factor"]:
        fails.append(f"too slow: {rt:.1f}x real time (need >= {h['realtime_factor']}x)")

    # type checks
    tt = np.arange(2 * SR) / SR
    sine = lambda f, a: (a * np.sin(2 * np.pi * f * tt))[None, :].astype(np.float32)  # noqa: E731
    if t in ("eq", "gain_pan"):
        x = sine(1000.0, 0.5)
        for pname, p in sets:
            try:
                y = call(x, p)
                fund, harm, other = _harmonic_split(y, 1000.0)
                thd = 10 * np.log10((harm + other) / (fund + 1e-30) + 1e-30)
                if fund > 1e-12 and thd > h["eq_thd_db"]:
                    fails.append(f"{t} added harmonics/noise ({thd:.1f} dB) at {pname}")
                    break
                pn = signals(2)["pink"]
                a, b = call(pn, p), call(pn * 2, p)
                err = 20 * np.log10(np.linalg.norm(b - 2 * a) / (np.linalg.norm(2 * a) + 1e-30) + 1e-30)
                if np.linalg.norm(a) > 1e-9 and err > -50:
                    fails.append(f"{t} is not linear at {pname} (error {err:.1f} dB)")
                    break
            except Exception as e:  # noqa: BLE001
                fails.append(f"type check raised {type(e).__name__}: {e}")
                break
    if t == "comp":
        for pname, p in sets:
            try:
                loud, quiet = sine(200.0, 0.5), sine(200.0, 0.01)
                gl = _rms_db(call(loud, p)[:, SR:]) - _rms_db(loud[:, SR:])
                gq = _rms_db(call(quiet, p)[:, SR:]) - _rms_db(quiet[:, SR:])
                if gl > gq + 0.5:
                    fails.append(f"comp expands (loud {gl:+.1f} dB vs quiet {gq:+.1f} dB) at {pname}")
                    break
            except Exception as e:  # noqa: BLE001
                fails.append(f"type check raised {type(e).__name__}: {e}")
                break
    if t in ("sat", "clip"):
        x = sine(1000.0, 0.5)
        worst_alias = -300.0
        for pname, p in sets:
            try:
                y = call(x, p)
                fund, harm, other = _harmonic_split(y, 1000.0)
                if fund < 1e-12:
                    continue
                al = 10 * np.log10(other / fund + 1e-30)
                worst_alias = max(worst_alias, al)
                if al > h["alias_db"]:
                    fails.append(f"{t} aliases ({al:.1f} dB) at {pname}: oversample >= 4x")
                    break
                dc = float(np.max(np.abs(y[:, SR // 2:].mean(axis=1))))
                if dc > 1e-3:
                    fails.append(f"{t} adds DC offset {dc:.4f} at {pname}")
                    break
            except Exception as e:  # noqa: BLE001
                fails.append(f"type check raised {type(e).__name__}: {e}")
                break
        info["worst_alias_db"] = round(worst_alias, 1)

    if t == "pitch":
        d = {k: float(v["default"]) for k, v in mod.PARAMS.items()}
        try:
            x = _voice(440.0)
            y = call(x, d)
            c_in, c_out = _cents(x), _cents(y)
            info["pitch_in_tune_shift_cents"] = round(c_out - c_in, 2)
            if abs(c_out - c_in) > 10:
                fails.append(f"pitch moved an in-tune A4 by {c_out - c_in:+.1f} cents at defaults")
            lv = _rms_db(y) - _rms_db(x)
            if abs(lv) > 1.0:
                fails.append(f"pitch changed the level of an in-tune note by {lv:+.1f} dB")
            if {"strength", "retune_ms", "key", "scale"} <= set(mod.PARAMS):
                p = dict(d, strength=1.0, retune_ms=0.0, key=9.0, scale=0.0)
                x2 = _voice(440.0 * 2 ** (40 / 1200))
                c2 = _cents(call(x2, p))
                info["pitch_corrected_cents"] = round(c2, 2)
                if abs(c2) > 10:
                    fails.append(f"pitch did not correct a +40 cent note (came out at {c2:+.1f} cents)")
        except Exception as e:  # noqa: BLE001
            fails.append(f"type check raised {type(e).__name__}: {e}")
    if t == "gate":
        for pname, p in sets:
            try:
                loud = sine(200.0, 0.5)
                gl = _rms_db(call(loud, p)[:, SR:]) - _rms_db(loud[:, SR:])
                if gl > 0.5:
                    fails.append(f"gate boosted a loud signal by {gl:+.1f} dB at {pname}")
                    break
            except Exception as e:  # noqa: BLE001
                fails.append(f"type check raised {type(e).__name__}: {e}")
                break

    return {"passed": not fails, "failures": fails[:20], "info": info}


def _voice(f0: float, dur: float = 2.0) -> np.ndarray:
    """Harmonic-rich steady note (a stand-in for a sung vowel)."""
    t = np.arange(int(SR * dur)) / SR
    ph = 2 * np.pi * f0 * t
    x = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 12))
    env = np.minimum(1, np.minimum(t / 0.05, (dur - t) / 0.05))
    return (0.3 * x * env)[None, :].astype(np.float32)


def _cents(y: np.ndarray) -> float:
    """Pitch of a steady note vs A4, in cents (FFT peak with parabolic interpolation)."""
    m = np.asarray(y, np.float64).mean(axis=0)
    m = m[len(m) // 4: 3 * len(m) // 4]
    w = np.hanning(len(m))
    nfft = 1 << 20
    Y = np.abs(np.fft.rfft(m * w, nfft))
    f = np.fft.rfftfreq(nfft, 1 / SR)
    band = (f > 300) & (f < 600)
    i = int(np.argmax(np.where(band, Y, 0)))
    a, b, c = np.log(Y[i - 1] + 1e-30), np.log(Y[i] + 1e-30), np.log(Y[i + 1] + 1e-30)
    off = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return float(1200 * np.log2((f[i] + off * (f[1] - f[0])) / 440.0))


def test_plugin(path: Path, cfg: dict, timeout_s: float = 300.0) -> dict:
    """Full harness: static check here, signal/type tests in a subprocess."""
    path = Path(path)
    src = path.read_text(encoding="utf-8")
    errs = static_check(src, cfg["harness"]["allowed_imports"])
    if errs:
        return {"passed": False, "stage": "static", "failures": errs, "info": {}}
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--worker", str(path.resolve())]
    try:
        p = subprocess.run(cmd, input=json.dumps(cfg), capture_output=True, text=True,
                           timeout=timeout_s, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {"passed": False, "stage": "signal", "failures": [f"timed out after {timeout_s:.0f} s"], "info": {}}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        tail = (p.stderr or p.stdout).strip().splitlines()[-5:]
        return {"passed": False, "stage": "signal", "failures": ["worker crashed: " + " | ".join(tail)], "info": {}}
    res = json.loads(lines[-1])
    res["stage"] = "signal"
    return res


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--worker":
    import warnings
    warnings.simplefilter("ignore")
    _cfg = json.loads(sys.stdin.read())
    _res = run_tests(Path(sys.argv[2]), _cfg)
    sys.stdout.write("\n" + json.dumps(_res) + "\n")
