"""Plugin contract, loading, session (chain) validation and hashing.

A plugin is a Python file with TYPE, PARAMS and process(audio, sr, params).
A chain (the whole session) is plain JSON:

    {"stems":  {"09_Vox1": [{"plugin": "stock:eq", "params": {...}}, ...,
                            {"plugin": "stock:gain_pan", "params": {...}}]},     # every track
     "buses":  {"vox": {"inputs": ["09_Vox1", ...],          # routed here instead of the master
                        "sends": {"10_Vox2": -12.0},          # post-fader aux sends (dB)
                        "chain": [{"plugin": "stock:comp", ...}]}},
     "master": [{"plugin": "stock:comp", "params": {...}}]}

("stems" is the historical key: it holds the raw tracks, one entry per file.)

Any param may be AUTOMATED instead of a number:
    "gain_db": {"auto": [[0.0, 0.0], [48.0, 9.0], [54.0, 0.0]], "curve": "step"}
points are [seconds, value]; "linear" ramps between points, "step" holds each value until the
next point (with a short crossfade). Before the first point the first value holds.

`plugin` is either "stock:<name>" or a project-relative path such as
"plugins/drums/003_eq.py". Agent-written files only load if the test harness
approved that exact file content (plugins/approved.json, keyed by sha256).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

from .config import STOCK_DIR

TYPES = ("eq", "comp", "sat", "clip", "gain_pan", "pitch", "gate", "space")
MAX_LANE_POINTS = 64
CURVES = ("linear", "step")
MAX_BUSES = 8


class PluginError(Exception):
    pass


class ChainError(Exception):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_module_cache: dict[str, ModuleType] = {}


def load_module(path: Path) -> ModuleType:
    """Import a plugin file (cached by content hash)."""
    path = Path(path)
    h = sha256_file(path)
    if h in _module_cache:
        return _module_cache[h]
    spec = importlib.util.spec_from_file_location(f"mixloop_plugin_{h[:16]}", path)
    if spec is None or spec.loader is None:
        raise PluginError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        raise PluginError(f"{path.name}: import failed: {type(e).__name__}: {e}") from e
    check_contract(mod, path.name)
    _module_cache[h] = mod
    return mod


def check_contract(mod: ModuleType, name: str = "plugin"):
    t = getattr(mod, "TYPE", None)
    if t not in TYPES:
        raise PluginError(f"{name}: TYPE must be one of {TYPES}, got {t!r}")
    params = getattr(mod, "PARAMS", None)
    if not isinstance(params, dict):
        raise PluginError(f"{name}: PARAMS must be a dict")
    for k, spec in params.items():
        if not isinstance(spec, dict) or not {"default", "min", "max"} <= set(spec):
            raise PluginError(f"{name}: PARAMS[{k!r}] needs default/min/max")
        lo, hi, d = spec["min"], spec["max"], spec["default"]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lo, hi, d)):
            raise PluginError(f"{name}: PARAMS[{k!r}] values must be finite numbers")
        if not lo <= d <= hi:
            raise PluginError(f"{name}: PARAMS[{k!r}] default {d} outside [{lo}, {hi}]")
    if not callable(getattr(mod, "process", None)):
        raise PluginError(f"{name}: missing process(audio, sr, params)")


def defaults(mod: ModuleType) -> dict:
    return {k: float(v["default"]) for k, v in mod.PARAMS.items()}


def stock_plugins() -> dict[str, ModuleType]:
    """Every stock plugin: {"stock:<name>": module}."""
    out = {}
    for f in sorted(STOCK_DIR.glob("*.py")):
        if f.name.startswith("_"):
            continue
        out[f"stock:{f.stem}"] = load_module(f)
    return out


# ------------------------------------------------------------- automation

def is_lane(v) -> bool:
    return isinstance(v, dict) and "auto" in v


def lane_value_at(lane, t: float) -> float:
    """Value of a param (number or lane) at time t (seconds)."""
    if not is_lane(lane):
        return float(lane)
    pts = lane["auto"]
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t < t1:
            if lane.get("curve", "linear") == "step" or t1 <= t0:
                return v0
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return pts[-1][1]


def _num(v) -> float:
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("not finite")
    return v


def _clamp_lane(k: str, v: dict, lo: float, hi: float, notes: list):
    raw = v.get("auto")
    if not isinstance(raw, list) or not raw:
        notes.append(f"param {k!r}: automation needs a non-empty 'auto' list")
        return None
    pts = []
    for pt in raw:
        try:
            if isinstance(pt, dict):
                t, val = _num(pt.get("t")), _num(pt.get("v", pt.get("value")))
            else:
                t, val = _num(pt[0]), _num(pt[1])
        except (TypeError, ValueError, IndexError, KeyError):
            notes.append(f"param {k!r}: bad automation point {pt!r} skipped")
            continue
        if val < lo or val > hi:
            notes.append(f"param {k!r} automation value {val} clamped to [{lo}, {hi}]")
        pts.append([round(max(t, 0.0), 4), round(min(max(val, lo), hi), 6)])
    if not pts:
        return None
    pts.sort(key=lambda p: p[0])
    dedup: list[list[float]] = []
    for p in pts:
        if dedup and abs(dedup[-1][0] - p[0]) < 1e-4:
            dedup[-1] = p
        else:
            dedup.append(p)
    if len(dedup) > MAX_LANE_POINTS:
        notes.append(f"param {k!r}: {len(dedup)} automation points, kept the first {MAX_LANE_POINTS}")
        dedup = dedup[:MAX_LANE_POINTS]
    curve = v.get("curve", "linear")
    if curve not in CURVES:
        notes.append(f"param {k!r}: curve {curve!r} unknown, using linear")
        curve = "linear"
    if len(dedup) == 1 or all(p[1] == dedup[0][1] for p in dedup):
        return dedup[0][1]                      # constant: store as a plain number
    return {"auto": dedup, "curve": curve}


def clamp_params(mod: ModuleType, params: dict | None) -> tuple[dict, list[str]]:
    """Fill defaults, clamp to PARAMS ranges (numbers or automation lanes). Returns (params, notes)."""
    out, notes = defaults(mod), []
    for k, v in (params or {}).items():
        if k not in mod.PARAMS:
            notes.append(f"unknown param {k!r} ignored")
            continue
        lo, hi = mod.PARAMS[k]["min"], mod.PARAMS[k]["max"]
        if is_lane(v):
            lane = _clamp_lane(k, v, lo, hi, notes)
            if lane is not None:
                out[k] = lane
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            notes.append(f"param {k!r}={v!r} not a number, default used")
            continue
        if not math.isfinite(v):
            notes.append(f"param {k!r} not finite, default used")
            continue
        if v < lo or v > hi:
            notes.append(f"param {k!r}={v} clamped to [{lo}, {hi}]")
        out[k] = min(max(v, lo), hi)
    return out, notes


# ------------------------------------------------------------------ store

class PluginStore:
    """Resolves plugin refs for one project and tracks harness approvals."""

    def __init__(self, project_root: Path):
        self.root = Path(project_root)
        self.approved_path = self.root / "plugins" / "approved.json"

    def approved(self) -> dict:
        if self.approved_path.exists():
            return json.loads(self.approved_path.read_text())
        return {}

    def approve(self, path: Path, report: dict):
        a = self.approved()
        rel = Path(path).resolve().relative_to(self.root).as_posix()
        a[sha256_file(path)] = {"path": rel, "type": report.get("type"), "passed": True}
        self.approved_path.parent.mkdir(parents=True, exist_ok=True)
        self.approved_path.write_text(json.dumps(a, indent=1))

    def resolve(self, ref: str) -> Path:
        if ref.startswith("stock:"):
            p = STOCK_DIR / f"{ref[6:]}.py"
            if not p.exists():
                raise PluginError(f"no stock plugin {ref}")
            return p
        p = (self.root / ref).resolve()
        if self.root.resolve() not in p.parents:
            raise PluginError(f"plugin path escapes project: {ref}")
        if not p.exists():
            raise PluginError(f"plugin file not found: {ref}")
        return p

    def load(self, ref: str) -> ModuleType:
        p = self.resolve(ref)
        if not ref.startswith("stock:") and sha256_file(p) not in self.approved():
            raise PluginError(f"{ref} has not passed the test harness")
        return load_module(p)

    def ref_hash(self, ref: str) -> str:
        return sha256_file(self.resolve(ref))


# ------------------------------------------------------------- validation

def validate_chain(chain: dict, store: PluginStore, stems: list[str], limits: dict) -> dict:
    """Check structure, routing and per-type limits; fill + clamp params. Returns a
    normalised deep copy. Raises ChainError."""
    if not isinstance(chain, dict) or "stems" not in chain:
        raise ChainError("chain needs a 'stems' object")
    out = {"stems": {}, "buses": {}, "master": []}
    notes: list[str] = []
    missing = set(stems) - set(chain["stems"])
    if missing:
        raise ChainError(f"chain missing stems: {sorted(missing)}")
    for name in stems:
        out["stems"][name], n = _validate_list(chain["stems"][name], store, limits,
                                               f"stem {name}", require_gain_pan=True)
        notes += n
    buses = chain.get("buses") or {}
    if not isinstance(buses, dict):
        raise ChainError("buses must be an object {name: {inputs, sends, chain}}")
    if len(buses) > MAX_BUSES:
        raise ChainError(f"{len(buses)} buses, max {MAX_BUSES}")
    routed: dict[str, str] = {}
    for bname, b in buses.items():
        if not isinstance(bname, str) or not bname or bname in stems or bname == "master":
            raise ChainError(f"bad bus name {bname!r} (must differ from track names and 'master')")
        if not isinstance(b, dict):
            raise ChainError(f"bus {bname}: must be an object")
        inputs = [str(x) for x in (b.get("inputs") or [])]
        for t in inputs:
            if t not in stems:
                raise ChainError(f"bus {bname}: input {t!r} is not a track")
            if t in routed:
                raise ChainError(f"track {t} is routed to two buses ({routed[t]}, {bname})")
            routed[t] = bname
        sends = {}
        raw_sends = b.get("sends") or {}
        if not isinstance(raw_sends, dict):
            raise ChainError(f"bus {bname}: sends must be an object {{track: dB}}")
        for t, lvl in raw_sends.items():
            if t not in stems:
                raise ChainError(f"bus {bname}: send from {t!r}, which is not a track")
            try:
                lvl = float(lvl)
            except (TypeError, ValueError):
                raise ChainError(f"bus {bname}: send level for {t} is not a number") from None
            if not math.isfinite(lvl):
                raise ChainError(f"bus {bname}: send level for {t} is not finite")
            if lvl < -60 or lvl > 6:
                notes.append(f"bus {bname}: send {t} {lvl} dB clamped to [-60, 6]")
            sends[t] = round(min(max(lvl, -60.0), 6.0), 4)
        if not inputs and not sends:
            raise ChainError(f"bus {bname}: no inputs and no sends")
        blist, n = _validate_list(b.get("chain") or [], store, limits, f"bus {bname}",
                                  require_gain_pan=False)
        notes += n
        out["buses"][bname] = {"inputs": sorted(inputs), "sends": dict(sorted(sends.items())),
                               "chain": blist}
    out["master"], n = _validate_list(chain.get("master", []), store, limits, "master",
                                      require_gain_pan=False)
    notes += n
    if notes:
        out["_notes"] = notes
    return out


def _validate_list(items, store, limits, where, require_gain_pan):
    if not isinstance(items, list):
        raise ChainError(f"{where}: chain must be a list")
    counts = {t: 0 for t in TYPES}
    out, notes = [], []
    for i, it in enumerate(items):
        if not isinstance(it, dict) or "plugin" not in it:
            raise ChainError(f"{where}[{i}]: needs a 'plugin' key")
        try:
            mod = store.load(it["plugin"])
        except PluginError as e:
            raise ChainError(f"{where}[{i}]: {e}") from e
        counts[mod.TYPE] += 1
        params, n = clamp_params(mod, it.get("params"))
        notes += [f"{where}[{i}] {m}" for m in n]
        out.append({"plugin": it["plugin"], "params": params})
    for t, c in counts.items():
        if c > limits.get(t, 0):
            raise ChainError(f"{where}: {c} x {t} exceeds limit {limits.get(t, 0)}")
    if require_gain_pan:
        if counts["gain_pan"] != 1:
            raise ChainError(f"{where}: needs exactly one gain_pan plugin")
        if store.load(out[-1]["plugin"]).TYPE != "gain_pan":
            raise ChainError(f"{where}: gain_pan must be the last plugin")
    return out, notes


# ---------------------------------------------------------------- hashing

def _canon_val(v):
    if is_lane(v):
        return {"auto": [[round(float(t), 4), round(float(x), 6)] for t, x in v["auto"]],
                "curve": v.get("curve", "linear")}
    return round(float(v), 6)


def _canon_list(items, store):
    return [{"plugin": it["plugin"], "sha": store.ref_hash(it["plugin"]),
             "params": {k: _canon_val(v) for k, v in sorted(it["params"].items())}}
            for it in items]


def stem_chain_hash(items: list, store: PluginStore) -> str:
    return sha256_text(json.dumps(_canon_list(items, store), sort_keys=True))


def bus_hash(bus: dict, store: PluginStore) -> str:
    return sha256_text(json.dumps({
        "inputs": sorted(bus.get("inputs", [])),
        "sends": {k: round(float(v), 4) for k, v in sorted(bus.get("sends", {}).items())},
        "chain": _canon_list(bus.get("chain", []), store)}, sort_keys=True))


def chain_hash(chain: dict, store: PluginStore) -> str:
    canon = {"stems": {k: _canon_list(v, store) for k, v in sorted(chain["stems"].items())},
             "master": _canon_list(chain.get("master", []), store)}
    if chain.get("buses"):     # sessions without buses hash exactly as before (caches stay valid)
        canon["buses"] = {k: bus_hash(b, store) for k, b in sorted(chain["buses"].items())}
    return sha256_text(json.dumps(canon, sort_keys=True))


# ---------------------------------------------------------------- helpers

def flat_chain(stems: list[str]) -> dict:
    """Unity gain, centre pan, no processing, no buses."""
    return {"stems": {s: [{"plugin": "stock:gain_pan", "params": {}}] for s in stems},
            "buses": {}, "master": []}


def with_stem(chain: dict, stem: str, items: list) -> dict:
    c = copy.deepcopy(chain)
    c["stems"][stem] = copy.deepcopy(items)
    c.pop("_notes", None)
    return c


def with_master(chain: dict, items: list) -> dict:
    c = copy.deepcopy(chain)
    c["master"] = copy.deepcopy(items)
    c.pop("_notes", None)
    return c


def with_bus(chain: dict, name: str, bus: dict | None) -> dict:
    """Add / replace a bus (None removes it)."""
    c = copy.deepcopy(chain)
    c.setdefault("buses", {})
    if bus is None:
        c["buses"].pop(name, None)
    else:
        c["buses"][name] = copy.deepcopy(bus)
    c.pop("_notes", None)
    return c


def _fmt_num(v) -> str:
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_val(v) -> str:
    if is_lane(v):
        pts = v.get("auto") or []
        cells = []
        for pt in pts[:8]:
            if isinstance(pt, dict):
                t, x = pt.get("t"), pt.get("v", pt.get("value"))
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                t, x = pt[0], pt[1]
            else:
                continue
            cells.append(f"{_fmt_num(t)}s:{_fmt_num(x)}")
        body = ">".join(cells) + (">..." if len(pts) > 8 else "")
        return f"auto[{v.get('curve', 'linear')} {body}]"
    return _fmt_num(v)


def describe_list(items: list) -> str:
    """Short human/LLM-readable summary of a chain list."""
    parts = []
    for it in items:
        name = it["plugin"].replace("stock:", "")
        ps = ", ".join(f"{k}={_fmt_val(v)}" for k, v in it.get("params", {}).items())
        parts.append(f"{name}({ps})")
    return " -> ".join(parts) if parts else "(empty)"


def describe_bus(name: str, bus: dict) -> str:
    ins = ", ".join(bus.get("inputs", [])) or "-"
    sends = ", ".join(f"{k}@{v:g}dB" for k, v in bus.get("sends", {}).items()) or "-"
    return f"bus {name}: inputs [{ins}] sends [{sends}] chain {describe_list(bus.get('chain', []))}"
