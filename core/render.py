"""tracks + session -> mix. Plain code, no LLM.

Signal flow (like a DAW):
    track -> its plugin chain (last: gain_pan) -> bus it is routed to, else the master
    track -> post-fader aux sends (dB) -> send buses
    bus   -> its plugin chain -> master
    master chain -> gain-stage to the target loudness -> fixed true-peak limiter

Automation: any param can be a lane ({"auto": [[t, v], ...], "curve": "linear"|"step"}).
  - Plugins that declare SAMPLE_AUTOMATION (e.g. stock gain_pan: gain_db, pan) get per-sample
    arrays: exact volume / pan rides. Step changes get a 10 ms ramp so they never click.
  - Any other automated param: the plugin is rendered once per distinct setting (full length,
    so filter/compressor state stays continuous) and the outputs are crossfaded at the
    breakpoints - linear crossfades across "linear" segments, 20 ms crossfades at "step" changes.

Per-track and per-bus outputs are cached in memory by chain hash, so a candidate that changes
one track only re-renders that track (and the bus it feeds).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from . import audio as A
from .plugins import PluginError, PluginStore, bus_hash, is_lane, lane_value_at, stem_chain_hash

MAX_SNAPSHOTS = 12
STEP_RAMP_S = 0.010
STEP_XFADE_S = 0.020


class RenderError(Exception):
    pass


@dataclass
class Render:
    stems: dict[str, np.ndarray]        # post-chain stereo track outputs (pre-bus)
    pre: np.ndarray                     # master bus output, before the final limiter
    final: np.ndarray                   # after the fixed limiter
    sr: int
    length_errors: list[str] = field(default_factory=list)
    stem_hashes: dict[str, str] = field(default_factory=dict)
    pre_lufs: float = float("nan")      # integrated loudness of `pre`, computed once
    buses: dict[str, np.ndarray] = field(default_factory=dict)   # post-chain bus outputs
    bus_hashes: dict[str, str] = field(default_factory=dict)


def lane_array(lane, n: int, sr: int) -> np.ndarray:
    """Per-sample values of a lane (float32, length n)."""
    pts = lane["auto"]
    ts = np.array([p[0] for p in pts], np.float64) * sr
    vs = np.array([p[1] for p in pts], np.float64)
    idx = np.arange(n, dtype=np.float64)
    if lane.get("curve", "linear") == "linear":
        return np.interp(idx, ts, vs).astype(np.float32)
    # step: hold, with a short linear ramp starting at each breakpoint
    out = np.full(n, vs[0], np.float64)
    ramp = max(1, int(STEP_RAMP_S * sr))
    for i in range(1, len(pts)):
        s = int(ts[i])
        if s >= n:
            break
        e = min(n, s + ramp)
        out[s:e] = vs[i - 1] + (vs[i] - vs[i - 1]) * (np.arange(e - s) + 1) / ramp
        out[e:] = vs[i]
    return out.astype(np.float32)


def _snapshot_weights(times: list[float], modes: list[str], n: int, sr: int) -> np.ndarray:
    """Weights (len(times), n) that crossfade between snapshot outputs. modes[i] is the
    transition from times[i] to times[i+1]: 'linear' (crossfade across the segment) or 'step'."""
    K = len(times)
    W = np.zeros((K, n), np.float32)
    s_idx = [min(n, max(0, int(t * sr))) for t in times]
    W[0, : s_idx[0]] = 1.0
    xf = max(1, int(STEP_XFADE_S * sr))
    for i in range(K):
        a = s_idx[i]
        b = s_idx[i + 1] if i + 1 < K else n
        if i + 1 < K and modes[i] == "linear" and b > a:
            r = np.linspace(0.0, 1.0, b - a, endpoint=False, dtype=np.float32)
            W[i, a:b] += 1.0 - r
            W[i + 1, a:b] += r
        else:
            W[i, a:b] += 1.0
    # smooth step changes: short crossfade centred on each step boundary
    for i in range(K - 1):
        if modes[i] != "step":
            continue
        c = s_idx[i + 1]
        a, b = max(0, c - xf // 2), min(n, c + xf // 2)
        if b > a:
            r = np.linspace(0.0, 1.0, b - a, dtype=np.float32)
            W[i, a:b] = 1.0 - r
            W[i + 1, a:b] = r
            for k in range(K):
                if k not in (i, i + 1):
                    W[k, a:b] = 0.0
    return W


class Renderer:
    def __init__(self, stems: dict[str, np.ndarray], sr: int, store: PluginStore,
                 ceiling_dbtp: float = -1.0, target_lufs: float = -14.0,
                 cache_size: int | None = None):
        self.target_lufs = target_lufs
        self.inputs = stems
        self.sr = sr
        self.n = next(iter(stems.values())).shape[1]
        self.store = store
        self.ceiling = ceiling_dbtp
        self.cache: OrderedDict[str, tuple[np.ndarray, list[str]]] = OrderedDict()
        self.cache_size = cache_size or (len(stems) + 8)
        self.bus_cache: OrderedDict[str, tuple[np.ndarray, list[str]]] = OrderedDict()
        self.bus_cache_size = 12

    # ---------------------------------------------------------- one plugin
    def _call(self, mod, x: np.ndarray, params: dict, where: str, i: int, ref: str) -> np.ndarray:
        try:
            y = mod.process(np.ascontiguousarray(x, dtype=np.float32), self.sr, dict(params))
        except Exception as e:  # noqa: BLE001
            raise RenderError(f"{where}[{i}] {ref} raised {type(e).__name__}: {e}") from e
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 2 or y.shape[0] not in (1, 2):
            raise RenderError(f"{where}[{i}] {ref} returned shape {y.shape}")
        return y

    def _run_item(self, x: np.ndarray, it: dict, where: str, i: int) -> np.ndarray:
        mod = self.store.load(it["plugin"])
        params = it["params"]
        lanes = {k: v for k, v in params.items() if is_lane(v)}
        if not lanes:
            return self._call(mod, x, params, where, i, it["plugin"])
        n = x.shape[1]
        sample_keys = set(getattr(mod, "SAMPLE_AUTOMATION", ()))
        if set(lanes) <= sample_keys:
            p = {k: (lane_array(v, n, self.sr) if is_lane(v) else v) for k, v in params.items()}
            return self._call(mod, x, p, where, i, it["plugin"])
        # snapshot rendering
        dur = n / self.sr
        times = sorted({min(max(float(t), 0.0), dur) for v in lanes.values() for t, _ in v["auto"]})
        snaps = [{k: (lane_value_at(v, t) if is_lane(v) else v) for k, v in params.items()} for t in times]
        modes = []
        for a, b in zip(times, times[1:]):
            lin = any(v.get("curve", "linear") == "linear" and lane_value_at(v, a) != lane_value_at(v, b)
                      for v in lanes.values())
            modes.append("linear" if lin else "step")
        keyed: dict[tuple, int] = {}
        renders: list[np.ndarray] = []
        idx = []
        for s in snaps:
            key = tuple(sorted((k, round(float(v), 9)) for k, v in s.items()))
            if key not in keyed:
                if len(renders) >= MAX_SNAPSHOTS:
                    raise RenderError(f"{where}[{i}] {it['plugin']}: more than {MAX_SNAPSHOTS} distinct "
                                      "automated settings; use fewer breakpoints for non-volume params")
                keyed[key] = len(renders)
                renders.append(self._call(mod, x, s, where, i, it["plugin"]))
            idx.append(keyed[key])
        ch = max(r.shape[0] for r in renders)
        renders = [A.to_stereo(r) if r.shape[0] != ch else r for r in renders]
        m = min(r.shape[1] for r in renders)
        W = _snapshot_weights(times, modes, m, self.sr)
        out = np.zeros((ch, m), np.float32)
        for k in range(len(times)):
            out += renders[idx[k]][:, :m] * W[k][None, :]
        return out

    def _run_list(self, x: np.ndarray, items: list, where: str) -> tuple[np.ndarray, list[str]]:
        errs = []
        for i, it in enumerate(items):
            n_in = x.shape[1]
            try:
                y = self._run_item(x, it, where, i)
            except PluginError as e:
                raise RenderError(f"{where}[{i}] {e}") from e
            if abs(y.shape[1] - n_in) > 0:
                errs.append(f"{where}[{i}] {it['plugin']}: length {y.shape[1]} vs {n_in}")
                y = y[:, :n_in] if y.shape[1] > n_in else np.pad(y, ((0, 0), (0, n_in - y.shape[1])))
            x = y
        return x, errs

    # ------------------------------------------------------------ tracks
    def render_stem(self, name: str, items: list) -> tuple[np.ndarray, list[str], str]:
        h = stem_chain_hash(items, self.store)
        key = f"{name}:{h}"
        if key in self.cache:
            self.cache.move_to_end(key)
            y, errs = self.cache[key]
            return y, errs, h
        y, errs = self._run_list(self.inputs[name], items, f"stem {name}")
        y = A.to_stereo(y)
        self.cache[key] = (y, errs)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return y, errs, h

    # ------------------------------------------------------------ routing
    def bus_input(self, bus: dict, outs: dict[str, np.ndarray]) -> np.ndarray:
        x = np.zeros((2, self.n), dtype=np.float32)
        for t in bus.get("inputs", []):
            x += outs[t]
        for t, lvl in bus.get("sends", {}).items():
            x += outs[t] * np.float32(A.undb(lvl))
        return x

    def route(self, outs: dict[str, np.ndarray], chain: dict, hashes: dict[str, str] | None = None
              ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, str], list[str]]:
        """Track outputs -> buses -> summed master input. With `hashes`, bus outputs are cached."""
        buses = chain.get("buses") or {}
        routed = {t for b in buses.values() for t in b.get("inputs", [])}
        mix = np.zeros((2, self.n), dtype=np.float32)
        for name, y in outs.items():
            if name not in routed:
                mix += y
        bus_out, bus_h, errs = {}, {}, []
        for bname, bus in buses.items():
            key = None
            if hashes is not None:
                members = sorted(set(bus.get("inputs", [])) | set(bus.get("sends", {})))
                key = bname + ":" + bus_hash(bus, self.store) + ":" + ":".join(hashes[m] for m in members)
            if key is not None and key in self.bus_cache:
                self.bus_cache.move_to_end(key)
                y, e = self.bus_cache[key]
            else:
                y, e = self._run_list(self.bus_input(bus, outs), bus.get("chain", []), f"bus {bname}")
                y = A.to_stereo(y)
                if key is not None:
                    self.bus_cache[key] = (y, e)
                    while len(self.bus_cache) > self.bus_cache_size:
                        self.bus_cache.popitem(last=False)
            bus_out[bname] = y
            bus_h[bname] = key or ""
            errs += e
            mix += y
        return mix, bus_out, bus_h, errs

    def render(self, chain: dict) -> Render:
        outs, errs, hashes = {}, [], {}
        for name in self.inputs:
            y, e, h = self.render_stem(name, chain["stems"][name])
            outs[name], hashes[name] = y, h
            errs += e
        mix, bus_out, bus_h, e = self.route(outs, chain, hashes)
        errs += e
        pre, final, pre_l, e = self.finish(mix, chain.get("master", []))
        errs += e
        return Render(stems=outs, pre=pre, final=final, sr=self.sr, length_errors=errs,
                      stem_hashes=hashes, pre_lufs=pre_l, buses=bus_out, bus_hashes=bus_h)

    def finish(self, mix: np.ndarray, master: list) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
        """Summed master input -> master chain -> fixed final step. Returns (pre, final, pre_lufs, errors)."""
        pre, errs = self._run_list(mix, master, "master")
        pre = A.to_stereo(pre)
        # Loudness is a fixed final step: gain-stage the bus to the target
        # loudness, then the fixed true-peak limiter. The overall level of the
        # pre-limiter mix can therefore never buy (or cost) anything.
        pre_l = A.lufs(pre, self.sr)
        staged = (pre * A.undb(self.target_lufs - pre_l)).astype(np.float32) if np.isfinite(pre_l) else pre
        final = A.limit(staged, self.sr, self.ceiling)
        return pre, final, pre_l, errs
