"""One candidate through the gauntlet: validate -> render -> guardrails ->
loudness-match -> Audiobox. Results are cached by full-chain hash."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from . import audio as A
from . import guardrails as G
from .config import Project
from .judge import Score, composite, make_judge
from .plugins import (ChainError, PluginStore, chain_hash, flat_chain, sha256_text,
                      validate_chain, with_stem)
from .render import Render, Renderer, RenderError


@dataclass
class Result:
    hash: str
    chain: dict
    ok: bool                         # passed validation, render and guardrails
    score: Score | None = None
    guard: dict | None = None
    error: str | None = None
    seconds: float = 0.0
    cached: bool = False
    render: Render | None = field(default=None, repr=False)

    @property
    def value(self) -> float:
        return self.score.score if (self.ok and self.score) else float("-inf")

    def reason_failed(self) -> str:
        if self.error:
            return self.error
        if self.guard and not self.guard["passed"]:
            return "; ".join(f"{c['name']}: {c['value']} (limit {c['limit']})"
                             for c in self.guard["checks"] if not c["passed"])
        return ""

    def summary(self) -> dict:
        d = {"hash": self.hash[:12], "ok": self.ok, "cached": self.cached,
             "seconds": round(self.seconds, 2)}
        if self.score:
            d["score"] = self.score.to_dict()
        if not self.ok:
            d["fail"] = self.reason_failed()
        return d


class Evaluator:
    def __init__(self, project: Project, cfg: dict, judge_kind: str = "audiobox"):
        self.project = project.ensure()
        self.cfg = cfg
        self.stems, self.sr = A.load_stems(project.source_dir, remove_dc=not project.raw)
        self.stem_names = list(self.stems)
        self.store = PluginStore(project.root)
        self.renderer = Renderer(self.stems, self.sr, self.store,
                                 cfg["audio"]["ceiling_dbtp"], cfg["audio"]["target_lufs"])
        self.judge = make_judge(cfg, judge_kind)
        self.judge_kind = judge_kind
        self._cache = self._load_cache()
        self.baseline: G.Baseline | None = None
        self.baseline_result: Result | None = None

    # -------------------------------------------------------------- cache
    def _load_cache(self) -> dict:
        c = {}
        if self.project.cache.exists():
            for line in self.project.cache.read_text().splitlines():
                try:
                    d = json.loads(line)
                    c[d["key"]] = d
                except (json.JSONDecodeError, KeyError):
                    continue
        return c

    def _cache_key(self, h: str) -> str:
        g = json.dumps([self.cfg["guardrails"], self.cfg["audio"],
                        self.cfg["score"]["window_s"], self.cfg["score"]["hop_s"]], sort_keys=True)
        return f"{self.judge_kind}:{h}:{sha256_text(g)[:8]}"

    def _cache_put(self, key: str, d: dict):
        d = {"key": key, **d}
        self._cache[key] = d
        with open(self.project.cache, "a") as f:
            f.write(json.dumps(d) + "\n")

    # ---------------------------------------------------------- baseline
    def setup_baseline(self) -> Result:
        """Flat sum: unity gain, centre pan, no processing."""
        chain = validate_chain(flat_chain(self.stem_names), self.store, self.stem_names,
                               self.cfg["limits"])
        r = self.renderer.render(chain)
        self.baseline = G.baseline_stats(r, self.cfg)
        res = self.evaluate(chain, keep_render=True, force_score=True)
        if not res.ok:
            # the baseline is the reference; report but carry on
            print(f"WARNING: flat baseline fails guardrails: {res.reason_failed()}")
        self.baseline_result = res
        return res

    # ---------------------------------------------------------- evaluate
    def evaluate(self, chain: dict, keep_render: bool = False, force_score: bool = False) -> Result:
        t0 = time.perf_counter()
        try:
            chain = validate_chain(chain, self.store, self.stem_names, self.cfg["limits"])
        except ChainError as e:
            return Result(hash="", chain=chain, ok=False, error=f"invalid chain: {e}")
        notes = chain.pop("_notes", None)
        h = chain_hash(chain, self.store)
        key = self._cache_key(h)
        if key in self._cache and not keep_render:
            d = self._cache[key]
            sc = composite(d["windows"], self.cfg) if d.get("windows") else None
            return Result(hash=h, chain=chain, ok=d["ok"], score=sc, guard=d.get("guard"),
                          error=d.get("error"), cached=True)
        if self.baseline is None:
            raise RuntimeError("call setup_baseline() first")
        try:
            r = self.renderer.render(chain)
        except RenderError as e:
            res = Result(hash=h, chain=chain, ok=False, error=f"render: {e}")
            self._cache_put(key, {"ok": False, "error": res.error})
            return res
        rep = G.check(r, self.baseline, self.cfg, bus_ref=self.bus_baseline_level, chain=chain)
        res = Result(hash=h, chain=chain, ok=rep.passed, guard=rep.to_dict(),
                     render=r if keep_render else None)
        if notes:
            res.guard["notes"] = notes
        windows = None
        if rep.passed or force_score:
            x, _ = A.normalize_lufs(r.final, self.sr, self.cfg["audio"]["target_lufs"])
            from .judge_worker import JudgeError
            try:
                res.score = self.judge.score(x, self.sr)
            except JudgeError as e:       # judge kept faulting: drop the candidate, don't cache
                res.ok, res.error = False, f"judge: {e}"
                return res
            windows = res.score.windows
        res.seconds = time.perf_counter() - t0
        self._cache_put(key, {"ok": res.ok, "guard": res.guard, "windows": windows})
        return res

    def perturbed(self, chain: dict) -> list[float]:
        """Composite scores of the mix under 6 inaudible variants (the re-test): +/-1 sample
        shift, +/-0.1 dB overall, +/-0.1 dB on the loudest track's fader. Cached by chain hash."""
        chain = validate_chain(chain, self.store, self.stem_names, self.cfg["limits"])
        chain.pop("_notes", None)
        h = chain_hash(chain, self.store)
        if not hasattr(self, "_pert"):
            self._pert = {}
        if h in self._pert:
            return self._pert[h]
        x = self.matched(chain)
        variants = [x * np.float32(A.undb(0.1)), x * np.float32(A.undb(-0.1)),
                    np.pad(x, ((0, 0), (1, 0)))[:, : x.shape[1]], np.pad(x, ((0, 0), (0, 1)))[:, 1:]]
        stem = max(self.stem_names, key=lambda s: float(np.mean(self.stems[s] ** 2)))
        for d in (0.1, -0.1):
            variants.append(self.matched(nudge_fader(chain, stem, d)))
        scores = [composite(self.judge.raw_windows(v, self.sr), self.cfg).score for v in variants]
        self._pert[h] = scores
        return scores

    def bus_baseline_level(self, inputs: tuple) -> float:
        """Flat-baseline level (vs the flat mix, LU) of a set of tracks summed."""
        key = tuple(sorted(inputs))
        if not hasattr(self, "_bus_ref"):
            self._bus_ref = {}
        if key not in self._bus_ref:
            flat = validate_chain(flat_chain(self.stem_names), self.store, self.stem_names, self.cfg["limits"])
            x = None
            for t in key:
                y, _, _ = self.renderer.render_stem(t, flat["stems"][t])
                x = y.copy() if x is None else x + y
            mix_l = self.baseline_result.render.pre_lufs if (self.baseline_result is not None and
                                                              self.baseline_result.render is not None) else None
            if mix_l is None or not np.isfinite(mix_l):
                mix_l = A.lufs(self.renderer.render(flat).pre, self.sr)
            l = A.lufs(x, self.sr) if x is not None else float("-inf")
            self._bus_ref[key] = l - mix_l if np.isfinite(l) else float("-inf")
        return self._bus_ref[key]

    def render(self, chain: dict) -> Render:
        chain = validate_chain(chain, self.store, self.stem_names, self.cfg["limits"])
        chain.pop("_notes", None)
        return self.renderer.render(chain)

    def matched(self, chain: dict) -> np.ndarray:
        """Final render, loudness-matched to the scoring target."""
        r = self.render(chain)
        x, _ = A.normalize_lufs(r.final, self.sr, self.cfg["audio"]["target_lufs"])
        return x

    # ------------------------------------------------------------- noise
    def measure_noise(self, chain: dict) -> dict:
        """Score the same mix with tiny inaudible changes. Returns spread
        (max - min) and the margin to use (noise_multiplier x spread)."""
        r = self.render(chain)
        x, _ = A.normalize_lufs(r.final, self.sr, self.cfg["audio"]["target_lufs"])
        variants = {
            "orig": x,
            "+0.1dB": x * np.float32(A.undb(0.1)),
            "-0.1dB": x * np.float32(A.undb(-0.1)),
            "shift+1": np.pad(x, ((0, 0), (1, 0)))[:, : x.shape[1]],
            "shift-1": np.pad(x, ((0, 0), (0, 1)))[:, 1:],
        }
        windows = {k: self.judge.raw_windows(v, self.sr) for k, v in variants.items()}
        # +/-0.1 dB on one stem's fader (survives loudness matching)
        stem = max(self.stem_names, key=lambda s: float(np.mean(self.stems[s] ** 2)))
        for d in (0.1, -0.1):
            items = [dict(it, params=dict(it["params"])) for it in chain["stems"][stem]]
            items[-1]["params"]["gain_db"] = items[-1]["params"].get("gain_db", 0.0) + d
            c = with_stem(chain, stem, items)
            windows[f"{stem}{d:+.1f}dB"] = self.judge.raw_windows(self.matched(c), self.sr)
        return noise_from_windows(windows, self.cfg)


def nudge_fader(chain: dict, stem: str, d_db: float) -> dict:
    """The same session with one track's fader moved by d_db (automation lanes shifted)."""
    items = [dict(it, params=dict(it["params"])) for it in chain["stems"][stem]]
    g = items[-1]["params"].get("gain_db", 0.0)
    if isinstance(g, dict) and "auto" in g:
        g = {**g, "auto": [[t, v + d_db] for t, v in g["auto"]]}
    else:
        g = float(g) + d_db
    items[-1]["params"]["gain_db"] = g
    return with_stem(chain, stem, items)


def noise_from_windows(windows: dict, cfg: dict) -> dict:
    """Spread of the composite over the tiny-change variants, for the CURRENT weighting.
    Kept separate so stored windows can be re-weighted without re-running the judge."""
    scores = {k: composite(w, cfg).score for k, w in windows.items()}
    vals = list(scores.values())
    spread = float(max(vals) - min(vals))
    s = cfg["score"]
    std = float(np.std(vals))
    margin = max(s["noise_multiplier"] * spread, s["min_margin"])
    out = {"scores": scores, "spread": spread, "std": std, "margin": margin, "windows": windows}
    if s.get("confirm"):
        out["screen_margin"] = max(s.get("screen_sigma", 3.5) * std, s["min_margin"])
        out["confirm_margin"] = max(s.get("confirm_sigma", 3.0) * std, s["min_margin"])
        out["margin"] = out["screen_margin"]
    return out
