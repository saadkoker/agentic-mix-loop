"""Stem agents. One fresh, short-context call per stem per round.

An agent sees its stem card, the mix card, its current chain, the
orchestrator's brief and a summary of what already failed for this stem. It
returns 1-3 candidate chains, each with a one-line reason. In code mode it may
also write new plugin files, which must pass the test harness (one retry with
the error) before they can be used.

RandomAgent is an offline stand-in (no API key needed): random moves within
the same rules. It doubles as a random-search baseline.
"""
from __future__ import annotations

import ast
import copy
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import STOCK_DIR
from .harness import test_plugin
from .llm import LLMError, OpenRouter
from .plugins import (TYPES, PluginError, PluginStore, describe_list, load_module, sha256_text,
                      stock_plugins)

CAND_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            # params as [{name, value}] pairs, not a free-form object: providers that enforce the
            # schema (Anthropic via OpenRouter) turn an object with no listed properties into {}
            "chain": {"type": "array", "items": {
                "type": "object",
                "properties": {"plugin": {"type": "string"},
                               "params": {"type": "array", "items": {
                                   "type": "object",
                                   "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                                   "required": ["name", "value"]}}},
                "required": ["plugin", "params"]}},
            "new_plugins": {"type": "array", "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "code": {"type": "string"}},
                "required": ["name", "code"]}},
        },
        "required": ["reason", "chain"]}}},
    "required": ["candidates"],
}

PALETTE = """Plugin palette (every plugin is exactly one of these types):
| type     | for                         | building blocks allowed                                    | max per chain |
| eq       | carving space, fixing tone  | pedalboard HighpassFilter/LowpassFilter/PeakFilter/LowShelfFilter/HighShelfFilter, or scipy biquads | {eq} |
| comp     | dynamics, punch             | pedalboard Compressor, or envelope follower + gain computer | {comp} |
| sat      | harmonics, warmth, density  | tanh / soft-clip curves, pedalboard Distortion; MUST oversample >= 4x | {sat} |
| clip     | taming peaks transparently  | hard/soft clipper, pedalboard Clipping; MUST oversample >= 4x | {clip} |
| gain_pan | balance + stereo placement  | static gain, constant-power pan (-100 L..+100 R), M/S width 0 (mono) .. 1.5 | exactly 1, LAST |"""

CONTRACT = '''Plugin file contract:
```python
TYPE = "eq"            # one of: eq, comp, sat, clip, gain_pan
PARAMS = {             # every knob, with hard ranges
    "low_cut_hz": {"default": 30.0, "min": 20.0, "max": 400.0},
}
def process(audio, sr, params):
    """audio: float32 array, shape (channels, samples). Returns same shape."""
```
Only `numpy`, `scipy.signal`, `pedalboard`, `math` may be imported. No file, network,
subprocess, print, dunder attributes, getattr/eval/exec. The harness rejects a plugin unless:
same shape out as in; no NaN/Inf; silence stays below -90 dBFS; gain change within +/-24 dB;
latency < 5 ms; >= 2x faster than real time; an eq or gain_pan must be linear and add no
harmonics; a comp must never expand; sat/clip must not alias (oversample >= 4x) and must not
add DC. Neutral parameter values should be an exact bypass.'''

RULES = """How candidates are judged:
- Each candidate changes ONLY this stem; the rest of the mix stays as the current best.
- The full mix is rendered, gain-staged to -14 LUFS, then a fixed -1 dBTP limiter. Overall
  loudness is therefore irrelevant: making things louder buys nothing.
- Hard guardrails (fail = thrown out): true peak, NaN/silence, length, DC offset, stereo
  correlation > 0.2 and > 0.8 below 150 Hz, PLR >= 8 dB before the limiter (don't crush the
  dynamics), every stem within 18 dB of its baseline level relative to the mix (you can't mute or
  bury a hard part), no clicks.
- Score = 0.5 mean(PQ) + 0.5 mean(CE) + 0.2 worst-window (PQ+CE)/2 from Audiobox Aesthetics,
  over 10 s windows. A change must beat the best by the measured judge noise margin.
- Small, musically motivated moves usually win. Don't repeat things listed as already failed."""


@dataclass
class Candidate:
    stem: str
    chain: list
    reason: str
    new_files: list[str] = field(default_factory=list)
    agent: str = "llm"


def plugin_spec(mod) -> dict:
    return {"type": mod.TYPE, "params": mod.PARAMS, "doc": (mod.__doc__ or "").strip()[:600]}


def stock_catalog() -> dict:
    return {ref: plugin_spec(mod) for ref, mod in stock_plugins().items()}


def custom_catalog(store: PluginStore, stem: str) -> dict:
    out = {}
    for sha, meta in store.approved().items():
        p = meta["path"]
        if p.startswith(f"plugins/{stem}/"):
            try:
                out[p] = plugin_spec(store.load(p))
            except PluginError:
                continue
    return out


def params_dict(p) -> dict:
    """Agent params -> {name: value}. Accepts the [{name, value}] list the schema asks for,
    or a plain dict (older replies, fake clients)."""
    if isinstance(p, dict):
        return p
    out = {}
    for kv in p or []:
        if isinstance(kv, dict) and "name" in kv:
            out[str(kv["name"])] = kv.get("value")
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_").lower()[:24] or "plugin"


def _peek_type(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "TYPE":
            if isinstance(n.value, ast.Constant) and n.value.value in TYPES:
                return n.value.value
    return None


class StemAgent:
    def __init__(self, cfg: dict, client: OpenRouter, store: PluginStore, project_root: Path,
                 write_code: bool, model: str | None = None):
        self.cfg = cfg
        self.client = client
        self.store = store
        self.root = Path(project_root)
        self.write_code = write_code
        self.model = model or cfg["models"]["stem_agent"]

    # ----------------------------------------------------------- prompt
    def _system(self, is_master: bool) -> str:
        lim = self.cfg["limits"]
        who = ("You own the MASTER BUS of a mix. Your chain runs on the summed stereo mix, before a "
               "fixed limiter you cannot change. gain_pan is optional here." if is_master else
               "You are the mix engineer for ONE stem in a multitrack mix. You can only change "
               "this stem's processing chain.")
        parts = [who, PALETTE.format(**lim), RULES]
        if self.write_code:
            parts.append("You MAY write new plugin files when stock processing can't do what you "
                         "need. Reference a new plugin in a chain as \"new:<name>\" and put its "
                         "source in new_plugins. Prefer tuning existing plugins: parameter "
                         "changes are cheap and safe.\n\n" + CONTRACT)
        else:
            parts.append("PARAMETERS ONLY this phase: use the stock plugins (and any approved "
                         "plugins listed) - do not write code; new_plugins must be empty.")
        parts.append('Reply with JSON: {"candidates": [{"reason": "<one line>", "chain": '
                     '[{"plugin": "stock:eq", "params": [{"name": "hp_hz", "value": 30}, ...]}, ..., '
                     '{"plugin": "stock:gain_pan", "params": [{"name": "gain_db", "value": -2}]}], '
                     '"new_plugins": [{"name": "...", "code": "..."}]}]}. '
                     "params is a list of {name, value} pairs using the plugin's PARAMS names. "
                     f"Return 1 to {self.cfg['loop']['candidates_per_agent']} candidates that "
                     "are genuinely different from each other. Params you omit take defaults.")
        return "\n\n".join(parts)

    def _appendix(self, stem: str, cur: list) -> list[str]:
        """Fixed technical reference, always attached, never written by the orchestrator."""
        custom = custom_catalog(self.store, stem)
        used_specs = {}
        for it in cur:
            try:
                used_specs[it["plugin"]] = plugin_spec(self.store.load(it["plugin"]))
            except PluginError:
                pass
        nl = "\n"
        out = ["# Technical appendix (fixed)",
               "## Your current chain (JSON)" + nl + json.dumps(cur, indent=1),
               "## Plugins in your current chain" + nl + json.dumps(used_specs, indent=1),
               "## Stock plugins" + nl + json.dumps(stock_catalog(), indent=1)]
        if custom:
            out.append("## Approved custom plugins for this stem" + nl + json.dumps(custom, indent=1))
        if self.write_code:
            src = [f"### {t}.py{nl}```python{nl}{(STOCK_DIR / f'{t}.py').read_text()}```"
                   for t in ("eq", "comp", "sat", "clip", "gain_pan")]
            out.append("## Stock plugin source (examples of the contract)" + nl + (nl + nl).join(src))
        return out

    def _fallback_task(self, stem: str, ctx: dict) -> list[str]:
        """Used only when no orchestrator wrote a prompt (Phase 2, or the plan call failed)."""
        nl = "\n"
        out = ["## Brief" + nl + (ctx.get("brief") or "Improve the mix."),
               "## Current best score" + nl + str(ctx.get("best_score", "?"))]
        if ctx.get("card"):
            out.append("## What is in this stem (description with timestamps)" + nl + json.dumps(ctx["card"], indent=1))
        if ctx.get("mix_card"):
            out.append("## The song (sections, description)" + nl + json.dumps(ctx["mix_card"], indent=1))
        if ctx.get("diagnosis"):
            out.append("## Audiobox diagnosis of the current best mix" + nl + json.dumps(ctx["diagnosis"], indent=1))
        if ctx.get("levels"):
            out.append("## Stem levels relative to the mix (LUFS)" + nl + json.dumps(ctx["levels"], indent=1))
        if ctx.get("other_stems"):
            out.append("## Other stems' current chains (you can't change them)" + nl +
                       nl.join(f"- {k}: {v}" for k, v in ctx["other_stems"].items()))
        if ctx.get("past"):
            out.append("## Already tried for this stem (don't repeat failures)" + nl +
                       nl.join(f"- {x}" for x in ctx["past"]))
        return out

    def _user(self, stem: str, ctx: dict) -> str:
        nl = "\n"
        head = f"# {'Master bus' if stem == 'master' else 'Stem: ' + stem}"
        if ctx.get("prompt"):
            task = ["## Your task (written by the orchestrator)" + nl + ctx["prompt"].strip()]
        else:
            task = self._fallback_task(stem, ctx)
        return (nl + nl).join([head] + task + self._appendix(stem, ctx["current_chain"]))

    # ----------------------------------------------------------- run
    def propose(self, stem: str, ctx: dict) -> tuple[list[Candidate], list[str]]:
        """Returns (usable candidates, log notes)."""
        is_master = stem == "master"
        msgs = [{"role": "system", "content": self._system(is_master)},
                {"role": "user", "content": self._user(stem, ctx)}]
        notes = []
        try:
            out, raw = self.client.chat(self.model, msgs, schema=CAND_SCHEMA, max_tokens=12000,
                                        tag=f"stem:{stem}")
        except LLMError as e:
            return [], [f"{stem}: agent call failed: {e}"]
        cands, failed = self._materialize(stem, out.get("candidates", []), ctx["current_chain"])
        notes += [f"{stem}: {f['why']}" for f in failed]
        if failed and self.write_code:
            # one retry with the errors
            fix = ("Some candidates were rejected:\n" +
                   "\n".join(f"- candidate {f['index'] + 1} ({f['reason']}): {f['why']}" for f in failed) +
                   "\n\nReturn fixed versions of ONLY those candidates (same JSON format).")
            msgs += [{"role": "assistant", "content": json.dumps(out)}, {"role": "user", "content": fix}]
            try:
                out2, _ = self.client.chat(self.model, msgs, schema=CAND_SCHEMA, max_tokens=12000,
                                           tag=f"stem:{stem}:retry")
                c2, failed2 = self._materialize(stem, out2.get("candidates", []), ctx["current_chain"])
                cands += c2
                notes += [f"{stem} (retry, dropped): {f['why']}" for f in failed2]
            except LLMError as e:
                notes.append(f"{stem}: retry failed: {e}")
        return cands[: self.cfg["loop"]["candidates_per_agent"]], notes

    def _is_gain_pan(self, ref: str) -> bool:
        try:
            return self.store.load(ref).TYPE == "gain_pan"
        except PluginError:
            return False

    def _repair_gain_pan(self, chain: list, current: list) -> list:
        """Agents sometimes forget that gain_pan must be last (or present):
        move it to the end, or keep the current one."""
        gps = [it for it in chain if self._is_gain_pan(it["plugin"])]
        rest = [it for it in chain if not self._is_gain_pan(it["plugin"])]
        if len(gps) == 1:
            return rest + gps
        if not gps and current:
            return rest + [copy.deepcopy(current[-1])]
        return chain

    def _materialize(self, stem: str, raw: list, current: list | None = None) -> tuple[list[Candidate], list[dict]]:
        current = current or []
        ok, failed = [], []
        for i, c in enumerate(raw[: self.cfg["loop"]["candidates_per_agent"]]):
            reason = str(c.get("reason", ""))[:300]
            chain = c.get("chain") or []
            new = {n.get("name", ""): n.get("code", "") for n in (c.get("new_plugins") or [])}
            if new and not self.write_code:
                failed.append({"index": i, "reason": reason, "why": "wrote code in params-only phase"})
                continue
            mapping, why, files = {}, None, []
            for name, code in new.items():
                path, err = self.write_plugin(stem, name, code)
                if err:
                    why = f"plugin '{name}' rejected: {err}"
                    break
                mapping[f"new:{name}"] = path
                files.append(path)
            if why:
                failed.append({"index": i, "reason": reason, "why": why})
                continue
            fixed = []
            for it in chain:
                ref = str(it.get("plugin", ""))
                ref = mapping.get(ref, ref)
                if ref.startswith("new:"):
                    why = f"chain references {ref} but no code was given"
                    break
                fixed.append({"plugin": ref, "params": params_dict(it.get("params"))})
            if why:
                failed.append({"index": i, "reason": reason, "why": why})
                continue
            if stem != "master":
                fixed = self._repair_gain_pan(fixed, current)
            ok.append(Candidate(stem=stem, chain=fixed, reason=reason, new_files=files))
        return ok, failed

    def write_plugin(self, stem: str, name: str, code: str) -> tuple[str | None, str | None]:
        """Write to plugins/<stem>/NNN_<type>_<name>.py (deduped by content) and run the harness."""
        t = _peek_type(code)
        if t is None:
            return None, "TYPE missing or not one of " + ", ".join(TYPES)
        d = self.root / "plugins" / stem
        d.mkdir(parents=True, exist_ok=True)
        h = sha256_text(code)
        approved = self.store.approved()
        for p in d.glob("*.py"):
            if sha256_text(p.read_text(encoding="utf-8")) == h:
                rel = p.relative_to(self.root).as_posix()
                if any(m["path"] == rel for m in approved.values()):
                    return rel, None
                path = p
                break
        else:
            n = 1 + max([int(p.name[:3]) for p in d.glob("[0-9][0-9][0-9]_*.py")] or [0])
            path = d / f"{n:03d}_{t}_{_slug(name)}.py"
            path.write_text(code, encoding="utf-8")
        rep = test_plugin(path, self.cfg)
        log = d / "harness_log.jsonl"
        with open(log, "a") as f:
            f.write(json.dumps({"file": path.name, **rep}) + "\n")
        if not rep["passed"]:
            return None, "; ".join(rep["failures"][:6])
        self.store.approve(path, {"type": t})
        return path.relative_to(self.root).as_posix(), None


class RandomAgent:
    """Offline agent: random moves under the same rules (stock plugins only)."""

    def __init__(self, cfg: dict, store: PluginStore, seed: int = 0):
        self.cfg = cfg
        self.store = store
        self.rng = random.Random(seed)

    def _rand_params(self, mod, scale: float) -> dict:
        out = {}
        for k, s in mod.PARAMS.items():
            lo, hi, d = s["min"], s["max"], s["default"]
            if self.rng.random() < 0.5:
                out[k] = min(hi, max(lo, d + self.rng.gauss(0, scale * (hi - lo))))
            else:
                out[k] = d
        return out

    def propose(self, stem: str, ctx: dict) -> tuple[list[Candidate], list[str]]:
        cur = ctx["current_chain"]
        is_master = stem == "master"
        n = self.cfg["loop"]["candidates_per_agent"]
        lim = self.cfg["limits"]
        cands = []
        for _ in range(n):
            ch = copy.deepcopy(cur)
            counts = {}
            for it in ch:
                counts[self.store.load(it["plugin"]).TYPE] = counts.get(self.store.load(it["plugin"]).TYPE, 0) + 1
            move = self.rng.random()
            body = ch if is_master else ch[:-1]
            if move < 0.5 and ch:
                it = self.rng.choice(ch)
                mod = self.store.load(it["plugin"])
                k = self.rng.choice(list(mod.PARAMS))
                s = mod.PARAMS[k]
                v = float(it["params"].get(k, s["default"]))
                it["params"] = dict(it["params"])
                it["params"][k] = min(s["max"], max(s["min"], v + self.rng.gauss(0, 0.08 * (s["max"] - s["min"]))))
                reason = f"nudge {it['plugin']} {k}"
            elif move < 0.85:
                free = [ref for ref, m in stock_plugins().items()
                        if m.TYPE not in ("gain_pan", "pitch") and counts.get(m.TYPE, 0) < lim.get(m.TYPE, 0)]
                if not free:
                    continue
                ref = self.rng.choice(free)
                mod = stock_plugins()[ref]
                new = {"plugin": ref, "params": self._rand_params(mod, 0.12)}
                body.insert(self.rng.randint(0, len(body)), new)
                reason = f"add {ref[6:]}"
            else:
                removable = [i for i, it in enumerate(body)]
                if not removable:
                    continue
                i = self.rng.choice(removable)
                reason = f"remove {body[i]['plugin']}"
                body.pop(i)
            ch = body if is_master else body + [ch[-1]]
            cands.append(Candidate(stem=stem, chain=ch, reason=reason, agent="random"))
        return cands, []


def summarize_chain(items: list) -> str:
    return describe_list(items)
