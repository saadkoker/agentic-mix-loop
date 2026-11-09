"""Tool-using mix agents: the engineer at the console of the agentic DAW.

One agent works on one target per round: a track, a bus ("bus:<name>") or the master. It gets
the orchestrator's prompt plus a fixed technical appendix, then works like an engineer:
measures (levels over time, pitch, spectrum, the judge's diagnosis), may write its own plugins
(each must pass the test harness), auditions drafts against the real judge (limited number per
round), and finally submits 1-3 candidates. The orchestrator's normal accept rule (beat the best
by the judge-noise margin, pass every guardrail) decides what is kept.
"""
from __future__ import annotations

import copy
import json

from . import tools as T
from .llm import LLMError
from .config import STOCK_DIR
from .plugins import describe_bus, describe_list, stock_plugins, with_bus, with_master, with_stem
from .stem_agent import Candidate, StemAgent, custom_catalog, plugin_spec

PARAM = {"type": "object",
         "properties": {"name": {"type": "string"},
                        "value": {"type": "number", "description": "constant value"},
                        "auto": {"type": "array", "description": "automation points instead of value",
                                 "items": {"type": "object",
                                           "properties": {"t": {"type": "number", "description": "seconds"},
                                                          "v": {"type": "number"}},
                                           "required": ["t", "v"]}},
                        "curve": {"type": "string", "enum": ["linear", "step"]}},
         "required": ["name"]}
ITEM = {"type": "object",
        "properties": {"plugin": {"type": "string", "description": "stock:<name> or a plugins/... path"},
                       "params": {"type": "array", "items": PARAM}},
        "required": ["plugin", "params"]}
CANDIDATE = {"type": "object",
             "properties": {
                 "reason": {"type": "string", "description": "one line: what and why"},
                 "chain": {"type": "array", "items": ITEM},
                 "inputs": {"type": "array", "items": {"type": "string"},
                            "description": "bus targets only: tracks routed into this bus"},
                 "sends": {"type": "array", "description": "bus targets only: aux sends into this bus",
                           "items": {"type": "object",
                                     "properties": {"track": {"type": "string"}, "db": {"type": "number"}},
                                     "required": ["track", "db"]}}},
             "required": ["reason", "chain"]}


def _fn(name, desc, props=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {}, "required": required or []}}}


SPAN = {"start": {"type": "number", "description": "seconds (optional)"},
        "end": {"type": "number", "description": "seconds (optional)"}}
TOOLS = [
    _fn("song_info", "Song length, tempo estimate and every track: channels, where it plays, level."),
    _fn("levels", "Phrase-by-phrase loudness of a track: raw (as recorded) and how loud it sits against "
                  "the current mix in each phrase. Use it to find parts that are too loud / buried.",
        {"track": {"type": "string"}, **SPAN}, ["track"]),
    _fn("pitch", "Pitch analysis of a monophonic track (vocal, bass, lead): notes with start time, "
                 "duration and cents off, pitch-class time histogram, key candidates.",
        {"track": {"type": "string"}, **SPAN}, ["track"]),
    _fn("spectrum", "Octave-band levels of a track, 'bus:<name>' or 'mix' in the current best mix, and "
                    "its share of the mix per band (masking).",
        {"target": {"type": "string"}, **SPAN}, ["target"]),
    _fn("judge_view", "The Audiobox judge's diagnosis of the current best mix: weakest windows, per-window "
                      "quality timeline, and this target's measured sensitivity (-/+ dB) and solo quality."),
    _fn("plugin_source", "Source code of a stock plugin (examples of the plugin contract), e.g. 'eq', 'tune'.",
        {"name": {"type": "string"}}, ["name"]),
    _fn("write_plugin", "Write a new plugin file. It is run through the test harness; on success you get "
                        "its path to use as 'plugin' in a chain, otherwise the failures to fix.",
        {"name": {"type": "string", "description": "short name, e.g. vocal_rider"},
         "code": {"type": "string", "description": "full Python source following the contract"}},
        ["name", "code"]),
    _fn("audition", "Render the full mix with this candidate for your target and score it with the real "
                    "judge (plus every guardrail). Returns the score change vs the current best. Limited "
                    "number per round - audition your most promising drafts.",
        {"candidate": CANDIDATE}, ["candidate"]),
    _fn("submit", "Finish: submit 1-3 candidates for your target. Ends your turn.",
        {"candidates": {"type": "array", "items": CANDIDATE, "minItems": 1, "maxItems": 3}},
        ["candidates"]),
]

CONTRACT = '''Plugin file contract (for write_plugin):
```python
TYPE = "comp"          # one of: eq, comp, sat, clip, gain_pan, pitch, gate, space
PARAMS = {             # every knob, with hard ranges (numbers only)
    "threshold_db": {"default": -20.0, "min": -60.0, "max": 0.0},
}
def process(audio, sr, params):
    """audio: float32 array (channels, samples). Return the same shape (gain_pan: 2 channels)."""
```
Only numpy, scipy.signal, pedalboard and math may be imported. No file / network / subprocess,
print, dunder attributes, getattr / eval / exec. The harness rejects a plugin unless: same shape;
no NaN/Inf; silence in -> silence out (< -90 dBFS); boosts <= +24 dB (cuts > 24 dB only allowed for
gain_pan / gate / space); latency < 5 ms (except space); >= 2x faster than real time; eq / gain_pan
linear with no harmonics; comp never expands; sat / clip oversample (aliasing < -45 dB) and add no
DC; pitch leaves an in-tune note alone; gate never boosts. Neutral params should be an exact bypass.
Automation: the renderer can automate any param of your plugin (it renders once per distinct
setting and crossfades), so keep params as plain numbers.'''

SYSTEM = """You are a mix engineer working inside an agentic DAW. You own ONE target this turn: a track,
a bus or the master. Work like an engineer at a console: measure first, then act.

The session: every raw track exactly as recorded, each with its own chain (last plugin: gain_pan, the
fader). Tracks can be routed into buses (group processing) and can feed send buses (reverb / delay) via
post-fader sends. The master chain runs on the sum, then a FIXED final step: gain-staged to -14 LUFS and
a -1 dBTP limiter. Overall loudness therefore buys nothing.

Tools: song_info, levels (phrase-by-phrase loudness, raw and in the mix), pitch (notes, cents off, key
candidates), spectrum (bands and masking), judge_view (the judge's diagnosis), plugin_source,
write_plugin (your own plugin, tested by the harness), audition (render + score a draft with the real
judge), submit (1-3 candidates; ends your turn).

You have automation. Any param can be a lane instead of a number:
  {"name": "gain_db", "auto": [{"t": 47.9, "v": 0}, {"t": 48.0, "v": 9}, {"t": 54.0, "v": 9}, {"t": 54.1, "v": 0}], "curve": "linear"}
gain_db / pan on gain_pan are rendered sample-accurately (volume rides, clip-gain regions, mutes down
to -60 dB between phrases); any other param switches per section with short crossfades. Use it when a
part's level or tone needs to differ between sections - a single static setting can't fix a part that
is too loud in one phrase and buried in the next.

Pitch correction: stock:tune (key, scale, strength, retune_ms, ...). Use the pitch tool to see what was
sung and decide key / scale / strength yourself; natural settings (strength 0.4-0.8, retune 40-120 ms)
usually beat hard tuning. Only use it on monophonic sources.

How candidates are judged: the full mix is rendered and scored by Audiobox Aesthetics
(0.7 PQ + 0.3 CE + 0.2 worst 10 s window). A change must beat the current best by the first bar
(margin_needed in audition results) AND survive a re-test under tiny inaudible variations - so a draft
that only just clears the bar may be judge jitter; prefer clear wins.
Hard guardrails reject a candidate outright: true peak, NaN/silence, length, DC, stereo correlation
(> 0.2; > 0.8 below 150 Hz), PLR >= 8 dB before the limiter, every track (and every bus) within 18 dB of
its original level relative to the mix (you cannot mute a whole part), clicks.

Rules: auditions are limited, so measure before you audition. Don't repeat what already failed. Submit
candidates that genuinely differ. Params you omit take their defaults; plugin refs are "stock:<name>"
or the path write_plugin returned."""


def _params(pairs) -> dict:
    """[{name, value | auto, curve}] (or a dict) -> {name: number | lane}."""
    if isinstance(pairs, dict):
        return pairs
    out = {}
    for kv in pairs or []:
        if not isinstance(kv, dict) or "name" not in kv:
            continue
        if kv.get("auto"):
            pts = []
            for pt in kv["auto"]:
                if isinstance(pt, dict):
                    pts.append([pt.get("t"), pt.get("v", pt.get("value"))])
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    pts.append([pt[0], pt[1]])
            out[str(kv["name"])] = {"auto": pts, "curve": kv.get("curve", "linear")}
        elif "value" in kv:
            out[str(kv["name"])] = kv["value"]
    return out


class ToolAgent:
    def __init__(self, cfg: dict, client, orch, model: str | None = None):
        self.cfg = cfg
        self.client = client
        self.o = orch
        self.model = model or cfg["models"]["stem_agent"]
        self.helper = StemAgent(cfg, client, orch.ev.store, orch.p.root, write_code=True)
        L = cfg["loop"]
        self.max_steps = int(L.get("agent_max_steps", 14))
        self.max_auditions = int(L.get("agent_auditions", 4))
        self.max_writes = int(L.get("agent_plugin_writes", 4))

    # ------------------------------------------------------------ appendix
    def _current(self, target: str, ctx: dict):
        best = self.o.best_chain
        if target == "master":
            return best.get("master", [])
        if target.startswith("bus:"):
            name = target[4:]
            cur = copy.deepcopy((best.get("buses") or {}).get(name))
            if cur is None:
                cur = {"inputs": ctx.get("inputs") or [], "sends": ctx.get("sends") or {}, "chain": []}
            return cur
        return best["stems"][target]

    def _appendix(self, target: str, cur) -> str:
        nl = "\n"
        best = self.o.best_chain
        stock = {ref: plugin_spec(m) for ref, m in stock_plugins().items()}
        key = target.replace(":", "_")
        custom = custom_catalog(self.o.ev.store, key)
        buses = best.get("buses") or {}
        parts = ["# Technical appendix (fixed)",
                 f"## Your target: {target}",
                 "## Its current setting (JSON)" + nl + json.dumps(cur, indent=1),
                 "## Session routing" + nl + (nl.join(describe_bus(k, b) for k, b in buses.items())
                                              or "no buses yet: every track feeds the master directly"),
                 "## Tracks" + nl + ", ".join(self.o.stems),
                 "## Stock plugins" + nl + json.dumps(stock, indent=1)]
        if custom:
            parts.append("## Approved custom plugins for this target" + nl + json.dumps(custom, indent=1))
        parts.append(CONTRACT)
        if target.startswith("bus:"):
            parts.append("## Bus candidates" + nl + "Give `inputs` (tracks routed into the bus; each track "
                         "can be in only one bus) and/or `sends` (post-fader aux sends, dB) plus the bus "
                         "`chain`. A send bus (reverb/delay) usually has only sends and a wet-only effect "
                         "(mix = 1). The bus chain may end with gain_pan as the bus fader.")
        return (nl + nl).join(parts)

    # ------------------------------------------------------------ helpers
    def _to_change(self, target: str, cand: dict, cur):
        """Candidate JSON -> (chain-or-bus for the Candidate, full session chain)."""
        items = [{"plugin": str(it.get("plugin", "")), "params": _params(it.get("params"))}
                 for it in (cand.get("chain") or []) if isinstance(it, dict)]
        best = self.o.best_chain
        if target == "master":
            return items, with_master(best, items)
        if target.startswith("bus:"):
            name = target[4:]
            inputs = cand.get("inputs")
            if inputs is None:
                inputs = cur.get("inputs", [])
            sends = cand.get("sends")
            if sends is None:
                sends = cur.get("sends", {})
            if isinstance(sends, list):
                sends = {str(s.get("track")): s.get("db") for s in sends if isinstance(s, dict)}
            bus = {"inputs": [self.o.match_stem(t) or t for t in inputs],
                   "sends": {(self.o.match_stem(k) or k): v for k, v in sends.items()},
                   "chain": items}
            return bus, with_bus(best, name, bus)
        items = self.helper._repair_gain_pan(items, cur)
        return items, with_stem(best, target, items)

    def _score(self, full: dict) -> dict:
        res = self.o.ev.evaluate(full)
        best = self.o.state["best_score"]
        out = {"ok": res.ok, "margin_needed": round(self.o.margin, 4)}
        if res.score is not None:
            s = res.score
            out.update({"score_change": round(s.score - best["score"], 4),
                        "PQ_change": round(s.PQ - best["PQ"], 4), "CE_change": round(s.CE - best["CE"], 4),
                        "worst_window": round(s.worst, 3), "worst_window_at": T.mmss(s.worst_at_s),
                        "passes_first_bar": bool(res.ok and s.score > best["score"] + self.o.margin),
                        "note": "to be kept it must also pass a re-test (averaged over tiny inaudible "
                                "variations) - aim clearly above the first bar"})
        if not res.ok:
            out["rejected_because"] = res.reason_failed()[:400]
        notes = (res.guard or {}).get("notes")
        if notes:
            out["param_notes"] = notes[:8]
        return out

    # --------------------------------------------------------------- run
    def propose(self, target: str, ctx: dict) -> tuple[list[Candidate], list[str]]:
        cur = self._current(target, ctx)
        prompt = ctx.get("prompt") or ctx.get("brief") or "Improve this target's contribution to the mix."
        past = ctx.get("past") or []
        user = (f"# Target: {target}\n\n## Your task (written by the orchestrator)\n{prompt.strip()}\n\n"
                + ("## Already tried for this target\n" + "\n".join(f"- {x}" for x in past) + "\n\n" if past else "")
                + self._appendix(target, cur))
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        notes: list[str] = []
        self._aud = self._wr = 0
        render = self.o.ev.render(self.o.best_chain)
        new_files: list[str] = []
        submitted = None
        for step in range(self.max_steps):
            force = step == self.max_steps - 1
            choice = {"type": "function", "function": {"name": "submit"}} if force else "auto"
            try:
                msg = self.client.chat_tools(self.model, msgs, TOOLS, tool_choice=choice, max_tokens=12000,
                                             tag=f"agent:{target}")
            except LLMError as e:
                notes.append(f"{target}: agent call failed: {e}")
                break
            keep = {k: msg[k] for k in ("role", "content", "tool_calls", "reasoning_details") if k in msg}
            keep.setdefault("role", "assistant")
            if keep.get("content") is None:
                keep["content"] = ""
            msgs.append(keep)
            calls = msg.get("tool_calls") or []
            if not calls:
                msgs.append({"role": "user", "content": "Use the tools as needed, then call submit with 1-3 candidates."})
                continue
            for call in calls:
                fn = (call.get("function") or {}).get("name", "")
                try:
                    args = json.loads((call.get("function") or {}).get("arguments") or "{}")
                except json.JSONDecodeError as e:
                    args, err = {}, f"arguments were not valid JSON: {e}"
                else:
                    err = None
                if fn == "submit" and err is None:
                    submitted = args.get("candidates") or []
                    result = {"received": len(submitted)}
                elif err:
                    result = {"error": err}
                else:
                    try:
                        result = self._tool(fn, args, target, cur, render, ctx)
                    except Exception as e:  # noqa: BLE001
                        result = {"error": f"{type(e).__name__}: {e}"}
                    if fn == "write_plugin" and result.get("path"):
                        new_files.append(result["path"])
                msgs.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": json.dumps(result)[:14000]})
            if submitted is not None:
                break
        cands = []
        for i, c in enumerate((submitted or [])[: self.cfg["loop"]["candidates_per_agent"]]):
            if not isinstance(c, dict):
                continue
            try:
                change, full = self._to_change(target, c, cur)
            except Exception as e:  # noqa: BLE001
                notes.append(f"{target}: candidate {i + 1} unusable: {e}")
                continue
            cand = Candidate(stem=target, chain=change, reason=str(c.get("reason", ""))[:300],
                             new_files=new_files, agent="tools")
            cand.full_chain = full  # type: ignore[attr-defined]
            cands.append(cand)
        if submitted is None:
            notes.append(f"{target}: agent never submitted")
        notes.append(f"{target}: {self._aud} audition(s), {self._wr} plugin write(s), {len(cands)} candidate(s)")
        return cands, notes

    def _tool(self, fn: str, a: dict, target: str, cur, render, ctx) -> dict:
        ev = self.o.ev
        if fn == "song_info":
            return T.song_info(ev, self.o.cards())
        if fn in ("levels", "pitch"):
            track = self.o.match_stem(a.get("track"))
            if track is None:
                return {"error": f"unknown track {a.get('track')!r}; tracks: {', '.join(self.o.stems)}"}
            if fn == "levels":
                return T.levels(ev, render, track, a.get("start"), a.get("end"))
            return T.pitch(ev, track, a.get("start"), a.get("end"))
        if fn == "spectrum":
            tgt = str(a.get("target", ""))
            if tgt not in ("mix",) and not tgt.startswith("bus:"):
                tgt = self.o.match_stem(tgt) or tgt
            return T.spectrum(ev, render, tgt, a.get("start"), a.get("end"))
        if fn == "judge_view":
            return T.judge_view(self.o.state.get("diagnosis"), target)
        if fn == "plugin_source":
            name = str(a.get("name", "")).replace("stock:", "").replace(".py", "")
            p = STOCK_DIR / f"{name}.py"
            if not p.exists() or name.startswith("_"):
                return {"error": f"no stock plugin {name!r}; stock: {', '.join(r[6:] for r in stock_plugins())}"}
            return {"name": name, "source": p.read_text(encoding="utf-8")}
        if fn == "write_plugin":
            if self._wr >= self.max_writes:
                return {"error": "no plugin writes left this round"}
            self._wr += 1
            key = target.replace(":", "_")
            path, err = self.helper.write_plugin(key, str(a.get("name", "plugin")), str(a.get("code", "")))
            return {"path": path} if path else {"failures": err}
        if fn == "audition":
            if self._aud >= self.max_auditions:
                return {"error": "no auditions left this round: submit"}
            change, full = self._to_change(target, a.get("candidate") or {}, cur)
            self._aud += 1
            out = self._score(full)
            out["auditions_left"] = self.max_auditions - self._aud
            out["your_change"] = describe_bus(target[4:], change) if target.startswith("bus:") else describe_list(change)
            return out
        return {"error": f"unknown tool {fn!r}"}
