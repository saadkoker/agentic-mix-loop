"""The orchestrator owns the mix. It never touches audio itself: it decides
what to try, sends stem agents to try it, and keeps or throws away the result
based on the score.

Phases (config [loop].phase or --phase):
  2  one agent, params only: round-robin one stem per round, no orchestrator LLM, no Gemini
  3  orchestrator + stem agents, params only
  4  + Gemini description cards (what is in each stem, with timestamps)
  5  + agents may write their own plugins behind the test harness

The orchestrator (Opus) writes every subagent's task prompt. The critique
comes from the judge itself (core/diagnose.py). Opt-in ([audit] enabled): a
periodic outside ear (core/outside_ear.py) points the orchestrator at problems a
second model hears repeatedly; it never scores or vetoes anything.
--agent random swaps every LLM role for offline heuristics (smoke tests,
random-search baseline).
"""
from __future__ import annotations

import copy
import json
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio as A
from . import guardrails as G
from .config import Project
from .evaluate import Evaluator, Result
from .diagnose import diagnose
from .outside_ear import outside_ear
from .gemini import Listener, load_cards, safe
from .listen import blind_ab, interactive
from .llm import BudgetExceeded, LLMError, OpenRouter, Spend
from .plugins import describe_bus, describe_list, with_master, with_stem
from .stem_agent import Candidate, RandomAgent, StemAgent

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "targets": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "stem": {"type": "string",
                         "description": "a track name, 'bus:<name>' (new or existing bus) or 'master'"},
                "prompt": {"type": "string"},
                "inputs": {"type": "array", "items": {"type": "string"},
                           "description": "new bus only: tracks to route into it"},
                "sends": {"type": "array", "description": "new send bus only: aux sends into it",
                          "items": {"type": "object",
                                    "properties": {"track": {"type": "string"}, "db": {"type": "number"}},
                                    "required": ["track", "db"]}}},
            "required": ["stem", "prompt"]}},
        "notes": {"type": "string"}},
    "required": ["targets"],
}

BALANCE_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "stems": {"type": "array", "items": {
                "type": "object",
                "properties": {"stem": {"type": "string"}, "gain_db": {"type": "number"},
                               "pan": {"type": "number"}, "width": {"type": "number"}},
                "required": ["stem"]}}},
        "required": ["reason", "stems"]}}},
    "required": ["candidates"],
}

ORCH_SYSTEM = """You are the orchestrator (mix lead) of an agentic DAW. You never touch audio.
The session holds every raw track exactly as recorded. Each round you choose which targets to work on
and WRITE THE COMPLETE TASK PROMPT for each target's agent. A target is a track, a bus ("bus:<name>",
existing or new - give `inputs` for a group bus, `sends` for a send bus such as a shared reverb) or the
master (master rounds are scheduled for you).

Each agent is a fresh tool-using model at the console. It has measurement tools (phrase-by-phrase
levels raw and in the mix, pitch analysis with key candidates, spectrum/masking, the judge's
diagnosis), can write its own plugins behind a test harness, can audition drafts against the real
judge a few times, and then submits 1-3 candidates. Its kit: EQ, compressor, de-esser, saturation,
clipper, gate, tuner (pitch correction), reverb, delay, gain/pan/width - and AUTOMATION of any param
(volume rides, clip-gain regions, mutes between phrases, per-section settings). It sees ONLY your
prompt plus a fixed technical appendix (its current setting, routing, plugin catalog, contract). It does
NOT see the diagnosis, descriptions, other tracks or the history unless you say what matters - so be
concrete: the goal, which sections (timestamps) to care about, which tools to use first, what to try,
what already failed, how its candidates should differ.

How results are judged: candidates are rendered, checked against hard guardrails (incl. every track and
bus within 18 dB of its original level vs the mix), loudness-matched to -14 LUFS and scored by Audiobox
Aesthetics (score = 0.7 PQ + 0.3 CE + 0.2 worst-window). A change must beat the best by the first
bar (acceptance_margin) and then survive a re-test: re-scored under tiny inaudible variations, its
AVERAGE gain must still clear the confirm margin. Changes that only won by judge jitter fail the re-test. The ONLY critique is the Audiobox diagnosis you are given: weakest windows,
per-track score sensitivity to -/+ dB, and solo quality. Trust it over intuition. Raw multitracks often
have parts recorded at very different levels from phrase to phrase: a single fader can't fix that,
automation can. The track descriptions are factual (what plays and when); they contain no opinions."""

ORCH_AUDIT = """

OUTSIDE LISTENER (`outside_listener` in the context, when present): every ~10 rounds, and when the loop
stalls, a different AI model listens to the current best mix several times with no reference and no
knowledge of what changed, and lists what sounds wrong. You get only the mix problems it reported in
EVERY listen, each with measurements for that time range (section loudness vs the whole song, the
loudest tracks there, the tracks that moved most vs the flat mix there, band balance, bus levels).
Treat them as leads on WHERE to look, not as goals: check each against its numbers and ignore any the
numbers contradict. For the rest, aim targets at them (e.g. a volume ride on the track that rose most
in that section), give the agent the timestamps and the numbers, and have it measure first. The judge
still decides every accept exactly as before; a fix it won't confirm after a couple of tries is not
worth more rounds. `rounds_ago` says how stale the notes are."""


@dataclass
class Phase:
    n: int
    orchestrator_llm: bool
    gemini: bool
    write_code: bool
    single_agent: bool


def phase_flags(n: int, agent_kind: str) -> Phase:
    llm = agent_kind == "llm"
    return Phase(n=n, orchestrator_llm=llm and n >= 3, gemini=llm and n >= 4,
                 write_code=llm and n >= 5, single_agent=n <= 2)


def fmt_delta(v: float) -> str:
    return f"{v:+.4f}"


class Orchestrator:
    def __init__(self, project: Project, cfg: dict, agent_kind: str = "llm",
                 judge_kind: str = "audiobox", listen: bool = True, seed: int = 0):
        self.p = project.ensure()
        self.cfg = cfg
        self.L = cfg["loop"]
        self.agent_kind = agent_kind
        self.phase = phase_flags(int(self.L["phase"]), agent_kind)
        self.listen_enabled = listen
        self.rng = random.Random(seed)
        self.ev = Evaluator(project, cfg, judge_kind)
        self.stems = self.ev.stem_names
        self.spend = Spend(self.L["max_usd"], self.p.root / "llm_log.jsonl")
        self.client = OpenRouter(cfg, self.spend)
        if agent_kind == "llm" and not self.client.available:
            raise SystemExit(f"{cfg['models']['api_key_env']} is not set. Set it, or run with "
                             "--agent random for an offline test.")
        self.listener = Listener(cfg, self.client, self.p.cards / "log")
        self.agent_mode = self.L.get("agent_mode", "tools") if agent_kind == "llm" else "random"
        if agent_kind == "llm" and self.agent_mode == "tools":
            from .daw_agent import ToolAgent
            self.agent = ToolAgent(cfg, self.client, self)
            self.master_agent = ToolAgent(cfg, self.client, self, model=cfg["models"]["orchestrator"])
        elif agent_kind == "llm":
            self.agent = StemAgent(cfg, self.client, self.ev.store, self.p.root, self.phase.write_code)
            self.master_agent = StemAgent(cfg, self.client, self.ev.store, self.p.root,
                                          self.phase.write_code, model=cfg["models"]["orchestrator"])
        else:
            self.agent = self.master_agent = RandomAgent(cfg, self.ev.store, seed)
        self.state = self._load_state()
        self.A = cfg.get("audit", {})
        self.audit_on = bool(self.A.get("enabled")) and agent_kind == "llm" and self.phase.orchestrator_llm

    # ================================================================ state
    def _load_state(self) -> dict:
        if self.p.state.exists():
            return json.loads(self.p.state.read_text())
        return {}

    def _save_state(self):
        self.state["spend_usd"] = self.state.get("spend_prev", 0.0) + self.spend.total
        self.p.state.write_text(json.dumps(self.state, indent=1))
        self.p.best.write_text(json.dumps(self.state["best_chain"], indent=1))

    def _log(self, entry: dict):
        entry = {"t": round(time.time(), 1), **entry}
        with open(self.p.history, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _git(self, msg: str):
        root = str(self.p.root)
        try:
            if not (self.p.root / ".git").exists():
                subprocess.run(["git", "init", "-q"], cwd=root, check=True)
                (self.p.root / ".gitignore").write_text(
                    "stems/\nrenders/\nreference/\nscore_cache.jsonl\n*.wav\n*.flac\n__pycache__/\n")
            subprocess.run(["git", "add", "-A", "plugins", "chains", "history.jsonl", "state.json",
                            ".gitignore"], cwd=root, check=True, capture_output=True)
            r = subprocess.run(["git", "commit", "-q", "-m", msg], cwd=root, capture_output=True, text=True)
            if r.returncode != 0 and "nothing to commit" not in r.stdout + r.stderr:
                if not self.state.get("git_warned"):
                    print(f"  git commit failed: {(r.stderr or r.stdout).strip()[:200]}")
                    self.state["git_warned"] = True
        except (OSError, subprocess.CalledProcessError) as e:
            if not self.state.get("git_warned"):
                print(f"  git unavailable: {e}")
                self.state["git_warned"] = True

    # ============================================================ helpers
    @property
    def best_chain(self) -> dict:
        return self.state["best_chain"]

    @property
    def best_value(self) -> float:
        return self.state["best_score"]["score"]

    @property
    def margin(self) -> float:
        return self.state["margin"]

    def _levels(self, chain: dict) -> dict:
        r = self.ev.render(chain)
        return {k: round(v, 2) for k, v in G.stem_rel_levels(r).items()}

    def _score_str(self, s: dict) -> str:
        b = self.state["baseline_score"]["score"]
        return (f"{s['score']:.4f} ({fmt_delta(s['score'] - b)} vs flat baseline; "
                f"PQ {s['PQ']:.3f}, CE {s['CE']:.3f}, worst window {s['worst']:.3f} at "
                f"{s['worst_at_s']:.0f}s; PC {s['PC']:.3f}, CU {s['CU']:.3f})")

    def _past(self, stem: str, line: str):
        self.state.setdefault("past", {}).setdefault(stem, []).append(line)
        self.state["past"][stem] = self.state["past"][stem][-14:]

    def cards(self) -> dict:
        return load_cards(self.p.cards) if self.phase.gemini else {}

    # ============================================================== setup
    def setup(self):
        if self.state.get("best_chain"):
            print(f"Resuming at round {self.state['round']}, best {self.best_value:.4f}")
            self.ev.setup_baseline()   # rebuild guardrail baseline stats (cached score)
            self.state["spend_prev"] = self.state.get("spend_usd", 0.0)
            self.state["phase"] = self.phase.n
            if self.phase.gemini:
                self.listening_pass()  # cards for any stem that doesn't have one yet
            if (not self.state.get("diagnosis")
                    or self.state.get("accepted_since_diagnosis", 0) >= self.L["diagnose_every"]):
                self.run_diagnosis()      # also redo one that was due when the last run stopped
            return
        print(f"Stems ({len(self.stems)}): {', '.join(self.stems)}  sr={self.ev.sr}  "
              f"length {self.ev.renderer.n / self.ev.sr:.1f}s")
        print("Scoring the flat baseline ...")
        base = self.ev.setup_baseline()
        if base.score is None:
            raise SystemExit("baseline could not be scored: " + base.reason_failed())
        print(f"  baseline {base.score.short()}")
        noise = self._noise(base.chain)
        self.state = {
            "phase": self.phase.n, "agent": self.agent_kind, "round": 0, "started": time.time(),
            "elapsed_s": 0.0, "baseline_chain": base.chain, "baseline_score": base.score.to_dict(),
            "best_chain": base.chain, "best_score": base.score.to_dict(), "best_hash": base.hash,
            "margin": noise["margin"], "confirm_margin": noise.get("confirm_margin"), "noise": noise,
            "accepted": 0, "accepted_since_diagnosis": 0,
            "accepted_since_listen": 0, "rounds_without_accept": 0,
            "score_by_round": [base.score.score], "diagnosis": None, "past": {}, "flags": [],
            "checkpoints": [{"id": 0, "round": 0, "chain": base.chain,
                             "score": base.score.score, "liked": True, "n_accepted": 0}],
            "liked_ckpt": 0, "old_votes_in_row": 0,
            "holdout_ref": {"round": 0, "chain": base.chain, "score": base.score.to_dict()},
            "levels": self._levels(base.chain), "spend_usd": 0.0,
        }
        self._log({"event": "baseline", "hash": base.hash[:12], "score": base.score.to_dict(),
                   "noise": noise})
        src = ' (from phase0.json)' if noise.get('source') == 'phase0' else ''
        if noise.get("confirm_margin"):
            print(f"  judge noise std {noise['std']:.4f} -> first bar {noise['margin']:.4f}, re-test must "
                  f"confirm an averaged gain > {noise['confirm_margin']:.4f}{src}")
        else:
            print(f"  judge noise spread {noise['spread']:.4f} -> acceptance margin {noise['margin']:.4f}{src}")
        A.write(self.p.renders / "baseline.wav", self.ev.render(base.chain).final, self.ev.sr)
        if self.phase.gemini:
            self.listening_pass()
        self.run_diagnosis()
        self._save_state()
        self._git("baseline")

    def _noise(self, chain: dict) -> dict:
        """Judge noise for the CURRENT score weighting. Phase 0's stored windows are
        re-weighted without re-running the judge; otherwise it is measured now."""
        from .evaluate import noise_from_windows
        n = None
        if self.p.phase0.exists():
            d = json.loads(self.p.phase0.read_text())
            if d.get("judge") == self.ev.judge_kind and d.get("noise", {}).get("windows"):
                n = noise_from_windows(d["noise"]["windows"], self.cfg)
                n["source"] = "phase0"
        if n is None:
            print("Measuring judge noise (tiny inaudible changes) ...")
            n = self.ev.measure_noise(chain)
        n.pop("windows", None)       # keep state.json small
        return n

    def listening_pass(self, force: bool = False):
        """Gemini Job 1: a card for every full stem, then the flat mix."""
        have = self.cards()
        todo = [s for s in self.stems if force or s not in have]
        if not todo and "_mix" in have and not force:
            return
        print(f"Gemini listening pass: {len(todo)} stems + mix ...")
        for s in todo:
            card = safe(self.listener.card, s, self.ev.stems[s], self.ev.sr)
            if card:
                (self.p.cards / f"{s}.json").write_text(json.dumps(card, indent=1))
                print(f"  {s}: {card.get('role', '')}"[:160])
        if force or "_mix" not in have:
            mix = self.ev.render(self.state["baseline_chain"]).final
            card = safe(self.listener.card, "_mix", mix, self.ev.sr, is_mix=True, stem_names=self.stems)
            if card:
                (self.p.cards / "_mix.json").write_text(json.dumps(card, indent=1))

    # ============================================================ planning
    def plan(self, master: bool = False) -> list[dict]:
        """Targets for this round, each with the full prompt for its agent."""
        n = 1 if (self.phase.single_agent or master) else self.L["stems_per_round"]
        if not self.phase.orchestrator_llm:
            return self._fallback_plan(n, master)
        cards = self.cards()
        ctx = {
            "round": self.state["round"],
            "best": self._score_str(self.state["best_score"]),
            "acceptance_margin": self.margin,
            "audiobox_diagnosis": self.state.get("diagnosis"),
            "song_description": cards.get("_mix"),
            "tracks": {s: {"description": cards.get(s),
                           "channels": int(self.ev.stems[s].shape[0]),
                           "chain": describe_list(self.best_chain["stems"][s]),
                           "level_vs_mix_lufs": self.state["levels"].get(s),
                           "history": self.state.get("past", {}).get(s, [])} for s in self.stems},
            "buses": {b: {"setting": describe_bus(b, v),
                          "history": self.state.get("past", {}).get(f"bus:{b}", [])}
                      for b, v in (self.best_chain.get("buses") or {}).items()},
            "master_chain": describe_list(self.best_chain.get("master", [])),
            "master_history": self.state.get("past", {}).get("master", []),
            "recent_best_scores": self.state["score_by_round"][-10:],
            "accepted_so_far": self.state.get("accepted_log", [])[-10:],
        }
        if self.audit_on and self.state.get("outside_ear", {}).get("issues"):
            oe = self.state["outside_ear"]
            ctx["outside_listener"] = {"rounds_ago": self.state["round"] - oe["round"],
                                       "accepted_since": oe.get("accepted_since", 0),
                                       "issues": oe["issues"]}
        if master:
            ask = ("This is a MASTER BUS round. Write the complete prompt for the master-bus agent "
                   "(its chain runs on the summed mix before the fixed limiter). Return exactly one "
                   "target with stem \"master\".")
        else:
            ask = (f"Pick 1-{n} targets (tracks or buses) to work on this round. For each, write the "
                   "complete prompt for its agent. Use exact track names; buses as 'bus:<name>'.")
        msg = f"Round {self.state['round']}. {ask} JSON.\n\n" + json.dumps(ctx, indent=1)
        try:
            out, _ = self.client.chat(self.cfg["models"]["orchestrator"],
                                      [{"role": "system", "content": ORCH_SYSTEM + (ORCH_AUDIT if self.audit_on else "")},
                                       {"role": "user", "content": msg}],
                                      schema=PLAN_SCHEMA, max_tokens=8000, tag="plan")
            targets = []
            for t in out.get("targets", []):
                stem = "master" if master else self.match_target(t.get("stem"))
                if stem and t.get("prompt"):
                    tgt = {"stem": stem, "prompt": t["prompt"]}
                    if stem.startswith("bus:"):
                        tgt["inputs"] = [x for x in (self.match_stem(i) for i in (t.get("inputs") or [])) if x]
                        tgt["sends"] = {self.match_stem(x.get("track")): x.get("db")
                                        for x in (t.get("sends") or [])
                                        if isinstance(x, dict) and self.match_stem(x.get("track"))}
                    targets.append(tgt)
            targets = targets[:n]
            if targets:
                self._log({"event": "plan", "round": self.state["round"], "targets": targets,
                           "notes": out.get("notes")})
                return targets
        except LLMError as e:
            print(f"  plan failed ({e}); falling back to the template prompt")
        return self._fallback_plan(n, master)

    def _fallback_plan(self, n: int, master: bool) -> list[dict]:
        """No orchestrator model: order stems by the diagnosis gradient, then round-robin."""
        if master:
            return [{"stem": "master", "brief": "Improve the whole mix on the master bus without "
                                                "crushing dynamics."}]
        diag = self.state.get("diagnosis") or {}
        step = diag.get("step_db", 3)
        gain = {s: max(d.get(f"minus{step:g}dB", 0), d.get(f"plus{step:g}dB", 0))
                for s, d in diag.get("stems", {}).items()}
        r = self.state["round"]
        if gain and r % 2 == 0:
            picks = sorted(self.stems, key=lambda s: -gain.get(s, 0))[:n]
        else:
            picks = [self.stems[(r * n + i) % len(self.stems)] for i in range(n)]
        out = []
        for s in dict.fromkeys(picks):
            d = diag.get("stems", {}).get(s, {})
            brief = ""
            if d:
                brief = (f"Audiobox: this stem -{step:g} dB -> {d.get(f'minus{step:g}dB', 0):+.4f}, "
                         f"+{step:g} dB -> {d.get(f'plus{step:g}dB', 0):+.4f}; "
                         f"solo PQ {d.get('solo_PQ', '?')}.")
            out.append({"stem": s, "brief": brief})
        return out

    def stem_ctx(self, stem: str, target: dict) -> dict:
        cards = self.cards()
        chain = self.best_chain
        if stem == "master":
            cur = chain["master"]
        elif stem.startswith("bus:"):
            cur = (chain.get("buses") or {}).get(stem[4:], {}).get("chain", [])
        else:
            cur = chain["stems"][stem]
        return {
            "current_chain": copy.deepcopy(cur),
            "inputs": target.get("inputs"),
            "sends": target.get("sends"),
            "prompt": target.get("prompt"),            # written by the orchestrator
            # everything below is only used by the fallback template (no orchestrator prompt)
            "brief": target.get("brief", ""),
            "best_score": self._score_str(self.state["best_score"]),
            "card": cards.get(stem) if stem in self.stems else None,
            "mix_card": cards.get("_mix"),
            "diagnosis": self.state.get("diagnosis"),
            "levels": self.state.get("levels"),
            "other_stems": {s: describe_list(v) for s, v in chain["stems"].items() if s != stem},
            "past": self.state.get("past", {}).get(stem, []),
        }

    # ============================================================== balance
    def balance_round(self) -> list[Candidate]:
        """Round 1: gain/pan/width across all stems at once, by the orchestrator."""
        cands = []
        if self.phase.orchestrator_llm:
            cards = self.cards()
            info = {s: {"description": cards.get(s), "level_vs_mix_lufs": self.state["levels"].get(s),
                        "channels": int(self.ev.stems[s].shape[0])} for s in self.stems}
            msg = ("Round 1 is static balance only: propose up to 3 different gain/pan/width settings "
                   "for ALL stems at once (gain_db -20..12, pan -100..100, width 0..1.5; width only "
                   "matters for stereo stems; fold bass/kick/lead vocal toward mono if stereo). A good "
                   "static balance gives most of a mix's quality. Use the Audiobox diagnosis: its per-stem "
                   "-3/+3 dB sensitivities are measured score changes. Every stem must stay within 18 dB of "
                   "its current level relative to the mix. JSON.\n\n" + json.dumps(
                       {"stems": info, "song_description": cards.get("_mix"),
                        "audiobox_diagnosis": self.state.get("diagnosis"),
                        "baseline": self._score_str(self.state["best_score"])}, indent=1))
            try:
                out, _ = self.client.chat(self.cfg["models"]["orchestrator"],
                                          [{"role": "system", "content": ORCH_SYSTEM},
                                           {"role": "user", "content": msg}],
                                          schema=BALANCE_SCHEMA, max_tokens=6000, tag="balance")
                self._log({"event": "balance_reply", "round": self.state["round"], "reply": out})
                for c in out.get("candidates", [])[:3]:
                    cand = self._balance_candidate(c.get("stems", {}), c.get("reason", ""))
                    if cand is not None:
                        cands.append(cand)
            except LLMError as e:
                print(f"  balance call failed: {e}")
        if not cands:
            for i in range(3):
                setting = {s: {"gain_db": self.rng.uniform(-4, 3),
                               "pan": self.rng.uniform(-50, 50) if self.ev.stems[s].shape[0] == 1 else 0.0,
                               "width": self.rng.uniform(0.7, 1.2)} for s in self.stems}
                cands.append(self._balance_candidate(setting, f"random balance {i + 1}"))
        return cands

    def match_target(self, name) -> str | None:
        """A track, 'bus:<name>' or 'master'."""
        if not isinstance(name, str):
            return None
        if name.strip().lower() == "master":
            return "master"
        if name.startswith("bus:") or name.startswith("bus "):
            b = re.sub(r"[^A-Za-z0-9_-]+", "_", name[4:].strip()).strip("_")[:32]
            return f"bus:{b}" if b and b not in self.stems else None
        if name in (self.best_chain.get("buses") or {}):
            return f"bus:{name}"
        return self.match_stem(name)

    def match_stem(self, name) -> str | None:
        """Exact stem name, or a forgiving match ('LeadVox1' -> '02_LeadVox1')."""
        if not isinstance(name, str):
            return None
        if name in self.stems:
            return name
        norm = lambda x: re.sub(r"[^a-z0-9]", "", re.sub(r"^\d+[_ -]*", "", x.lower()))  # noqa: E731
        hits = [s for s in self.stems if norm(s) == norm(name)]
        if len(hits) == 1:
            return hits[0]
        hits = [s for s in self.stems if norm(name) and (norm(name) in norm(s) or norm(s) in norm(name))]
        return hits[0] if len(hits) == 1 else None

    def _normalize_setting(self, setting) -> dict:
        """Accept {stem: {...}} or [{"stem": ..., ...}] and fuzzy stem names."""
        items = []
        if isinstance(setting, dict):
            items = [(k, v) for k, v in setting.items()]
        elif isinstance(setting, list):
            for it in setting:
                if isinstance(it, dict):
                    items.append((it.get("stem") or it.get("name"), it))
        out = {}
        for k, v in items:
            s = self.match_stem(k)
            if s and isinstance(v, dict):
                vals = {}
                for key in ("gain_db", "pan", "width"):
                    try:
                        if key in v:
                            vals[key] = float(v[key])
                    except (TypeError, ValueError):
                        pass
                if vals:
                    out[s] = {**out.get(s, {}), **vals}
        return out

    def _balance_candidate(self, setting, reason: str) -> Candidate | None:
        setting = self._normalize_setting(setting)
        if not setting:
            print(f"  balance candidate dropped (no stem names matched): {reason[:60]}")
            return None
        chain = copy.deepcopy(self.best_chain)
        for s, vals in setting.items():
            gp = chain["stems"][s][-1]
            gp["params"] = {**gp["params"], **vals}
        c = Candidate(stem="*balance*", chain=[], reason=reason, agent=self.agent_kind)
        c.full_chain = chain  # type: ignore[attr-defined]
        return c

    # ============================================================== rounds
    def _candidate_chain(self, c: Candidate) -> dict:
        if getattr(c, "full_chain", None) is not None:
            return c.full_chain  # type: ignore[attr-defined]
        if c.stem == "master":
            return with_master(self.best_chain, c.chain)
        return with_stem(self.best_chain, c.stem, c.chain)

    def run_round(self):
        st = self.state
        st["round"] += 1
        r = st["round"]
        start_best = self.best_value
        print(f"\n--- Round {r}  best {self.best_value:.4f}  (margin {self.margin:.4f}, "
              f"spend ${self.spend.total + st.get('spend_prev', 0):.2f}) ---")
        proposals: list[Candidate] = []
        notes: list[str] = []
        if r == 1 and not self.phase.single_agent:
            print("  static balance pass (all stems, gain/pan/width)")
            proposals = self.balance_round()
        elif r % self.L["master_every"] == 0 and not self.phase.single_agent:
            print("  master bus pass")
            t = self.plan(master=True)[0]
            proposals, notes = self.master_agent.propose("master", self.stem_ctx("master", t))
        else:
            targets = self.plan()
            print("  targets: " + "; ".join(f"{t['stem']}: {(t.get('prompt') or t.get('brief') or '')[:90]}"
                                            for t in targets))
            for t in targets:
                c, n = self.agent.propose(t["stem"], self.stem_ctx(t["stem"], t))
                proposals += c
                notes += n
        for n in notes:
            print(f"  note: {n}")
            self._log({"event": "note", "round": r, "note": n})

        results: list[tuple[Candidate, Result]] = []
        for c in proposals:
            res = self.ev.evaluate(self._candidate_chain(c))
            results.append((c, res))
            d = res.value - self.best_value if res.ok else None
            tag = f"{d:+.4f}" if d is not None else "FAIL " + res.reason_failed()[:120]
            print(f"  [{c.stem}] {c.reason[:70]:<70} {tag}{' (cached)' if res.cached else ''}")

        # best this round, then optional combine of winners from different stems
        winners = [(c, res) for c, res in results if res.ok and res.value > self.best_value + self.margin]
        best_c, best_res = max(results, key=lambda cr: cr[1].value, default=(None, None))
        accepted_hash = None
        self._retest = {}
        combo_pick = None
        if winners:
            best_c, best_res = max(winners, key=lambda cr: cr[1].value)
            if self.L["combine_winners"]:
                per_stem = {}
                for c, res in sorted(winners, key=lambda cr: -cr[1].value):
                    if c.stem not in per_stem and c.stem not in ("*balance*",):
                        per_stem[c.stem] = (c, res)
                if len(per_stem) >= 2:
                    (c1, r1), (c2, r2) = list(per_stem.values())[:2]
                    combo_chain = copy.deepcopy(self._candidate_chain(c1))
                    if c2.stem == "master":
                        combo_chain["master"] = copy.deepcopy(c2.chain)
                    elif c2.stem.startswith("bus:"):
                        combo_chain.setdefault("buses", {})[c2.stem[4:]] = copy.deepcopy(c2.chain)
                    else:
                        combo_chain["stems"][c2.stem] = copy.deepcopy(c2.chain)
                    combo = Candidate(stem=f"{c1.stem}+{c2.stem}", chain=[],
                                      reason=f"combine: {c1.reason[:60]} | {c2.reason[:60]}",
                                      agent="combine")
                    combo.full_chain = combo_chain  # type: ignore[attr-defined]
                    cres = self.ev.evaluate(combo_chain)
                    results.append((combo, cres))
                    print(f"  [combine] {c1.stem} + {c2.stem}: "
                          f"{(cres.value - self.best_value):+.4f}" if cres.ok else
                          f"  [combine] FAIL {cres.reason_failed()[:100]}")
                    if cres.ok and cres.value > best_res.value + self.margin:
                        combo_pick = (combo, cres)
            ranked = sorted(winners, key=lambda cr: -cr[1].value)
            if combo_pick:
                ranked.insert(0, combo_pick)
            tries = int(self.cfg["score"].get("confirm_tries", 2)) if self.state.get("confirm_margin") else 1
            for c, res in ranked[:tries]:
                if self.confirm(c, res):
                    accepted_hash = res.hash
                    self._accept(c, res)
                    break

        for c, res in results:
            own = c.stem in self.stems or c.stem == "master" or c.stem.startswith("bus:")
            if c.stem.startswith("bus:") and isinstance(c.chain, dict):
                desc = describe_bus(c.stem[4:], c.chain)
            else:
                desc = describe_list(c.chain) if own else ""
            delta = res.value - start_best if res.ok else None
            self._log({
                "event": "candidate", "round": r, "stem": c.stem, "agent": c.agent,
                "hash": res.hash[:12], "reason": c.reason, "chain": desc,
                "score": res.score.to_dict() if res.score else None, "delta": delta,
                "guard": [ch for ch in (res.guard or {}).get("checks", []) if not ch["passed"]],
                "error": res.error, "accepted": bool(res.ok and res.hash == accepted_hash),
                "new_files": c.new_files, "cached": res.cached,
            })
            if own:
                rt = self._retest.get(res.hash)
                if rt and res.hash != accepted_hash:
                    self._past(c.stem, f"r{r} passed the first bar ({fmt_delta(delta)}) but FAILED the re-test "
                                       f"({rt:+.4f} averaged over inaudible variants): {c.reason} :: {desc[:200]}")
                    continue
                if res.ok and res.hash == accepted_hash:
                    self._past(c.stem, f"r{r} ACCEPTED ({fmt_delta(delta)}): {c.reason} :: {desc[:220]}")
                elif not res.ok:
                    self._past(c.stem, f"r{r} failed ({res.reason_failed()[:140]}): {c.reason} :: {desc[:220]}")
                else:
                    self._past(c.stem, f"r{r} not better ({fmt_delta(delta)}): {c.reason} :: {desc[:220]}")

        if accepted_hash is None:
            st["rounds_without_accept"] += 1
        st["score_by_round"].append(self.best_value)
        self._save_state()

        # periodic jobs
        if accepted_hash and st["accepted_since_diagnosis"] >= self.L["diagnose_every"]:
            self.run_diagnosis()
        if accepted_hash and st["accepted_since_listen"] >= self.L["listen_every"]:
            self.listening_check()
        if r % self.L["holdout_every"] == 0:
            self.holdout()
        if self.audit_on:
            last = st.get("last_audit_round", 0)
            rwa = st["rounds_without_accept"]
            if r - last >= int(self.A.get("every", 10)):
                self.run_outside_ear(f"every {self.A.get('every', 10)} rounds")
            elif rwa >= int(self.A.get("plateau", 4)) and last <= r - rwa:
                self.run_outside_ear(f"{rwa} rounds without an accept")

    def confirm(self, c: Candidate, res: Result) -> bool:
        """Re-test: candidate and current best, both under the same 6 inaudible variants."""
        cm = self.state.get("confirm_margin")
        if not cm:
            return True
        from .judge_worker import JudgeError
        t0 = time.time()
        try:
            cand = self.ev.perturbed(res.chain)
            base = self.ev.perturbed(self.best_chain)
        except JudgeError as e:
            print(f"  re-test [{c.stem}] failed to score ({e}); not accepted")
            return False
        gain = float(np.mean(cand) - np.mean(base))
        ok = gain > cm
        self._retest[res.hash] = gain
        print(f"  re-test [{c.stem}]: {res.value - self.best_value:+.4f} first score, {gain:+.4f} averaged over "
              f"6 inaudible variants (need > {cm:.4f}) -> {'CONFIRMED' if ok else 'not confirmed'} "
              f"({time.time() - t0:.0f} s)")
        self._log({"event": "retest", "round": self.state["round"], "stem": c.stem, "hash": res.hash[:12],
                   "first": res.value - self.best_value, "averaged": gain, "need": cm, "confirmed": ok,
                   "cand_scores": cand, "best_scores": base})
        return ok

    def _accept(self, c: Candidate, res: Result):
        if not res.ok or res.score is None:
            raise ValueError(f"refusing to accept a failed candidate: {res.reason_failed()}")
        st = self.state
        st["prev_best"] = self.best_value
        print(f"  ACCEPT [{c.stem}] {c.reason[:80]}  {self.best_value:.4f} -> {res.value:.4f}")
        st["best_chain"] = res.chain
        st["best_score"] = res.score.to_dict()
        st["best_hash"] = res.hash
        st["accepted"] += 1
        st["accepted_since_diagnosis"] = st.get("accepted_since_diagnosis", 0) + 1
        st["accepted_since_listen"] += 1
        st["rounds_without_accept"] = 0
        if st.get("outside_ear"):
            st["outside_ear"]["accepted_since"] = st["outside_ear"].get("accepted_since", 0) + 1
        st["levels"] = self._levels(res.chain)
        st.setdefault("accepted_log", []).append({
            "round": st["round"], "stem": c.stem, "reason": c.reason, "hash": res.hash[:12],
            "score": res.value, "delta": res.value - st["prev_best"]})
        self._save_state()
        self._git(f"round {st['round']}: {c.stem}: {c.reason[:60]} ({res.value:.4f})")

    # ======================================================= periodic jobs
    def run_diagnosis(self):
        """The judge critiques the current best: weak windows, per-stem gradient, solo quality."""
        print("  Audiobox diagnosis of the current best ...")
        t0 = time.time()
        d = diagnose(self.ev, self.best_chain, self.cfg, step_db=self.L["diagnose_step_db"],
                     min_gain=self.margin / 2)
        self.state["diagnosis"] = d
        self.state["accepted_since_diagnosis"] = 0
        for h in d["hints"]:
            print(f"    - {h}")
        print(f"    ({time.time() - t0:.0f} s)")
        self._log({"event": "diagnosis", "round": self.state["round"], "diagnosis": d})
        self._save_state()

    def run_outside_ear(self, why: str):
        """A second model listens to the current best (neutral prompt, no reference); repeated mix
        problems + measurements go to the orchestrator as leads. Never affects scoring."""
        st = self.state
        print(f"  Outside ear ({self.A.get('model')}, {why}) ...")
        t0 = time.time()
        try:
            oe = outside_ear(self.ev, self.client, self.cfg, self.best_chain, st["baseline_chain"], st["round"])
        except (LLMError, BudgetExceeded) as e:
            if isinstance(e, BudgetExceeded):
                raise
            print(f"    failed: {e}")
            return
        oe["best_score"] = self.best_value
        oe["accepted_since"] = 0
        st["outside_ear"] = oe
        st["last_audit_round"] = st["round"]
        st.setdefault("outside_ear_log", []).append(
            {"round": st["round"], "why": why, "score": self.best_value,
             "issues": [{k: i[k] for k in ("time", "issue", "severity")} for i in oe["issues"]],
             "dropped": [{k: i.get(k) for k in ("time", "issue", "why_dropped")} for i in oe.get("dropped", [])],
             "heard": [len(r["issues"]) for r in oe.get("runs", [])]})
        for i in oe["issues"]:
            m = i["measured"]
            print(f"    - [{i['severity']}] {i['time']}: {i['issue']}  "
                  f"(section {m['range_loudness_vs_whole_song_LU']['best']:+} LU vs song, "
                  f"flat {m['range_loudness_vs_whole_song_LU']['flat']:+})")
        for i in oe.get("dropped", []):
            print(f"    (dropped: {i.get('time')}: {i.get('issue')} - {i.get('why_dropped')})")
        if not oe["issues"]:
            print(f"    nothing heard in every listen{' (' + oe['error'] + ')' if oe.get('error') else ''}")
        print(f"    (listens heard {[len(r['issues']) for r in oe.get('runs', [])]} issues; "
              f"{time.time() - t0:.0f} s)")
        self._log({"event": "outside_ear", "round": st["round"], "why": why, **oe})
        self._save_state()

    def listening_check(self):
        st = self.state
        if not self.listen_enabled or not interactive():
            print("  (listening check due - skipped: not interactive or --no-listen. "
                  "Run `python mixloop.py listen <song>` any time.)")
            self._log({"event": "listen_skipped", "round": st["round"]})
            st["accepted_since_listen"] = 0
            st["listen_pending"] = True
            self._save_state()
            return
        ck = next(c for c in st["checkpoints"] if c["id"] == st["liked_ckpt"])
        old = self.ev.matched(ck["chain"])
        new = self.ev.matched(self.best_chain)
        pick = blind_ab(old, new, self.ev.sr, self.p.renders / "listen", self.p.votes,
                        self.L["listen_clip_s"], {"round": st["round"], "old_ckpt": ck["id"],
                                                  "old_score": ck["score"], "new_score": self.best_value})
        self.apply_vote(pick)

    def apply_vote(self, pick: str):
        st = self.state
        new_id = max(c["id"] for c in st["checkpoints"]) + 1
        st["checkpoints"].append({"id": new_id, "round": st["round"], "chain": self.best_chain,
                                  "score": self.best_value, "liked": pick != "old",
                                  "n_accepted": len(st.get("accepted_log", []))})
        st["accepted_since_listen"] = 0
        st["listen_pending"] = False
        if pick == "old":
            st["old_votes_in_row"] += 1
            print(f"  You preferred the older mix ({st['old_votes_in_row']} in a row).")
            if st["old_votes_in_row"] >= 2:
                ck = next(c for c in st["checkpoints"] if c["id"] == st["liked_ckpt"])
                fooled = st.get("accepted_log", [])[ck.get("n_accepted", 0):]
                print(f"  ROLLBACK to checkpoint {ck['id']} (round {ck['round']}); "
                      f"{len(fooled)} accepted changes marked judge-fooled.")
                self._log({"event": "rollback", "round": st["round"], "to_ckpt": ck["id"],
                           "judge_fooled": [a["hash"] for a in fooled]})
                for a in fooled:
                    a["judge_fooled"] = True
                res = self.ev.evaluate(ck["chain"])
                st["best_chain"] = ck["chain"]
                st["best_score"] = res.score.to_dict()
                st["best_hash"] = res.hash
                st["levels"] = self._levels(ck["chain"])
                st["old_votes_in_row"] = 0
                self._git(f"rollback to checkpoint {ck['id']}")
        else:
            st["old_votes_in_row"] = 0
            st["liked_ckpt"] = new_id
        self._log({"event": "listen", "round": st["round"], "pick": pick, "ckpt": new_id})
        self._save_state()

    def holdout(self):
        """Overfitting check: PC/CU vs the best from N rounds ago."""
        st = self.state
        ref = st["holdout_ref"]
        now = st["best_score"]
        flags = []
        rise = (now["PQ"] + now["CE"]) - (ref["score"]["PQ"] + ref["score"]["CE"])
        if rise > self.margin:
            for ax in ("PC", "CU"):
                if now[ax] < ref["score"][ax] - self.margin:
                    flags.append(f"{ax} fell {now[ax] - ref['score'][ax]:+.3f} while PQ+CE rose {rise:+.3f}")
        for f in flags:
            print(f"  HOLD-OUT FLAG: {f}")
            st["flags"].append({"round": st["round"], "flag": f})
        self._log({"event": "holdout", "round": st["round"], "ref_round": ref["round"], "flags": flags})
        st["holdout_ref"] = {"round": st["round"], "chain": self.best_chain, "score": now}
        self._save_state()

    # ================================================================= run
    def should_stop(self, t0: float) -> str | None:
        st, L = self.state, self.L
        if st["round"] >= L["max_rounds"]:
            return f"round cap {L['max_rounds']}"
        if st["rounds_without_accept"] >= L["plateau_rounds"]:
            return f"plateau: {L['plateau_rounds']} rounds without an accepted change"
        h = st["score_by_round"]
        if len(h) > L["slow_window"] and h[-1] - h[-1 - L["slow_window"]] < L["slow_delta"]:
            return f"slow: < {L['slow_delta']} gain over the last {L['slow_window']} rounds"
        if st.get("elapsed_s", 0) + (time.time() - t0) > L["max_hours"] * 3600:
            return f"wall-clock cap {L['max_hours']} h"
        if self.spend.total + st.get("spend_prev", 0) >= L["max_usd"]:
            return f"LLM spend cap ${L['max_usd']}"
        return None

    def run(self, max_rounds: int | None = None):
        if max_rounds is not None:
            self.L["max_rounds"] = max_rounds
        self.setup()
        t0 = time.time()
        reason = None
        try:
            while True:
                reason = self.should_stop(t0)
                if reason:
                    break
                self.run_round()
        except BudgetExceeded as e:
            reason = str(e)
        except KeyboardInterrupt:
            reason = "you stopped it (Ctrl-C)"
        self.state["elapsed_s"] = self.state.get("elapsed_s", 0.0) + (time.time() - t0)
        self.state["stop_reason"] = reason
        self._save_state()
        print(f"\nSTOP: {reason}")
        if (self.state.get("accepted_since_listen", 0) > 0 or self.state.get("listen_pending")) \
                and self.listen_enabled and interactive():
            print("Final listening check:")
            self.listening_check()
        from .report import final_report
        path = final_report(self.p, self.cfg, self.ev, self.state)
        self._git(f"final: {self.best_value:.4f}")
        print(f"Report: {path}")
        return path
