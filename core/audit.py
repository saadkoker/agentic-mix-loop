"""Gemini audit: the old A/B veto, run on the final mix, plus known-answer controls.

For each song:
  final      flat vs best mix          (x2 repeats, each run in both orders)
  identical  flat vs flat              (right answer: same)
  crushed    flat vs over-compressed   (right answer: flat)
  harsh      flat vs harsh top end     (right answer: flat; Audiobox misses this one)

Every comparison is the veto's: full songs, loudness-matched, A then B, asked twice with the
order swapped. A verdict only counts when both orders agree. Nothing here feeds the loop.
"""
from __future__ import annotations

import json
import time

from . import audio as A
from .config import Project
from .evaluate import Evaluator
from .gemini import Listener
from .llm import OpenRouter, Spend
from .phase0 import variants

# expected: which side should win ("old" = first argument = flat), or "same"
CONTROLS = [("identical", "flat", "same"), ("crushed", "over_compressed", "old"),
            ("harsh", "harsh_top", "old")]


def correct(r: dict, expect: str | None) -> bool | None:
    """Both orders must give the expected answer. ab() reports 'same'+'same' as a split
    (it only names a winner), so check the two runs directly."""
    if not expect:
        return None
    return r["run1"] == expect and r["run2"] == expect


def audit_song(project: Project, cfg: dict, repeats: int = 2) -> dict:
    ev = Evaluator(project, cfg, "dummy")          # renders only; no GPU judge needed
    state = json.loads(project.state.read_text())
    sr = ev.sr
    v = variants(ev, cfg)
    flat = v["flat"]
    best = ev.matched(state["best_chain"])
    best, _ = A.normalize_lufs(best, sr, cfg["audio"]["target_lufs"])
    spend = Spend(5.0, project.root / "llm_log.jsonl")
    lis = Listener(cfg, OpenRouter(cfg, spend), project.cards / "log")
    rows = []

    def run(name, old, new, expect):
        t0 = time.time()
        r = lis.ab(old, new, sr)
        ok = correct(r, expect)
        rows.append({"pair": name, "expect": expect, **r, "correct": ok,
                     "seconds": round(time.time() - t0, 1)})
        print(f"  {name:12s} verdict {r['verdict']:6s} (order1 {r['run1']}, order2 {r['run2']})"
              + (f"  expected {expect}: {'OK' if ok else 'WRONG'}" if expect else ""), flush=True)
        for w in r["why"]:
            print(f"      - {w}", flush=True)

    print(f"\n=== Gemini audit: {project.root.name} ({cfg['models']['listener']}) ===", flush=True)
    for i in range(repeats):
        run(f"final#{i + 1}", flat, best, None)           # old = flat, new = best mix
    for name, var, expect in CONTROLS:
        if var in v:
            run(name, flat, v[var], expect)
    out = {"song": project.root.name, "model": cfg["models"]["listener"], "rows": rows,
           "usd": round(spend.total, 4),
           "score": {"flat": state["baseline_score"]["score"], "best": state["best_score"]["score"]}}
    (project.root / "gemini_audit.json").write_text(json.dumps(out, indent=1))
    print(f"  spend ${spend.total:.3f} -> {project.root / 'gemini_audit.json'}", flush=True)
    return out


# ------------------------------------------------------------------ open listen
OPEN_PROMPT = ("Listen to this song. Is there anything wrong with the mix? List every issue you hear, "
               "each with its time range and a short description. If you hear nothing wrong, return an "
               "empty list.")
OPEN_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {"type": "array", "items": {
            "type": "object",
            "properties": {"time": {"type": "string"}, "issue": {"type": "string"},
                           "severity": {"type": "string", "enum": ["minor", "moderate", "major"]}},
            "required": ["time", "issue", "severity"]}},
        "overall": {"type": "string"}},
    "required": ["issues", "overall"],
}


def open_listen(project: Project, cfg: dict, wav, tag: str) -> dict:
    """One neutral, single-mix question: no reference, no system prompt, nothing about what changed."""
    x, sr = A.read(wav)
    spend = Spend(2.0, project.root / "llm_log.jsonl")
    lis = Listener(cfg, OpenRouter(cfg, spend), None)
    parts = lis._parts(x, sr)
    content = [{"type": "text", "text": OPEN_PROMPT}] + lis._audio_content("Song", parts, x.shape[1] / sr)
    out, _ = lis.client.chat(lis.model, [{"role": "user", "content": content}], schema=OPEN_SCHEMA,
                             max_tokens=4000, tag=f"open:{tag}")
    out["usd"] = round(spend.total, 4)
    return out
