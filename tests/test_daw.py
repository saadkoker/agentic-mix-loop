"""Agentic DAW path with a fake OpenRouter client (no key, no network, dummy judge):
raw import (files untouched), tool-using agents (measure, write a plugin, audition, submit with
automation), a bus target created by the orchestrator, the master pass, and forced submit.

    python tests/test_daw.py
"""
import hashlib
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from core import audio as A  # noqa: E402
from core import llm  # noqa: E402
from core.config import Project, load_config  # noqa: E402
from core.importer import import_raw  # noqa: E402

SR = 44100
GOOD_SAT = '''"""Soft saturation with 4x oversampling."""
import numpy as np
from scipy import signal
TYPE = "sat"
PARAMS = {"drive_db": {"default": 0.0, "min": 0.0, "max": 12.0}}
def process(audio, sr, params):
    d = 10 ** (params["drive_db"] / 20)
    if d <= 1.0:
        return np.asarray(audio, dtype=np.float32).copy()
    up = signal.resample_poly(audio, 4, 1, axis=1)
    up = np.tanh(d * up) / d
    return signal.resample_poly(up, 1, 4, axis=1)[:, : audio.shape[1]].astype(np.float32)
'''

calls, tool_results, forced = [], {}, []


def make_raw_session(src: Path, seconds: float = 20.0):
    """Delivered files: different lengths, one 24-bit PCM, one mono vocal-like line."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(0)
    beat = t % 0.5
    kick = (np.sin(2 * np.pi * 55 * beat) * np.exp(-beat * 18)).astype(np.float32)
    hat = np.stack([rng.standard_normal(n), rng.standard_normal(n)]) * np.exp(-((t * 4) % 1) * 40) * 0.05
    pad = 0.08 * np.stack([np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t),
                           np.sin(2 * np.pi * 220.5 * t) + np.sin(2 * np.pi * 330 * t)])
    # "vocal": a sharp A4 sung loud in the first phrase, quiet in the second
    f = 440 * 2 ** (30 / 1200)
    ph = 2 * np.pi * f * t
    voice = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 10)) * 0.2
    env = np.zeros(n)
    env[int(1 * SR):int(6 * SR)] = 1.0
    env[int(9 * SR):int(14 * SR)] = 0.1
    src.mkdir(parents=True)
    A.write(src / "01_Kick.wav", kick[None] * 0.8, SR, "PCM_24")
    A.write(src / "02_Hats.wav", hat.astype(np.float32), SR, "FLOAT")
    A.write(src / "03_Pad.wav", pad.astype(np.float32)[:, : n - SR], SR, "FLOAT")   # 1 s shorter
    A.write(src / "04_Vox.wav", (voice * env)[None].astype(np.float32), SR, "FLOAT")


def fake_chat(self, model, messages, schema=None, max_tokens=8000, temperature=None, tag="", reasoning=None):
    calls.append(("chat", tag, model))
    self.spend.add(0.001, {"model": model, "tag": tag})
    user = messages[-1]["content"]
    if tag.startswith("card"):
        return {"stem": "x", "role": "test role", "elements": [{"name": "a", "character": "b", "when": "0:00-0:20"}]}, {}
    if tag == "balance":
        return {"candidates": [{"reason": "vox up", "stems": [{"stem": "04_Vox", "gain_db": 2}]}]}, {}
    if tag == "plan":
        if "MASTER BUS round" in user:
            return {"targets": [{"stem": "master", "prompt": "MASTER TASK"}]}, {}
        if '"buses": {}' in user or "bus:music" not in user:
            return {"targets": [{"stem": "04_Vox", "prompt": "VOX TASK: fix the level jump between phrases, tune it."},
                                {"stem": "bus:music", "prompt": "BUS TASK: glue pad and hats.",
                                 "inputs": ["03_Pad", "Hats"]}]}, {}
        return {"targets": [{"stem": "01_Kick", "prompt": "KICK TASK (agent never submits)"}]}, {}
    raise AssertionError(f"unexpected chat tag {tag}")


def _call(i, name, args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def fake_tools(self, model, messages, tools, tool_choice="auto", max_tokens=8000, tag="", temperature=None):
    calls.append(("tools", tag, model))
    self.spend.add(0.001, {"model": model, "tag": tag})
    target = tag.split(":", 1)[1]
    step = sum(1 for m in messages if m.get("role") == "assistant")
    for m in messages:
        if m.get("role") == "tool":
            tool_results.setdefault(target, {})[m["tool_call_id"]] = json.loads(m["content"])
    if isinstance(tool_choice, dict):
        forced.append(target)
        return {"role": "assistant", "content": "", "tool_calls": [_call(99, "submit", {"candidates": [
            {"reason": "forced submit", "chain": [{"plugin": "stock:gain_pan", "params": [{"name": "gain_db", "value": -1}]}]}]})]}
    if target == "04_Vox":
        if step == 0:
            return {"role": "assistant", "content": "measuring", "tool_calls": [
                _call(1, "song_info", {}), _call(2, "levels", {"track": "04_Vox"}),
                _call(3, "pitch", {"track": "Vox"}), _call(4, "spectrum", {"target": "04_Vox"}),
                _call(5, "judge_view", {})]}
        if step == 1:
            return {"role": "assistant", "content": "", "tool_calls": [
                _call(6, "write_plugin", {"name": "warm", "code": GOOD_SAT}),
                _call(7, "plugin_source", {"name": "tune"})]}
        if step == 2:
            lane = [{"t": 0, "v": 0}, {"t": 8.0, "v": 0}, {"t": 8.1, "v": 12}, {"t": 15, "v": 12}, {"t": 15.1, "v": 0}]
            return {"role": "assistant", "content": "", "tool_calls": [_call(8, "audition", {"candidate": {
                "reason": "ride the quiet phrase up", "chain": [
                    {"plugin": "stock:gain_pan", "params": [{"name": "gain_db", "auto": lane, "curve": "linear"}]}]}})]}
        path = tool_results["04_Vox"]["c6"].get("path")
        lane = [{"t": 0, "v": 0}, {"t": 8.0, "v": 0}, {"t": 8.1, "v": 12}, {"t": 15, "v": 12}, {"t": 15.1, "v": 0}]
        return {"role": "assistant", "content": "", "tool_calls": [_call(9, "submit", {"candidates": [
            {"reason": "ride + tune", "chain": [
                {"plugin": "stock:tune", "params": [{"name": "key", "value": 9}, {"name": "scale", "value": 0},
                                                    {"name": "strength", "value": 1.0}, {"name": "retune_ms", "value": 20}]},
                {"plugin": "stock:gain_pan", "params": [{"name": "gain_db", "auto": lane, "curve": "linear"}]}]},
            {"reason": "custom warmth", "chain": [
                {"plugin": path, "params": [{"name": "drive_db", "value": 4}]},
                {"plugin": "stock:gain_pan", "params": []}]}]})]}
    if target == "bus:music":
        return {"role": "assistant", "content": "", "tool_calls": [_call(1, "submit", {"candidates": [
            {"reason": "glue comp", "inputs": ["03_Pad", "02_Hats"], "chain": [
                {"plugin": "stock:comp", "params": [{"name": "threshold_db", "value": -24}, {"name": "ratio", "value": 2}]},
                {"plugin": "stock:gain_pan", "params": [{"name": "gain_db", "value": 1}]}]}]})]}
    if target == "master":
        return {"role": "assistant", "content": "", "tool_calls": [_call(1, "submit", {"candidates": [
            {"reason": "tilt", "chain": [{"plugin": "stock:eq", "params": [{"name": "high_shelf_db", "value": 1}]}]}]})]}
    # 01_Kick: keeps measuring, never submits -> forced on the last step
    return {"role": "assistant", "content": "", "tool_calls": [_call(step, "levels", {"track": "01_Kick"})]}


def main():
    llm.OpenRouter.chat = fake_chat
    llm.OpenRouter.chat_tools = fake_tools
    llm.OpenRouter.available = property(lambda self: True)
    import os
    os.environ.setdefault("OPENROUTER_API_KEY", "fake")
    from core.orchestrator import Orchestrator

    tmp = Path(tempfile.mkdtemp(prefix="mixloop_daw_"))
    fails = []

    def ok(c, m):
        print(f"  {'ok  ' if c else 'FAIL'} {m}")
        if not c:
            fails.append(m)

    try:
        src, song = tmp / "delivered", tmp / "song"
        make_raw_session(src)
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()}
        import_raw(src, song)
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (song / "tracks").iterdir()}
        ok(before == after, "raw import: every file copied byte-for-byte, nothing grouped or converted")
        pr = Project(song)
        cfg = load_config(None, {"loop": {"phase": 5, "diagnose_every": 1, "master_every": 3,
                                          "agent_mode": "tools", "agent_max_steps": 5}})
        o = Orchestrator(pr, cfg, agent_kind="llm", judge_kind="dummy", listen=False)
        ok(sorted(o.stems) == ["01_Kick", "02_Hats", "03_Pad", "04_Vox"], f"tracks = files ({o.stems})")
        ok(len({x.shape[1] for x in o.ev.stems.values()}) == 1, "shorter track padded in memory only")
        o.setup()
        o.run_round()      # 1: balance
        o.run_round()      # 2: vox agent (tools) + new bus
        r = tool_results.get("04_Vox", {})
        ok("tempo_bpm_estimate" in r.get("c1", {}) and "04_Vox" in r["c1"].get("tracks", {}), "song_info tool")
        lv = r.get("c2", {})
        ok(len(lv.get("phrases", [])) >= 2 and lv.get("phrase_level_spread_db", 0) > 15,
           f"levels tool sees the two phrases ~20 dB apart ({lv.get('phrase_level_spread_db')} dB)")
        pt = r.get("c3", {})
        ok(pt.get("notes") and abs(pt["notes"][0]["cents"] - 30) < 6 and pt["notes"][0]["note"].startswith("A"),
           f"pitch tool: A sung ~+30 cents ({pt.get('notes', [{}])[0] if pt.get('notes') else pt})")
        ok("share_of_mix_db" in r.get("c4", {}), "spectrum tool")
        ok("weakest_windows" in r.get("c5", {}), "judge_view tool")
        ok(str(r.get("c6", {}).get("path", "")).startswith("plugins/04_Vox/"), f"write_plugin passes harness ({r.get('c6')})")
        ok("def process" in r.get("c7", {}).get("source", ""), "plugin_source tool")
        au = r.get("c8", {})
        ok("score_change" in au and au.get("auditions_left") == cfg["loop"]["agent_auditions"] - 1,
           f"audition scores a draft with automation ({au})")
        hist = [json.loads(ln) for ln in pr.history.read_text().splitlines()]
        r2 = [h for h in hist if h.get("event") == "candidate" and h["round"] == 2]
        ok(any("auto[" in (h.get("chain") or "") and "tune(" in (h.get("chain") or "") for h in r2),
           "submitted candidate carries volume automation + tuner")
        ok(any("plugins/04_Vox/" in (h.get("chain") or "") for h in r2), "submitted candidate uses the agent's plugin")
        bus = [h for h in r2 if h["stem"] == "bus:music"]
        ok(bus and bus[0]["ok"] if "ok" in bus[0] else bus and bus[0].get("score") is not None,
           f"bus candidate rendered and scored ({bus[0].get('chain') if bus else None})")
        ok(all(not (h.get("error") or "").startswith("invalid chain") for h in r2),
           f"all round-2 candidates valid ({[h.get('error') for h in r2 if h.get('error')]})")
        o.run_round()      # 3: master
        ok(any(c[0] == "tools" and c[1] == "agent:master" and c[2] == cfg["models"]["orchestrator"] for c in calls),
           "master agent is tool-using and runs on the orchestrator model")
        o.run_round()      # 4: kick agent never submits -> forced submit on the last step
        ok("01_Kick" in forced, "agent that never submits is forced to submit on its last step")
        if o.best_chain.get("buses"):
            ok("music" in o.best_chain["buses"], "accepted bus is in the session")
        rr = o.ev.render(o.best_chain)
        ok(np.all(np.isfinite(rr.final)), "best session renders")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL OK" if not fails else f"{len(fails)} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
