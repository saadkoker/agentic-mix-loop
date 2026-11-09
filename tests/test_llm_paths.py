"""Exercise every LLM code path (orchestrator plan + balance + master, stem
agent incl. writing a plugin and the harness retry, Gemini cards / critique /
A/B) with a fake OpenRouter client. No key or network needed.

    python tests/test_llm_paths.py
"""
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
warnings.filterwarnings("ignore")

from core.config import Project, load_config  # noqa: E402
from core import llm  # noqa: E402
from test_loop import make_song  # noqa: E402

GOOD_SAT = '''"""Soft saturation with 4x oversampling."""
import numpy as np
from scipy import signal
TYPE = "sat"
PARAMS = {"drive_db": {"default": 3.0, "min": 0.0, "max": 12.0}}
def process(audio, sr, params):
    d = 10 ** (params["drive_db"] / 20)
    up = signal.resample_poly(audio, 4, 1, axis=1)
    up = np.tanh(d * up) / d
    return signal.resample_poly(up, 1, 4, axis=1)[:, : audio.shape[1]].astype(np.float32)
'''
BAD_SAT = '''import numpy as np
TYPE = "sat"
PARAMS = {"drive_db": {"default": 12.0, "min": 0.0, "max": 24.0}}
def process(audio, sr, params):
    d = 10 ** (params["drive_db"] / 20)
    return (np.tanh(d * audio) / d).astype(np.float32)
'''

calls = []
sent = []
OPUS_PROMPT = 'OPUS-WRITTEN TASK: cut 300 Hz on the pad by 2-4 dB, three variants.'


def fake_chat(self, model, messages, schema=None, max_tokens=8000, temperature=None, tag="", reasoning=None):
    calls.append((tag, model))
    sent.append((tag, list(messages)))
    self.spend.add(0.001, {"model": model, "tag": tag})
    user = messages[-1]["content"]
    if tag.startswith("card"):
        has_audio = any(p.get("type") == "input_audio" for p in user) if isinstance(user, list) else False
        assert has_audio, "card request must carry audio"
        return {"stem": "x", "role": "test role",
                "elements": [{"name": "tone", "character": "pure", "when": "0:00-0:30"}]}, {}
    if tag.startswith("ab"):
        return {"better": "B", "confidence": 0.7, "why": "clearer"}, {}
    if tag == "plan":
        assert "audiobox_diagnosis" in user, "orchestrator must see the Audiobox diagnosis"
        if "MASTER BUS round" in user:
            return {"targets": [{"stem": "master", "prompt": "OPUS MASTER TASK: gentle glue comp."}]}, {}
        return {"targets": [{"stem": "pad", "prompt": OPUS_PROMPT},
                            {"stem": "nope", "prompt": "invalid stem is dropped"}]}, {}
    if tag == "balance":
        return {"candidates": [{"reason": "bass up", "stems": [{"stem": "bass", "gain_db": 2}, {"stem": "PAD", "pan": 20}]}]}, {}
    if tag.startswith("stem:") and tag.endswith(":retry"):
        return {"candidates": [{"reason": "fixed: oversampled", "chain": [
            {"plugin": "new:warm", "params": {"drive_db": 4}}, {"plugin": "stock:gain_pan", "params": {}}],
            "new_plugins": [{"name": "warm", "code": GOOD_SAT}]}]}, {}
    if tag.startswith("stem:"):
        return {"candidates": [
            {"reason": "cut mud (gain_pan first on purpose -> repaired)", "chain": [
                {"plugin": "stock:gain_pan", "params": {"gain_db": -1}},
                {"plugin": "stock:eq", "params": {"peak1_hz": 300, "peak1_db": -3}}]},
            {"reason": "naive saturation (should fail harness)", "chain": [
                {"plugin": "new:fuzz", "params": {}}, {"plugin": "stock:gain_pan", "params": {}}],
             "new_plugins": [{"name": "fuzz", "code": BAD_SAT}]},
        ]}, {}
    raise AssertionError(f"unexpected tag {tag}")


def main():
    llm.OpenRouter.chat = fake_chat
    llm.OpenRouter.available = property(lambda self: True)
    import os
    os.environ.setdefault("OPENROUTER_API_KEY", "fake")
    from core.orchestrator import Orchestrator

    tmp = Path(tempfile.mkdtemp(prefix="mixloop_llm_"))
    fails = []

    def ok(c, m):
        print(f"  {'ok  ' if c else 'FAIL'} {m}")
        if not c:
            fails.append(m)

    try:
        make_song(tmp, seconds=20)
        pr = Project(tmp)
        cfg = load_config(None, {"loop": {"phase": 5, "diagnose_every": 1, "master_every": 3, "agent_mode": "oneshot"}})
        o = Orchestrator(pr, cfg, agent_kind="llm", judge_kind="dummy", listen=False)
        o.setup()
        cards = list((pr.cards).glob("*.json"))
        bal = [m for t, m in sent if t == "balance"]
        ok(len(cards) == 5, f"cards for 4 stems + mix ({len(cards)})")
        o.run_round()   # round 1: balance
        ok(any(t == "balance" for t, _ in calls), "balance call made by orchestrator model")
        ok(o.state.get("diagnosis") and o.state["diagnosis"]["stems"], "Audiobox diagnosis computed at setup")
        ok(not any(t.startswith("critique") or t.startswith("ab") for t, _ in calls), "no Gemini critique or A/B in the loop")
        card = json.loads((pr.cards / "pad.json").read_text())
        ok(set(card) == {"stem", "role", "elements"}, f"cards are description-only ({sorted(card)})")
        o.run_round()   # round 2: plan + stem agent (+ harness retry)
        tags = [t for t, _ in calls]
        ok("plan" in tags, "plan call")
        ok(any(t == "stem:pad" for t in tags) and any(t == "stem:pad:retry" for t in tags),
           "stem agent called, harness failure triggered one retry")
        stem_msgs = [m for t, m in sent if t == "stem:pad"]
        um = stem_msgs[0][-1]["content"] if stem_msgs else ""
        ok(OPUS_PROMPT in um and "Technical appendix" in um, "stem agent gets the orchestrator's prompt + fixed appendix")
        ok("Audiobox diagnosis" not in um and "Already tried" not in um,
           "stem agent sees only what the orchestrator chose (no auto-attached context)")
        files = sorted(p.name for p in (pr.plugins / "pad").glob("*.py"))
        approved = json.loads((pr.plugins / "approved.json").read_text())
        ok(any("warm" in f for f in files) and any("warm" in m["path"] for m in approved.values()),
           f"fixed plugin written and approved ({files})")
        ok(not any("fuzz" in m["path"] for m in approved.values()), "naive saturation rejected")
        hist = [json.loads(ln) for ln in pr.history.read_text().splitlines()]
        r2 = [h for h in hist if h.get("event") == "candidate" and h["round"] == 2]
        ok(len(r2) >= 2, f"round 2 candidates evaluated ({len(r2)})")
        ok(all(not (h["error"] or "").startswith("invalid chain") for h in r2), "gain_pan order auto-repaired")
        o.run_round()   # round 3: master pass (orchestrator model)
        ok(any(t == "stem:master" and m == cfg["models"]["orchestrator"] for t, m in calls),
           "master bus uses the orchestrator model")
        mm = [m for t, m in sent if t == "stem:master"]
        ok(mm and "OPUS MASTER TASK" in mm[0][-1]["content"], "orchestrator wrote the master-bus prompt")
        ok(o.spend.total > 0 and (pr.root / "llm_log.jsonl").exists(), f"spend tracked (${o.spend.total:.3f})")
        ab = o.listener.ab(o.ev.matched(o.state["baseline_chain"]), o.ev.matched(o.best_chain), o.ev.sr)
        ok(ab["verdict"] == "split", f"Phase 0 A/B experiment: B both times = split ({ab['verdict']})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL OK" if not fails else f"{len(fails)} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
