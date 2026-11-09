"""End-to-end logic tests on a small synthetic song, using the dummy judge and
the offline random agent (no GPU or API key needed).

    python tests/test_loop.py
"""
import copy
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
from core.config import Project, load_config  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from core.plugins import with_master, with_stem  # noqa: E402
from core.stem_agent import Candidate  # noqa: E402

SR = 44100


def make_song(root: Path, seconds: float = 30.0):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(0)
    beat = (t % 0.5)
    kick = (np.sin(2 * np.pi * 55 * beat) * np.exp(-beat * 18)).astype(np.float32)
    bass = (0.3 * np.sin(2 * np.pi * 55 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t))).astype(np.float32)
    hat_env = np.exp(-((t * 4) % 1) * 40)
    hat = np.stack([rng.standard_normal(n) * hat_env * 0.05, rng.standard_normal(n) * hat_env * 0.05])
    pad = 0.08 * np.stack([np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t),
                           np.sin(2 * np.pi * 220.5 * t) + np.sin(2 * np.pi * 330 * t)])
    stems = root / "stems"
    stems.mkdir(parents=True)
    A.write(stems / "kick.wav", kick[None] * 0.8, SR, "FLOAT")
    A.write(stems / "bass.wav", bass[None], SR, "FLOAT")
    A.write(stems / "hats.wav", hat.astype(np.float32), SR, "FLOAT")
    A.write(stems / "pad.wav", pad.astype(np.float32), SR, "FLOAT")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="mixloop_test_"))
    fails = []

    def ok(cond, msg):
        print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
        if not cond:
            fails.append(msg)

    try:
        make_song(tmp)
        pr = Project(tmp)
        cfg = load_config(None, {"loop": {"phase": 3, "listen_every": 1, "holdout_every": 3}})
        o = Orchestrator(pr, cfg, agent_kind="random", judge_kind="dummy", listen=False, seed=1)
        o.setup()
        ok(o.state["baseline_score"]["score"] > 0, "baseline scored")
        ok(o.margin >= cfg["score"]["min_margin"], f"margin set ({o.margin:.4f})")
        ev = o.ev
        flat = o.best_chain

        # loudness invariance: every stem +6 dB -> same score
        loud = copy.deepcopy(flat)
        for s in loud["stems"]:
            loud["stems"][s][-1]["params"]["gain_db"] = 6.0
        r0, r1 = ev.evaluate(flat), ev.evaluate(loud)
        ok(r1.ok and abs(r1.value - r0.value) < 1e-3, f"whole-mix gain buys nothing ({r1.value - r0.value:+.5f})")

        # stem presence: bury the bass
        buried = with_stem(flat, "bass", [{"plugin": "stock:eq", "params": {"lp_hz": 2000, "low_shelf_db": -12}},
                                          {"plugin": "stock:gain_pan", "params": {"gain_db": -20}}])
        rb = ev.evaluate(buried)
        ok(not rb.ok and "stem_presence" in rb.reason_failed(), f"burying a stem fails ({rb.reason_failed()[:60]})")

        # PLR: crush the master
        crushed = with_master(flat, [{"plugin": "stock:comp", "params": {"threshold_db": -40, "ratio": 10,
                                                                          "attack_ms": 0.1, "makeup_db": 18}},
                                     {"plugin": "stock:clip", "params": {"input_db": 12, "ceiling_db": -24, "softness": 0}}])
        rc = ev.evaluate(crushed)
        ok(not rc.ok and "plr" in rc.reason_failed(), f"crushing the master fails PLR ({rc.reason_failed()[:60]})")

        # chain rules
        bad = with_stem(flat, "pad", [{"plugin": "stock:gain_pan", "params": {}},
                                      {"plugin": "stock:eq", "params": {}}])
        ok(not ev.evaluate(bad).ok, "gain_pan must be last")
        too_many = with_stem(flat, "pad", [{"plugin": "stock:sat", "params": {}}] * 2 +
                             [{"plugin": "stock:gain_pan", "params": {}}])
        ok(not ev.evaluate(too_many).ok, "per-type limit enforced")
        ok(ev.evaluate(flat).cached, "repeat chain is served from cache")

        # a few real rounds
        o.L["max_rounds"] = 4
        for _ in range(4):
            o.run_round()
        hist = pr.history.read_text().splitlines()
        ok(len(hist) > 5, f"history written ({len(hist)} lines)")

        # forced accept -> checkpoint -> two 'old' votes -> rollback
        first = o.best_chain
        for i in range(2):
            ch = with_stem(o.best_chain, "pad", [{"plugin": "stock:gain_pan", "params": {"gain_db": -1.0 - i}}])
            res = ev.evaluate(ch)
            assert res.ok, res.reason_failed()
            o._accept(Candidate(stem="pad", chain=ch["stems"]["pad"], reason=f"test accept {i}"), res)
            if i == 0:
                o.apply_vote("new")            # you liked change 0
                liked = o.best_chain
        o.apply_vote("old")
        ok(o.state["old_votes_in_row"] == 1, "one 'old' vote doesn't roll back")
        ch = with_stem(o.best_chain, "pad", [{"plugin": "stock:gain_pan", "params": {"gain_db": -4.0}}])
        o._accept(Candidate(stem="pad", chain=ch["stems"]["pad"], reason="test accept 2"), ev.evaluate(ch))
        o.apply_vote("old")
        ok(o.best_chain == liked, "two 'old' votes in a row roll back to the last liked checkpoint")
        fooled = [a for a in o.state["accepted_log"] if a.get("judge_fooled")]
        ok(len(fooled) >= 2, f"changes after the checkpoint marked judge-fooled ({len(fooled)})")
        ok(first != liked, "sanity: checkpoint differs from start")

        o.holdout()
        ok(any('"holdout"' in ln for ln in pr.history.read_text().splitlines()), "hold-out check logged")

        from core.report import final_report
        rp = final_report(pr, cfg, ev, o.state)
        ok(rp.exists() and (pr.renders / "final" / "3_best_mix_loud.wav").exists(), "report + three renders")
        loud_x, _ = A.read(pr.renders / "final" / "3_best_mix_loud.wav")
        ok(A.true_peak(loud_x) <= -0.99, f"loud master true peak {A.true_peak(loud_x):.2f} dBTP")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL OK" if not fails else f"{len(fails)} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
