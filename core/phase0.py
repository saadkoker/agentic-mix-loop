"""Phase 0: test the judge before building anything on top of it.

Per song: score the flat sum, the pro mix (any audio file in reference/, if
you have one), and deliberately broken versions: vocal -10 dB, bass muted,
everything over-compressed, harsh top end. Measure judge noise. Optionally
run Gemini's A/B on the same pairs, and your own blind votes to see whether
Gemini agrees with you well above chance (the condition for turning the
Gemini veto on).

Pass for a song: the flat sum (and the pro mix, if present) beats every broken
version by more than the noise margin. Pass overall: most songs pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pedalboard as pb

from . import audio as A
from .config import Project
from .evaluate import Evaluator
from .gemini import Listener, safe
from .importer import classify
from .listen import blind_ab, interactive
from .llm import OpenRouter, Spend
from .plugins import flat_chain


def _finish(pre: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Same fixed final step as the renderer, then loudness-match."""
    staged, _ = A.normalize_lufs(pre, sr, cfg["audio"]["target_lufs"])
    final = A.limit(staged, sr, cfg["audio"]["ceiling_dbtp"])
    return A.normalize_lufs(final, sr, cfg["audio"]["target_lufs"])[0]


def variants(ev: Evaluator, cfg: dict) -> dict[str, np.ndarray]:
    sr = ev.sr
    r = ev.render(flat_chain(ev.stem_names))
    stems = r.stems
    kind = {s: classify(s) for s in stems}
    vocal = [s for s, k in kind.items() if k in ("vocals",)] or [s for s, k in kind.items() if k == "backing_vocals"]
    bass = [s for s, k in kind.items() if k == "bass"]
    flat = sum(stems.values())
    out = {"flat": _finish(flat, sr, cfg)}
    if vocal:
        g = np.float32(A.undb(-10))
        out["vocal_-10dB"] = _finish(sum(y * (g if s in vocal else 1) for s, y in stems.items()), sr, cfg)
    if bass:
        out["bass_muted"] = _finish(sum(y for s, y in stems.items() if s not in bass), sr, cfg)
    crushed = pb.Pedalboard([pb.Gain(gain_db=-A.sample_peak(flat)),
                             pb.Compressor(threshold_db=-32, ratio=10, attack_ms=0.5, release_ms=40),
                             pb.Gain(gain_db=20), pb.Clipping(threshold_db=-3)])(flat, sr)
    out["over_compressed"] = _finish(np.asarray(crushed, np.float32), sr, cfg)
    harsh = pb.Pedalboard([pb.HighShelfFilter(cutoff_frequency_hz=5000, gain_db=10, q=0.707),
                           pb.PeakFilter(cutoff_frequency_hz=3000, gain_db=6, q=1.0)])(flat, sr)
    out["harsh_top"] = _finish(np.asarray(harsh, np.float32), sr, cfg)
    return out


def references(project: Project, sr: int, cfg: dict) -> dict[str, np.ndarray]:
    out = {}
    if project.reference.exists():
        for p in sorted(project.reference.iterdir()):
            if p.suffix.lower() in (".wav", ".flac", ".aif", ".aiff"):
                x, rsr = A.read(p)
                if rsr != sr:
                    from scipy import signal
                    g = np.gcd(rsr, sr)
                    x = signal.resample_poly(x, sr // g, rsr // g, axis=1).astype(np.float32)
                out[f"pro:{p.stem}"] = A.normalize_lufs(A.to_stereo(x), sr, cfg["audio"]["target_lufs"])[0]
    return out


def run_song(project: Project, cfg: dict, judge_kind: str = "audiobox",
             gemini: bool = False, human: bool = False) -> dict:
    print(f"\n=== Phase 0: {project.root.name} ===")
    ev = Evaluator(project, cfg, judge_kind)
    sr = ev.sr
    v = variants(ev, cfg)
    refs = references(project, sr, cfg)
    scores = {}
    for name, x in {**refs, **v}.items():
        s = ev.judge.score(x, sr)
        scores[name] = s.to_dict(windows=True)     # windows kept: any weighting can be re-tested
        print(f"  {name:<18} {s.short()}")
    ev.setup_baseline()
    noise = ev.measure_noise(ev.baseline_result.chain)
    m = noise["margin"]
    print(f"  noise spread {noise['spread']:.4f} -> margin {m:.4f}")

    broken = [k for k in v if k != "flat"]
    goods = ["flat"] + list(refs)
    checks = []
    for g in goods:
        for b in broken:
            d = scores[g]["score"] - scores[b]["score"]
            checks.append({"good": g, "broken": b, "delta": d, "ok": d > m})
    for r_ in refs:
        d = scores[r_]["score"] - scores["flat"]["score"]
        checks.append({"good": r_, "broken": "flat", "delta": d, "ok": d > m})
    for c in checks:
        print(f"  {'ok ' if c['ok'] else 'BAD'} {c['good']} vs {c['broken']}: {c['delta']:+.4f}")
    passed = all(c["ok"] for c in checks)

    result = {"song": project.root.name, "judge": judge_kind, "scores": scores, "noise": noise,
              "checks": checks, "passed": passed}

    pairs = [(c["good"], c["broken"]) for c in checks]
    allv = {**refs, **v}
    if gemini:
        client = OpenRouter(cfg, Spend(cfg["loop"]["max_usd"], project.root / "llm_log.jsonl"))
        if not client.available:
            print("  Gemini A/B skipped: OPENROUTER_API_KEY not set")
        else:
            lis = Listener(cfg, client, project.cards / "log")
            g_res = []
            for good, bad in pairs:
                ab = safe(lis.ab, allv[bad], allv[good], sr)   # old = broken, new = good
                if ab:
                    g_res.append({"good": good, "broken": bad, "verdict": ab["verdict"],
                                  "agrees_with_expected": ab["verdict"] == "new"})
                    print(f"  gemini {good} vs {bad}: {ab['verdict']}")
            result["gemini_ab"] = g_res
            if g_res:
                result["gemini_expected_rate"] = float(np.mean([g["agrees_with_expected"] for g in g_res]))
    if human:
        if not interactive():
            print("  human A/B skipped: not an interactive terminal")
        else:
            h_res = []
            for good, bad in pairs:
                pick = blind_ab(allv[bad], allv[good], sr, project.renders / "listen", project.votes,
                                cfg["loop"]["listen_clip_s"], {"phase0": True, "good": good, "broken": bad})
                h_res.append({"good": good, "broken": bad, "pick": pick})
            result["human_ab"] = h_res
            if result.get("gemini_ab"):
                gm = {(g["good"], g["broken"]): g["verdict"] for g in result["gemini_ab"]}
                agree = [gm.get((h["good"], h["broken"])) == h["pick"] for h in h_res
                         if h["pick"] != "same" and gm.get((h["good"], h["broken"])) in ("new", "old")]
                if agree:
                    rate = float(np.mean(agree))
                    result["gemini_human_agreement"] = rate
                    print(f"  Gemini agrees with you on {rate:.0%} of {len(agree)} decided pairs "
                          f"(chance is 50%; Gemini A/B is an experiment only and never used by the loop)")
    project.phase0.write_text(json.dumps(result, indent=1))
    print(f"  {'PASS' if passed else 'FAIL'} -> {project.phase0}")
    return result


def summary(results: list[dict]) -> bool:
    n = len(results)
    k = sum(r["passed"] for r in results)
    print(f"\n=== Phase 0 summary: {k}/{n} songs pass ===")
    for r in results:
        bad = [f"{c['good']}>{c['broken']}" for c in r["checks"] if not c["ok"]]
        print(f"  {r['song']:<24} {'PASS' if r['passed'] else 'FAIL'}  margin {r['noise']['margin']:.4f}"
              f"{'  failed: ' + ', '.join(bad) if bad else ''}")
    ok = k > n / 2
    print("Judge tracks the obvious cases on most songs: go on to Phase 1." if ok else
          "The judge can't tell these apart on most songs: stop here and rethink the judge.")
    return ok
