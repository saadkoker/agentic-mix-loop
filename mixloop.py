"""Agentic mix loop - command line.

  python mixloop.py import <zip|folder> --name <song> [--raw | --groups auto|none|file.json]
  python mixloop.py phase0 <song> [<song> ...] [--gemini] [--human]
  python mixloop.py score <song> [--chain chains/x.json] [--out mix.wav]      (Phase 1)
  python mixloop.py harness <plugin.py>
  python mixloop.py cards <song> [--force]                     (Gemini: describe each stem)
  python mixloop.py diagnose <song>                            (Audiobox critique of the best mix)
  python mixloop.py run <song> [--phase 2..5] [--agent llm|random] [--max-rounds N] [--no-listen] [--fresh]
  python mixloop.py listen <song>                                             (blind A/B now)
  python mixloop.py report <song>
  python mixloop.py audit <song> [<song> ...] [--repeats N]   (Gemini A/B on the final mix + known-answer controls)

<song> is a folder under songs/ or a path. All LLM calls go through OpenRouter
(set OPENROUTER_API_KEY).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", message=".*zero-padding.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

from core.config import Project, load_config  # noqa: E402


def song(name: str) -> Project:
    p = Path(name)
    if not p.exists():
        p = ROOT / "songs" / name
    return Project(p)


def cmd_import(a):
    from core.importer import import_raw, import_session
    dest = ROOT / "songs" / a.name
    src = Path(a.src)
    print(f"Importing {src.name} -> {dest}{' (raw: files copied exactly as delivered)' if a.raw else ''}")
    sources = import_raw(src, dest) if a.raw else import_session(src, dest, a.groups)
    for k, v in sources.items():
        print(f"  {k}: {len(v)} track(s): {', '.join(v[:6])}{' ...' if len(v) > 6 else ''}")
    Project(dest).ensure()


def cmd_phase0(a):
    from core.phase0 import run_song, summary
    cfg = load_config()
    res = [run_song(song(s).ensure(), load_config(song(s).root), a.judge, a.gemini, a.human) for s in a.songs]
    ok = summary(res)
    (ROOT / "phase0_summary.json").write_text(json.dumps(
        [{"song": r["song"], "passed": r["passed"], "margin": r["noise"]["margin"],
          **{k: r[k] for k in ("gemini_expected_rate", "gemini_human_agreement") if k in r}} for r in res], indent=1))
    sys.exit(0 if ok else 1)


def cmd_score(a):
    from core.evaluate import Evaluator
    from core import audio as A
    pr = song(a.song)
    cfg = load_config(pr.root)
    ev = Evaluator(pr, cfg, a.judge)
    base = ev.setup_baseline()
    print(f"flat baseline: {base.score.short() if base.score else base.reason_failed()}")
    if a.chain:
        chain = json.loads((pr.root / a.chain if not Path(a.chain).exists() else Path(a.chain)).read_text())
        res = ev.evaluate(chain, keep_render=True)
    else:
        res = base
    print(f"\nchain {res.hash[:12]}  {'PASS' if res.ok else 'FAIL'}")
    for c in (res.guard or {}).get("checks", []):
        print(f"  {'ok  ' if c['passed'] else 'FAIL'} {c['name']:<16} {c['value']!s:<14} {c['limit']} {c['note']}")
    if res.error:
        print(f"  error: {res.error}")
    if res.score:
        print(f"\n{res.score.short()}")
        if base.score:
            print(f"change vs flat baseline: {res.score.score - base.score.score:+.4f}")
        if a.windows:
            for w in res.score.windows:
                print(f"  {w['t']:6.1f}s  PQ {w['PQ']:.3f}  CE {w['CE']:.3f}  PC {w['PC']:.3f}  CU {w['CU']:.3f}")
    if a.out and res.render is not None:
        A.write(Path(a.out), res.render.final, ev.sr)
        print(f"wrote {a.out}")


def cmd_axes(a):
    from core.axes import study
    dirs = [song(s).root for s in a.songs] if a.songs else sorted((ROOT / "songs").iterdir())
    out = study([d for d in dirs if d.is_dir()], load_config())
    (ROOT / "axes_study.json").write_text(json.dumps(out, indent=1))


def cmd_harness(a):
    from core.harness import test_plugin
    r = test_plugin(Path(a.plugin), load_config())
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["passed"] else 1)


def _orch(a, **kw):
    from core.orchestrator import Orchestrator
    pr = song(a.song)
    over = {"loop": {}}
    if getattr(a, "phase", None):
        over["loop"]["phase"] = a.phase
    if getattr(a, "max_usd", None):
        over["loop"]["max_usd"] = a.max_usd
    cfg = load_config(pr.root, over)
    return Orchestrator(pr, cfg, agent_kind=getattr(a, "agent", "llm"), judge_kind=a.judge,
                        listen=not getattr(a, "no_listen", False), seed=getattr(a, "seed", 0), **kw)


def cmd_run(a):
    pr = song(a.song)
    if a.fresh:
        for p in (pr.state, pr.history, pr.best):
            if p.exists():
                p.unlink()
        if pr.checkpoints.exists():
            shutil.rmtree(pr.checkpoints)
    o = _orch(a)
    o.run(a.max_rounds)


def cmd_cards(a):
    o = _orch(a)
    o.setup()
    o.listening_pass(force=a.force)


def cmd_diagnose(a):
    a.agent = "random"     # the judge critiques; no LLM needed
    o = _orch(a)
    resumed = o.p.state.exists()
    o.setup()              # a fresh setup already runs one
    if resumed:
        o.run_diagnosis()


def cmd_listen(a):
    a.agent = "random"     # no LLM needed to listen
    o = _orch(a)
    o.setup()
    o.listen_enabled = True
    o.listening_check()


def cmd_audit(a):
    from core.audit import audit_song
    for name in a.songs:
        pr = song(name)
        audit_song(pr, load_config(pr.root), a.repeats)


def cmd_report(a):
    from core.report import final_report
    a.agent = "random"
    o = _orch(a)
    o.setup()
    print(final_report(o.p, o.cfg, o.ev, o.state))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, song_arg=True):
        p = sub.add_parser(name)
        if song_arg:
            p.add_argument("song")
        p.add_argument("--judge", default="audiobox", choices=["audiobox", "dummy"])
        p.set_defaults(fn=fn)
        return p

    p = sub.add_parser("import")
    p.add_argument("src")
    p.add_argument("--name", required=True)
    p.add_argument("--groups", default="auto")
    p.add_argument("--raw", action="store_true",
                   help="copy every track exactly as delivered (no grouping, conversion, padding)")
    p.set_defaults(fn=cmd_import)

    p = add("phase0", cmd_phase0, song_arg=False)
    p.add_argument("songs", nargs="+")
    p.add_argument("--gemini", action="store_true")
    p.add_argument("--human", action="store_true")

    p = sub.add_parser("audit", help="Gemini A/B (the old veto) on the final mix, with controls")
    p.add_argument("songs", nargs="+")
    p.add_argument("--repeats", type=int, default=2)
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("axes", help="compare Audiobox axis weightings on stored Phase 0 data")
    p.add_argument("songs", nargs="*")
    p.set_defaults(fn=cmd_axes)

    p = add("score", cmd_score)
    p.add_argument("--chain")
    p.add_argument("--out")
    p.add_argument("--windows", action="store_true")

    p = sub.add_parser("harness")
    p.add_argument("plugin")
    p.set_defaults(fn=cmd_harness)

    p = add("run", cmd_run)
    p.add_argument("--phase", type=int, choices=[2, 3, 4, 5])
    p.add_argument("--agent", default="llm", choices=["llm", "random"])
    p.add_argument("--max-rounds", type=int)
    p.add_argument("--max-usd", type=float)
    p.add_argument("--no-listen", action="store_true")
    p.add_argument("--fresh", action="store_true", help="discard saved state and start over")
    p.add_argument("--seed", type=int, default=0)

    p = add("cards", cmd_cards)
    p.add_argument("--force", action="store_true")
    add("diagnose", cmd_diagnose)
    add("listen", cmd_listen)
    add("report", cmd_report)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
