"""On stop: three renders and a short report."""
from __future__ import annotations

import json
from pathlib import Path

from . import audio as A
from .plugins import describe_bus, describe_list


def _plot(scores: list[float], baseline: float, accepted: list[dict], path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
    ax.plot(range(len(scores)), scores, color="#2a6fdb", lw=1.8, label="best score")
    ax.axhline(baseline, color="#888", lw=1, ls="--", label="flat baseline")
    for a in accepted:
        ax.plot(a["round"], a["score"], "o", color="#e07b00", ms=4)
    ax.set_xlabel("round")
    ax.set_ylabel("score")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def final_report(project, cfg: dict, ev, state: dict) -> Path:
    out = project.renders / "final"
    out.mkdir(parents=True, exist_ok=True)
    sr = ev.sr
    base = ev.render(state["baseline_chain"]).final
    best_r = ev.render(state["best_chain"])
    A.write(out / "1_flat_baseline.wav", base, sr)
    A.write(out / "2_best_mix.wav", best_r.final, sr)
    loud = A.limit_to_lufs(best_r.pre, sr, cfg["audio"]["loud_master_lufs"], cfg["audio"]["ceiling_dbtp"])
    A.write(out / "3_best_mix_loud.wav", loud, sr)
    (project.chains / "final.json").write_text(json.dumps(state["best_chain"], indent=1))

    b, s = state["baseline_score"], state["best_score"]
    plot = _plot(state["score_by_round"], b["score"], state.get("accepted_log", []), out / "score_history.png")
    L = []
    L.append(f"# Mix report: {project.root.name}\n")
    L.append(f"Stopped: {state.get('stop_reason')}  \nRounds: {state['round']}, accepted changes: "
             f"{state['accepted']}, LLM spend: ${state.get('spend_usd', 0):.2f}, wall-clock "
             f"{state.get('elapsed_s', 0) / 60:.1f} min  \nPhase {state.get('phase')}, agent "
             f"{state.get('agent')}, acceptance margin {state['margin']:.4f} "
             f"(judge noise spread {state['noise']['spread']:.4f})\n")
    L.append("## Scores\n")
    L.append("| | score | PQ | CE | worst window | PC | CU |\n|---|---|---|---|---|---|---|")
    for name, d in (("flat baseline", b), ("best mix", s)):
        L.append(f"| {name} | {d['score']:.4f} | {d['PQ']:.3f} | {d['CE']:.3f} | "
                 f"{d['worst']:.3f} @ {d['worst_at_s']:.0f}s | {d['PC']:.3f} | {d['CU']:.3f} |")
    L.append(f"\nChange vs baseline: **{s['score'] - b['score']:+.4f}**\n")
    loud_l = A.lufs(loud, sr)
    L.append("## Files\n")
    L.append(f"- `renders/final/1_flat_baseline.wav` (flat sum through the fixed limiter)")
    L.append(f"- `renders/final/2_best_mix.wav` (best mix at {cfg['audio']['target_lufs']} LUFS)")
    L.append(f"- `renders/final/3_best_mix_loud.wav` (final limiter pushed to {loud_l:.1f} LUFS, "
             f"{A.true_peak(loud):.2f} dBTP)")
    L.append(f"- `chains/final.json` (rebuild exactly with `python mixloop.py score {project.root.name} "
             "--chain chains/final.json`)\n")
    if plot:
        L.append("![score history](renders/final/score_history.png)\n")
    L.append("## Accepted changes\n")
    acc = state.get("accepted_log", [])
    if not acc:
        L.append("None.\n")
    for a in acc:
        flag = " **(judge fooled: rolled back)**" if a.get("judge_fooled") else ""
        L.append(f"- round {a['round']} `{a['stem']}` {a['delta']:+.4f} -> {a['score']:.4f}: {a['reason']}{flag}")
    if state.get("flags"):
        L.append("\n## Hold-out flags\n")
        L += [f"- round {f['round']}: {f['flag']}" for f in state["flags"]]
    if state.get("outside_ear_log"):
        L.append("\n## Outside ear (leads only, never scored)\n")
        for a in state["outside_ear_log"]:
            L.append(f"- round {a['round']} ({a['why']}, best {a['score']:.4f}; "
                     f"listens heard {a.get('heard')} issues):")
            L += ([f"  - [{i['severity']}] {i['time']}: {i['issue']}" for i in a["issues"]]
                  or ["  - nothing heard in every listen"])
            L += [f"  - dropped: {i['time']}: {i['issue']} ({i['why_dropped']})" for i in a.get("dropped", [])]
    L.append("\n## Final chains\n")
    for stem, items in state["best_chain"]["stems"].items():
        L.append(f"- **{stem}**: {describe_list(items)}")
    for bname, bus in (state["best_chain"].get("buses") or {}).items():
        L.append(f"- **{describe_bus(bname, bus)}**")
    L.append(f"- **master**: {describe_list(state['best_chain'].get('master', []))} -> fixed limiter "
             f"{cfg['audio']['ceiling_dbtp']} dBTP")
    path = project.root / "report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
