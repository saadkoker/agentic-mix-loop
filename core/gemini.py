"""Gemini describes; it does not judge (via OpenRouter).

Its only job in the loop: a description card for each full stem and for the
song (what plays, what it is, when), with timestamps. No problems, no quality
opinions, no frequency claims - the critique comes from the judge itself
(core/diagnose.py), so agents never spend rounds pleasing a second opinion.

ab() is kept only as an opt-in Phase 0 experiment (phase0 --gemini) to
measure whether Gemini can tell mixes apart at all. It never affects the loop.

Gemini always hears the whole recording. Audio is cut into consecutive parts
labelled with absolute times; if all parts fit in one request they are sent
together, otherwise each group of parts is described on its own and a final
text call stitches the partial descriptions into one.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .llm import LLMError, OpenRouter, audio_part

# ------------------------------------------------------------------ schemas

CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "stem": {"type": "string"},
        "role": {"type": "string"},
        "elements": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "character": {"type": "string"},
                           "when": {"type": "string"}},
            "required": ["name", "character", "when"]}},
    },
    "required": ["stem", "role", "elements"],
}

AB_SCHEMA = {
    "type": "object",
    "properties": {"better": {"type": "string", "enum": ["A", "B", "same"]},
                   "confidence": {"type": "number"},
                   "why": {"type": "string"}},
    "required": ["better", "why"],
}

ENGINEER = ("You are a careful listener describing music for a mix engineer. Describe what "
            "you hear - instruments, sounds, parts and when they happen. Do not judge quality, "
            "do not list problems or suggest processing, and do not guess frequencies.")
AB_SYSTEM = ("You are an experienced mix engineer comparing two mixes. Be honest; say 'same' "
             "when you cannot tell.")


def mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60}:{t % 60:02d}"


def activity_map(x: np.ndarray, sr: int, block_s: float = 1.0, thresh_db: float = -50.0) -> str:
    """Where the audio is active, e.g. '0:00-0:31 silent; 0:31-2:10 playing'."""
    m = x.mean(axis=0)
    b = int(block_s * sr)
    nb = max(1, len(m) // b)
    rms = np.sqrt(np.mean(m[: nb * b].reshape(nb, b) ** 2, axis=1) + 1e-20)
    on = 20 * np.log10(rms) > thresh_db
    spans, start = [], 0
    for i in range(1, nb + 1):
        if i == nb or on[i] != on[start]:
            spans.append(f"{mmss(start * block_s)}-{mmss(i * block_s)} {'playing' if on[start] else 'silent'}")
            start = i
    return "; ".join(spans)


class Listener:
    def __init__(self, cfg: dict, client: OpenRouter, log_dir: Path | None = None):
        self.cfg = cfg
        self.m = cfg["models"]
        self.client = client
        self.model = self.m["listener"]
        self.log_dir = log_dir

    @property
    def available(self) -> bool:
        return self.client.available

    # ------------------------------------------------------------ audio
    def _encode(self, x: np.ndarray, sr: int) -> bytes:
        """Mono at models.audio_sr by default; audio_channels = 2 keeps stereo (the audit uses it)."""
        x = np.atleast_2d(x)
        stereo = int(self.m.get("audio_channels", 1)) == 2 and x.shape[0] == 2
        y = x.astype(np.float64) if stereo else x.mean(axis=0, keepdims=True).astype(np.float64)
        tsr = int(self.m["audio_sr"])
        if sr != tsr:
            g = np.gcd(sr, tsr)
            y = signal.resample_poly(y, tsr // g, sr // g, axis=1)
        pk = np.max(np.abs(y)) if y.size else 0
        if pk > 0.999:
            y = y * (0.999 / pk)
        y = (y.T if stereo else y[0]).astype(np.float32)
        buf = io.BytesIO()
        fmt = self.m.get("audio_format", "mp3").lower()
        if fmt == "mp3":
            sf.write(buf, y, tsr, format="MP3")
        else:
            sf.write(buf, y, tsr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def _parts(self, x: np.ndarray, sr: int) -> list[tuple[float, float, bytes]]:
        """Consecutive parts covering the whole recording."""
        n = x.shape[1]
        step = int(self.m["chunk_s"] * sr)
        out = []
        for s in range(0, n, step):
            e = min(n, s + step)
            if e - s < sr * 0.5 and out:      # fold a tiny tail into the previous part
                t0, _, _ = out.pop()
                s = int(t0 * sr)
            out.append((s / sr, e / sr, self._encode(x[:, s:e], sr)))
        return out

    def _groups(self, parts: list) -> list[list]:
        """Pack parts into requests under the size limit (base64 = 4/3)."""
        limit = self.m["max_request_mb"] * 1e6 * 0.75
        groups, cur, size = [], [], 0
        for p in parts:
            if cur and size + len(p[2]) > limit:
                groups.append(cur)
                cur, size = [], 0
            cur.append(p)
            size += len(p[2])
        if cur:
            groups.append(cur)
        return groups

    def _audio_content(self, label: str, parts: list, total_s: float) -> list:
        fmt = "mp3" if self.m.get("audio_format", "mp3").lower() == "mp3" else "wav"
        content = []
        for i, (t0, t1, data) in enumerate(parts):
            content.append({"type": "text", "text":
                            f"{label} part {i + 1}/{len(parts)}: {mmss(t0)}-{mmss(t1)} "
                            f"of {mmss(total_s)}"})
            content.append(audio_part(data, fmt))
        return content

    def _log(self, name: str, obj):
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / f"{name}.json").write_text(json.dumps(obj, indent=1))

    def _ask(self, content: list, schema: dict, tag: str):
        msgs = [{"role": "system", "content": ENGINEER}, {"role": "user", "content": content}]
        out, _ = self.client.chat(self.model, msgs, schema=schema, max_tokens=6000, tag=tag,
                                  temperature=0.2)
        return out

    # ---------------------------------------------------------- job 1
    def card(self, name: str, x: np.ndarray, sr: int, is_mix: bool = False,
             stem_names: list[str] | None = None) -> dict:
        total = x.shape[1] / sr
        parts = self._parts(x, sr)
        groups = self._groups(parts)
        what = "the flat (unprocessed) mix of all stems" if is_mix else f"the stem '{name}'"
        extra = (f" The stems in it are: {', '.join(stem_names)}. For the mix, describe the song "
                 "sections (intro, verse, hook, breakdown, outro...) and which stems play in each."
                 if is_mix and stem_names else "")
        instr = (
            f"You are hearing {what}, the FULL recording, {mmss(total)} long, delivered as "
            f"consecutive parts. Treat the parts as one continuous recording and use absolute "
            f"timestamps of the full recording (m:ss). Activity map (from levels): "
            f"{activity_map(x, sr)}.{extra}\n\n"
            "Describe what is in it, in time order. List each distinct element or part (e.g. "
            "kick, snare, hi-hat roll, FX riser, lead line, ad-lib, verse, chorus, breakdown) with "
            "a short plain description of what it is and how it sounds, and exactly when it "
            "occurs (m:ss-m:ss, several ranges allowed); for a loop or FX track list them in order "
            "('first X, then Y, then Z'). Give the stem's role in the song. Describe only: no "
            "problems, no quality judgments, no processing advice, no frequency numbers. "
            f"Set \"stem\" to \"{name}\". Reply as JSON.")
        if len(groups) == 1:
            card = self._ask([{"type": "text", "text": instr}] +
                             self._audio_content("Audio", groups[0], total), CARD_SCHEMA,
                             f"card:{name}")
        else:
            partials = []
            for gi, g in enumerate(groups):
                content = [{"type": "text", "text": instr + f"\n\nThis request covers "
                            f"{mmss(g[0][0])}-{mmss(g[-1][1])} only (section {gi + 1}/{len(groups)})."}]
                partials.append(self._ask(content + self._audio_content("Audio", g, total),
                                          CARD_SCHEMA, f"card:{name}:{gi}"))
            card = self._stitch(name, partials)
        card["stem"] = name
        self._log(f"card_{name}", card)
        return card

    def _stitch(self, name: str, partials: list[dict]) -> dict:
        txt = ("These JSON cards describe consecutive sections of ONE recording "
               f"('{name}'), in order, with absolute timestamps. Merge them into a single card: "
               "join elements that continue across section boundaries (widen their 'when'), "
               "keep everything else.\n\n" +
               json.dumps(partials, indent=1))
        msgs = [{"role": "system", "content": ENGINEER}, {"role": "user", "content": txt}]
        out, _ = self.client.chat(self.model, msgs, schema=CARD_SCHEMA, max_tokens=6000,
                                  tag=f"stitch:{name}", temperature=0.0)
        return out

    # ------------------------------------ Phase 0 experiment only
    def _ab_once(self, a: np.ndarray, b: np.ndarray, sr: int) -> list[dict]:
        total = a.shape[1] / sr
        pa, pb = self._parts(a, sr), self._parts(b, sr)
        # group by time range so each request compares the same section of both
        limit = self.m["max_request_mb"] * 1e6 * 0.75
        groups, cur, size = [], [], 0
        for x, y in zip(pa, pb):
            s = len(x[2]) + len(y[2])
            if cur and size + s > limit:
                groups.append(cur)
                cur, size = [], 0
            cur.append((x, y))
            size += s
        if cur:
            groups.append(cur)
        verdicts = []
        for gi, g in enumerate(groups):
            sec = f"{mmss(g[0][0][0])}-{mmss(g[-1][0][1])}"
            instr = (f"Two mixes of the same song, loudness-matched. You will hear mix A ({sec}), then "
                     f"mix B (same section). Which mix sounds better overall - balance, clarity, tone, "
                     f"punch, space? Answer A, B, or same if you honestly can't tell. Confidence 0-1. JSON.")
            content = ([{"type": "text", "text": instr}] +
                       self._audio_content("Mix A", [p[0] for p in g], total) +
                       self._audio_content("Mix B", [p[1] for p in g], total))
            msgs = [{"role": "system", "content": AB_SYSTEM}, {"role": "user", "content": content}]
            out, _ = self.client.chat(self.model, msgs, schema=AB_SCHEMA, max_tokens=3000,
                                      tag=f"ab:{gi}", temperature=0.2)
            verdicts.append(out)
        return verdicts

    @staticmethod
    def _tally(verdicts: list[dict]) -> str:
        score = sum({"A": 1, "B": -1}.get(v.get("better"), 0) for v in verdicts)
        return "A" if score > 0 else "B" if score < 0 else "same"

    def ab(self, old: np.ndarray, new: np.ndarray, sr: int) -> dict:
        """Run twice with the order swapped. Returns {'verdict': 'new'|'old'|'split', ...}."""
        r1 = self._ab_once(old, new, sr)      # A = old
        r2 = self._ab_once(new, old, sr)      # A = new
        p1 = {"A": "old", "B": "new"}.get(self._tally(r1), "same")
        p2 = {"A": "new", "B": "old"}.get(self._tally(r2), "same")
        if p1 == p2 and p1 in ("new", "old"):
            verdict = p1
        else:
            verdict = "split"
        out = {"verdict": verdict, "run1": p1, "run2": p2,
               "why": [v.get("why", "") for v in r1 + r2]}
        self._log("ab_latest", out)
        return out


def load_cards(card_dir: Path) -> dict:
    out = {}
    for p in sorted(Path(card_dir).glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
    return out


def safe(fn, *a, default=None, **kw):
    try:
        return fn(*a, **kw)
    except LLMError as e:
        print(f"  gemini: {e}")
        return default
