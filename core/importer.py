"""Turn a multitrack session (zip or folder of WAVs) into a song project.

Raw sessions often have 15-40 tracks of different lengths. The loop wants
one agent per *stem*, so by default tracks are grouped into stems by name
(drums, perc, bass, guitars, keys, synths, strings, fx, vocals, backing
vocals, other) and summed flat. Use --groups none to keep every track, or a
JSON file {"stem name": ["track name", ...]} for your own grouping.
Tracks are zero-padded at the end to the longest length.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

RULES = [  # first match wins; keywords match whole tokens (or token prefix if >= 4 chars)
    ("backing_vocals", ["backing", "bvox", "bgv", "bv", "harmony", "harmonies", "choir", "adlib", "adlibs", "bvs"]),
    ("vocals", ["vox", "vocal", "vocals", "voice", "leadvox", "lead"]),
    ("perc", ["perc", "percussion", "shaker", "tamb", "tambourine", "bongo", "conga", "clave",
              "cowbell", "chimes", "chime", "triangle"]),
    ("drums", ["kick", "kik", "snare", "snr", "hat", "hihat", "hh", "tom", "toms", "overhead",
               "overheads", "oh", "room", "cymbal", "cymbals", "ride", "crash", "drum", "drums",
               "beat", "loop", "clap", "claps", "snap", "boom"]),
    ("bass", ["bass", "sub", "808"]),
    ("guitars", ["gtr", "guitar", "guitars", "acoustic", "ag", "eg"]),
    ("keys", ["piano", "keys", "key", "organ", "hammond", "rhodes", "ep", "wurli", "clav",
              "mellotron", "elecpiano", "pno"]),
    ("strings", ["string", "strings", "violin", "viola", "cello", "harp", "orchestra"]),
    ("synths", ["synth", "synths", "pad", "pads", "arp", "pluck"]),
    ("fx", ["sfx", "fx", "riser", "noise", "cartoon", "sweep", "impact"]),
]


def tokens(name: str) -> list[str]:
    s = re.sub(r"\.(wav|aif|aiff|flac)$", "", name, flags=re.I)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)|(\d)([A-Za-z])", lambda m: " ".join(g for g in m.groups() if g), s)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def classify(name: str) -> str:
    toks = tokens(name)
    for group, kws in RULES:
        for t in toks:
            for k in kws:
                if t == k or (len(k) >= 4 and t.startswith(k)):
                    return group
    return "other"


def side_hint(name: str) -> str | None:
    toks = tokens(name)
    if toks and toks[-1] in ("l", "left") or "left" in toks:
        return "L"
    if toks and toks[-1] in ("r", "right") or "right" in toks:
        return "R"
    return None


def _read_tracks(src: Path) -> dict[str, tuple[np.ndarray, int]]:
    out = {}
    if src.is_file() and src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            for info in z.infolist():
                n = info.filename
                base = n.rsplit("/", 1)[-1]
                if "__MACOSX" in n or base.startswith("._") or not base.lower().endswith((".wav", ".flac", ".aif", ".aiff")):
                    continue
                data, sr = sf.read(io.BytesIO(z.read(info)), dtype="float32", always_2d=True)
                out[base.rsplit(".", 1)[0]] = (data.T.copy(), sr)
    else:
        for p in sorted(Path(src).rglob("*")):
            if p.suffix.lower() in (".wav", ".flac", ".aif", ".aiff") and not p.name.startswith("._"):
                data, sr = sf.read(str(p), dtype="float32", always_2d=True)
                out[p.stem] = (data.T.copy(), sr)
    if not out:
        raise SystemExit(f"no audio files found in {src}")
    return out


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")[:60]


def import_session(src: Path, dest: Path, groups: str = "auto") -> dict:
    tracks = _read_tracks(Path(src))
    srs = {sr for _, sr in tracks.values()}
    if len(srs) != 1:
        raise SystemExit(f"tracks have mixed sample rates {srs}")
    sr = srs.pop()
    n = max(a.shape[1] for a, _ in tracks.values())
    tracks = {k: np.pad(a, ((0, 0), (0, n - a.shape[1]))) for k, (a, _) in tracks.items()}

    if groups == "none":
        mapping = {_safe(k): [k] for k in tracks}
    elif groups == "auto":
        mapping: dict[str, list[str]] = {}
        for k in sorted(tracks):
            mapping.setdefault(classify(k), []).append(k)
    else:
        mapping = json.loads(Path(groups).read_text())
        missing = [t for v in mapping.values() for t in v if t not in tracks]
        if missing:
            raise SystemExit(f"groups file names unknown tracks: {missing}")

    stem_dir = Path(dest) / "stems"
    stem_dir.mkdir(parents=True, exist_ok=True)
    for old in stem_dir.glob("*.wav"):
        old.unlink()
    order = [g for g, _ in RULES] + ["other"]
    sources = {}
    for i, (g, members) in enumerate(sorted(mapping.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99)):
        hints = {m: side_hint(m) for m in members}
        stereo = len(members) > 1 and (any(tracks[m].shape[0] == 2 for m in members) or any(hints.values()))
        stereo = stereo or (len(members) == 1 and tracks[members[0]].shape[0] == 2)
        mix = np.zeros((2 if stereo else 1, n), np.float32)
        for m in members:
            a = tracks[m]
            if stereo and a.shape[0] == 1:
                h = hints[m]
                if h == "L":
                    a = np.vstack([a[0], np.zeros(n, np.float32)])
                elif h == "R":
                    a = np.vstack([np.zeros(n, np.float32), a[0]])
                else:
                    a = np.vstack([a[0], a[0]]) * np.float32(0.70710678)   # centre, -3 dB pan law
            mix += a
        name = _safe(g) if groups == "none" else f"{i + 1:02d}_{_safe(g)}"
        sf.write(str(stem_dir / f"{name}.wav"), mix.T, sr, subtype="FLOAT")
        sources[name] = members
    (Path(dest) / "sources.json").write_text(json.dumps(sources, indent=1))
    return sources


def import_raw(src: Path, dest: Path) -> dict:
    """Copy every audio file exactly as delivered (same bytes, format, length, name) into
    <dest>/tracks/. Nothing is grouped, converted, padded or cleaned; the renderer pads
    shorter tracks in memory only."""
    src, dest = Path(src), Path(dest)
    out_dir = dest / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():
        old.unlink()
    exts = (".wav", ".flac", ".aif", ".aiff")
    copied = {}

    def put(name: str, data: bytes):
        stem, ext = name.rsplit(".", 1)
        final = f"{stem}.{ext}"
        i = 2
        while final.rsplit(".", 1)[0] in copied:
            final = f"{stem}_{i}.{ext}"
            i += 1
        (out_dir / final).write_bytes(data)
        copied[final.rsplit(".", 1)[0]] = name

    if src.is_file() and src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            for info in z.infolist():
                n = info.filename
                base = n.rsplit("/", 1)[-1]
                if "__MACOSX" in n or base.startswith("._") or not base.lower().endswith(exts):
                    continue
                put(base, z.read(info))
    else:
        for q in sorted(src.rglob("*")):
            if q.suffix.lower() in exts and not q.name.startswith("._"):
                put(q.name, q.read_bytes())
    if not copied:
        raise SystemExit(f"no audio files found in {src}")
    (dest / "sources.json").write_text(json.dumps({k: [v] for k, v in copied.items()}, indent=1))
    return {k: [v] for k, v in copied.items()}
