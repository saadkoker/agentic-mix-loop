"""Config loading and the song-project folder layout."""
from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.toml"
STOCK_DIR = Path(__file__).resolve().parent / "stock"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(project: Path | None = None, overrides: dict | None = None) -> dict:
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    if project is not None and (project / "config.toml").exists():
        with open(project / "config.toml", "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


@dataclass
class Project:
    """One song. Mirrors the folder layout in the build plan."""

    root: Path

    def __post_init__(self):
        self.root = Path(self.root).resolve()

    stems = property(lambda s: s.root / "stems")
    tracks = property(lambda s: s.root / "tracks")      # raw session: the files exactly as delivered

    @property
    def raw(self) -> bool:
        return self.tracks.is_dir() and any(self.tracks.iterdir())

    @property
    def source_dir(self) -> Path:
        return self.tracks if self.raw else self.stems
    cards = property(lambda s: s.root / "cards")
    plugins = property(lambda s: s.root / "plugins")
    chains = property(lambda s: s.root / "chains")
    renders = property(lambda s: s.root / "renders")
    history = property(lambda s: s.root / "history.jsonl")
    votes = property(lambda s: s.root / "votes.jsonl")
    state = property(lambda s: s.root / "state.json")
    cache = property(lambda s: s.root / "score_cache.jsonl")
    phase0 = property(lambda s: s.root / "phase0.json")
    best = property(lambda s: s.root / "chains" / "best.json")
    checkpoints = property(lambda s: s.root / "chains" / "checkpoints")
    reference = property(lambda s: s.root / "reference")

    def ensure(self):
        for d in (self.stems, self.cards, self.plugins, self.chains, self.renders,
                  self.checkpoints, self.reference):
            d.mkdir(parents=True, exist_ok=True)
        return self
