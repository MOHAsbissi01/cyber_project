"""Configuration loader: reads config.toml once, resolves paths against the project root."""
from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config() -> dict:
    path = Path(os.environ.get("CYBERINTEL_CONFIG", ROOT / "config.toml"))
    with open(path, "rb") as f:
        return tomllib.load(f)


def cfg(section: str, key: str | None = None):
    c = load_config()[section]
    return c if key is None else c[key]


def path(name: str, *parts: str, create: bool = False) -> Path:
    """Resolve a configured directory/file (`paths.<name>`) against the project root."""
    p = ROOT / cfg("paths", name)
    p = p.joinpath(*parts)
    if create:
        (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
    return p
