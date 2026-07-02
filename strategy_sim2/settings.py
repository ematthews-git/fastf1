"""Configuration loading for strategy_sim2.

All configuration lives in ``config/settings.yaml``. Paths inside it are relative to
the repository root; :func:`resolve_path` turns them into absolute paths.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
DEFAULT_SETTINGS_PATH = PKG_DIR / "config" / "settings.yaml"


@lru_cache(maxsize=8)
def load_settings(path: str | None = None) -> dict[str, Any]:
    """Load and cache the YAML settings as a plain dict."""
    p = Path(path) if path else DEFAULT_SETTINGS_PATH
    with open(p, "r") as f:
        return yaml.safe_load(f)


def resolve_path(rel: str | Path) -> Path:
    """Resolve a repo-relative path from settings to an absolute path."""
    rel = Path(rel)
    return rel if rel.is_absolute() else (REPO_ROOT / rel)


def cache_dir(cfg: dict | None = None) -> Path:
    """Return the FastF1 cache dir, creating it if needed."""
    cfg = cfg or load_settings()
    p = resolve_path(cfg["data"]["cache_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p
