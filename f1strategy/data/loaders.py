"""FastF1 session loading — thin, cache-aware, and robust to missing sessions.

Kept deliberately small and side-effect-light so a backend pipeline can either
call these directly or inject its own already-loaded sessions everywhere the rest
of the package accepts a ``Session``.
"""

from __future__ import annotations

import os
import functools
from typing import Optional

import fastf1 as f1

PRACTICE_SESSIONS: tuple[str, ...] = ("FP1", "FP2", "FP3")

_CACHE_ENABLED = False


def enable_cache(path: Optional[str] = None) -> None:
    """Enable FastF1's on-disk cache once (idempotent).

    Path resolution: explicit arg > ``FASTF1_CACHE_DIR`` env > ``./f1_cache``.
    """
    global _CACHE_ENABLED
    if _CACHE_ENABLED:
        return
    cache_dir = path or os.environ.get("FASTF1_CACHE_DIR", "./f1_cache")
    os.makedirs(cache_dir, exist_ok=True)
    f1.Cache.enable_cache(cache_dir)
    try:
        f1.set_log_level("ERROR")  # quiet FastF1's load/network chatter
    except Exception:  # noqa: BLE001  (older/newer fastf1 may lack this)
        pass
    _CACHE_ENABLED = True


def _load(year: int, track, kind: str, messages: bool = False):
    enable_cache()
    s = f1.get_session(year, track, kind)
    # weather is unused (wetness is inferred from compounds); skipping it is faster
    # and dodges an occasional weather-stream load failure.
    s.load(laps=True, telemetry=False, weather=False, messages=messages)
    return s


def load_race(year: int, track, messages: bool = False):
    """Load a race session (laps + optionally race-control messages).

    ``messages=True`` is needed for safety-car detection; leave False for speed
    when only laps/results are required.
    """
    return _load(year, track, "R", messages=messages)


def load_results(session):
    """Grid/finish/status table keyed by 3-letter driver code.

    Returns ``{ABB: {"grid": int, "finish": int, "status": str,
    "classified": str, "laps": int, "team": str}}``. GridPosition 0 (pit-lane
    start) is mapped to the back of the grid.
    """
    res = session.results
    if res is None or not len(res):
        return {}
    n = len(res)
    out: dict[str, dict] = {}
    for _, row in res.iterrows():
        abb = row.get("Abbreviation")
        if not isinstance(abb, str) or not abb:
            continue
        grid = row.get("GridPosition")
        grid = int(grid) if grid and grid > 0 else n  # 0/NaN pit-lane start -> back
        finish = row.get("Position")
        finish = int(finish) if finish and finish == finish else n
        out[abb] = {
            "grid": grid,
            "finish": finish,
            "status": str(row.get("Status", "")),
            "classified": str(row.get("ClassifiedPosition", "")),
            "laps": int(row["Laps"]) if row.get("Laps") == row.get("Laps") else 0,
            "team": str(row.get("TeamName", "")),
        }
    return out


def load_quali(year: int, track):
    """Load a qualifying session (for the prediction grid)."""
    return _load(year, track, "Q")


def grid_order(session) -> list[str]:
    """Ordered driver codes front-to-back from a session's results.

    Prefers ``GridPosition`` (a race session) and falls back to ``Position`` (a
    qualifying session). Pit-lane/0 entries are dropped.
    """
    res = getattr(session, "results", None)
    if res is None or not len(res):
        return []
    col = "GridPosition" if ("GridPosition" in res.columns
                             and (res["GridPosition"] > 0).any()) else "Position"
    r = res.dropna(subset=[col])
    r = r[r[col] > 0].sort_values(col)
    return [a for a in r["Abbreviation"].tolist() if isinstance(a, str) and a]


def load_practice_sessions(year: int, track) -> list[tuple[str, object]]:
    """Load every available practice session for a weekend.

    Returns ``[(name, session), ...]``. Missing sessions (sprint weekends lack
    FP2/FP3, data not yet published, etc.) are silently skipped.
    """
    out: list[tuple[str, object]] = []
    for name in PRACTICE_SESSIONS:
        try:
            s = _load(year, track, name)
            # accessing .laps raises DataNotLoadedError for a not-yet-run weekend, so
            # keep it inside the guard (a post-practice prediction before FP happens)
            if s.laps is not None and len(s.laps):
                out.append((name, s))
        except Exception:  # noqa: BLE001  (missing / future / sprint-weekend sessions)
            continue
    return out


@functools.lru_cache(maxsize=256)
def _event_name_cached(year: int, track: str) -> Optional[str]:
    enable_cache()
    try:
        return f1.get_event(year, track)["EventName"]
    except Exception:  # noqa: BLE001
        return None


def event_name(year: int, track, fallback: Optional[str] = None) -> str:
    """Best-effort human event name (e.g. 'Spanish Grand Prix')."""
    name = _event_name_cached(year, str(track))
    return name or (fallback or f"{str(track).title()} GP")
