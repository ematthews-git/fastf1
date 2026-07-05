"""Canonical data schema, compound handling, and FastF1 track-status parsing.

We keep tabular data in pandas DataFrames with the canonical column names defined
here, and use small dataclasses only for scalar metadata objects.

FastF1 ``TrackStatus`` is a *concatenation* of the status codes that were active
during a lap (e.g. ``'671'`` = codes 6, 7 and 1 within the same lap), so status
tests are substring tests, not equality tests.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- Tyre compounds -------------------------------------------------------
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")
# Relative softness rank within a weekend (higher = softer = more grip, less life).
COMPOUND_RANK = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}


def normalize_compound(c: object) -> str:
    """Map a raw FastF1 compound string to SOFT/MEDIUM/HARD/INTERMEDIATE/WET/UNKNOWN."""
    if c is None:
        return "UNKNOWN"
    s = str(c).strip().upper()
    if s in DRY_COMPOUNDS or s in WET_COMPOUNDS:
        return s
    return "UNKNOWN"


# --- Circuit names ---------------------------------------------------------
# FastF1's event Location is not stable across seasons for the same venue (Monaco is
# "Monaco" in 2024-25 but "Monte Carlo" in 2021/2026). An un-normalised name silently
# SPLITS a circuit's history across two keys: priors, deg fits and circuit_rules each
# see only a fraction of the data. Canonical names below match the 2026 schedule.
CIRCUIT_ALIASES = {"Monaco": "Monte Carlo"}


def normalize_circuit(name: object) -> str:
    s = str(name).strip()
    return CIRCUIT_ALIASES.get(s, s)


# --- Track status codes ---------------------------------------------------
# 1 green, 2 yellow, 4 safety car, 5 red flag, 6 VSC deployed, 7 VSC ending.
TS_GREEN, TS_YELLOW, TS_SC, TS_RED, TS_VSC = "1", "2", "4", "5", "67"


def _ts(status: object) -> str:
    if status is None:
        return ""
    try:
        import math

        if isinstance(status, float) and math.isnan(status):
            return ""
    except TypeError:
        pass
    return str(status)


def ts_is_sc(status: object) -> bool:
    return TS_SC in _ts(status)


def ts_is_red(status: object) -> bool:
    return TS_RED in _ts(status)


def ts_is_vsc(status: object) -> bool:
    s = _ts(status)
    return ("6" in s) or ("7" in s)


def ts_is_green(status: object) -> bool:
    """True if the lap ran entirely under green/clear (only codes 1 or 3)."""
    s = _ts(status)
    return len(s) > 0 and all(ch in "13" for ch in s)


# --- Canonical cleaned-lap columns ---------------------------------------
LAP_COLUMNS = [
    "year", "round", "circuit", "event_format", "session",
    "driver", "driver_number", "team",
    "lap_number", "total_laps", "laps_remaining", "lap_time_s",
    "stint", "compound", "tyre_life", "fresh_tyre", "position",
    "is_green", "is_sc", "is_vsc", "is_inlap", "is_outlap", "is_clean",
]

RESULT_COLUMNS = [
    "year", "round", "circuit", "driver", "team",
    "grid", "finish_position", "classified", "status",
    "dnf", "dns", "points", "laps_completed", "best_quali_s",
]


@dataclass(frozen=True)
class SessionRef:
    year: int
    round: int
    session: str  # 'R', 'Q', 'FP1', 'S', ...


@dataclass
class SessionMeta:
    year: int
    round: int
    session: str
    circuit: str
    event_name: str
    event_format: str
    event_date: str
    n_laps: int = 0
    n_drivers: int = 0
    is_dry: bool = True
    reason: str = ""
    included: bool = True
