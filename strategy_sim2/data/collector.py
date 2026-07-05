"""FastF1 data collection: load sessions and turn them into tidy DataFrames.

Everything downstream consumes the canonical frames returned here, never raw FastF1
objects, so the rest of the codebase is insulated from FastF1's schema.
"""
from __future__ import annotations

import warnings
from typing import Optional

warnings.filterwarnings("ignore")

import fastf1
import numpy as np
import pandas as pd

try:
    from fastf1.exceptions import RateLimitExceededError
except Exception:  # pragma: no cover - fastf1 version guard
    class RateLimitExceededError(Exception):
        pass

from strategy_sim2.data import schema
from strategy_sim2.settings import cache_dir, load_settings

_CACHE_READY = False
_RATE_LIMITED = False  # set once FastF1's hourly API budget is exhausted this run


def rate_limited() -> bool:
    return _RATE_LIMITED


def ensure_cache(cfg: dict | None = None) -> None:
    global _CACHE_READY
    if not _CACHE_READY:
        fastf1.Cache.enable_cache(str(cache_dir(cfg or load_settings())))
        _CACHE_READY = True


def get_schedule(year: int) -> pd.DataFrame:
    ensure_cache()
    return fastf1.get_event_schedule(year, include_testing=False)


def load_session(year: int, rnd: int, session: str = "R",
                 *, weather: bool = True, messages: bool = False):
    """Load a FastF1 session (no telemetry). Returns None if data is unavailable.

    Cached sessions load without network. Once the hourly API budget is exhausted
    we stop attempting further network loads this run (returning None) instead of
    crashing, so partial builds degrade gracefully.
    """
    global _RATE_LIMITED
    ensure_cache()
    try:
        ses = fastf1.get_session(year, rnd, session)
        ses.load(telemetry=False, weather=weather, messages=messages)
    except RateLimitExceededError:
        _RATE_LIMITED = True
        return None
    except Exception:
        return None
    if ses.laps is None or len(ses.laps) == 0:
        return None
    return ses


def _seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def session_laps(ses) -> pd.DataFrame:
    """Return a tidy per-lap frame with canonical columns and parsed flags.

    Includes *all* laps (with green/sc/vsc/in-lap/out-lap flags); ``is_clean`` is
    filled in later by :mod:`strategy_sim2.data.clean`.
    """
    laps = ses.laps
    ev = ses.event
    ts = laps["TrackStatus"].astype("string").fillna("")

    out = pd.DataFrame({
        "year": int(ev["EventDate"].year),
        "round": int(ev["RoundNumber"]),
        "circuit": schema.normalize_circuit(ev["Location"]),
        "event_format": str(ev["EventFormat"]),
        "session": str(ses.name),
        "driver": laps["Driver"].astype("string"),
        "driver_number": laps["DriverNumber"].astype("string"),
        "team": laps["Team"].astype("string"),
        "lap_number": laps["LapNumber"].astype("float"),
        "lap_time_s": _seconds(laps["LapTime"]),
        "stint": laps["Stint"].astype("float"),
        "compound": laps["Compound"].map(schema.normalize_compound).astype("string"),
        "tyre_life": laps["TyreLife"].astype("float"),
        "fresh_tyre": laps["FreshTyre"].astype("boolean"),
        "position": laps["Position"].astype("float"),
        "is_sc": ts.str.contains("4", regex=False),
        "is_red": ts.str.contains("5", regex=False),
        "is_vsc": ts.str.contains("6", regex=False) | ts.str.contains("7", regex=False),
        "is_green": ts.str.fullmatch(r"[13]+").fillna(False),
        "is_inlap": laps["PitInTime"].notna(),
        "is_outlap": laps["PitOutTime"].notna(),
        "deleted": laps["Deleted"].astype("boolean").fillna(False),
        "is_accurate": laps["IsAccurate"].astype("boolean").fillna(False),
    })

    total_laps = int(np.nanmax(out["lap_number"].to_numpy())) if len(out) else 0
    out["total_laps"] = total_laps
    out["laps_remaining"] = total_laps - out["lap_number"]
    out["is_clean"] = False  # set by clean.clean_laps
    return out


_DNS_STATUS = {"Did not start", "Withdrew", "Did not qualify", "Did not prequalify"}


def _classify(classified_position: object, status: object) -> tuple[bool, bool]:
    """Return (classified_finisher, dns) from FastF1 result fields.

    ``ClassifiedPosition`` is the reliable signal: a number (incl. lapped runners)
    means classified; 'R' retired, 'N' not classified, 'E' excluded -> DNF;
    'W' withdrew, 'F' failed to qualify -> did-not-start; 'D' disqualified is
    treated as a finisher (a penalty, not a reliability retirement).
    """
    cp = str(classified_position).strip()
    if cp in {"W", "F"}:
        return False, True
    if cp in {"R", "N", "E"}:
        return False, False
    if cp.lstrip("-").isdigit() or cp == "D":
        return True, False
    # Fallback to Status text when ClassifiedPosition is blank.
    s = str(status).strip()
    if s in _DNS_STATUS:
        return False, True
    if s in {"Finished", "Lapped"} or (s.startswith("+") and "Lap" in s):
        return True, False
    if s == "Retired":
        return False, False
    return True, False  # default: assume classified rather than invent a DNF


def session_results(ses) -> pd.DataFrame:
    """Tidy race/qualifying results: grid, finish, DNF, points, best quali time."""
    res = ses.results.copy()
    ev = ses.event

    q_cols = [c for c in ("Q1", "Q2", "Q3") if c in res.columns]
    if q_cols:
        q_secs = pd.concat([_seconds(res[c]) for c in q_cols], axis=1)
        best_quali = q_secs.min(axis=1, skipna=True)
    else:
        best_quali = pd.Series(np.nan, index=res.index)

    cpos = res["ClassifiedPosition"] if "ClassifiedPosition" in res.columns else res["Position"]
    pairs = [_classify(cp, st) for cp, st in zip(cpos, res["Status"])] if len(res) else []
    finished = pd.Series([p[0] for p in pairs], index=res.index)
    dns = pd.Series([p[1] for p in pairs], index=res.index)

    out = pd.DataFrame({
        "year": int(ev["EventDate"].year),
        "round": int(ev["RoundNumber"]),
        "circuit": schema.normalize_circuit(ev["Location"]),
        "driver": res["Abbreviation"].astype("string"),
        "team": res["TeamName"].astype("string"),
        "grid": res["GridPosition"].astype("float"),
        "finish_position": res["Position"].astype("float"),
        "status": res["Status"].astype("string"),
        "points": res["Points"].astype("float"),
        "laps_completed": res["Laps"].astype("float") if "Laps" in res.columns else np.nan,
        "best_quali_s": best_quali.astype("float").values,
    })
    out["dns"] = dns.values
    out["classified"] = finished.values
    out["dnf"] = (~finished.values) & (~dns.values)
    return out.reset_index(drop=True)


def weather_summary(ses) -> dict:
    """Small dry/wet-relevant weather summary for a session."""
    try:
        w = ses.weather_data
    except Exception:
        return {"rainfall_any": False, "track_temp_min": np.nan, "track_temp_max": np.nan}
    return {
        "rainfall_any": bool(w["Rainfall"].any()) if "Rainfall" in w else False,
        "track_temp_min": float(w["TrackTemp"].min()) if "TrackTemp" in w else np.nan,
        "track_temp_max": float(w["TrackTemp"].max()) if "TrackTemp" in w else np.nan,
    }


def session_meta(ses) -> schema.SessionMeta:
    ev = ses.event
    laps = session_laps(ses)
    return schema.SessionMeta(
        year=int(ev["EventDate"].year),
        round=int(ev["RoundNumber"]),
        session=str(ses.name),
        circuit=schema.normalize_circuit(ev["Location"]),
        event_name=str(ev["EventName"]),
        event_format=str(ev["EventFormat"]),
        event_date=str(ev["EventDate"].date()),
        n_laps=int(laps["total_laps"].iloc[0]) if len(laps) else 0,
        n_drivers=int(laps["driver"].nunique()) if len(laps) else 0,
    )
