"""Clean-lap builder: mark the green-flag, representative laps used for fitting.

A lap is ``is_clean`` (usable for fuel/tyre estimation) when it is a full green-flag
racing lap on a slick compound, not an in/out lap, not lap 1, not deleted, accurate,
and not a slow outlier relative to its stint (traffic, lift-and-coast, mistakes).
Cleaned per-race frames are cached to parquet so parameter fitting need not reload.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from strategy_sim2.data import collector, session_filter
from strategy_sim2.data.schema import DRY_COMPOUNDS
from strategy_sim2.settings import load_settings, resolve_path


def clean_laps(laps: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Add/overwrite the ``is_clean`` column on a tidy lap frame."""
    cfg = cfg or load_settings()
    c = cfg["cleaning"]
    df = laps.copy()

    base = (
        df["is_green"].fillna(False)
        & df["lap_time_s"].notna()
        & df["compound"].isin(DRY_COMPOUNDS)
        & ~df["is_inlap"].fillna(False)
        & ~df["is_outlap"].fillna(False)
    )
    if c.get("drop_lap1", True):
        base &= df["lap_number"] > 1
    if c.get("drop_deleted", True):
        base &= ~df["deleted"].fillna(False)
    if c.get("require_accurate", True):
        base &= df["is_accurate"].fillna(False)

    df["is_clean"] = False
    sub = df.loc[base, ["driver", "stint", "lap_time_s"]]
    if len(sub):
        med = sub.groupby(["driver", "stint"])["lap_time_s"].transform("median")
        thresh = med * float(c.get("outlier_pct_of_stint_median", 1.07))
        df.loc[sub.index, "is_clean"] = (sub["lap_time_s"] <= thresh).values
    return df


def _derived_path(cfg: dict, year: int, rnd: int) -> Path:
    d = resolve_path(cfg["data"]["derived_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d / f"laps_{year}_{rnd:02d}.pkl"


def get_clean_race(year: int, rnd: int, cfg: dict | None = None,
                   use_cache: bool = True) -> pd.DataFrame | None:
    """Return the cleaned lap frame for one race, caching to disk. None if absent."""
    cfg = cfg or load_settings()
    path = _derived_path(cfg, year, rnd)
    if use_cache and path.exists():
        return pd.read_pickle(path)
    ses = collector.load_session(year, rnd, "R", weather=False, messages=False)
    if ses is None:
        return None
    laps = clean_laps(collector.session_laps(ses), cfg)
    laps.to_pickle(path)
    return laps


def load_training_laps(cfg: dict | None = None, clean_only: bool = True) -> pd.DataFrame:
    """Concatenate cleaned laps across all *included* dry races in the manifest."""
    cfg = cfg or load_settings()
    races = session_filter.included_races(cfg)
    frames = []
    for _, r in races.iterrows():
        df = get_clean_race(int(r["year"]), int(r["round"]), cfg)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    alllaps = pd.concat(frames, ignore_index=True)
    return alllaps[alllaps["is_clean"]].reset_index(drop=True) if clean_only else alllaps
