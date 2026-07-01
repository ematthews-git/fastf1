"""Lap cleaning, stint reconstruction, and green-flag filtering.

FastF1 laps have two quirks we always repair: missing ``Compound`` (sometimes the
literal string ``'nan'``) and a ``Stint`` column that we prefer to recompute from
pit-in events for robustness. Everything downstream (tyre model, observations)
builds on :func:`clean_laps`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import COMPOUNDS

# Track-status digits meaning not-green (SC / red / VSC deployed / VSC ending).
NON_GREEN = "[4567]"
WET_COMPOUNDS = ("INTERMEDIATE", "WET")


def add_laptime_seconds(laps: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a float ``LapTime_s`` column."""
    d = laps.copy()
    d["LapTime_s"] = d["LapTime"].dt.total_seconds()
    return d


def clean_laps(session) -> pd.DataFrame:
    """Cleaned laps with a robust ``stint_id`` and repaired ``Compound``.

    * Compound: literal ``'nan'`` -> NaN, then filled within each stint by the
      stint's modal (known) compound.
    * stint_id: a new stint begins on the lap *after* one carrying a ``PitInTime``
      (more reliable than FastF1's ``Stint`` across odd sessions).
    * Adds ``LapTime_s``.
    """
    d = session.laps.copy()
    d["Compound"] = d["Compound"].replace("nan", np.nan)
    d = d.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    # prev lap (same driver) pitted -> this lap starts a new stint
    d["_pp"] = d.groupby("Driver")["PitInTime"].shift().notna()
    d["stint_id"] = d.groupby("Driver")["_pp"].cumsum().astype(int)
    d["Compound"] = d.groupby(["Driver", "stint_id"])["Compound"].transform(
        lambda x: x.fillna(x.dropna().mode().iloc[0]) if x.notna().any() else x
    )
    d = add_laptime_seconds(d)
    return d


def green_laps(laps: pd.DataFrame, drop_start: int = 2,
               dry_only: bool = True, outlier_pct: float = 1.07) -> pd.DataFrame:
    """Clean green-flag racing laps for pace/degradation fitting.

    Keeps timed, non-in/out laps run under green flag, past the stint warm-up
    (``TyreLife > drop_start``), optionally on dry compounds only, and drops gross
    outliers slower than ``outlier_pct`` of the field median (traffic/lockups).
    """
    d = laps if "LapTime_s" in laps.columns else add_laptime_seconds(laps)
    mask = (
        d["LapTime_s"].notna()
        & d["PitInTime"].isna()
        & d["PitOutTime"].isna()
        & ~d["TrackStatus"].astype(str).str.contains(NON_GREEN, regex=True)
        & (d["TyreLife"] > drop_start)
    )
    if dry_only:
        mask &= d["Compound"].isin(COMPOUNDS)
    if "Deleted" in d.columns:
        mask &= ~(d["Deleted"] == True)  # noqa: E712  (NaN kept)
    d = d[mask].copy()
    if len(d) and outlier_pct:
        cut = d["LapTime_s"].median() * outlier_pct
        d = d[d["LapTime_s"] <= cut].copy()
    return d


def stint_table(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-(driver, stint) summary with the pit lap that ended each stint.

    Columns: Driver, stint_id, Compound, start_lap, end_lap, n_laps,
    start_tyre_life, pit_lap (the lap a stop happened on, or NA for the final
    stint / a stint that ended at the flag).
    """
    d = laps if "stint_id" in laps.columns else laps.assign(stint_id=laps["Stint"])
    rows = []
    for (drv, sid), st in d.groupby(["Driver", "stint_id"]):
        st = st.sort_values("LapNumber")
        comp = st["Compound"].dropna()
        comp = comp.mode().iloc[0] if len(comp) else np.nan
        last = st.iloc[-1]
        pit_lap = int(last["LapNumber"]) if pd.notna(last["PitInTime"]) else pd.NA
        rows.append({
            "Driver": drv,
            "stint_id": int(sid),
            "Compound": comp,
            "start_lap": int(st["LapNumber"].min()),
            "end_lap": int(st["LapNumber"].max()),
            "n_laps": int(len(st)),
            "start_tyre_life": (float(st["TyreLife"].iloc[0])
                                if pd.notna(st["TyreLife"].iloc[0]) else np.nan),
            "pit_lap": pit_lap,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["Driver", "stint_id"]).reset_index(drop=True)
    return out


def race_lap_count(laps: pd.DataFrame) -> int:
    """Scheduled race distance in laps (winner's lap count)."""
    return int(laps["LapNumber"].max())


def wet_fraction(laps: pd.DataFrame) -> float:
    """Share of timed laps run on intermediate/wet tyres (race-wetness proxy)."""
    d = laps if "LapTime_s" in laps.columns else add_laptime_seconds(laps)
    timed = d[d["LapTime_s"].notna()]
    if not len(timed):
        return 0.0
    wet = timed["Compound"].isin(WET_COMPOUNDS).sum()
    return float(wet) / float(len(timed))
