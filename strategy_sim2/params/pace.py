"""Per-driver base (clean-air, low-fuel) pace for a target weekend.

Base pace is the fastest single-lap time a driver can produce, matching the paper's
use of the qualifying time. In post-quali mode we read it straight from the
qualifying session; the preliminary mode (previous-year proxy) lives in
``context/prelim.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_sim2.data import collector


def quali_pace(year: int, rnd: int) -> dict[str, float]:
    """Best qualifying lap (seconds) per driver abbreviation for a completed weekend."""
    ses = collector.load_session(year, rnd, "Q", weather=False, messages=False)
    if ses is None:
        return {}
    res = collector.session_results(ses)
    out = {}
    for _, row in res.iterrows():
        q = row.get("best_quali_s")
        if pd.notna(q):
            out[str(row["driver"])] = float(q)
    return out


def season_form_delta(year: int, lap_model, cfg: dict | None = None) -> dict[str, float]:
    """Per-driver current-season pace delta (s), circuit-neutral, from race laps.

    Each clean lap is corrected for fuel, compound offset and tyre age, then made
    relative to that race's field median, and averaged per driver across the season.
    Negative = faster. This is a stable pre-weekend form signal (many races) that
    needs no new session loads.
    """
    from strategy_sim2.params import dataset

    laps = dataset.training_laps(cfg, years=[year])
    if not len(laps):
        return {}
    recs = []
    for row in laps.itertuples():
        fuel = lap_model.fuel_coef(row.circuit)
        corr = (row.lap_time_s - fuel * row.laps_remaining
                - lap_model.pace_offset(row.compound, row.circuit)
                - lap_model.deg(row.compound, row.tyre_life, row.circuit, row.driver))
        recs.append((row.round, row.driver, corr))
    df = pd.DataFrame(recs, columns=["round", "driver", "corr"])
    df["rel"] = df["corr"] - df.groupby("round")["corr"].transform("median")
    per = df.groupby(["round", "driver"])["rel"].median().reset_index()
    return {str(d): float(v) for d, v in per.groupby("driver")["rel"].mean().items()}


def pace_from_race_median(year: int, rnd: int, clean_laps: pd.DataFrame | None = None,
                          top_frac: float = 0.3) -> dict[str, float]:
    """Fallback base pace: each driver's fast-end race pace (mean of quickest laps).

    Used when qualifying is unavailable; expressed on the same scale as lap times.
    """
    if clean_laps is None:
        from strategy_sim2.data import clean as _clean
        clean_laps = _clean.get_clean_race(year, rnd)
    if clean_laps is None or not len(clean_laps):
        return {}
    laps = clean_laps[clean_laps["is_clean"]] if "is_clean" in clean_laps else clean_laps
    out = {}
    for drv, g in laps.groupby("driver"):
        t = g["lap_time_s"].to_numpy(float)
        if len(t):
            k = max(1, int(len(t) * top_frac))
            out[str(drv)] = float(np.mean(np.sort(t)[:k]))
    return out
