"""Pit-stop time loss per circuit.

Placeholder hand-curated values for now — the user has a dedicated pit-loss
function that will drop in behind :func:`pit_loss` later. A data-derived
:func:`measure_pit_loss` is provided as a sanity fallback / cross-check.

"Pit loss" here = total green-flag time lost by making a stop vs staying out
(pit-lane delta + stationary time), in seconds.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .data.laps import clean_laps, green_laps

DEFAULT_PIT_LOSS = 21.0

# Approximate green-flag pit loss by circuit (seconds). Keyed by lowercase
# substrings matched against the track name. GENERIC / FOR TESTING ONLY.
_PIT_LOSS: dict[str, float] = {
    "bahrain": 22.5,
    "jeddah": 20.0, "saudi": 20.0,
    "australia": 19.5, "melbourne": 19.5, "albert": 19.5,
    "japan": 22.0, "suzuka": 22.0,
    "china": 22.0, "shanghai": 22.0,
    "miami": 20.5,
    "imola": 27.5, "emilia": 27.5,       # notably long, slow pit lane
    "monaco": 20.0,
    "canada": 17.0, "montreal": 17.0, "gilles": 17.0,   # famously quick pit lane
    "spain": 21.0, "barcelona": 21.0, "catal": 21.0,
    "austria": 20.0, "spielberg": 20.0, "red bull ring": 20.0,
    "britain": 20.5, "silverstone": 20.5,
    "hungar": 20.0, "budapest": 20.0,
    "belgium": 19.0, "spa": 19.0,
    "netherlands": 21.0, "zandvoort": 21.0,
    "italy": 24.0, "monza": 24.0,
    "azerbaijan": 19.5, "baku": 19.5,
    "singapore": 28.0, "marina bay": 28.0,
    "austin": 21.0, "cota": 21.0, "united states": 21.0,
    "mexico": 21.5, "hermanos": 21.5,
    "brazil": 20.5, "interlagos": 20.5, "paulo": 20.5,
    "vegas": 19.5,
    "qatar": 25.0, "losail": 25.0, "lusail": 25.0,
    "abu dhabi": 21.0, "yas": 21.0,
}


def pit_loss(track: str, year: Optional[int] = None) -> float:
    """Curated green-flag pit loss (s) for a circuit; DEFAULT if unknown.

    Signature is intentionally ``(track, year)`` so the user's real pit-loss
    function can replace this module without touching callers.
    """
    key = str(track).lower()
    for sub, val in _PIT_LOSS.items():
        if sub in key:
            return val
    return DEFAULT_PIT_LOSS


def measure_pit_loss(session) -> float:
    """Data-derived pit loss from a race (median in+out lap delta vs base pace).

    Excludes stops made under SC/VSC/red. Robust median. Returns NaN if the race
    has no clean green stops. Use as a sanity check against :func:`pit_loss`.
    """
    d = clean_laps(session)
    d["ts"] = d["TrackStatus"].astype(str)
    losses = []
    for _, dd in d.groupby("Driver"):
        dd = dd.sort_values("LapNumber")
        base = green_laps(dd)["LapTime_s"].median()
        if not np.isfinite(base):
            continue
        for lap in dd.loc[dd["PitInTime"].notna(), "LapNumber"]:
            ri = dd[dd["LapNumber"] == lap]
            ro = dd[dd["LapNumber"] == lap + 1]
            if not len(ri) or not len(ro):
                continue
            if ri["ts"].str.contains("[4567]").any() or ro["ts"].str.contains("[4567]").any():
                continue
            il, ol = ri["LapTime_s"].iloc[0], ro["LapTime_s"].iloc[0]
            if np.isfinite(il) and np.isfinite(ol):
                losses.append((il - base) + (ol - base))
    losses = np.array([x for x in losses if np.isfinite(x)])
    if not len(losses):
        return float("nan")
    return float(np.median(losses[losses < np.median(losses) * 1.3]))
