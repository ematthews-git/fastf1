"""Per-track overtaking difficulty — how costly it is to be stuck in traffic.

This is what gives track position its value: at a low-ease circuit (Monaco) a
faster car can be trapped behind a slower one for many laps, so a strategy that
sheds track position (an extra stop, rejoining in a pack) is badly punished; at a
high-ease circuit (Monza) traffic clears quickly and clean-air pace dominates.

Hand-curated table for now (GENERIC / FOR TESTING), same swappable pattern as
``pitloss.py`` / ``safetycar.py``; a data-driven estimator is provided.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

# Overtaking ease in [0, 1]: 0 ≈ impossible (Monaco), 1 ≈ trivial. Keyed by lowercase
# substrings matched against the track name.
OVERTAKE_EASE: dict[str, float] = {
    "monaco": 0.05,
    "hungar": 0.15, "budapest": 0.15,
    "singapore": 0.18, "marina bay": 0.18,
    "zandvoort": 0.20, "netherlands": 0.20,
    "imola": 0.22, "emilia": 0.22,
    "spain": 0.30, "barcelona": 0.30, "catal": 0.30,
    "japan": 0.35, "suzuka": 0.35,
    "australia": 0.38, "melbourne": 0.38,
    "qatar": 0.38, "losail": 0.38, "lusail": 0.38,
    "abu dhabi": 0.40, "yas": 0.40,
    "mexico": 0.42, "hermanos": 0.42,
    "miami": 0.45,
    "saudi": 0.52, "jeddah": 0.52,
    "britain": 0.55, "silverstone": 0.55,
    "united states": 0.55, "austin": 0.55, "cota": 0.55,
    "brazil": 0.55, "interlagos": 0.55, "paulo": 0.55,
    "china": 0.58, "shanghai": 0.58,
    "canada": 0.60, "montreal": 0.60,
    "austria": 0.62, "spielberg": 0.62,
    "bahrain": 0.65,
    "belgium": 0.70, "spa": 0.70,
    "vegas": 0.75,
    "azerbaijan": 0.80, "baku": 0.80,
    "italy": 0.85, "monza": 0.85,
}
DEFAULT_EASE = 0.45

# Pace advantage (s/lap) that makes passing routine on an *easy* track. On harder
# tracks the advantage actually needed scales up as ease falls, so a much faster car
# still gets through anywhere while a marginally faster one is trapped at Monaco.
DELTA_REF = 0.55


def overtake_ease(track: str) -> float:
    key = str(track).lower()
    for sub, val in OVERTAKE_EASE.items():
        if sub in key:
            return val
    return DEFAULT_EASE


def pass_probability(pace_delta: float, ease: float, overtake_scale: float = 1.0) -> float:
    """Per-lap probability a follower that is ``pace_delta`` s/lap faster clears the
    car ahead.

    Ease sets the pace advantage *needed*, not a cap on the probability: a much faster
    car gets through even at Monaco, while a marginally faster one is trapped there but
    passes freely at Monza. Zero if the follower isn't actually faster.
    """
    if pace_delta <= 0.0:
        return 0.0
    delta_needed = DELTA_REF / max(ease * overtake_scale, 0.03)
    return float(min(0.98, 1.0 - math.exp(-pace_delta / delta_needed)))


# ---------------------------------------------------------------- data-driven prior


def measure_overtake_ease(session, threshold: float = 0.6) -> Optional[float]:
    """Rough data-driven ease from on-track position volatility.

    Uses the mean per-lap change in classified position across green laps as a proxy
    for how much overtaking a circuit permits, squashed into [0,1]. Slower path (needs
    the ``Position`` column); use to refresh :data:`OVERTAKE_EASE`, not on the hot path.
    """
    laps = getattr(session, "laps", None)
    if laps is None or not len(laps) or "Position" not in laps.columns:
        return None
    d = laps.dropna(subset=["Position", "LapNumber"]).copy()
    changes = []
    for _, g in d.groupby("Driver"):
        g = g.sort_values("LapNumber")
        changes.append(g["Position"].diff().abs().mean())
    import numpy as np
    m = np.nanmean([c for c in changes if c == c]) if changes else np.nan
    if not np.isfinite(m):
        return None
    return float(min(1.0, m / threshold))
