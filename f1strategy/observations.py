"""Empirical strategy extraction — the calibration TARGET.

For each historical race we distil what the *front-runners* actually did, because
that is what the simulator's optimum will be scored against. The subtlety the user
flagged: a driver who finished 4th from 12th on the grid ran a reactive recovery
race, not a free strategic optimum — so every driver gets a **strategic-freedom
weight** from grid position, finish position and how far they moved, and reference
aggregates are weighted by it. DNFs, lapped cars and wet / red-flag races are
dropped outright.

The public object is :class:`RaceObservation`; build one with
:func:`race_observation`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .data.laps import clean_laps, stint_table, race_lap_count, wet_fraction
from .data.loaders import load_results, event_name
from .config import COMPOUNDS

# Strategic-freedom weighting decay constants (positions). Larger = gentler decay.
TAU_FINISH = 4.0     # reward finishing near the front
TAU_GRID = 6.0       # reward starting near the front (free air, own strategy)
TAU_RECOVERY = 5.0   # penalise large grid->finish moves (reactive races)

WET_MAX = 0.15               # >15% of laps on inters/wets -> treat race as wet
MIN_REF_WEIGHT_FRAC = 0.25   # driver counts as a "reference" if weight >= frac*max
MIN_REF_DRIVERS = 3          # need at least this many clean front-runners
MIN_STRAT_STINT = 5          # a stint shorter than this = reactive stop (incident/SC/
                             # penalty/fastest-lap), so the driver's race isn't a clean
                             # strategic reference and is excluded


@dataclass
class RaceObservation:
    """The reference strategy a prediction for this race should reproduce."""

    track: str
    year: int
    event_name: str
    n_laps: int
    representative: bool
    reason: str = ""                       # why it was excluded, if it was
    ref_stops: Optional[int] = None        # weighted-modal stop count
    ref_sequence: tuple[str, ...] = ()     # top weighted compound sequence
    ref_pit_laps: tuple[int, ...] = ()     # weighted-median pit laps at ref_stops
    pit_windows: dict = field(default_factory=dict)   # stop idx -> {median, lo, hi}
    sequences: list = field(default_factory=list)     # [(seq, weight_frac), ...]
    stop_distribution: dict = field(default_factory=dict)  # n_stops -> weight_frac
    weight: float = 0.0                    # loss weight (agreement * data volume)
    drivers: Optional[pd.DataFrame] = field(default=None, repr=False)

    def label(self) -> str:
        seq = "→".join(s[0] for s in self.ref_sequence)
        pits = ",".join(map(str, self.ref_pit_laps)) or "-"
        return f"{self.ref_stops}-stop {seq} @ {pits}"


def strategic_freedom_weight(grid: int, finish: int) -> float:
    """Weight in (0, 1] favouring clean front-running, own-strategy drives."""
    wf = math.exp(-(finish - 1) / TAU_FINISH)
    wg = math.exp(-(grid - 1) / TAU_GRID)
    wr = math.exp(-abs(grid - finish) / TAU_RECOVERY)
    return wf * wg * wr


def driver_strategies(session, laps: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per-driver strategy + classification for one race.

    Columns: Driver, grid, finish, n_stops, sequence (tuple), pit_laps (tuple),
    dnf, lapped, freedom (weight), eligible (clean, non-lapped finisher).
    """
    laps = clean_laps(session) if laps is None else laps
    st = stint_table(laps)
    results = load_results(session)
    n_drivers = max(len(results), 1)

    rows = []
    for drv, g in st.groupby("Driver"):
        g = g.sort_values("stint_id")
        seq = tuple(c for c in g["Compound"].tolist() if isinstance(c, str) and c in COMPOUNDS)
        pit_laps = tuple(int(x) for x in g["pit_lap"].dropna().tolist())
        stint_lengths = [int(x) for x in g["n_laps"].tolist()]
        has_short_stint = any(L < MIN_STRAT_STINT for L in stint_lengths)
        info = results.get(drv, {})
        grid = info.get("grid", n_drivers)
        finish = info.get("finish", n_drivers)
        classified = info.get("classified", "")
        status = info.get("status", "").lower()
        dnf = not str(classified).isdigit()
        lapped = ("lap" in status) and ("finished" not in status)
        rows.append({
            "Driver": drv,
            "grid": grid,
            "finish": finish,
            "n_stops": len(pit_laps),
            "sequence": seq,
            "pit_laps": pit_laps,
            "dnf": dnf,
            "lapped": lapped,
            "freedom": strategic_freedom_weight(grid, finish),
            # a clean strategic reference: finished on the lead lap with no reactive
            # (very short) stints
            "eligible": (not dnf) and (not lapped) and len(seq) >= 1 and not has_short_stint,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("finish").reset_index(drop=True)
    return df


def _weighted_mode(pairs: list[tuple]) -> tuple[object, dict]:
    agg: dict = defaultdict(float)
    for value, w in pairs:
        agg[value] += w
    total = sum(agg.values()) or 1.0
    frac = {k: v / total for k, v in agg.items()}
    top = max(agg, key=agg.get)
    return top, frac


def _weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return float(np.interp(q, cw, v))


def _representative(session, laps: pd.DataFrame) -> tuple[bool, str]:
    """Reject wet, red-flag-warped, or too-thin races for calibration."""
    if wet_fraction(laps) > WET_MAX:
        return False, "wet race"
    if laps["TrackStatus"].astype(str).str.contains("5", regex=False).any():
        return False, "red flag"     # free tyre change under red warps pit timing
    return True, ""


def race_observation(session, track: str, year: int) -> RaceObservation:
    """Build the reference strategy for one race (the calibration target)."""
    laps = clean_laps(session)
    n_laps = race_lap_count(laps)
    fallback = None
    ev = getattr(session, "event", None)
    if ev is not None:
        try:
            fallback = ev["EventName"]
        except Exception:  # noqa: BLE001
            fallback = None
    name = event_name(year, track, fallback=fallback)
    rep, reason = _representative(session, laps)
    drivers = driver_strategies(session, laps)

    obs = RaceObservation(track=track, year=year, event_name=name, n_laps=n_laps,
                          representative=rep, reason=reason, drivers=drivers)
    if not rep or not len(drivers):
        return obs

    clean = drivers[drivers["eligible"]].copy()
    if len(clean) < MIN_REF_DRIVERS:
        obs.representative = False
        obs.reason = "too few clean front-runners"
        return obs

    # Restrict to genuine reference drivers: freedom weight within a band of the best.
    wmax = clean["freedom"].max()
    ref = clean[clean["freedom"] >= MIN_REF_WEIGHT_FRAC * wmax].copy()

    # Reference stop count (weighted modal) + full stop distribution.
    ref_stops, stop_frac = _weighted_mode(list(zip(ref["n_stops"], ref["freedom"])))
    ref_stops = int(ref_stops)

    # Top compound sequence by weight (among all reference drivers).
    _, seq_frac = _weighted_mode(list(zip(ref["sequence"], ref["freedom"])))
    sequences = sorted(seq_frac.items(), key=lambda kv: kv[1], reverse=True)
    ref_sequence = sequences[0][0] if sequences else ()

    # Pit windows: among reference drivers running the modal stop count, take the
    # weighted-median pit lap (and 10-90% spread) for each stop index.
    at_ref = ref[ref["n_stops"] == ref_stops]
    windows: dict = {}
    ref_pit_laps: list[int] = []
    for i in range(ref_stops):
        laps_i = [(d["pit_laps"][i], d["freedom"]) for _, d in at_ref.iterrows()
                  if len(d["pit_laps"]) > i]
        if not laps_i:
            continue
        vals = [a for a, _ in laps_i]
        wts = [b for _, b in laps_i]
        med = _weighted_quantile(vals, wts, 0.5)
        lo = _weighted_quantile(vals, wts, 0.10)
        hi = _weighted_quantile(vals, wts, 0.90)
        windows[i] = {"median": round(med, 1), "lo": round(lo, 1), "hi": round(hi, 1)}
        ref_pit_laps.append(int(round(med)))

    # Confidence: agreement on the modal stop count, tempered by how much clean data
    # backs it (saturates around a handful of reference drivers).
    agreement = stop_frac.get(ref_stops, 0.0)
    volume = 1.0 - math.exp(-ref["freedom"].sum() / 1.5)
    obs.ref_stops = ref_stops
    obs.ref_sequence = ref_sequence
    obs.ref_pit_laps = tuple(ref_pit_laps)
    obs.pit_windows = windows
    obs.sequences = sequences
    obs.stop_distribution = dict(sorted(stop_frac.items()))
    obs.weight = float(agreement * volume)
    return obs
