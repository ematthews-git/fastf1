"""Parametric tyre model — per-compound pace + degradation, race+practice hybrid.

Predictive by construction: a target year's model is built only from seasons
*strictly before* it (no look-ahead), so it works for a race that hasn't happened.
Optionally folds the target weekend's own FP long-runs in for the post-practice
"updated" prediction.

Per compound we produce a :class:`CompoundModel` with:
  * ``deg``        — degradation, s/lap
  * ``base_offset`` — fresh-lap pace vs the fastest available compound, s (>=0)

Estimation (all on clean green laps):
  1. ``fuel_rate`` — fuel-burn + track-evolution, s/lap, via joint OLS with
     per-season intercepts (so car evolution doesn't leak into the fuel term).
  2. degradation — HARD from race data (run to the end of its stints, fully
     observed); MEDIUM/SOFT from practice long-runs when available (races truncate
     their degraded phase because teams pit them before the cliff), else race.
  3. base offsets — fuel/season-corrected fresh pushing pace per compound, blended
     with a softness ladder prior for robustness.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from .config import COMPOUNDS, SOFTNESS, CompoundModel
from .data.laps import clean_laps, green_laps
from .data import loaders

MAX_HISTORY = 4
MIN_STINT_LAPS = 8       # min clean laps for a race stint to yield a slope
MIN_ELIGIBLE_LAPS = 30   # min pooled clean laps for a compound to be modellable
DEG_FLOOR = 0.008        # degradation is physically non-negative; a negative estimate is
                         # a thin-data/fuel-correction artifact, so floor it at ~0
FUEL_BOUNDS = (-0.14, -0.02)
FUEL_FALLBACK = -0.055
MIN_LONGRUN = 6          # min laps for a practice run to count as a long run
FUEL_ADD_FALLBACK = -0.05
LADDER_STEP = 0.6        # s of fresh pace per compound step (prior)
OFFSET_PRIOR_WEIGHT = 0.5  # blend weight: measured fresh-pace gap vs ladder prior


# --------------------------------------------------------------------------- pooling


GROUND_EFFECT_ERA = 2022   # first season of the current regulation set


def _season_race_pool(track, target_year, seasons_back, loader,
                      min_season: int = GROUND_EFFECT_ERA) -> tuple[pd.DataFrame, list[int]]:
    """Concat clean green race laps from prior seasons (no look-ahead).

    ``min_season`` floors the pool so a target never reaches across a regulation
    boundary into structurally-different cars (default: the ground-effect era).
    """
    frames, seasons = [], []
    for y in range(target_year - 1, target_year - 1 - seasons_back, -1):
        if y < min_season:
            break
        try:
            s = loader(y, track)
        except Exception:  # noqa: BLE001  (race may not exist / not cached)
            continue
        if getattr(s, "laps", None) is None or not len(s.laps):
            continue
        g = green_laps(clean_laps(s))
        if not len(g):
            continue
        g = g.assign(year=y)
        frames.append(g)
        seasons.append(y)
    if not frames:
        return pd.DataFrame(), []
    return pd.concat(frames, ignore_index=True), seasons


# ------------------------------------------------------------------- fuel / deg fits


def estimate_fuel_rate(pool: pd.DataFrame) -> float:
    """Fuel+track-evo rate (s/lap, negative) via joint OLS with season intercepts."""
    g = pool.dropna(subset=["LapNumber", "TyreLife", "Compound", "LapTime_s"])
    if len(g) < 50:
        return FUEL_FALLBACK
    years = sorted(g["year"].unique())
    cols = [(g["year"] == y).astype(float).values for y in years]
    cols.append(g["LapNumber"].astype(float).values)                 # fuel
    for c in COMPOUNDS:
        cols.append(((g["Compound"] == c) * g["TyreLife"]).astype(float).values)
    X = np.column_stack(cols)
    try:
        beta, *_ = np.linalg.lstsq(X, g["LapTime_s"].values, rcond=None)
        fuel = float(beta[len(years)])
    except Exception:  # noqa: BLE001
        return FUEL_FALLBACK
    if not np.isfinite(fuel):
        return FUEL_FALLBACK
    return float(np.clip(fuel, *FUEL_BOUNDS))


def _wmedian(vals, weights) -> float:
    v = np.asarray(vals, float)
    w = np.asarray(weights, float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, cw[-1] / 2.0)])


def race_stint_deg(pool: pd.DataFrame, fuel_rate: float) -> tuple[dict, dict]:
    """Per-compound degradation from race stints (fuel-corrected robust slopes)."""
    g = pool.dropna(subset=["LapNumber", "TyreLife", "Compound", "LapTime_s"]).copy()
    g["fc"] = g["LapTime_s"] - fuel_rate * g["LapNumber"]
    acc = defaultdict(list)
    for (_, _, _), st in g.groupby(["year", "Driver", "stint_id"]):
        if len(st) < MIN_STINT_LAPS:
            continue
        comp = st["Compound"].mode().iloc[0]
        if comp not in COMPOUNDS:
            continue
        x = st["TyreLife"].astype(float).values
        if x.std() < 1e-6:
            continue
        slope = float(np.polyfit(x, st["fc"].values, 1)[0])
        acc[comp].append((slope, len(st)))
    deg = {c: _wmedian([s for s, _ in a], [n for _, n in a]) for c, a in acc.items()}
    return deg, {c: len(a) for c, a in acc.items()}


def base_offsets(pool: pd.DataFrame, fuel_rate: float) -> dict:
    """Fresh-lap pace offset per compound (s vs fastest), blended with a ladder.

    Measured part: fuel + per-season-median corrected pushing pace on fresh tyres
    (low tyre life), taken as a low quantile (fast laps). Prior part: a softness
    ladder (softer = faster fresh). Robust to a compound being thin in the data.
    """
    g = pool.dropna(subset=["LapNumber", "TyreLife", "Compound", "LapTime_s"]).copy()
    g["fc"] = g["LapTime_s"] - fuel_rate * g["LapNumber"]
    g["fc"] -= g.groupby("year")["fc"].transform("median")   # cross-season comparable
    fresh = g[g["TyreLife"] <= 6]
    meas = {}
    for c in COMPOUNDS:
        gc = fresh[fresh["Compound"] == c]["fc"]
        if len(gc) >= 8:
            meas[c] = float(gc.quantile(0.20))   # pushing (fast) fresh pace
    present = [c for c in COMPOUNDS if c in meas]
    ladder = {c: -(SOFTNESS[c] - 1) * LADDER_STEP for c in COMPOUNDS}  # softer = faster
    if present:
        m0 = min(meas[c] for c in present)
        measured = {c: (meas[c] - m0 if c in meas else None) for c in COMPOUNDS}
    else:
        measured = {c: None for c in COMPOUNDS}
    l0 = min(ladder[c] for c in COMPOUNDS)
    prior = {c: ladder[c] - l0 for c in COMPOUNDS}
    out = {}
    for c in COMPOUNDS:
        if measured[c] is not None:
            out[c] = OFFSET_PRIOR_WEIGHT * prior[c] + (1 - OFFSET_PRIOR_WEIGHT) * measured[c]
        else:
            out[c] = prior[c]
    # renormalise so fastest available = 0
    lo = min(out.values())
    return {c: out[c] - lo for c in COMPOUNDS}


# ---------------------------------------------------------------- practice long runs


def _run_slope(run: pd.DataFrame) -> Optional[tuple[float, int]]:
    """Robust deg slope of one practice run vs tyre age (Theil-Sen)."""
    r = run.copy()
    med = r["LapTime_s"].median()
    r = r[r["LapTime_s"] <= med * 1.03]        # drop cool-down / backing-off laps
    if len(r) < MIN_LONGRUN:
        return None
    x = r["TyreLife"].astype(float).values
    if x.max() - x.min() < MIN_LONGRUN - 1:
        return None
    slope = float(theilslopes(r["LapTime_s"].values, x)[0])
    return slope, len(r)


def practice_deg(track, years, fuel_rate, loader) -> tuple[dict, dict]:
    """Per-compound deg from practice race-sim long runs, pooled over years."""
    add_back = fuel_rate if np.isfinite(fuel_rate) else FUEL_ADD_FALLBACK
    acc = defaultdict(list)
    for y in years:
        for _, s in loaders.load_practice_sessions(y, track):
            g = green_laps(clean_laps(s), drop_start=1)
            for (_, _), run in g.groupby(["Driver", "stint_id"]):
                if len(run) < MIN_LONGRUN:
                    continue
                comp = run["Compound"].mode().iloc[0]
                if comp not in COMPOUNDS:
                    continue
                res = _run_slope(run)
                if res is None:
                    continue
                slope, n = res
                acc[comp].append((slope - add_back, n))   # recover intrinsic deg
    deg = {c: _wmedian([d for d, _ in a], [n for _, n in a]) for c, a in acc.items()}
    return deg, {c: len(a) for c, a in acc.items()}


# ------------------------------------------------------------------------ assemble


def _hybrid_deg(race_deg: dict, prac_deg: dict) -> dict:
    """HARD from race, MEDIUM/SOFT from practice when available, else fall back."""
    out = {}
    for c in COMPOUNDS:
        r, p = race_deg.get(c), prac_deg.get(c)
        primary, backup = (r, p) if c == "HARD" else (p, r)
        val = primary if (primary is not None and np.isfinite(primary)) else backup
        if val is not None and np.isfinite(val):
            out[c] = val
    return out


def build_tyre_model(track, target_year, seasons_back: int = MAX_HISTORY,
                     use_practice: bool = False, target_practice: bool = False,
                     loader: Optional[Callable] = None,
                     min_season: int = GROUND_EFFECT_ERA) -> dict:
    """Build the per-compound tyre model + fuel rate for one race.

    Returns a dict: ``{"compounds": {slot: CompoundModel}, "fuel_rate": float,
    "seasons_used": [...], "sources": {...}}``. ``compounds`` only includes slots
    with enough data to model.
    """
    loader = loader or loaders.load_race
    pool, seasons = _season_race_pool(track, target_year, seasons_back, loader, min_season)
    if not len(pool):
        return {"compounds": {}, "fuel_rate": FUEL_FALLBACK, "seasons_used": [], "sources": {}}

    fuel = estimate_fuel_rate(pool)
    race_deg, race_n = race_stint_deg(pool, fuel)
    offsets = base_offsets(pool, fuel)
    laps_per = {c: int((pool["Compound"] == c).sum()) for c in COMPOUNDS}

    prac_deg, prac_n = {}, {}
    if use_practice or target_practice:
        pyears = [y for y in range(target_year - 1, target_year - 1 - seasons_back, -1)
                  if y >= min_season]
        if target_practice:
            pyears = [target_year] + pyears
        prac_deg, prac_n = practice_deg(track, pyears, fuel, loader)

    hybrid = _hybrid_deg(race_deg, prac_deg)

    compounds: dict[str, CompoundModel] = {}
    sources: dict[str, str] = {}
    for c in COMPOUNDS:
        if c not in hybrid or laps_per[c] < MIN_ELIGIBLE_LAPS:
            continue
        used_prac = (c != "HARD") and (c in prac_deg) and np.isfinite(prac_deg.get(c, np.nan))
        src = "practice" if used_prac else "race"
        compounds[c] = CompoundModel(
            slot=c, deg=float(max(hybrid[c], DEG_FLOOR)), base_offset=float(offsets[c]),
            n=laps_per[c], source=src, cx=None,
        )
        sources[c] = src

    return {
        "compounds": compounds,
        "fuel_rate": fuel,
        "seasons_used": seasons,
        "sources": sources,
        "race_deg": race_deg,
        "practice_deg": prac_deg,
        "practice_n": prac_n,
    }
