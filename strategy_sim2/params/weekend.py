"""Weekend-specific tyre model from practice / sprint long runs.

Historical parameters can't see a regime change (new-generation tyres behaving
differently at this circuit *this* weekend). Teams read that from Friday long runs —
so do we:

* **Deg slopes**: within each ≥`min_run_laps` green stint, lap time regressed on tyre
  age; the falling fuel load biases the within-stint slope down by ~`fuel_coef`, so we
  add it back. Pooled across drivers/sessions per compound.
* **Pace offsets**: drivers who ran two compounds in the *same session* give a paired
  fuel/deg-corrected level delta vs MEDIUM (cross-session levels are not comparable —
  unknown fuel loads).
* **Usage shares**: the fraction of long-run laps per compound. Teams rehearse the
  tyres they intend to race — a strong prior on race compound choice.

Weekend estimates are EB-blended with the historical circuit model by lap count
(`_shrink`), and exposed via :class:`WeekendLapModel`, a thin adapter that overrides
`pace_offset`/`deg` for the target circuit and delegates everything else.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy_sim2.data import collector
from strategy_sim2.data.schema import DRY_COMPOUNDS
from strategy_sim2.params.lapmodel import COMPOUNDS, LapModel, _shrink
from strategy_sim2.settings import load_settings

_SESSIONS = ["FP1", "FP2", "FP3", "S"]  # unavailable ones (e.g. FP3 on sprint) skip


@dataclass
class WeekendTyres:
    circuit: str
    deg_slopes: dict = field(default_factory=dict)   # compound -> weekend slope (s/lap)
    offsets: dict = field(default_factory=dict)      # compound -> offset vs MEDIUM (s)
    usage: dict = field(default_factory=dict)        # compound -> long-run lap share
    n_laps: dict = field(default_factory=dict)       # compound -> long-run laps observed
    n_runs: int = 0

    def usage_prior(self, compound: str, alpha: float = 8.0) -> float:
        total = sum(self.n_laps.values())
        return (self.n_laps.get(compound, 0) + alpha) / (total + alpha * len(COMPOUNDS))


def _long_runs(year: int, rnd: int, min_run_laps: int) -> pd.DataFrame:
    """Green, dry-compound, plausible laps from all practice sessions, in stints of
    >= min_run_laps laps. Columns: session, driver, stint, compound, age, lap_time_s."""
    frames = []
    for ses_name in _SESSIONS:
        ses = collector.load_session(year, rnd, ses_name, weather=False, messages=False)
        if ses is None:
            continue
        laps = collector.session_laps(ses)
        ok = (laps["is_green"].fillna(False)
              & laps["lap_time_s"].notna()
              & laps["compound"].isin(DRY_COMPOUNDS)
              & ~laps["is_inlap"].fillna(False)
              & ~laps["is_outlap"].fillna(False)
              & ~laps["deleted"].fillna(False))
        sub = laps[ok].copy()
        if not len(sub):
            continue
        # drop slow non-representative laps within a stint (traffic, aborted runs)
        med = sub.groupby(["driver", "stint"])["lap_time_s"].transform("median")
        sub = sub[sub["lap_time_s"] <= med * 1.05]
        counts = sub.groupby(["driver", "stint"])["lap_time_s"].transform("size")
        sub = sub[counts >= min_run_laps]
        if len(sub):
            sub["session"] = ses_name
            frames.append(sub[["session", "driver", "stint", "compound",
                               "tyre_life", "lap_time_s"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).rename(columns={"tyre_life": "age"})


def fit_weekend(year: int, rnd: int, circuit: str, lap_model: LapModel,
                cfg: dict | None = None) -> WeekendTyres | None:
    cfg = cfg or load_settings()
    wcfg = cfg.get("weekend", {})
    min_run = int(wcfg.get("min_run_laps", 5))
    runs = _long_runs(year, rnd, min_run)
    wt = WeekendTyres(circuit=circuit)
    if not len(runs):
        return None
    fuel = lap_model.fuel_coef(circuit)

    # --- deg slopes: within-stint regression + fuel-burn correction ---
    for comp, g in runs.groupby("compound"):
        slopes, weights = [], []
        for _, st in g.groupby(["session", "driver", "stint"]):
            if len(st) < min_run or st["age"].nunique() < 3:
                continue
            x = st["age"].to_numpy(float)
            y = st["lap_time_s"].to_numpy(float)
            b = np.polyfit(x, y, 1)[0]
            slopes.append(b + fuel)  # fuel burn masks ~fuel_coef of true deg
            weights.append(len(st))
        if slopes:
            wt.deg_slopes[comp] = float(np.clip(np.average(slopes, weights=weights), 0.0, 0.5))
            wt.n_laps[comp] = int(sum(weights))
            wt.n_runs += len(slopes)

    # Practice long runs systematically over-state absolute deg (race-start fuel, green
    # track), so only the RELATIVE compound comparison is trustworthy: rescale weekend
    # slopes so their mean matches the historical circuit severity, preserving the
    # weekend's compound ordering.
    if wt.deg_slopes:
        hist_mean = float(np.mean([lap_model.deg_slope(c, circuit) for c in wt.deg_slopes]))
        wk_mean = float(np.mean(list(wt.deg_slopes.values())))
        if wk_mean > 1e-6 and hist_mean > 1e-6:
            scale = float(np.clip(hist_mean / wk_mean, 0.1, 3.0))
            wt.deg_slopes = {c: v * scale for c, v in wt.deg_slopes.items()}

    # --- usage shares ---
    total = sum(wt.n_laps.values())
    if total:
        wt.usage = {c: wt.n_laps.get(c, 0) / total for c in COMPOUNDS}

    # --- offsets: same-session, same-driver cross-compound level deltas vs MEDIUM ---
    lvl = []
    for (ses_name, drv, _), st in runs.groupby(["session", "driver", "stint"]):
        comp = st["compound"].iloc[0]
        slope = wt.deg_slopes.get(comp, lap_model.deg_slope(comp, circuit))
        corrected = st["lap_time_s"] - slope * st["age"]
        lvl.append((ses_name, drv, comp, float(corrected.median())))
    lv = pd.DataFrame(lvl, columns=["session", "driver", "compound", "level"])
    piv = lv.groupby(["session", "driver", "compound"])["level"].median().unstack("compound")
    if "MEDIUM" in piv.columns:
        for c in COMPOUNDS:
            if c == "MEDIUM" or c not in piv.columns:
                continue
            dev = (piv[c] - piv["MEDIUM"]).dropna()
            if len(dev) >= 3:
                wt.offsets[c] = float(np.clip(dev.mean(), -1.5, 1.5))
        wt.offsets["MEDIUM"] = 0.0
    return wt


class WeekendLapModel:
    """LapModel adapter: weekend-blended deg/offsets for one circuit, else delegate."""

    def __init__(self, base: LapModel, weekend: WeekendTyres, k_laps: float = 60.0):
        self._base = base
        self._wk = weekend
        self._k = float(k_laps)

    def __getattr__(self, name):  # noqa: D105 — delegate everything else
        return getattr(self._base, name)

    def _blend_slope(self, compound: str, circuit: str, driver=None) -> float:
        hist = self._base.deg_slope(compound, circuit, driver)
        wk = self._wk.deg_slopes.get(compound)
        if circuit != self._wk.circuit or wk is None:
            return hist
        n = self._wk.n_laps.get(compound, 0)
        return _shrink(wk, hist, n, self._k)

    def deg_slope(self, compound, circuit=None, driver=None) -> float:
        return self._blend_slope(compound, circuit, driver)

    def deg(self, compound, age, circuit=None, driver=None) -> float:
        return (self._blend_slope(compound, circuit, driver) * float(age)
                + self._base._cliff(compound, age, circuit))

    def pace_offset(self, compound, circuit=None) -> float:
        hist = self._base.pace_offset(compound, circuit)
        wk = self._wk.offsets.get(compound)
        if circuit != self._wk.circuit or wk is None:
            return hist
        n = self._wk.n_laps.get(compound, 0)
        return _shrink(wk, hist, n, self._k)

    def deg_severity(self, circuit=None) -> float:
        return float(np.mean([self.deg(c, 20, circuit) / 20.0 for c in COMPOUNDS]))
