"""Opening-lap position changes (the start "mixing" of the field).

Following the paper, each driver has an empirical distribution of positions
gained/lost on lap 1, smoothed with a Gaussian kernel so we can draw continuous
outcomes. Drivers with few observations fall back to the pooled field distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy_sim2.settings import load_settings


@dataclass
class StartModel:
    deltas_by_driver: dict[str, np.ndarray] = field(default_factory=dict)
    pooled: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    sigma: float = 1.0
    min_samples: int = 5

    def sample_gain(self, driver: str | None, rng: np.random.Generator) -> float:
        """Positions gained on lap 1 (positive = moved forward)."""
        arr = self.deltas_by_driver.get(driver)
        if arr is None or len(arr) < self.min_samples:
            arr = self.pooled
        base = float(rng.choice(arr))
        return base + rng.normal(0.0, self.sigma)


def fit_start(results: pd.DataFrame, lap1: pd.DataFrame,
              cfg: dict | None = None) -> StartModel:
    """Fit from race results (grid) joined to lap-1 positions.

    ``results``: per-(year, round, driver) with columns year, round, driver, grid.
    ``lap1``:    per-(year, round, driver) lap-1 positions, columns year, round,
                 driver, position.
    """
    cfg = cfg or load_settings()
    sigma = float(cfg.get("params", {}).get("start_sigma", 1.0))
    min_samples = int(cfg.get("params", {}).get("start_min_samples", 5))

    merged = results.merge(lap1, on=["year", "round", "driver"], how="inner")
    merged = merged.dropna(subset=["grid", "position"])
    merged = merged[merged["grid"] > 0]  # grid 0 = pit-lane start, exclude
    merged["gain"] = merged["grid"].astype(float) - merged["position"].astype(float)

    pooled = merged["gain"].to_numpy(float)
    deltas = {str(d): g["gain"].to_numpy(float) for d, g in merged.groupby("driver")}
    return StartModel(deltas_by_driver=deltas,
                      pooled=pooled if len(pooled) else np.array([0.0]),
                      sigma=sigma, min_samples=min_samples)
