"""Driver retirement (DNF) model: hierarchical Beta-Bernoulli.

Per the paper, a driver's DNF probability is estimated with a conjugate Beta prior
whose parameters come from the whole field (empirical Bayes), so drivers who never
retired still get a small non-zero probability and sparse records are regularised.
Recent seasons are weighted more heavily (a new reg set changes reliability).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy_sim2.settings import load_settings


@dataclass
class DNFModel:
    prob_by_driver: dict[str, float] = field(default_factory=dict)
    global_prob: float = 0.12
    prior_alpha: float = 2.0
    prior_beta: float = 14.0
    p_first_lap: float = 0.25  # share of retirements that happen on lap 1

    def dnf_prob(self, driver: str | None = None) -> float:
        return float(self.prob_by_driver.get(driver, self.global_prob))

    def sample_retire_lap(self, total_laps: int, rng: np.random.Generator) -> int:
        """Lap on which a retiring driver stops (lap 1 over-weighted)."""
        if total_laps <= 1 or rng.random() < self.p_first_lap:
            return 1
        return int(rng.integers(2, total_laps + 1))


def _method_of_moments(fractions: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(fractions))
    var = float(np.var(fractions))
    mu = min(max(mu, 1e-3), 1 - 1e-3)
    if var <= 1e-9:
        var = mu * (1 - mu) / 8.0  # fall back to a moderately informative prior
    common = mu * (1 - mu) / var - 1.0
    common = max(common, 1.0)
    return mu * common, (1 - mu) * common


def fit_dnf(results: pd.DataFrame, cfg: dict | None = None) -> DNFModel:
    """Fit from a per-(driver, race) results frame with columns year, driver, dnf,
    dns, laps_completed."""
    cfg = cfg or load_settings()
    end_year = cfg["training"]["end_year"]
    decay = float(cfg["training"]["recency_decay"])
    min_races = int(cfg.get("params", {}).get("dnf_min_races", 4))

    df = results[~results["dns"].fillna(False)].copy()
    df = df.dropna(subset=["driver"])
    df["w"] = decay ** (end_year - df["year"].astype(int))

    # Empirical-Bayes prior from raw (unweighted) per-driver DNF fractions.
    per = df.groupby("driver").agg(z=("dnf", "sum"), n=("dnf", "size"))
    fr = per[per["n"] >= min_races]
    fractions = (fr["z"] / fr["n"]).to_numpy() if len(fr) >= 4 else (per["z"] / per["n"]).to_numpy()
    alpha, beta = _method_of_moments(fractions)

    # Recency-weighted posterior mean per driver.
    wagg = df.groupby("driver").apply(
        lambda g: pd.Series({"zw": float((g["w"] * g["dnf"]).sum()),
                             "nw": float(g["w"].sum())}),
        include_groups=False,
    )
    prob_by_driver = {
        str(d): float((row["zw"] + alpha) / (row["nw"] + alpha + beta))
        for d, row in wagg.iterrows()
    }
    global_prob = float(alpha / (alpha + beta))

    # Share of retirements occurring on lap 1 (mixing accidents at the start).
    dnfs = df[df["dnf"]]
    if len(dnfs) and dnfs["laps_completed"].notna().any():
        p_first = float((dnfs["laps_completed"].fillna(99) <= 1).mean())
    else:
        p_first = 0.25

    return DNFModel(prob_by_driver=prob_by_driver, global_prob=global_prob,
                    prior_alpha=alpha, prior_beta=beta, p_first_lap=p_first)
