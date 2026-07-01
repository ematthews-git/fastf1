"""Forward model — modelled race time for a given strategy.

We only ever *compare* strategies within one race, so the returned time is
relative: the fuel-burn contribution (a function of absolute lap number, identical
for every strategy over the same race distance) is omitted because it cancels.
What differs between strategies is how laps are split across compounds/tyre ages
and the number of stops — which is exactly what this scores.

Speed: per compound we precompute a prefix-cost table (cost of a fresh stint of
length L), so evaluating any strategy is O(number of stints).
"""

from __future__ import annotations

import numpy as np

from .config import TrackContext, GlobalParams, SimConfig, StrategyResult

INF = float("inf")


def build_tables(ctx: TrackContext, params: GlobalParams) -> dict[str, np.ndarray]:
    """Prefix cost per compound: ``table[c][L]`` = cost of a fresh L-lap stint.

    cost(age) = base_offset[c] + deg[c]*deg_scale*age + stint_risk*max(0, age - knee)
    summed over ages 1..L, with a uniform knee = risk_free_life. The linear risk term
    biases pit laps earlier (pit before the cliff) without the token-min-stint
    pathology a per-compound quadratic cliff induced (see README).
    """
    n = ctx.n_laps
    ages = np.arange(1, n + 1, dtype=float)
    knee = params.risk_free_life
    risk = params.stint_risk * np.maximum(0.0, ages - knee)
    tables: dict[str, np.ndarray] = {}
    for c, cm in ctx.compounds.items():
        # deg_scale corrects the *race-measurement* under-read only. Practice-sourced
        # degradation is already unbiased, so it is not re-scaled — this is what stops
        # the post-practice update from double-correcting what practice already fixed.
        scale = params.deg_scale if cm.source == "race" else 1.0
        per_lap = cm.base_offset + cm.deg * scale * ages + risk
        prefix = np.concatenate(([0.0], np.cumsum(per_lap)))   # prefix[L] = sum of first L
        tables[c] = prefix
    return tables


def stint_bounds(pit_laps: tuple[int, ...], n_laps: int) -> list[int]:
    return [0, *pit_laps, n_laps]


def race_time(seq: tuple[str, ...], pit_laps: tuple[int, ...],
              ctx: TrackContext, params: GlobalParams, config: SimConfig,
              tables: dict[str, np.ndarray] | None = None) -> float:
    """Modelled (relative) race time for one strategy; INF if infeasible."""
    if len(seq) != len(pit_laps) + 1:
        return INF
    if tables is None:
        tables = build_tables(ctx, params)
    bounds = stint_bounds(pit_laps, ctx.n_laps)
    total = 0.0
    for i, c in enumerate(seq):
        length = bounds[i + 1] - bounds[i]
        if length < config.min_stint or length <= 0:
            return INF
        if c not in tables:
            return INF
        total += float(tables[c][length])
    n_stops = len(pit_laps)
    total += n_stops * (ctx.pit_loss + params.pit_stop_penalty)
    total += sc_adjustment(seq, pit_laps, ctx, params, config)
    return total


def sc_adjustment(seq, pit_laps, ctx, params, config) -> float:
    """Safety-car expected-value adjustment. Implemented in M4 (safetycar.py).

    Returns 0 when SC hedging is off or no SC model is attached, so the
    deterministic simulator is exactly the ``use_sc=False`` case.
    """
    if not config.use_sc or ctx.sc is None:
        return 0.0
    from . import safetycar  # lazy import; only needed when hedging is on
    return safetycar.sc_adjustment(seq, pit_laps, ctx, params, config)


def make_result(seq, pit_laps, ctx, params, config,
                tables=None) -> StrategyResult:
    t = race_time(seq, pit_laps, ctx, params, config, tables)
    return StrategyResult(compounds=tuple(seq), pit_laps=tuple(pit_laps),
                          race_time=t, n_stops=len(pit_laps))
