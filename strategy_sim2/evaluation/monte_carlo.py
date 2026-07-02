"""Monte-Carlo evaluation of a driver's candidate strategies.

For a target driver we run many stochastic races. Competitors sample their own
strategies from the shared candidate pool (weighted by plausibility); the target's
strategy is the one under test. Common random numbers are used across the target's
candidates within a given sim index (same competitor strategies and same race-luck
seed) so differences reflect strategy, not noise. Evaluation returns raw per-candidate
outcome matrices; ranking/selection happens elsewhere.
"""
from __future__ import annotations

import numpy as np

from strategy_sim2.context.postquali import WeekendContext
from strategy_sim2.generation.generator import Candidate
from strategy_sim2.sim.race import Entry, RaceContext, simulate_race

_MASK = (1 << 63) - 1


def _entries_for(wctx: WeekendContext, strat_map: dict[str, Candidate]) -> list[Entry]:
    return [Entry(d, wctx.teams.get(d, ""), wctx.grid[d], wctx.base_pace[d],
                  strat_map[d].compounds, strat_map[d].pit_laps)
            for d in wctx.drivers()]


def evaluate_driver(wctx: WeekendContext, target: str, pool: list[Candidate],
                    n_sims: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (finish[K, S], race_time[K, S]) for the target's K candidates."""
    cfg = wctx.cfg
    competitors = [d for d in wctx.drivers() if d != target]
    weights = np.array([c.prior for c in pool], dtype=float)
    weights = weights / weights.sum()
    K, S = len(pool), n_sims
    finish = np.full((K, S), np.nan)
    rtime = np.full((K, S), np.nan)

    for s in range(S):
        strat_rng = np.random.default_rng((seed * 1_000_003 + s) & _MASK)
        idx = strat_rng.choice(K, size=len(competitors), p=weights)
        comp_strat = {d: pool[i] for d, i in zip(competitors, idx)}
        for k, cand in enumerate(pool):
            sim_rng = np.random.default_rng((seed * 7919 + s) & _MASK)  # CRN across candidates
            strat_map = dict(comp_strat)
            strat_map[target] = cand
            ctx = RaceContext(wctx.profile, wctx.params.lap, wctx.params.dnf,
                              wctx.params.start, _entries_for(wctx, strat_map), cfg)
            res = simulate_race(ctx, sim_rng)
            finish[k, s] = res.finish_position[target]
            rtime[k, s] = res.race_time[target]
    return finish, rtime
