"""Strategy search — the argmin over strategies, plus a probability distribution.

For every legal compound sequence (1..max_stops stops, respecting the two-compound
rule) we place the pit laps optimally with a DP over the prefix-cost tables, then
rank sequences by modelled race time. A softmax over race-time gaps turns the
ranking into P(stop count) and pit-lap windows (enriched with Monte-Carlo in M4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np

from .config import TrackContext, GlobalParams, SimConfig, StrategyResult
from . import simulator

INF = float("inf")


@dataclass
class OptimizeResult:
    ranked: list[StrategyResult]
    optimal: StrategyResult
    p_by_stops: dict[int, float]
    pit_windows: dict[int, tuple[int, int]]
    exp_position: float = 0.0
    position_dist: dict = field(default_factory=dict)
    tables: dict = field(default=None, repr=False)


def candidate_sequences(available: list[str], config: SimConfig) -> list[tuple[str, ...]]:
    """All legal compound sequences for 1..max_stops stops.

    Adjacent repeats are allowed (teams do run the same compound twice); the
    two-compound rule requires at least two *distinct* dry compounds overall.
    """
    seqs: list[tuple[str, ...]] = []
    for nstop in range(1, config.max_stops + 1):
        for seq in product(available, repeat=nstop + 1):
            if config.require_two_compounds and len(set(seq)) < 2:
                continue
            seqs.append(seq)
    return seqs


def best_pit_laps(seq: tuple[str, ...], ctx: TrackContext, config: SimConfig,
                  tables: dict) -> Optional[tuple[tuple[int, ...], float]]:
    """Pace-optimal pit laps for a fixed compound sequence via DP.

    Minimises the decomposable stint (pace) cost; pit-count cost is constant and
    SC (if any) is layered on by the caller/refiner. Returns (pit_laps, pace_cost)
    or None if no feasible partition exists.
    """
    n = ctx.n_laps
    ms = config.min_stint
    S = len(seq)                       # number of stints
    if S == 1:
        length = n
        if length < ms or seq[0] not in tables:
            return None
        return (), float(tables[seq[0]][length])

    # dp[i][lap] = (min cost of first i stints ending at `lap`, prev lap)
    dp: list[dict[int, tuple[float, int]]] = [dict() for _ in range(S + 1)]
    dp[0][0] = (0.0, -1)
    for i in range(S):
        c = seq[i]
        if c not in tables:
            return None
        prefix = tables[c]
        remaining = S - (i + 1)
        for a, (ca, _) in dp[i].items():
            if i == S - 1:
                b_range = (n,)
            else:
                b_range = range(a + ms, n - remaining * ms + 1)
            for b in b_range:
                length = b - a
                if length < ms:
                    continue
                cost = ca + float(prefix[length])
                cur = dp[i + 1].get(b)
                if cur is None or cost < cur[0]:
                    dp[i + 1][b] = (cost, a)
    if n not in dp[S]:
        return None
    # backtrack the cut points
    bounds = [n]
    lap = n
    for i in range(S, 0, -1):
        _, prev = dp[i][lap]
        bounds.append(prev)
        lap = prev
    bounds.reverse()                   # [0, p1, ..., n]
    return tuple(bounds[1:-1]), dp[S][n][0]


def _refine_pitlaps(seq, pit_laps, ctx, params, config, tables, window=6):
    """Coordinate-descent refine of pit laps against the FULL objective.

    Needed once the SC EV term (which depends on *when* you stop, not just stint
    lengths) is active; a no-op when SC is off. Cheap: small window, few passes.
    """
    if not pit_laps:
        return pit_laps, simulator.race_time(seq, pit_laps, ctx, params, config, tables)
    best = list(pit_laps)
    best_t = simulator.race_time(seq, tuple(best), ctx, params, config, tables)
    for _ in range(3):
        improved = False
        for k in range(len(best)):
            lo = (best[k - 1] if k > 0 else 0) + config.min_stint
            hi = (best[k + 1] if k + 1 < len(best) else ctx.n_laps) - config.min_stint
            for cand in range(max(lo, best[k] - window), min(hi, best[k] + window) + 1):
                trial = list(best)
                trial[k] = cand
                t = simulator.race_time(seq, tuple(trial), ctx, params, config, tables)
                if t < best_t - 1e-9:
                    best_t, best[k], improved = t, cand, True
        if not improved:
            break
    return tuple(best), best_t


def _softmax_probs(times: np.ndarray, temp: float) -> np.ndarray:
    finite = np.isfinite(times)
    p = np.zeros_like(times)
    if not finite.any():
        return p
    t = times[finite]
    z = -(t - t.min()) / max(temp, 1e-6)
    e = np.exp(z - z.max())
    p[finite] = e / e.sum()
    return p


def _pit_windows(results: list[StrategyResult], n_stops: int) -> dict[int, tuple[int, int]]:
    """Probability-weighted central pit-lap window per stop index (modal stops)."""
    rows = [r for r in results if r.n_stops == n_stops and np.isfinite(r.race_time)]
    windows: dict[int, tuple[int, int]] = {}
    for i in range(n_stops):
        laps = np.array([r.pit_laps[i] for r in rows if len(r.pit_laps) > i], float)
        w = np.array([r.prob for r in rows if len(r.pit_laps) > i], float)
        if not len(laps) or w.sum() <= 0:
            continue
        order = np.argsort(laps)
        laps, w = laps[order], w[order]
        cw = np.cumsum(w) / w.sum()
        lo = float(np.interp(0.15, cw, laps))
        hi = float(np.interp(0.85, cw, laps))
        windows[i] = (int(round(lo)), int(round(hi)))
    return windows


def _rescore_traffic(results: list[StrategyResult], ctx: TrackContext,
                     params: GlobalParams, config: SimConfig, tables: dict
                     ) -> list[StrategyResult]:
    """Re-score the top pace candidates by expected traffic-inclusive outcome.

    Only the top-K by clean-air time are raced through the field (traffic reorders
    the competitive set but rarely promotes an off-pace strategy) — plus the best
    candidate of *each* stop count, so the stop-count distribution stays honest.
    Returns only the re-scored subset; ``race_time`` now holds the traffic-inclusive
    time (a different scale from clean-air time, so the two are never mixed).
    """
    from . import racesim
    perlap = racesim.perlap_costs(tables)
    ghosts = racesim.precompute_ghosts(ctx.field, ctx, perlap)
    fg, fpo = ctx.field.focal_grid, ctx.field.focal_pace_offset

    results.sort(key=lambda r: r.race_time)
    seed: dict[int, StrategyResult] = {}
    for r in results:
        seed.setdefault(r.n_stops, r)                 # best clean-air per stop count
    topk = list(seed.values())
    for r in results[:config.traffic_topk]:
        if r not in topk:
            topk.append(r)

    for r in topk:
        out = racesim.simulate_focal(r.compounds, r.pit_laps, ctx, params, perlap,
                                     ghosts, fg, fpo)
        sc = (simulator.sc_adjustment(r.compounds, r.pit_laps, ctx, params, config)
              if config.use_sc and ctx.sc is not None else 0.0)
        r.race_time = out.exp_time + sc
        r.exp_position = out.exp_position
    return topk


def optimize(ctx: TrackContext, params: GlobalParams, config: SimConfig) -> OptimizeResult:
    """Rank every legal strategy and summarise the distribution.

    Clean-air time when ``use_traffic`` is off or no field is attached (exactly v1);
    otherwise the top pace candidates are re-scored by expected traffic-inclusive
    outcome (see :func:`_rescore_traffic`) and ranked by that.
    """
    available = ctx.available
    tables = simulator.build_tables(ctx, params)
    results: list[StrategyResult] = []
    for seq in candidate_sequences(available, config):
        placed = best_pit_laps(seq, ctx, config, tables)
        if placed is None:
            continue
        pit_laps, _ = placed
        if config.use_sc and ctx.sc is not None:
            pit_laps, _ = _refine_pitlaps(seq, pit_laps, ctx, params, config, tables)
        t = simulator.race_time(seq, pit_laps, ctx, params, config, tables)
        if not np.isfinite(t):
            continue
        results.append(StrategyResult(compounds=seq, pit_laps=pit_laps,
                                      race_time=t, n_stops=len(pit_laps)))
    if not results:
        raise ValueError("No feasible strategy — check compounds/among min_stint vs n_laps.")

    traffic = config.use_traffic and ctx.field is not None
    if traffic:
        results = _rescore_traffic(results, ctx, params, config, tables)

    results.sort(key=lambda r: r.race_time)
    best = results[0].race_time
    times = np.array([r.race_time for r in results])
    probs = _softmax_probs(times, config.softmax_temp)
    for r, p in zip(results, probs):
        r.delta_to_best = r.race_time - best
        r.prob = float(p)

    # P(stop count) from the BEST strategy of each stop count, not a sum over all
    # permutations — otherwise a count with more near-optimal sequences (3-stop has
    # far more than 1-stop) would win on multiplicity rather than on pace.
    best_by_stop: dict[int, float] = {}
    for r in results:
        if r.n_stops not in best_by_stop or r.race_time < best_by_stop[r.n_stops]:
            best_by_stop[r.n_stops] = r.race_time
    stop_keys = sorted(best_by_stop)
    stop_probs = _softmax_probs(np.array([best_by_stop[k] for k in stop_keys]),
                                config.softmax_temp)
    p_by_stops = {k: float(p) for k, p in zip(stop_keys, stop_probs)}

    optimal = results[0]
    windows = _pit_windows(results, optimal.n_stops)
    pos_dist: dict[int, float] = {}
    if traffic:
        for r in results:
            k = int(round(r.exp_position))
            pos_dist[k] = pos_dist.get(k, 0.0) + r.prob
        pos_dist = dict(sorted(pos_dist.items()))
    return OptimizeResult(ranked=results, optimal=optimal, p_by_stops=p_by_stops,
                          pit_windows=windows, exp_position=optimal.exp_position,
                          position_dist=pos_dist, tables=tables)
