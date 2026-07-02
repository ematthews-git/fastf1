"""Ghost-field race simulation — expected outcome of a strategy against the field.

The focal car races *through* a field of fixed "ghost" rivals (one-way: rivals run
their own strategies and are not held up by the focal car). Everything is in
relative-to-field-pace seconds; the common-mode fuel effect is dropped because it
cancels in the gaps between cars. What remains and differs between strategies:
compound/age pace, pit losses, grid start position, and — the whole point — time
lost stuck behind a car the focal cannot pass.

Traffic is a smooth **expected value**, not Bernoulli rolls: catching a slower ghost
sets a ``stuck`` weight to 1; each lap the focal loses the pace it can't use plus
``dirty_air_loss``, scaled by ``stuck``, which decays by the per-lap pass probability
(floored so no car is trapped forever). On an easy track it clears in ~1 lap; at Monaco
it stays trapped for many. Deterministic, smooth for calibration, fast (a per-lap numpy
scan over ghosts). ``pass_prob`` comes from the per-track overtaking model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TrackContext, GlobalParams, SimConfig, FieldModel, SOFTNESS
from . import simulator, overtaking

GRID_SPACING = 0.35   # track-position seconds per grid slot at the start
HOLD_GAP = 0.45       # min following gap (s) a blocked car sits behind the ghost ahead
MIN_ESCAPE = 0.08     # floor on per-lap pass chance: even at Monaco a car isn't trapped
                      # behind a single rival forever (the rival pits, DRS, a mistake)


@dataclass
class RaceOutcome:
    exp_time: float       # focal traffic-inclusive relative race time (lower better)
    exp_position: float   # focal expected finishing position (1 = win)


def perlap_costs(tables: dict) -> dict[str, np.ndarray]:
    """Per-lap (not cumulative) relative cost by compound: cost at tyre age a = arr[a-1]."""
    return {c: np.diff(prefix) for c, prefix in tables.items()}


def _resolve(compound: str, perlap: dict) -> np.ndarray:
    """Per-lap costs for a compound, falling back to the nearest available by softness."""
    if compound in perlap:
        return perlap[compound]
    avail = list(perlap)
    j = min(avail, key=lambda k: abs(SOFTNESS.get(k, 2) - SOFTNESS.get(compound, 2)))
    return perlap[j]


def car_trajectory(grid: int, pace_offset: float, seq, pit_laps, ctx: TrackContext,
                   perlap: dict) -> np.ndarray:
    """Cumulative relative race time by lap (length n_laps+1) for one car/strategy."""
    n = ctx.n_laps
    cum = np.empty(n + 1)
    cum[0] = (grid - 1) * GRID_SPACING
    if not seq:                                  # unknown strategy -> flat single stint
        seq, pit_laps = (ctx.available[:1] or ["MEDIUM"], ())
    bounds = [0, *pit_laps, n]
    lap = 0
    for i, c in enumerate(seq):
        pc = _resolve(c, perlap)
        length = bounds[i + 1] - bounds[i]
        for a in range(1, length + 1):
            lap += 1
            cum[lap] = cum[lap - 1] + pace_offset + pc[min(a - 1, len(pc) - 1)]
        if i < len(seq) - 1:
            cum[lap] += ctx.pit_loss              # pit at the end of the stint
    return cum


def precompute_ghosts(field: FieldModel, ctx: TrackContext, perlap: dict) -> np.ndarray:
    """Fixed ghost trajectories, shape (n_rivals, n_laps+1)."""
    if not field or not field.rivals:
        return np.empty((0, ctx.n_laps + 1))
    return np.stack([car_trajectory(r.grid, r.pace_offset, r.seq, r.pit_laps, ctx, perlap)
                     for r in field.rivals])


def simulate_focal(focal_seq, focal_pit_laps, ctx: TrackContext, params: GlobalParams,
                   perlap: dict, ghosts: np.ndarray, focal_grid: int,
                   focal_pace_offset: float) -> RaceOutcome:
    """Race the focal strategy through the fixed ghost field; expected time + position.

    Traffic model: when the focal catches a slower ghost within ``dirty_air_gap``, a
    ``stuck`` weight (starts 1 on catching) makes it lose the pace it can't use plus
    ``dirty_air_loss`` each lap; ``stuck`` decays by the per-lap pass probability, so on
    an easy track it clears in ~1 lap and on Monaco it stays trapped for many. When a
    ghost pits it drops behind and the focal is freed (the traffic clears / undercut).
    """
    n = ctx.n_laps
    gap = params.dirty_air_gap
    ease = ctx.overtake_ease
    ghost_lt = np.diff(ghosts, axis=1) if len(ghosts) else np.empty((0, n))  # (R, n)

    bounds = [0, *focal_pit_laps, n]
    focal_cum = (focal_grid - 1) * GRID_SPACING
    blocker, stuck = -1, 0.0
    lap = 0
    for i, c in enumerate(focal_seq):
        pc = _resolve(c, perlap)
        length = bounds[i + 1] - bounds[i]
        is_pit_stint = i < len(focal_seq) - 1
        for a in range(1, length + 1):
            lap += 1
            clean_lt = focal_pace_offset + pc[min(a - 1, len(pc) - 1)]
            if a == length and is_pit_stint:
                clean_lt += ctx.pit_loss
            prov = focal_cum + clean_lt
            j = -1
            if len(ghosts):
                gc = ghosts[:, lap]
                # ghosts ahead within dirty air that the focal is faster than (catching)
                mask = (gc < prov) & (prov - gc < gap) & (ghost_lt[:, lap - 1] > clean_lt)
                idx = np.where(mask)[0]
                if len(idx):
                    j = int(idx[np.argmax(gc[idx])])          # closest such ghost ahead
            if j >= 0:
                if j != blocker:                              # newly caught a car to pass
                    blocker, stuck = j, 1.0
                pace_deficit = max(0.0, ghost_lt[j, lap - 1] - clean_lt)
                focal_cum = prov + stuck * (pace_deficit + params.dirty_air_loss)
                pp = overtaking.pass_probability(ghost_lt[j, lap - 1] - clean_lt,
                                                 ease, params.overtake_scale)
                stuck *= (1.0 - max(pp, MIN_ESCAPE))          # escape chance this lap
            else:
                blocker, stuck = -1, 0.0
                focal_cum = prov
    position = 1.0 + float((ghosts[:, n] < focal_cum).sum()) if len(ghosts) else 1.0
    return RaceOutcome(exp_time=float(focal_cum), exp_position=position)


def expected_outcome(focal_seq, focal_pit_laps, ctx: TrackContext, params: GlobalParams,
                     config: SimConfig, tables: dict | None = None,
                     perlap: dict | None = None, ghosts: np.ndarray | None = None
                     ) -> RaceOutcome:
    """Expected traffic-inclusive outcome for one focal strategy.

    ``perlap``/``ghosts`` are precomputed once per (ctx, params) by the optimizer and
    passed in for speed. SC hedging (if on) is layered on the focal time via the
    existing per-track EV term so the SC toggle keeps working alongside traffic.
    """
    if tables is None:
        tables = simulator.build_tables(ctx, params)
    if perlap is None:
        perlap = perlap_costs(tables)
    field = ctx.field
    if ghosts is None:
        ghosts = precompute_ghosts(field, ctx, perlap)
    out = simulate_focal(focal_seq, focal_pit_laps, ctx, params, perlap, ghosts,
                         field.focal_grid, field.focal_pace_offset)
    if config.use_sc and ctx.sc is not None:
        out.exp_time += simulator.sc_adjustment(focal_seq, focal_pit_laps, ctx, params, config)
    return out
