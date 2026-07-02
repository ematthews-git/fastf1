"""Build the competitive field the focal strategy is raced against.

Two regimes:
  * ``field_from_session`` — the **real** field for calibration/backtest: every
    classified rival's grid, mean pace and *actual* strategy, from FastF1. Rivals are
    run through the same lap-time model as the focal car (in relative-to-field-pace
    units) so gaps are consistent — faithful to real data without the unit-mixing of
    replaying raw cumulative times.
  * ``field_from_grid`` / ``field_parametric`` — a **modelled** field for prediction
    from a grid order + a pace spread (``field_from_quali`` in predict.py wires the
    qualifying grid in).

Pace is expressed as ``pace_offset`` = mean green lap-time minus the field reference
(s/lap); the focal car is one more car placed by its own grid + pace.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import FieldModel, RivalCar, TrackContext, GlobalParams
from .data.laps import clean_laps
from .observations import driver_strategies


def _grid_pace(grid: int, n: int, field_spread: float) -> float:
    """Pace offset (s/lap vs field reference) implied by a grid slot; front = faster."""
    ref_slot = (n - 1) / 2.0
    return (grid - 1 - ref_slot) / max(ref_slot, 1.0) * field_spread


def field_from_session(session, ctx: TrackContext, focal_driver: str,
                       params: Optional[GlobalParams] = None) -> Optional[FieldModel]:
    """Real field of classified rivals relative to ``focal_driver`` (the focal car).

    Rivals = classified, non-DNF drivers other than the focal, each with their real
    grid and real compound strategy. Pace is taken from grid order (front = faster,
    scale ``field_spread``) rather than noisy race-median lap times, which misrank the
    front — and this keeps the real and predicted fields on the same footing. Returns
    None if the focal driver has no grid/strategy.
    """
    spread = (params or GlobalParams()).field_spread
    laps = clean_laps(session)
    ds = driver_strategies(session, laps).set_index("Driver")
    if focal_driver not in ds.index:
        return None
    n = max(len(ds), 2)

    rivals: list[RivalCar] = []
    for drv, row in ds.iterrows():
        if drv == focal_driver:
            continue
        if row["dnf"] or len(row["sequence"]) < 1:      # only clean, classified ghosts
            continue
        rivals.append(RivalCar(
            driver=drv, grid=int(row["grid"]),
            pace_offset=_grid_pace(int(row["grid"]), n, spread),
            seq=tuple(row["sequence"]), pit_laps=tuple(row["pit_laps"]),
        ))
    if not rivals:
        return None
    fg = int(ds.loc[focal_driver, "grid"])
    return FieldModel(focal_grid=fg, focal_pace_offset=_grid_pace(fg, n, spread),
                      rivals=rivals, kind="real")


def field_from_grid(grid_order: list[str], ctx: TrackContext, params: GlobalParams,
                    focal_grid: int = 2, rival_strategy=None,
                    pace_by_driver: Optional[dict] = None) -> FieldModel:
    """Modelled field from an ordered grid (front first).

    ``pace_by_driver`` (s/lap vs reference) is used when known (e.g. from practice);
    otherwise pace is spread linearly across the grid at scale ``params.field_spread``.
    Rivals without a known strategy get ``rival_strategy`` (a representative line);
    the caller (predict) supplies the clean-air optimum.
    """
    n = max(len(grid_order), 2)
    ref_slot = (n - 1) / 2.0
    rivals: list[RivalCar] = []
    for i, drv in enumerate(grid_order):
        grid = i + 1
        if pace_by_driver and drv in pace_by_driver:
            po = float(pace_by_driver[drv])
        else:
            po = (grid - 1 - ref_slot) / max(ref_slot, 1.0) * params.field_spread
        if grid == focal_grid:
            continue                                     # this slot is the focal car
        seq, pit = (rival_strategy or ((), ()))
        rivals.append(RivalCar(driver=drv, grid=grid, pace_offset=po,
                               seq=tuple(seq), pit_laps=tuple(pit)))
    focal_po = (focal_grid - 1 - ref_slot) / max(ref_slot, 1.0) * params.field_spread
    if pace_by_driver:
        # focal driver not identifiable by slot here; keep the spread estimate
        pass
    return FieldModel(focal_grid=focal_grid, focal_pace_offset=focal_po,
                      rivals=rivals, kind="quali")


def field_parametric(ctx: TrackContext, params: GlobalParams, focal_grid: int = 2,
                     n_cars: int = 20, rival_strategy=None) -> FieldModel:
    """Generic grid (anonymous drivers) with a linear pace spread — full fallback."""
    order = [f"C{i+1:02d}" for i in range(n_cars)]
    fm = field_from_grid(order, ctx, params, focal_grid=focal_grid,
                         rival_strategy=rival_strategy)
    fm.kind = "parametric"
    return fm


def field_from_quali(track: str, year: int, ctx: TrackContext, params: GlobalParams,
                     focal_grid: int = 2, rival_strategy=None) -> FieldModel:
    """Prediction field from the qualifying grid; parametric fallback if unavailable."""
    from .data import loaders
    order = []
    try:
        order = loaders.grid_order(loaders.load_quali(year, track))
    except Exception:  # noqa: BLE001  (quali not run yet / not published)
        order = []
    if len(order) < 6:
        return field_parametric(ctx, params, focal_grid=focal_grid, rival_strategy=rival_strategy)
    return field_from_grid(order, ctx, params, focal_grid=focal_grid,
                           rival_strategy=rival_strategy)
