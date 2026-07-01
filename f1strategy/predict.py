"""Online entrypoint — predict the strategy for one (upcoming or historical) race.

Two-phase use case:
  * base prediction  — ``predict_strategy(track, year)`` before the weekend, tyre
    inputs from prior seasons only (no look-ahead).
  * updated prediction — ``predict_strategy(track, year, use_practice=True)`` after
    free practice, folding the weekend's own FP long-runs into the tyre inputs.

The same calibrated global parameters are used in both phases; only the measured
per-track inputs refresh. ``use_sc`` selects the SC-on / SC-off calibrated profile
and toggles the safety-car hedging term.
"""

from __future__ import annotations

from typing import Optional

from .config import (GlobalParams, SimConfig, TrackContext, StrategyPrediction)
from .data.loaders import load_race, event_name
from .data.laps import clean_laps, race_lap_count
from .tyre import build_tyre_model
from .pitloss import pit_loss
from .dataset import expected_lap_count
from . import optimizer, safetycar, calibrate


def build_context(track: str, year: int, use_practice: bool = False, use_sc: bool = True,
                  seasons_back: int = 3, n_laps: Optional[int] = None) -> TrackContext:
    """Assemble the measured per-race inputs for a prediction (no look-ahead).

    Prior-years practice long-runs are *always* folded in (available pre-weekend and
    the regime the global parameters were calibrated on). ``use_practice`` controls
    only whether the **target weekend's own FP** is added too — i.e. the post-practice
    update vs the base prediction.
    """
    tm = build_tyre_model(track, year, seasons_back=seasons_back,
                          use_practice=True, target_practice=use_practice)
    if len(tm["compounds"]) < 2:
        raise ValueError(f"Not enough tyre history to model {track} {year} "
                         f"(need >=2 well-sampled compounds; got {list(tm['compounds'])}).")
    if n_laps is None:
        n_laps = expected_lap_count(track, year, load_race)
    if not n_laps:
        raise ValueError(f"Could not determine race distance for {track} {year}.")
    pl = pit_loss(track, year)
    sc = safetycar.sc_model(track, pl) if use_sc else None
    return TrackContext(
        track=track, year=year, event_name=event_name(year, track),
        n_laps=n_laps, pit_loss=pl, fuel_rate=tm["fuel_rate"],
        compounds=tm["compounds"], sc=sc, seasons_used=tuple(tm["seasons_used"]),
        notes=f"tyre sources: {tm['sources']}",
    )


def predict_strategy(track: str, year: int, use_practice: bool = False,
                     use_sc: bool = True, params: Optional[GlobalParams] = None,
                     config: Optional[SimConfig] = None,
                     seasons_back: int = 3) -> StrategyPrediction:
    """Predict the optimal strategy + distribution for a race.

    ``params`` defaults to the calibrated profile matching ``use_sc``; pass an
    explicit :class:`GlobalParams` to override (e.g. for experiments).
    """
    if config is None:
        config = SimConfig(use_sc=use_sc, use_practice=use_practice)
    else:
        config.use_sc, config.use_practice = use_sc, use_practice
    if params is None:
        params = calibrate.load_params(sc_on=use_sc)

    ctx = build_context(track, year, use_practice=use_practice, use_sc=use_sc,
                        seasons_back=seasons_back)
    res = optimizer.optimize(ctx, params, config)

    return StrategyPrediction(
        track=track, year=year, event_name=ctx.event_name,
        optimal=res.optimal, ranked=res.ranked,
        p_by_stops=res.p_by_stops, pit_windows=res.pit_windows,
        context=ctx, used_practice=use_practice, used_sc=use_sc,
    )
