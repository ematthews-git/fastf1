"""Unit tests for the ghost-field traffic simulator (synthetic, no I/O)."""

import numpy as np
import pytest

from f1strategy.config import (GlobalParams, SimConfig, TrackContext, CompoundModel,
                               FieldModel, RivalCar)
from f1strategy import simulator, racesim


def make_ctx(ease=0.5, n_laps=40):
    compounds = {
        "MEDIUM": CompoundModel(slot="MEDIUM", deg=0.05, base_offset=0.0, source="race"),
        "HARD": CompoundModel(slot="HARD", deg=0.03, base_offset=0.5, source="race"),
    }
    return TrackContext(track="T", year=2025, event_name="T", n_laps=n_laps, pit_loss=20.0,
                        fuel_rate=-0.05, compounds=compounds, overtake_ease=ease)


def _run(ctx, seq, pit, params, rivals, focal_grid=2, focal_pace=0.0):
    tables = simulator.build_tables(ctx, params)
    perlap = racesim.perlap_costs(tables)
    field = FieldModel(focal_grid=focal_grid, focal_pace_offset=focal_pace, rivals=rivals)
    ctx.field = field
    ghosts = racesim.precompute_ghosts(field, ctx, perlap)
    return racesim.simulate_focal(seq, pit, ctx, params, perlap, ghosts, focal_grid, focal_pace)


def test_stuck_behind_slower_car_costs_time_on_hard_track():
    p = GlobalParams(dirty_air_loss=0.5, overtake_scale=1.0, dirty_air_gap=1.5)
    seq, pit = ("MEDIUM", "HARD"), (20,)
    # a slightly slower ghost that starts just ahead -> focal catches it
    ghost = [RivalCar("G", grid=1, pace_offset=0.25, seq=("MEDIUM", "HARD"), pit_laps=(20,))]
    hard = _run(make_ctx(ease=0.05), seq, pit, p, ghost)   # Monaco-like
    easy = _run(make_ctx(ease=0.95), seq, pit, p, ghost)   # Monza-like
    assert hard.exp_time > easy.exp_time + 1.0, "traffic must cost more where passing is hard"


def test_no_ghosts_is_pure_pace_plus_grid():
    p = GlobalParams()
    ctx = make_ctx()
    out = _run(ctx, ("MEDIUM", "HARD"), (20,), p, rivals=[], focal_grid=3)
    tables = simulator.build_tables(ctx, p)
    perlap = racesim.perlap_costs(tables)
    traj = racesim.car_trajectory(3, 0.0, ("MEDIUM", "HARD"), (20,), ctx, perlap)
    assert out.exp_time == pytest.approx(traj[-1])   # no field -> just the trajectory
    assert out.exp_position == 1.0


def test_more_dirty_air_loss_costs_more():
    seq, pit = ("MEDIUM", "HARD"), (20,)
    ghost = [RivalCar("G", grid=1, pace_offset=0.25, seq=("MEDIUM", "HARD"), pit_laps=(20,))]
    lo = _run(make_ctx(ease=0.1), seq, pit, GlobalParams(dirty_air_loss=0.1), ghost)
    hi = _run(make_ctx(ease=0.1), seq, pit, GlobalParams(dirty_air_loss=0.9), ghost)
    assert hi.exp_time > lo.exp_time


def test_faster_focal_finishes_ahead_of_slower_field():
    # focal much faster than two slower ghosts -> should finish P1 despite starting P3
    p = GlobalParams()
    ghosts = [RivalCar("A", grid=1, pace_offset=0.6, seq=("MEDIUM", "HARD"), pit_laps=(20,)),
              RivalCar("B", grid=2, pace_offset=0.8, seq=("MEDIUM", "HARD"), pit_laps=(20,))]
    out = _run(make_ctx(ease=0.8), ("MEDIUM", "HARD"), (20,), p, ghosts, focal_grid=3, focal_pace=0.0)
    assert out.exp_position == 1.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
