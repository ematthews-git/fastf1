"""Fast unit tests for the simulator/optimizer core (synthetic contexts, no I/O)."""

import math

import pytest

from f1strategy.config import (GlobalParams, SimConfig, TrackContext, CompoundModel,
                               SCModel, COMPOUNDS)
from f1strategy import simulator, optimizer


def make_ctx(degs, offsets, n_laps=50, pit_loss=22.0, sources=None, sc=None):
    sources = sources or {c: "race" for c in degs}
    compounds = {c: CompoundModel(slot=c, deg=degs[c], base_offset=offsets[c],
                                  n=500, source=sources[c]) for c in degs}
    return TrackContext(track="Test", year=2025, event_name="Test GP", n_laps=n_laps,
                        pit_loss=pit_loss, fuel_rate=-0.05, compounds=compounds, sc=sc)


# medium-deg 2-compound context typical of a real circuit
DEGS = {"SOFT": 0.14, "MEDIUM": 0.08, "HARD": 0.05}
OFFS = {"SOFT": 0.0, "MEDIUM": 0.6, "HARD": 1.1}


# ------------------------------------------------------------------- config


def test_globalparams_roundtrip():
    p = GlobalParams(deg_scale=1.3, pit_stop_penalty=4.5, stint_risk=0.2)
    assert GlobalParams.from_dict(p.to_dict()) == p
    assert "SEARCH_SPACE" not in p.to_dict()


# ---------------------------------------------------------------- enumeration


def test_two_compound_rule():
    cfg = SimConfig(max_stops=3)
    seqs = optimizer.candidate_sequences(list(COMPOUNDS), cfg)
    assert seqs, "should enumerate strategies"
    assert all(len(set(s)) >= 2 for s in seqs), "every strategy uses >=2 compounds"
    assert all(1 <= len(s) - 1 <= 3 for s in seqs)


def test_optimizer_returns_only_legal_and_feasible():
    ctx = make_ctx(DEGS, OFFS)
    cfg = SimConfig(use_sc=False, min_stint=8, max_stops=3)
    res = optimizer.optimize(ctx, GlobalParams(), cfg)
    for r in res.ranked:
        assert len(set(r.compounds)) >= 2
        bounds = [0, *r.pit_laps, ctx.n_laps]
        lengths = [b - a for a, b in zip(bounds, bounds[1:])]
        assert all(L >= cfg.min_stint for L in lengths), "min-stint respected"
        assert sum(lengths) == ctx.n_laps
        assert math.isfinite(r.race_time)


# ------------------------------------------------------------------ simulator


def test_race_time_increases_with_degradation():
    ctx = make_ctx(DEGS, OFFS)
    cfg = SimConfig(use_sc=False)
    seq, pits = ("SOFT", "MEDIUM"), (25,)
    t_lo = simulator.race_time(seq, pits, ctx, GlobalParams(deg_scale=1.0), cfg)
    t_hi = simulator.race_time(seq, pits, ctx, GlobalParams(deg_scale=1.6), cfg)
    assert t_hi > t_lo


def test_infeasible_when_stint_too_short():
    ctx = make_ctx(DEGS, OFFS, n_laps=50)
    cfg = SimConfig(use_sc=False, min_stint=10)
    # a stop on lap 3 leaves a 3-lap opening stint < min_stint
    assert simulator.race_time(("SOFT", "MEDIUM"), (3,), ctx, GlobalParams(), cfg) == math.inf


def test_more_degradation_means_more_or_equal_stops():
    ctx = make_ctx(DEGS, OFFS, n_laps=60)
    cfg = SimConfig(use_sc=False, max_stops=3, min_stint=8)
    lo = optimizer.optimize(ctx, GlobalParams(deg_scale=0.8), cfg).optimal.n_stops
    hi = optimizer.optimize(ctx, GlobalParams(deg_scale=2.0), cfg).optimal.n_stops
    assert hi >= lo


def test_higher_pit_penalty_means_fewer_or_equal_stops():
    ctx = make_ctx(DEGS, OFFS, n_laps=60)
    cfg = SimConfig(use_sc=False, max_stops=3, min_stint=8)
    few = optimizer.optimize(ctx, GlobalParams(pit_stop_penalty=25.0), cfg).optimal.n_stops
    many = optimizer.optimize(ctx, GlobalParams(pit_stop_penalty=0.0), cfg).optimal.n_stops
    assert few <= many


# --------------------------------------------------- source-aware deg scaling


def test_deg_scale_only_touches_race_sourced():
    # a practice-sourced compound must be unaffected by deg_scale
    ctx = make_ctx({"SOFT": 0.14, "MEDIUM": 0.08}, {"SOFT": 0.0, "MEDIUM": 0.6},
                   sources={"SOFT": "practice", "MEDIUM": "race"})
    t1 = simulator.build_tables(ctx, GlobalParams(deg_scale=1.0))["SOFT"]
    t2 = simulator.build_tables(ctx, GlobalParams(deg_scale=1.8))["SOFT"]
    assert (t1 == t2).all(), "practice-sourced compound unchanged by deg_scale"
    m1 = simulator.build_tables(ctx, GlobalParams(deg_scale=1.0))["MEDIUM"]
    m2 = simulator.build_tables(ctx, GlobalParams(deg_scale=1.8))["MEDIUM"]
    assert (m2 > m1)[1:].all(), "race-sourced compound scaled up"


# ------------------------------------------------------------------ safety car


def test_sc_hedging_pushes_stops_later():
    sc = SCModel(p_race=0.7, exp_count=0.9, pit_loss_under_sc=10.0)
    ctx = make_ctx(DEGS, OFFS, n_laps=55, sc=sc)
    cfg = SimConfig(use_sc=True, max_stops=1, min_stint=8)   # force a 1-stop
    early = optimizer.optimize(ctx, GlobalParams(sc_influence=0.0), cfg).optimal.pit_laps[0]
    late = optimizer.optimize(ctx, GlobalParams(sc_influence=1.0), cfg).optimal.pit_laps[0]
    assert late >= early
    assert late > early or sc.p_race == 0    # strictly later given real SC risk


def test_sc_off_is_deterministic_zero_adjustment():
    sc = SCModel(p_race=0.7, exp_count=0.9, pit_loss_under_sc=10.0)
    ctx = make_ctx(DEGS, OFFS, sc=sc)
    cfg = SimConfig(use_sc=False)
    assert simulator.sc_adjustment(("SOFT", "MEDIUM"), (25,), ctx, GlobalParams(sc_influence=1.0), cfg) == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))  # noqa: F821
