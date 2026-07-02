"""JSON reporting for the pre-race strategy page.

Emits a structured, machine-readable feed: run metadata (incl. the data manifest
summary) plus, per driver, the ranked strategy candidates with their expected finish,
finishing-position distribution, key probabilities, confidence interval and probability
of being the optimal choice. JSON only, by design.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from strategy_sim2.context.postquali import WeekendContext
from strategy_sim2.selection.selector import SelectedStrategy
from strategy_sim2.settings import load_settings, resolve_path


def _round(x, n=3):
    return None if x is None or (isinstance(x, float) and x != x) else round(float(x), n)


def _strategy_dict(s: SelectedStrategy, n_positions: int, cfg: dict) -> dict:
    o = s.outcome
    lo, hi = o.finish_ci(cfg["selection"]["ci_lo"], cfg["selection"]["ci_hi"])
    dist = {str(p): _round(v, 4) for p, v in o.distribution(n_positions).items() if v > 0.0}
    return {
        "rank": s.rank,
        "n_stops": s.candidate.n_stops,
        "compounds": list(s.candidate.compounds),
        "start_compound": s.candidate.start_compound,
        "planned_pit_laps": list(s.candidate.pit_laps),
        "stint_lengths": list(s.candidate.stint_lengths),
        "expected_finish": _round(o.mean_finish_classified, 2),  # given the driver finishes
        "expected_finish_all": _round(o.mean_finish, 2),         # incl. DNF sims
        "median_finish": _round(o.median_finish, 1),
        "finish_ci": [_round(lo, 1), _round(hi, 1)],
        "p_win": _round(o.p_win),
        "p_podium": _round(o.p_podium),
        "p_points": _round(o.p_points),
        "p_dnf": _round(o.p_dnf),
        "p_optimal": _round(s.p_optimal),
        "mean_race_time_s": _round(o.mean_race_time, 2),
        "plausibility_prior": _round(s.candidate.prior, 5),
        "finish_distribution": dist,
    }


def build_report(wctx: WeekendContext, per_driver: dict[str, list[SelectedStrategy]],
                 n_sims: int, seed: int, cfg: dict | None = None) -> dict:
    cfg = cfg or load_settings()
    p = wctx.profile
    n_pos = len(wctx.drivers())
    drivers = {}
    for d in wctx.drivers():
        drivers[d] = {
            "grid": wctx.grid[d],
            "team": wctx.teams.get(d, ""),
            "base_pace_s": _round(wctx.base_pace.get(d), 3),
            "candidates": [_strategy_dict(s, n_pos, cfg) for s in per_driver.get(d, [])],
        }
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": wctx.mode, "year": wctx.year, "round": wctx.round,
            "circuit": wctx.circuit, "n_drivers": n_pos,
            "n_sims": n_sims, "seed": seed,
            "allocation": list(wctx.allocation),
            "training_window": {k: cfg["training"][k] for k in ("start_year", "end_year")},
        },
        "circuit_profile": {
            "n_laps": p.n_laps, "pit_loss_s": _round(p.pit_loss, 2),
            "base_lap_time_s": _round(p.base_lap_time, 2),
            "deg_severity": _round(p.deg_severity, 4),
            "sc_prob": _round(p.sc_prob, 2), "vsc_prob": _round(p.vsc_prob, 2),
            "overtaking_difficulty": _round(p.overtaking_difficulty, 2),
            "n_races_in_history": p.n_races, "fallback": p.fallback,
        },
        "drivers": drivers,
    }


def write_report(report: dict, year: int, rnd: int, mode: str,
                 cfg: dict | None = None) -> str:
    cfg = cfg or load_settings()
    out_dir = resolve_path("strategy_sim2/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{year}_{rnd:02d}_{mode}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return str(path)
