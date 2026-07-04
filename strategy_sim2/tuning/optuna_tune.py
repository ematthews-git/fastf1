"""Out-of-sample tuning of the engine's free parameters with Optuna.

Objective = strategy-prediction quality (stop-count top-1, compound-set top-k, order
top-k) on TRAIN folds of past dry races; the winning configuration is then scored once
on the 2026 HOLD-OUT, so overfit settings are rejected by construction.

Efficiency notes:
* Training datasets are loaded once per fold cutoff and cached; each trial refits only
  the lap model (fit-time knobs: cliff_rate / knee_percentile / offset_prior_weight).
* Params are fit at each fold-year start (before=(year, 1)) rather than per race — a
  slightly pessimistic but much cheaper approximation of the expanding window.
* A driver subsample per race caps simulation cost.

Run:  venv/bin/python -m strategy_sim2.tuning.optuna_tune --trials 25 --sims 80
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from strategy_sim2.data import session_filter
from strategy_sim2.params import circuit, dataset
from strategy_sim2.params.dnf import fit_dnf
from strategy_sim2.params.estimate import ParameterSet
from strategy_sim2.params.lapmodel import fit_lap_model
from strategy_sim2.params.startline import fit_start
from strategy_sim2.settings import load_settings
from strategy_sim2.validation.backtest import strategy_backtest_race

_DATA_CACHE: dict = {}


def _fold_data(cfg, before):
    key = tuple(before)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = (
            dataset.training_laps(cfg, before=before),
            dataset.training_results(cfg, before=before),
            dataset.training_lap1(cfg, before=before),
        )
    return _DATA_CACHE[key]


def _fit_for(cfg, before) -> ParameterSet:
    laps, results, lap1 = _fold_data(cfg, before)
    return ParameterSet(lap=fit_lap_model(laps, cfg), dnf=fit_dnf(results, cfg),
                        start=fit_start(results, lap1, cfg),
                        n_laps_rows=len(laps), n_result_rows=len(results))


def _apply(cfg: dict, p: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["params"]["cliff_rate"] = p["cliff_rate"]
    c["params"]["knee_percentile"] = p["knee_percentile"]
    c["params"]["offset_prior_weight"] = p["offset_prior_weight"]
    c["generation"]["shortlist_prior_weight"] = p["shortlist_prior_weight"]
    c["generation"]["undercut_shift_laps"] = p["undercut_shift_laps"]
    c["selection"]["prior_weight"] = p["prior_weight"]
    c["selection"]["stops_weight"] = p["stops_weight"]
    c.setdefault("prior", {})
    c["prior"]["start_blend_k"] = p["start_blend_k"]
    c["prior"]["pattern_blend_k"] = p["pattern_blend_k"]
    return c


def _score_rows(rows: list[dict]) -> float:
    if not rows:
        return -1e9
    df = pd.DataFrame(rows)
    stop = pd.to_numeric(df["stop1"]).mean()
    setk = pd.to_numeric(df["set_topk"]).mean()
    ordk = pd.to_numeric(df["ord_topk"]).mean()
    return float(1.0 * stop + 1.0 * setk + 0.5 * ordk)


def _eval_fold(cfg, races: list[tuple[int, int]], n_sims: int,
               max_drivers: int | None) -> float:
    rows = []
    fits: dict[int, ParameterSet] = {}
    for year, rnd in races:
        if year not in fits:
            fits[year] = _fit_for(cfg, (year, 1))
        ps = fits[year]
        profiles = circuit.build_circuit_profiles(ps.lap, cfg, save=False, before=(year, 1))
        rr = strategy_backtest_race(year, rnd, ps, profiles, cfg, n_sims)
        if max_drivers and len(rr) > max_drivers:
            rr = rr[:: max(1, len(rr) // max_drivers)][:max_drivers]
        rows.extend(rr)
    return _score_rows(rows)


def _dry_races(cfg, year: int) -> list[tuple[int, int]]:
    races = session_filter.included_races(cfg)
    return [(year, int(r)) for r in races[races["year"] == year]["round"]]


def tune(n_trials: int = 25, n_sims: int = 80, max_drivers: int = 10,
         train_races: list[tuple[int, int]] | None = None,
         holdout_races: list[tuple[int, int]] | None = None, seed: int = 0):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_settings()
    if train_races is None:  # spread across 2024+2025 for era coverage
        r24, r25 = _dry_races(cfg, 2024), _dry_races(cfg, 2025)
        train_races = r24[::4][:4] + r25[::4][:4]
    if holdout_races is None:
        holdout_races = _dry_races(cfg, 2026)
    print(f"train fold: {train_races}\nholdout: {holdout_races}", flush=True)

    def objective(trial):
        p = {
            "cliff_rate": trial.suggest_float("cliff_rate", 0.005, 0.08, log=True),
            "knee_percentile": trial.suggest_float("knee_percentile", 60, 92),
            "offset_prior_weight": trial.suggest_float("offset_prior_weight", 0.2, 0.9),
            "shortlist_prior_weight": trial.suggest_float("shortlist_prior_weight", 1.0, 15.0),
            "undercut_shift_laps": trial.suggest_float("undercut_shift_laps", 0.0, 5.0),
            "prior_weight": trial.suggest_float("prior_weight", 0.2, 3.0),
            "stops_weight": trial.suggest_float("stops_weight", 0.0, 2.0),
            "start_blend_k": trial.suggest_float("start_blend_k", 3.0, 40.0, log=True),
            "pattern_blend_k": trial.suggest_float("pattern_blend_k", 3.0, 40.0, log=True),
        }
        s = _eval_fold(_apply(cfg, p), train_races, n_sims, max_drivers)
        print(f"  trial {trial.number}: score={s:.3f} {p}", flush=True)
        return s

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best_cfg = _apply(cfg, study.best_params)
    base_hold = _eval_fold(cfg, holdout_races, n_sims, max_drivers)
    best_hold = _eval_fold(best_cfg, holdout_races, n_sims, max_drivers)
    print(f"\nbest params: {study.best_params}")
    print(f"train score: {study.best_value:.3f}")
    print(f"holdout: default-config={base_hold:.3f}  tuned={best_hold:.3f}")
    return study.best_params, study.best_value, best_hold


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--sims", type=int, default=80)
    ap.add_argument("--max-drivers", type=int, default=10)
    args = ap.parse_args()
    tune(n_trials=args.trials, n_sims=args.sims, max_drivers=args.max_drivers)
