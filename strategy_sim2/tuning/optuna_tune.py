"""Out-of-sample tuning of the simulator's free parameters with Optuna.

Estimation parameters are learned from data; a handful of *simulator* parameters are not
directly observable (overtaking probabilities/thresholds, the plausibility weight in
selection). We tune those against backtest quality on a TRAIN fold of races and report
the winning configuration's score on a disjoint HELD-OUT fold, so overfit settings are
rejected by construction. Historical parameters are fit once and reused across trials
(the tunables don't affect estimation), which keeps trials cheap.
"""
from __future__ import annotations

import copy

import numpy as np

from strategy_sim2.params import circuit, estimate
from strategy_sim2.settings import load_settings
from strategy_sim2.validation.backtest import backtest_race


def _apply(cfg: dict, params: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["overtaking"]["pass_prob_easy"] = params["pass_prob_easy"]
    c["overtaking"]["pass_prob_hard"] = params["pass_prob_hard"]
    c["overtaking"]["threshold_easy"] = params["threshold_easy"]
    c["overtaking"]["threshold_hard"] = params["threshold_hard"]
    c["selection"]["prior_weight"] = params["prior_weight"]
    return c


def _score(rows: list[dict]) -> float:
    """Higher is better: reward finish correlation and strategy recall."""
    if not rows:
        return -1e9
    sp = np.nanmean([r["finish_spearman"] for r in rows])
    rc = np.nanmean([r["recall_in_topk"] for r in rows])
    return float(np.nan_to_num(sp) + np.nan_to_num(rc))


def _fold_score(cfg, ps, profiles, year, rounds, n_sims) -> float:
    rows = [backtest_race(year, r, ps, profiles, cfg, n_sims) for r in rounds]
    return _score([r for r in rows if r])


def tune(train_year: int, train_rounds: list[int], holdout_year: int,
         holdout_rounds: list[int], param_years: list[int],
         n_trials: int = 20, n_sims: int = 80, seed: int = 0):
    import optuna

    cfg = load_settings()
    ps = estimate.fit_all(cfg, years=param_years, use_cache=False)
    profiles = circuit.build_circuit_profiles(ps.lap, cfg, save=False, years=param_years)

    def objective(trial):
        params = {
            "pass_prob_easy": trial.suggest_float("pass_prob_easy", 0.4, 0.9),
            "pass_prob_hard": trial.suggest_float("pass_prob_hard", 0.03, 0.3),
            "threshold_easy": trial.suggest_float("threshold_easy", 0.0, 0.3),
            "threshold_hard": trial.suggest_float("threshold_hard", 0.3, 1.2),
            "prior_weight": trial.suggest_float("prior_weight", 0.0, 1.5),
        }
        return _fold_score(_apply(cfg, params), ps, profiles, train_year, train_rounds, n_sims)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best_cfg = _apply(cfg, study.best_params)
    train_s = study.best_value
    holdout_s = _fold_score(best_cfg, ps, profiles, holdout_year, holdout_rounds, n_sims)
    print(f"best params: {study.best_params}")
    print(f"train score={train_s:.3f}  holdout score={holdout_s:.3f}")
    return study.best_params, train_s, holdout_s


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--sims", type=int, default=80)
    ap.add_argument("--train-rounds", type=int, nargs="*", default=[1, 9])
    ap.add_argument("--holdout-rounds", type=int, nargs="*", default=[1, 9])
    args = ap.parse_args()
    # tune sim params on 2023 races (params from 2021-2022); report on 2024 races
    tune(train_year=2023, train_rounds=args.train_rounds,
         holdout_year=2024, holdout_rounds=args.holdout_rounds,
         param_years=[2021, 2022], n_trials=args.trials, n_sims=args.sims)
