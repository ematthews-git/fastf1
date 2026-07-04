"""Fast selection-parameter tuning on CACHED simulations.

The selection knobs (plausibility_prior_exp, plausibility_comp_exp, comp_temperature,
comp_gate_positions, order_novelty, clone_novelty) only affect ``select()``, which runs on
the already-computed per-driver Monte-Carlo outputs. So we simulate the evaluation set ONCE,
cache (pool, finish, rtime, actual strategy) per driver, then run hundreds of Optuna trials
that re-run only ``select()`` — seconds, not hours (contrast the full-refit tuner in
``optuna_tune.py`` which re-fits and re-sims every trial).

Run:
  venv/bin/python -m strategy_sim2.tuning.tune_selection --year 2025 \
      --rounds 2 3 7 11 15 20 21 23 --sims 200 --trials 400
"""
from __future__ import annotations

import copy
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_sim2.context.postquali import build_postquali_context
from strategy_sim2.data import clean, collector, session_filter
from strategy_sim2.evaluation.monte_carlo import evaluate_driver
from strategy_sim2.generation.generator import generate_candidates, shortlist
from strategy_sim2.params import circuit, estimate
from strategy_sim2.selection.selector import select, _family
from strategy_sim2.settings import load_settings, resolve_path
from strategy_sim2.validation.backtest import actual_strategies


def _sim_race(year: int, rnd: int, ps, profiles, cfg, n_sims: int) -> list[dict]:
    """Per-driver cached sim payload for one race (classified finishers with a realised
    strategy). Stores everything ``select()`` needs plus the ground-truth strategy."""
    raw = clean.get_clean_race(year, rnd, cfg)
    if raw is None:
        return []
    actual = actual_strategies(raw, cfg)
    res = collector.session_results(collector.load_session(year, rnd, "R", weather=False))
    classified = {str(x["driver"]) for _, x in res.iterrows() if x["classified"]}

    wctx = build_postquali_context(year, rnd, ps, profiles, cfg)
    pool = shortlist(generate_candidates(wctx.profile, wctx.params.lap, wctx.prior, wctx.allocation, cfg),
                     k=int(cfg["generation"]["shortlist_k"]),
                     w_prior=float(cfg["generation"].get("shortlist_prior_weight", 6.0)),
                     rep_prior_weight=float(cfg["generation"].get("shortlist_rep_prior_weight", 1.0)))
    pool_fams = {_family(c) for c in pool}
    n_pos = len(wctx.drivers())
    out = []
    for i, d in enumerate(wctx.drivers()):
        if d not in actual or d not in classified:
            continue
        fin, rt = evaluate_driver(wctx, d, pool, n_sims, int(cfg["simulation"]["seed"]) + i)
        a = actual[d]
        out.append({"pool": pool, "finish": fin, "rtime": rt, "n_pos": n_pos,
                    "a_set": tuple(sorted(a["compounds"])), "a_seq": tuple(a["compounds"]),
                    "a_stops": a["n_stops"], "in_short": a["family"] in pool_fams})
    return out


def build_cache(year: int, rounds: list[int], n_sims: int, cfg: dict) -> list[dict]:
    """Simulate the evaluation set ONCE (params fit once, before the test year)."""
    ps = estimate.fit_all(cfg, before=(year, 1), use_cache=False)
    profiles = circuit.build_circuit_profiles(ps.lap, cfg, save=False, before=(year, 1))
    data, t0 = [], time.time()
    for rnd in rounds:
        rows = _sim_race(year, rnd, ps, profiles, cfg, n_sims)
        data.extend(rows)
        print(f"  simmed {year}R{rnd}: {len(rows)} drivers [{time.time()-t0:.0f}s]", flush=True)
    return data


def _apply_selection(cfg: dict, p: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["selection"].update(p)
    return c


def _score(cache: list[dict], cfg: dict) -> dict:
    setk = ordk = stop1 = set1 = 0
    for r in cache:
        sel = select(r["pool"], r["finish"], r["rtime"], cfg, r["n_pos"])
        sets = {tuple(sorted(s.candidate.compounds)) for s in sel}
        seqs = {s.candidate.compounds for s in sel}
        setk += r["a_set"] in sets
        ordk += r["a_seq"] in seqs
        stop1 += sel[0].candidate.n_stops == r["a_stops"]
        set1 += tuple(sorted(sel[0].candidate.compounds)) == r["a_set"]
    n = max(1, len(cache))
    return {"setk": 100 * setk / n, "ordk": 100 * ordk / n,
            "stop1": 100 * stop1 / n, "set1": 100 * set1 / n}


def tune(year: int = 2025, rounds: list[int] | None = None, n_sims: int = 200,
         n_trials: int = 400, seed: int = 0, cfg: dict | None = None,
         w_setk: float = 1.0, w_ordk: float = 0.5) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = cfg or load_settings()
    if rounds is None:
        rounds = [int(r) for r in session_filter.included_races(cfg)
                  .query("year == @year")["round"]]

    cache_path = resolve_path(cfg["data"]["derived_dir"]) / f"seltune_{year}_{n_sims}.pkl"
    key = (year, tuple(sorted(rounds)), n_sims)
    if cache_path.exists():
        saved = pickle.load(open(cache_path, "rb"))
        cache = saved["cache"] if saved.get("key") == key else None
    else:
        cache = None
    if cache is None:
        print(f"building sim cache ({year} rounds {rounds}, {n_sims} sims)...", flush=True)
        cache = build_cache(year, rounds, n_sims, cfg)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump({"key": key, "cache": cache}, open(cache_path, "wb"))
    print(f"cache: {len(cache)} driver-races\n", flush=True)

    base = _score(cache, cfg)
    print(f"baseline (current cfg): setk={base['setk']:.1f} ordk={base['ordk']:.1f} "
          f"set1={base['set1']:.1f} stop1={base['stop1']:.1f}", flush=True)

    def objective(trial):
        p = {
            "plausibility_prior_exp": trial.suggest_float("plausibility_prior_exp", 0.3, 2.5),
            "plausibility_comp_exp": trial.suggest_float("plausibility_comp_exp", 0.0, 2.0),
            "comp_temperature": trial.suggest_float("comp_temperature", 0.8, 6.0),
            "comp_gate_positions": trial.suggest_float("comp_gate_positions", 3.0, 15.0),
            "order_novelty": trial.suggest_float("order_novelty", 0.05, 0.9),
            "clone_novelty": trial.suggest_float("clone_novelty", 0.0, 0.3),
        }
        m = _score(cache, _apply_selection(cfg, p))
        return w_setk * m["setk"] + w_ordk * m["ordk"]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    best = _score(cache, _apply_selection(cfg, study.best_params))
    print(f"\n{n_trials} trials in {time.time()-t0:.0f}s")
    print(f"baseline : setk={base['setk']:.1f} ordk={base['ordk']:.1f} set1={base['set1']:.1f} stop1={base['stop1']:.1f}")
    print(f"tuned    : setk={best['setk']:.1f} ordk={best['ordk']:.1f} set1={best['set1']:.1f} stop1={best['stop1']:.1f}")
    print(f"best params: {study.best_params}")
    return {"baseline": base, "tuned": best, "params": study.best_params}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--rounds", type=int, nargs="*", default=None)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--trials", type=int, default=400)
    args = ap.parse_args()
    tune(year=args.year, rounds=args.rounds, n_sims=args.sims, n_trials=args.trials)
