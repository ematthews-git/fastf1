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
from strategy_sim2.generation.generator import build_pool
from strategy_sim2.params import circuit, estimate
from strategy_sim2.selection.selector import field_display, plausibility_mass, select, _family
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
    pool = build_pool(wctx, cfg)
    pool_fams = {_family(c) for c in pool}
    n_pos = len(wctx.drivers())
    out = []
    for i, d in enumerate(wctx.drivers()):
        if d not in actual or d not in classified:
            continue
        fin, rt = evaluate_driver(wctx, d, pool, n_sims, int(cfg["simulation"]["seed"]) + i)
        a = actual[d]
        out.append({"race": f"{year}R{rnd}", "pool": pool, "finish": fin, "rtime": rt,
                    "n_pos": n_pos,
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


def _score(cache: list[dict], cfg: dict, near_margin: int = 2) -> dict:
    """Per-driver metrics (continuity) + the race-level PRODUCT metric: is the race's
    modal ordered strategy (or a near-tied second) in the field-aggregated shown 5?"""
    from collections import Counter, defaultdict

    setk = ordk = stop1 = set1 = 0
    by_race: dict[str, list[dict]] = defaultdict(list)
    q_race: dict[str, np.ndarray] = {}
    for r in cache:
        sel = select(r["pool"], r["finish"], r["rtime"], cfg, r["n_pos"])
        sets = {tuple(sorted(s.candidate.compounds)) for s in sel}
        seqs = {s.candidate.compounds for s in sel}
        setk += r["a_set"] in sets
        ordk += r["a_seq"] in seqs
        stop1 += sel[0].candidate.n_stops == r["a_stops"]
        set1 += tuple(sorted(sel[0].candidate.compounds)) == r["a_set"]
        q_d, _, _ = plausibility_mass(r["pool"], r["finish"], r["rtime"], cfg, r["n_pos"])
        q_race[r["race"]] = q_race.get(r["race"], 0.0) + q_d
        by_race[r["race"]].append(r)

    modal_ord = modal_set = 0
    for race, rs in by_race.items():
        pool = rs[0]["pool"]
        idx = field_display(pool, q_race[race] / len(rs), cfg)
        shown_seq = {pool[i].compounds for i in idx}
        shown_set = {tuple(sorted(pool[i].compounds)) for i in idx}
        counts = Counter(r["a_seq"] for r in rs).most_common()
        modal, modal_n = counts[0]
        targets = [modal]
        if len(counts) > 1 and counts[1][1] >= modal_n - near_margin:
            targets.append(counts[1][0])
        modal_ord += any(t in shown_seq for t in targets)
        modal_set += any(tuple(sorted(t)) in shown_set for t in targets)

    n = max(1, len(cache))
    nr = max(1, len(by_race))
    return {"setk": 100 * setk / n, "ordk": 100 * ordk / n,
            "stop1": 100 * stop1 / n, "set1": 100 * set1 / n,
            "modal_ord": 100 * modal_ord / nr, "modal_set": 100 * modal_set / nr}


def tune(year: int = 2025, rounds: list[int] | None = None, n_sims: int = 200,
         n_trials: int = 400, seed: int = 0, cfg: dict | None = None,
         w_modal_set: float = 0.25, w_tiebreak: float = 0.02) -> dict:
    """Objective = race-level modal_ord@5 (the product metric) + w_modal_set * modal_set@5,
    with a small per-driver ord/set tiebreak so trials aren't flat between the coarse
    race-level steps (8 tuning races -> 12.5% quanta)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = cfg or load_settings()
    if rounds is None:
        rounds = [int(r) for r in session_filter.included_races(cfg)
                  .query("year == @year")["round"]]

    cache_path = resolve_path(cfg["data"]["derived_dir"]) / f"seltune_{year}_{n_sims}.pkl"
    key = ("v2", year, tuple(sorted(rounds)), n_sims)  # v2: payload gained the race key
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

    def fmt(m: dict) -> str:
        return (f"modal_ord={m['modal_ord']:.1f} modal_set={m['modal_set']:.1f} | "
                f"setk={m['setk']:.1f} ordk={m['ordk']:.1f} "
                f"set1={m['set1']:.1f} stop1={m['stop1']:.1f}")

    base = _score(cache, cfg)
    print(f"baseline (current cfg): {fmt(base)}", flush=True)

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
        return (m["modal_ord"] + w_modal_set * m["modal_set"]
                + w_tiebreak * (m["ordk"] + m["setk"]))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    best = _score(cache, _apply_selection(cfg, study.best_params))
    print(f"\n{n_trials} trials in {time.time()-t0:.0f}s")
    print(f"baseline : {fmt(base)}")
    print(f"tuned    : {fmt(best)}")
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
