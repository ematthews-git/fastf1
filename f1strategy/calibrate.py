"""The inverse problem — tune global parameters so simulated optima match reality.

Loss (smooth, probabilistic): for each race, the negative log-likelihood-style
mismatch between the model's strategy distribution and the observed front-runner
reference. Cross-entropy on the stop-count distribution is primary (it uses the
*whole* observed split, so genuinely-divided races like Monza aren't over-punished),
plus soft pit-lap and compound terms. Optimised with Optuna TPE; overfitting is
controlled by a tiny parameter count, priors regularisation, and leave-one-track-out
cross-validation.
"""

from __future__ import annotations

import os
import json
import math
from collections import Counter
from datetime import date
from typing import Optional

import numpy as np
import optuna

from .config import (GlobalParams, SimConfig, SEARCH_SPACE, SC_ONLY_PARAMS,
                     TRAFFIC_ONLY_PARAMS, CLEANAIR_ONLY_PARAMS)
from . import optimizer, backtest

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Loss component weights. Stop-count cross-entropy is primary; the pit term is in
# laps (interpretable) with a small weight; compound mismatch is a light tie-breaker.
W_STOP, W_PIT, W_COMP = 1.0, 0.05, 0.3
PIT_MISS_LAPS = 15.0        # lap-equivalent penalty when the model can't field the
                            # observed stop count at all
REG_LAMBDA = 0.03           # pull thinly-constrained params toward their priors
PARAMS_DIR = "params"


# ------------------------------------------------------------------------- loss


def race_loss(case, params: GlobalParams, config: SimConfig) -> float:
    obs = case.obs
    res = optimizer.optimize(case.ctx, params, config)

    # (a) cross-entropy on stop-count distribution (primary, smooth).
    ce = 0.0
    for k, pk in obs.stop_distribution.items():
        mk = res.p_by_stops.get(k, 0.0)
        ce -= pk * math.log(max(mk, 1e-6))

    # (b) pit-lap error in LAPS and (c) compound multiset mismatch in [0,1], both
    #     evaluated on the model's best strategy at the observed stop count.
    at = [r for r in res.ranked if r.n_stops == obs.ref_stops]
    if at and obs.ref_pit_laps and len(at[0].pit_laps) == len(obs.ref_pit_laps):
        pit_term = sum(abs(a - b) for a, b in zip(at[0].pit_laps, obs.ref_pit_laps)) \
            / len(obs.ref_pit_laps)
        cp, co = Counter(at[0].compounds), Counter(obs.ref_sequence)
        diff = sum((cp - co).values()) + sum((co - cp).values())
        comp_term = diff / (sum(cp.values()) + sum(co.values()))
    else:
        pit_term, comp_term = PIT_MISS_LAPS, 1.0   # can't field the observed stop count

    return W_STOP * ce + W_PIT * pit_term + W_COMP * comp_term


def _regularisation(params: GlobalParams, active: list[str]) -> float:
    pen = 0.0
    d = params.to_dict()
    for name in active:
        lo, hi, prior = SEARCH_SPACE[name]
        pen += ((d[name] - prior) / (hi - lo)) ** 2
    return REG_LAMBDA * pen


def dataset_loss(cases, params: GlobalParams, config: SimConfig,
                 active: Optional[list] = None) -> float:
    num = den = 0.0
    for case in cases:
        if not case.usable:
            continue
        w = case.obs.weight
        num += w * race_loss(case, params, config)
        den += w
    base = num / den if den else 1e9
    return base + (_regularisation(params, active) if active else 0.0)


# ---------------------------------------------------------------------- optuna


def _active_params(sc_on: bool, use_traffic: bool) -> list[str]:
    active = []
    for p in SEARCH_SPACE:
        if p in SC_ONLY_PARAMS and not sc_on:
            continue
        if p in TRAFFIC_ONLY_PARAMS and not use_traffic:
            continue
        if p in CLEANAIR_ONLY_PARAMS and use_traffic:      # pit cost is emergent under traffic
            continue
        active.append(p)
    return active


def _objective(cases, config: SimConfig, sc_on: bool, active: list[str]):
    def obj(trial: optuna.Trial) -> float:
        kw = {p: trial.suggest_float(p, *SEARCH_SPACE[p][:2]) for p in active}
        return dataset_loss(cases, GlobalParams.from_dict(kw), config, active)
    return obj


def calibrate(cases, config: SimConfig, sc_on: Optional[bool] = None,
              n_trials: int = 200, seed: int = 0, verbose: bool = True) -> GlobalParams:
    """Fit global parameters on all provided cases; returns the best GlobalParams."""
    sc_on = config.use_sc if sc_on is None else sc_on
    active = _active_params(sc_on, config.use_traffic)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective(cases, config, sc_on, active), n_trials=n_trials,
                   show_progress_bar=False)
    params = GlobalParams.from_dict(study.best_params)
    if verbose:
        print(f"[calibrate] sc={'on' if sc_on else 'off'}  loss={study.best_value:.4f}  "
              f"{params.to_dict()}")
    return params


def cross_validate(cases, config: SimConfig, sc_on: Optional[bool] = None,
                   n_trials: int = 150, seed: int = 0, verbose: bool = True) -> dict:
    """Leave-one-track-out CV: calibrate on all-but-one track, test on the held-out.

    Returns held-out (out-of-sample) aggregate metrics + per-track rows. This is the
    honest estimate of how the model generalises to a circuit it wasn't tuned on.
    """
    sc_on = config.use_sc if sc_on is None else sc_on
    tracks = sorted({c.track for c in cases if c.usable})
    per_track, held_rows = [], []
    for held in tracks:
        train = [c for c in cases if c.track != held]
        test = [c for c in cases if c.track == held and c.usable]
        if not test or not any(c.usable for c in train):
            continue
        params = calibrate(train, config, sc_on, n_trials=n_trials, seed=seed, verbose=False)
        s = backtest.summarize(test, params, config)
        per_track.append({"track": held, "n": s["n"], "stop_acc": s["stop_acc"],
                          "pit_mae": s["pit_mae"], "comp_acc": s["comp_acc"],
                          "params": params.to_dict()})
        held_rows.extend(s["rows"])
        if verbose:
            print(f"  [CV] held-out {held:12s} n={s['n']}  stop-acc {s['stop_acc']:.0%}  "
                  f"pit MAE {s['pit_mae']:.1f}  comp {s['comp_acc']:.0%}")
    stop = float(np.mean([r["stop_ok"] for r in held_rows])) if held_rows else float("nan")
    comp = float(np.mean([r["comp_ok"] for r in held_rows])) if held_rows else float("nan")
    pit = [r["pit_err"] for r in held_rows if np.isfinite(r["pit_err"])]
    summary = {"cv_stop_acc": stop, "cv_comp_acc": comp,
               "cv_pit_mae": float(np.median(pit)) if pit else float("nan"),
               "n": len(held_rows), "per_track": per_track}
    if verbose:
        print(f"  [CV] OUT-OF-SAMPLE: stop-acc {stop:.0%} | pit MAE {summary['cv_pit_mae']:.1f} "
              f"laps | compound {comp:.0%}  (n={len(held_rows)})")
    return summary


# ------------------------------------------------------------------ persistence


def _params_path(sc_on: bool, use_traffic: bool) -> str:
    suffix = "_traffic" if use_traffic else ""
    return os.path.join(PARAMS_DIR, f"calibrated_sc_{'on' if sc_on else 'off'}{suffix}.json")


def save_params(params: GlobalParams, sc_on: bool, use_traffic: bool = False,
                meta: Optional[dict] = None) -> str:
    os.makedirs(PARAMS_DIR, exist_ok=True)
    path = _params_path(sc_on, use_traffic)
    with open(path, "w") as f:
        json.dump({"params": params.to_dict(), "sc_on": sc_on, "use_traffic": use_traffic,
                   "calibrated": str(date.today()), "meta": meta or {}}, f, indent=2)
    return path


def load_params(sc_on: bool, use_traffic: bool = False) -> GlobalParams:
    path = _params_path(sc_on, use_traffic)
    if not os.path.exists(path):
        path = _params_path(sc_on, False)          # fall back to the clean-air profile
    if not os.path.exists(path):
        return GlobalParams()                      # uncalibrated priors as a safe fallback
    with open(path) as f:
        return GlobalParams.from_dict(json.load(f)["params"])
