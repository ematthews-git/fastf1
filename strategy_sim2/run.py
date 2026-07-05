"""Orchestration entrypoint: context -> generation -> evaluation -> selection -> report.

Examples:
  venv/bin/python -m strategy_sim2.run --mode postquali --year 2026 --round 8
  venv/bin/python -m strategy_sim2.run --mode postquali --year 2026 --round 8 --sims 400
"""
from __future__ import annotations

import argparse
import time

from strategy_sim2.context.postquali import build_postquali_context
from strategy_sim2.evaluation.monte_carlo import evaluate_driver
from strategy_sim2.generation.generator import build_pool
from strategy_sim2.params import circuit, estimate
from strategy_sim2.report.reporter import build_report, write_report
from strategy_sim2.selection.selector import select
from strategy_sim2.settings import load_settings


def run(mode: str, year: int, rnd: int, n_sims: int | None = None,
        seed: int | None = None, drivers: list[str] | None = None,
        rebuild_params: bool = False, verbose: bool = True) -> str:
    cfg = load_settings()
    n_sims = n_sims or int(cfg["simulation"]["n_sims"])
    seed = seed if seed is not None else int(cfg["simulation"]["seed"])

    ps = estimate.fit_all(cfg, rebuild=rebuild_params)
    profiles = circuit.build_circuit_profiles(ps.lap, cfg)

    if mode == "postquali":
        wctx = build_postquali_context(year, rnd, ps, profiles, cfg)
    elif mode == "prelim":
        from strategy_sim2.context.prelim import build_prelim_context
        wctx = build_prelim_context(year, rnd, ps, profiles, cfg)
    else:
        raise ValueError(f"unknown mode: {mode}")

    pool = build_pool(wctx, cfg)
    if verbose:
        print(f"{wctx.circuit} {year} R{rnd} [{mode}]: {len(wctx.drivers())} drivers, "
              f"{len(pool)} candidates, {n_sims} sims/driver")

    targets = drivers or wctx.drivers()
    n_pos = len(wctx.drivers())
    per_driver = {}
    t0 = time.time()
    for i, d in enumerate(targets):
        finish, rtime = evaluate_driver(wctx, d, pool, n_sims, seed + i)
        per_driver[d] = select(pool, finish, rtime, cfg, n_pos, wctx.prior)
        if verbose:
            best = per_driver[d][0]
            print(f"  {d:4s} grid {wctx.grid[d]:2d} -> "
                  f"{'-'.join(x[0] for x in best.candidate.compounds):8s} "
                  f"E[fin]={best.outcome.mean_finish_classified:.2f}  [{time.time()-t0:.0f}s]")

    report = build_report(wctx, per_driver, n_sims, seed, cfg)
    path = write_report(report, year, rnd, mode, cfg)
    if verbose:
        print(f"wrote {path}  [{time.time()-t0:.0f}s total]")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="postquali", choices=["postquali", "prelim"])
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--drivers", nargs="*", default=None, help="subset of driver codes")
    ap.add_argument("--rebuild-params", action="store_true")
    args = ap.parse_args()
    run(args.mode, args.year, args.round, n_sims=args.sims, seed=args.seed,
        drivers=args.drivers, rebuild_params=args.rebuild_params)
