"""Command-line interface: predict / calibrate / backtest.

    python -m f1strategy.cli predict "Monza" 2025
    python -m f1strategy.cli predict "Monza" 2025 --practice --no-sc
    python -m f1strategy.cli calibrate --trials 200 --cv
    python -m f1strategy.cli calibrate --sc-on --practice
    python -m f1strategy.cli backtest
"""

from __future__ import annotations

import argparse
import sys

from .config import SimConfig, StrategyPrediction


def _print_prediction(pred: StrategyPrediction) -> None:
    ctx = pred.context
    phase = "post-practice" if pred.used_practice else "base"
    print(f"\n{pred.event_name} {pred.year}  —  {phase} prediction  "
          f"(SC hedging {'on' if pred.used_sc else 'off'})")
    print(f"  {ctx.n_laps} laps | pit loss {ctx.pit_loss:.1f}s | fuel {ctx.fuel_rate:+.3f}s/lap"
          f" | history {list(ctx.seasons_used)}"
          + (f" | SC prob {ctx.sc.p_race:.0%}" if ctx.sc else ""))
    print("  tyre model:")
    for c, cm in ctx.compounds.items():
        print(f"    {c:7s} deg {cm.deg:+.3f} s/lap   fresh +{cm.base_offset:.2f}s   "
              f"n={cm.n:4d}  [{cm.source}]")

    print(f"\n  OPTIMAL: {pred.optimal.label()}")
    dist = "  ".join(f"{k}-stop {v:.0%}" for k, v in sorted(pred.p_by_stops.items()))
    print(f"  P(stops): {dist}")
    if pred.used_traffic:
        pos = "  ".join(f"P{k} {v:.0%}" for k, v in sorted(pred.position_dist.items()) if v >= 0.03)
        print(f"  expected finish: ~P{pred.exp_position:.1f}  from grid P{pred.context.field.focal_grid}"
              f"  ({pos})")
    if pred.pit_windows:
        w = "  ".join(f"stop {i+1}: laps {lo}-{hi}" for i, (lo, hi) in pred.pit_windows.items())
        print(f"  pit windows: {w}")
    print("\n  ranked strategies:")
    print(f"    {'strategy':30s} {'Δ vs best':>10s} {'prob':>6s}")
    for r in pred.ranked[:6]:
        print(f"    {r.label():30s} {r.delta_to_best:9.1f}s {r.prob:5.0%}")


def cmd_predict(args) -> None:
    from .predict import predict_strategy
    pred = predict_strategy(args.track, args.year, use_practice=args.practice,
                            use_sc=not args.no_sc, use_traffic=args.traffic,
                            focal_grid=args.grid, seasons_back=args.seasons)
    _print_prediction(pred)


def cmd_calibrate(args) -> None:
    from .dataset import build_dataset
    from . import calibrate, backtest
    sc_on = args.sc_on
    config = SimConfig(use_sc=sc_on, use_traffic=args.traffic)
    cases = [c for c in build_dataset(use_practice=args.practice) if c.usable]
    print(f"{len(cases)} usable cases across {len(set(c.track for c in cases))} tracks")
    params = calibrate.calibrate(cases, config, sc_on=sc_on, n_trials=args.trials)
    backtest.report(cases, params, config, "in-sample")
    cv = None
    if args.cv:
        cv = calibrate.cross_validate(cases, config, sc_on=sc_on, n_trials=max(args.trials // 2, 60))
    path = calibrate.save_params(params, sc_on=sc_on, use_traffic=args.traffic, meta={
        "n_trials": args.trials, "n_cases": len(cases), "use_practice": args.practice,
        "tracks": sorted(set(c.track for c in cases)),
        "cv": {k: v for k, v in (cv or {}).items() if k != "per_track"} if cv else None,
    })
    print(f"saved -> {path}")


def cmd_backtest(args) -> None:
    from .dataset import build_dataset
    from . import calibrate, backtest
    sc_on = args.sc_on
    config = SimConfig(use_sc=sc_on, use_traffic=args.traffic)
    cases = [c for c in build_dataset(use_practice=args.practice) if c.usable]
    params = calibrate.load_params(sc_on=sc_on, use_traffic=args.traffic)
    label = f"SC {'on' if sc_on else 'off'}, traffic {'on' if args.traffic else 'off'}"
    backtest.report(cases, params, config, f"backtest ({label})")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="f1strategy", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("predict", help="predict one race's strategy")
    pp.add_argument("track")
    pp.add_argument("year", type=int)
    pp.add_argument("--practice", action="store_true", help="fold in FP long-runs (post-practice update)")
    pp.add_argument("--no-sc", action="store_true", help="disable safety-car hedging")
    pp.add_argument("--traffic", action="store_true",
                    help="optimise expected outcome vs the qualifying-grid field (track position)")
    pp.add_argument("--grid", type=int, default=2, help="focal grid slot for the traffic objective")
    pp.add_argument("--seasons", type=int, default=3, help="prior seasons of history to pool")
    pp.set_defaults(func=cmd_predict)

    pc = sub.add_parser("calibrate", help="fit global parameters to observed strategies")
    pc.add_argument("--sc-on", action="store_true", help="calibrate the SC-hedging profile")
    pc.add_argument("--traffic", action="store_true", help="calibrate the traffic-objective profile")
    pc.add_argument("--trials", type=int, default=200)
    pc.add_argument("--cv", action="store_true", help="also run leave-one-track-out CV")
    pc.add_argument("--practice", action="store_true", help="use practice-hybrid tyre inputs")
    pc.set_defaults(func=cmd_calibrate)

    pb = sub.add_parser("backtest", help="report predicted-vs-observed on the dataset")
    pb.add_argument("--sc-on", action="store_true")
    pb.add_argument("--traffic", action="store_true")
    pb.add_argument("--practice", action="store_true")
    pb.set_defaults(func=cmd_backtest)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
