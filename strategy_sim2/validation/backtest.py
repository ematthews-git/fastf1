"""Out-of-sample backtesting.

For each test race, parameters are trained only on *earlier* seasons (time-based split,
no leakage), the full pipeline is run in post-quali mode, and predictions are compared to
what actually happened. Reported metrics target the project's real goal — ranking
realistic, strong strategies highly — not race-time reproduction:

  * finish_spearman      : corr(predicted expected finish, actual finish)
  * stop_count_acc       : share of drivers whose top pick's stop count matched reality
  * recall_in_shortlist  : share whose actual strategy family was generated at all
  * recall_in_topk       : share whose actual strategy family made the selected 2-5
  * first_pit_mae        : |predicted first-stop lap - actual| (stop-count matches)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from strategy_sim2.context.postquali import build_postquali_context
from strategy_sim2.data import clean, collector, session_filter
from strategy_sim2.data.schema import DRY_COMPOUNDS
from strategy_sim2.evaluation.monte_carlo import evaluate_driver
from strategy_sim2.generation.generator import generate_candidates, shortlist
from strategy_sim2.params import circuit, estimate
from strategy_sim2.selection.selector import select
from strategy_sim2.settings import load_settings


def actual_strategies(raw: pd.DataFrame, cfg: dict | None = None) -> dict[str, dict]:
    """Per-driver realised strategy via the shared cleaner (red-flag / SC flurries merged)."""
    from strategy_sim2.data.strategy import extract_all
    cfg = cfg or load_settings()
    return extract_all(raw, int(cfg["cleaning"].get("min_strategic_stint", 5)))


def backtest_race(year: int, rnd: int, ps, profiles, cfg, n_sims: int) -> dict | None:
    raw = clean.get_clean_race(year, rnd, cfg)
    if raw is None:
        return None
    actual = actual_strategies(raw)
    res = collector.session_results(collector.load_session(year, rnd, "R", weather=False))
    actual_finish = {str(r["driver"]): float(r["finish_position"]) for _, r in res.iterrows()
                     if r["finish_position"] == r["finish_position"]}

    wctx = build_postquali_context(year, rnd, ps, profiles, cfg)
    pool = shortlist(generate_candidates(wctx.profile, wctx.params.lap, wctx.prior, wctx.allocation, cfg),
                     k=int(cfg["generation"]["shortlist_k"]),
                     w_prior=float(cfg["generation"].get("shortlist_prior_weight", 6.0)))
    pool_families = {(c.n_stops, tuple(sorted(c.compounds))) for c in pool}
    n_pos = len(wctx.drivers())

    pred_finish, top_stops, top_firstpit = {}, {}, {}
    recall_short, recall_top, stop_ok, pit_err = [], [], [], []
    for i, d in enumerate(wctx.drivers()):
        finish, rtime = evaluate_driver(wctx, d, pool, n_sims, int(cfg["simulation"]["seed"]) + i)
        sel = select(pool, finish, rtime, cfg, n_pos, wctx.prior)
        best = sel[0]
        pred_finish[d] = best.outcome.mean_finish_classified
        if d in actual:
            fam = actual[d]["family"]
            recall_short.append(fam in pool_families)
            recall_top.append(fam in {(s.candidate.n_stops, tuple(sorted(s.candidate.compounds)))
                                      for s in sel})
            stop_ok.append(best.candidate.n_stops == actual[d]["n_stops"])
            if best.candidate.n_stops == actual[d]["n_stops"] and best.candidate.pit_laps and actual[d]["pit_laps"]:
                pit_err.append(abs(best.candidate.pit_laps[0] - actual[d]["pit_laps"][0]))

    common = [d for d in pred_finish if d in actual_finish]
    rho = spearmanr([pred_finish[d] for d in common],
                    [actual_finish[d] for d in common]).statistic if len(common) > 2 else np.nan
    return {
        "year": year, "round": rnd, "circuit": wctx.circuit, "n_drivers": len(common),
        "finish_spearman": float(rho),
        "stop_count_acc": float(np.mean(stop_ok)) if stop_ok else np.nan,
        "recall_in_shortlist": float(np.mean(recall_short)) if recall_short else np.nan,
        "recall_in_topk": float(np.mean(recall_top)) if recall_top else np.nan,
        "first_pit_mae": float(np.mean(pit_err)) if pit_err else np.nan,
    }


def run_backtest(test_year: int, train_years: list[int] | None = None,
                 max_races: int | None = None, n_sims: int = 120,
                 cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_settings()
    start = int(cfg["training"]["start_year"])
    train_years = train_years or list(range(start, test_year))

    ps = estimate.fit_all(cfg, years=train_years, use_cache=False)
    profiles = circuit.build_circuit_profiles(ps.lap, cfg, save=False, years=train_years)

    races = session_filter.included_races(cfg)
    races = races[races["year"] == test_year]
    if max_races:
        races = races.head(max_races)

    rows = []
    for _, r in races.iterrows():
        m = backtest_race(test_year, int(r["round"]), ps, profiles, cfg, n_sims)
        if m:
            rows.append(m)
            print(f"  {m['circuit']:14s} rho={m['finish_spearman']:.3f} "
                  f"stopAcc={m['stop_count_acc']:.2f} recallShort={m['recall_in_shortlist']:.2f} "
                  f"recallTop={m['recall_in_topk']:.2f} pitMAE={m['first_pit_mae']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    if len(df):
        print("\n=== OOS backtest summary (train "
              f"{train_years[0]}-{train_years[-1]} -> test {test_year}) ===")
        for col in ["finish_spearman", "stop_count_acc", "recall_in_shortlist",
                    "recall_in_topk", "first_pit_mae"]:
            print(f"  {col:20s} mean={df[col].mean():.3f}")
    return df


# --------------------------------------------------------------------------
# Strategy-focused backtest (compound choice / order / stop count / pit windows)
# --------------------------------------------------------------------------

STRATEGY_FIELDS = [
    "race", "circuit", "driver", "grid", "actual_stops", "actual_comp", "actual_pits",
    "p1_stops", "p1_comp", "p1_pits", "set1", "ord1", "stop1",
    "set_topk", "ord_topk", "in_short", "pit_err_first", "pit_err_all",
]


def strategy_backtest_race(year: int, rnd: int, ps, profiles, cfg,
                           n_sims: int) -> list[dict]:
    """Per-driver strategy-prediction rows for one race (classified finishers only)."""
    import json as _json

    raw = clean.get_clean_race(year, rnd, cfg)
    if raw is None:
        return []
    actual = actual_strategies(raw, cfg)
    res = collector.session_results(collector.load_session(year, rnd, "R", weather=False))
    classified = {str(x["driver"]) for _, x in res.iterrows() if x["classified"]}

    wctx = build_postquali_context(year, rnd, ps, profiles, cfg)
    pool = shortlist(generate_candidates(wctx.profile, wctx.params.lap, wctx.prior, wctx.allocation, cfg),
                     k=int(cfg["generation"]["shortlist_k"]),
                     w_prior=float(cfg["generation"].get("shortlist_prior_weight", 6.0)))
    pool_fams = {(c.n_stops, tuple(sorted(c.compounds))) for c in pool}
    n_pos = len(wctx.drivers())

    rows = []
    for i, d in enumerate(wctx.drivers()):
        if d not in actual or d not in classified:
            continue
        a = actual[d]
        fin, rt = evaluate_driver(wctx, d, pool, n_sims, int(cfg["simulation"]["seed"]) + i)
        sel = select(pool, fin, rt, cfg, n_pos, wctx.prior)
        p1 = sel[0].candidate
        sel_ms = {tuple(sorted(s.candidate.compounds)) for s in sel}
        sel_seq = {s.candidate.compounds for s in sel}
        a_ms = tuple(sorted(a["compounds"]))

        pit_err_first, pit_err_all = "", ""
        if p1.n_stops == a["n_stops"] and p1.pit_laps and a["pit_laps"]:
            errs = [abs(pp - ap) for pp, ap in zip(p1.pit_laps, a["pit_laps"])]
            pit_err_first = errs[0]
            pit_err_all = _json.dumps(errs)

        rows.append({
            "race": f"{year}R{rnd}", "circuit": wctx.circuit, "driver": d,
            "grid": wctx.grid[d],
            "actual_stops": a["n_stops"], "actual_comp": "-".join(a["compounds"]),
            "actual_pits": _json.dumps(a["pit_laps"]),
            "p1_stops": p1.n_stops, "p1_comp": "-".join(p1.compounds),
            "p1_pits": _json.dumps(list(p1.pit_laps)),
            "set1": int(tuple(sorted(p1.compounds)) == a_ms),
            "ord1": int(p1.compounds == a["compounds"]),
            "stop1": int(p1.n_stops == a["n_stops"]),
            "set_topk": int(a_ms in sel_ms),
            "ord_topk": int(a["compounds"] in sel_seq),
            "in_short": int(a["family"] in pool_fams),
            "pit_err_first": pit_err_first, "pit_err_all": pit_err_all,
        })
    return rows


def summarize_strategy(df: pd.DataFrame) -> None:
    import json as _json

    def pct(col, d=df):
        return 100 * pd.to_numeric(d[col], errors="coerce").mean()

    print(f"\n=== STRATEGY BACKTEST: {len(df)} driver-races, {df['race'].nunique()} races ===")
    print(f"  stop-count top-1        : {pct('stop1'):5.1f}%")
    print(f"  compound-set top-1      : {pct('set1'):5.1f}%")
    print(f"  compound-order top-1    : {pct('ord1'):5.1f}%")
    print(f"  compound-set in top-k   : {pct('set_topk'):5.1f}%")
    print(f"  compound-order in top-k : {pct('ord_topk'):5.1f}%")
    print(f"  generation recall       : {pct('in_short'):5.1f}%")
    fe = pd.to_numeric(df["pit_err_first"], errors="coerce").dropna()
    if len(fe):
        print(f"  first-stop MAE          : {fe.mean():5.1f} laps "
              f"(±3: {100*(fe<=3).mean():.0f}%, ±5: {100*(fe<=5).mean():.0f}%)")
    per = df.groupby("circuit").agg(
        n=("driver", "size"),
        stop=("stop1", lambda s: 100 * pd.to_numeric(s).mean()),
        setk=("set_topk", lambda s: 100 * pd.to_numeric(s).mean()),
        recall=("in_short", lambda s: 100 * pd.to_numeric(s).mean()))
    print(per.round(0).to_string())


def run_strategy_backtest(test_year: int, rounds: list[int] | None = None,
                          n_sims: int = 200, out_csv: str | None = None,
                          cfg: dict | None = None) -> pd.DataFrame:
    """Expanding-window strategy backtest: for each test race, parameters and priors are
    fit on everything strictly BEFORE that race (incl. completed same-season rounds)."""
    import time as _time

    cfg = cfg or load_settings()
    races = session_filter.included_races(cfg)
    races = races[races["year"] == test_year].sort_values("round")
    if rounds is not None:
        races = races[races["round"].isin(rounds)]

    all_rows, t0 = [], _time.time()
    for _, r in races.iterrows():
        rnd = int(r["round"])
        ps = estimate.fit_all(cfg, before=(test_year, rnd), use_cache=False)
        profiles = circuit.build_circuit_profiles(ps.lap, cfg, save=False,
                                                  before=(test_year, rnd))
        rows = strategy_backtest_race(test_year, rnd, ps, profiles, cfg, n_sims)
        all_rows.extend(rows)
        print(f"{test_year}R{rnd} {r['circuit']}: {len(rows)} drivers "
              f"[{_time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(all_rows)
    if out_csv and len(df):
        df.to_csv(out_csv, index=False)
    if len(df):
        summarize_strategy(df)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", action="store_true",
                    help="strategy-accuracy backtest (expanding window)")
    ap.add_argument("--test-year", type=int, default=2024)
    ap.add_argument("--rounds", type=int, nargs="*", default=None)
    ap.add_argument("--max-races", type=int, default=5)
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()
    if args.strategy:
        run_strategy_backtest(args.test_year, rounds=args.rounds, n_sims=args.sims,
                              out_csv=args.out_csv)
    else:
        run_backtest(args.test_year, max_races=args.max_races, n_sims=args.sims)
