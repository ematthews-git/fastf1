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


def actual_strategies(raw: pd.DataFrame) -> dict[str, dict]:
    """Per-driver realised strategy: compound sequence, stop count, pit laps."""
    out = {}
    for drv, g in raw.groupby("driver"):
        g = g.dropna(subset=["stint"]).sort_values("lap_number")
        comps, pit_laps = [], []
        stints = list(g.groupby("stint"))
        for i, (_, sg) in enumerate(stints):
            mode = sg["compound"].mode()
            c = str(mode.iloc[0]) if len(mode) else None
            if c not in DRY_COMPOUNDS:
                comps = []  # wet stint -> discard (dry-only)
                break
            comps.append(c)
            if i < len(stints) - 1:
                pit_laps.append(int(sg["lap_number"].max()))
        if comps:
            out[str(drv)] = {"compounds": tuple(comps), "n_stops": len(comps) - 1,
                             "pit_laps": pit_laps, "family": (len(comps) - 1, tuple(sorted(comps)))}
    return out


def backtest_race(year: int, rnd: int, ps, profiles, cfg, n_sims: int) -> dict | None:
    raw = clean.get_clean_race(year, rnd, cfg)
    if raw is None:
        return None
    actual = actual_strategies(raw)
    res = collector.session_results(collector.load_session(year, rnd, "R", weather=False))
    actual_finish = {str(r["driver"]): float(r["finish_position"]) for _, r in res.iterrows()
                     if r["finish_position"] == r["finish_position"]}

    wctx = build_postquali_context(year, rnd, ps, profiles, cfg)
    pool = shortlist(generate_candidates(wctx.profile, ps.lap, wctx.prior, wctx.allocation, cfg),
                     k=int(cfg["generation"]["shortlist_k"]))
    pool_families = {(c.n_stops, tuple(sorted(c.compounds))) for c in pool}
    n_pos = len(wctx.drivers())

    pred_finish, top_stops, top_firstpit = {}, {}, {}
    recall_short, recall_top, stop_ok, pit_err = [], [], [], []
    for i, d in enumerate(wctx.drivers()):
        finish, rtime = evaluate_driver(wctx, d, pool, n_sims, int(cfg["simulation"]["seed"]) + i)
        sel = select(pool, finish, rtime, cfg, n_pos)
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-year", type=int, default=2024)
    ap.add_argument("--max-races", type=int, default=5)
    ap.add_argument("--sims", type=int, default=120)
    args = ap.parse_args()
    run_backtest(args.test_year, max_races=args.max_races, n_sims=args.sims)
