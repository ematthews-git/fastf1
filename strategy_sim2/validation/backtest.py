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
from strategy_sim2.generation.generator import build_pool
from strategy_sim2.params import circuit, estimate
from strategy_sim2.selection.selector import field_display, plausibility_mass, select
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
    pool = build_pool(wctx, cfg)
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
    "shown", "field_shown",
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
    pool = build_pool(wctx, cfg)
    pool_fams = {(c.n_stops, tuple(sorted(c.compounds))) for c in pool}
    n_pos = len(wctx.drivers())

    rows = []
    q_sum = np.zeros(len(pool))
    n_q = 0
    for i, d in enumerate(wctx.drivers()):
        if d not in actual or d not in classified:
            continue
        a = actual[d]
        fin, rt = evaluate_driver(wctx, d, pool, n_sims, int(cfg["simulation"]["seed"]) + i)
        q_d, _, _ = plausibility_mass(pool, fin, rt, cfg, n_pos)
        q_sum += q_d
        n_q += 1
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
            "shown": _json.dumps(["-".join(s.candidate.compounds) for s in sel]),
        })

    # Race-level display: the same greedy set-cover on field-aggregated plausibility mass.
    # This is what a per-race fan page shows and what the modal_ord@5/modal_set@5 metrics
    # score (the product goal: the race's modal strategy must be among the shown 5).
    if rows and n_q:
        idx = field_display(pool, q_sum / n_q, cfg)
        shown = _json.dumps(["-".join(pool[i].compounds) for i in idx])
        for r in rows:
            r["field_shown"] = shown
    return rows


def race_modal_metrics(df: pd.DataFrame, near_margin: int = 2) -> pd.DataFrame:
    """Race-level PRODUCT metric: is the race's modal ordered strategy — or a near-tied
    second (within ``near_margin`` runners) — in the race-level shown 5 (``field_shown``)?
    One row per race with modal_ord / modal_set hits."""
    import json as _json

    rows = []
    for race, g in df.groupby("race"):
        counts = g["actual_comp"].value_counts()
        modal, modal_n = str(counts.index[0]), int(counts.iloc[0])
        second = str(counts.index[1]) if len(counts) > 1 else None
        second_n = int(counts.iloc[1]) if len(counts) > 1 else 0
        near = second if (second is not None and second_n >= modal_n - near_margin) else None
        targets = [modal] + ([near] if near else [])

        fs = g["field_shown"].dropna()
        shown = _json.loads(fs.iloc[0]) if len(fs) else []
        shown_sets = {tuple(sorted(s.split("-"))) for s in shown}
        ord_hit = any(t in shown for t in targets)
        set_hit = any(tuple(sorted(t.split("-"))) in shown_sets for t in targets)
        rows.append({"race": race, "circuit": str(g["circuit"].iloc[0]),
                     "modal": modal, "modal_n": modal_n, "near": near or "",
                     "modal_ord": int(ord_hit), "modal_set": int(set_hit),
                     "field_shown": " | ".join(shown)})
    return pd.DataFrame(rows)


def summarize_strategy(df: pd.DataFrame, exclude: list[str] | None = None) -> None:
    # Aggregates are computed EXCLUDING circuits flagged as non-indicative (e.g. a 2026
    # compound-nomination shift breaks the label-keyed model at Barcelona); the per-circuit
    # table below still lists them so nothing is hidden.
    exclude = [str(x) for x in (exclude or [])]
    dfx = df[~df["circuit"].isin(exclude)] if exclude else df

    def pct(col, d=dfx):
        return 100 * pd.to_numeric(d[col], errors="coerce").mean()

    tag = f"  (excl. {', '.join(exclude)})" if exclude else ""
    print(f"\n=== STRATEGY BACKTEST: {len(dfx)} driver-races, {dfx['race'].nunique()} races{tag} ===")
    if "field_shown" in df.columns:
        rm_all = race_modal_metrics(df)
        rm = rm_all[~rm_all["circuit"].isin(exclude)] if exclude else rm_all
        print(f"  RACE modal-order in 5   : {100*rm['modal_ord'].mean():5.1f}%  "
              f"({int(rm['modal_ord'].sum())}/{len(rm)})   <- product metric")
        print(f"  RACE modal-set in 5     : {100*rm['modal_set'].mean():5.1f}%  "
              f"({int(rm['modal_set'].sum())}/{len(rm)})")
    print(f"  stop-count top-1        : {pct('stop1'):5.1f}%")
    print(f"  compound-set top-1      : {pct('set1'):5.1f}%")
    print(f"  compound-order top-1    : {pct('ord1'):5.1f}%")
    print(f"  compound-set in top-k   : {pct('set_topk'):5.1f}%")
    print(f"  compound-order in top-k : {pct('ord_topk'):5.1f}%")
    print(f"  generation recall       : {pct('in_short'):5.1f}%")
    fe = pd.to_numeric(dfx["pit_err_first"], errors="coerce").dropna()
    if len(fe):
        print(f"  first-stop MAE          : {fe.mean():5.1f} laps "
              f"(±3: {100*(fe<=3).mean():.0f}%, ±5: {100*(fe<=5).mean():.0f}%)")
    per = df.groupby("circuit").agg(
        n=("driver", "size"),
        stop=("stop1", lambda s: 100 * pd.to_numeric(s).mean()),
        setk=("set_topk", lambda s: 100 * pd.to_numeric(s).mean()),
        recall=("in_short", lambda s: 100 * pd.to_numeric(s).mean()))
    print(per.round(0).to_string())
    if "field_shown" in df.columns and len(rm_all):
        print("\nper-race modal (product metric):")
        cols = ["race", "circuit", "modal", "modal_n", "near", "modal_ord", "modal_set", "field_shown"]
        print(rm_all[cols].to_string(index=False))


def _strategy_race_job(test_year: int, rnd: int, n_sims: int, cfg: dict) -> list[dict]:
    """Fit the expanding-window params for one test race and return its per-driver rows.
    Module-level (picklable) so it can run in a worker process — each race is independent
    (its own pre-race param fit + sim), so the full backtest parallelises across rounds."""
    ps = estimate.fit_all(cfg, before=(test_year, rnd), use_cache=False)
    profiles = circuit.build_circuit_profiles(ps.lap, cfg, save=False, before=(test_year, rnd))
    return strategy_backtest_race(test_year, rnd, ps, profiles, cfg, n_sims)


def run_strategy_backtest(test_year: int, rounds: list[int] | None = None,
                          n_sims: int = 200, out_csv: str | None = None,
                          cfg: dict | None = None, workers: int = 5) -> pd.DataFrame:
    """Expanding-window strategy backtest: for each test race, parameters and priors are
    fit on everything strictly BEFORE that race (incl. completed same-season rounds).

    Rounds are independent, so up to ``workers`` run concurrently in separate processes
    (results are seed-fixed, so parallelism does not change them)."""
    import time as _time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cfg = cfg or load_settings()
    races = session_filter.included_races(cfg)
    races = races[races["year"] == test_year].sort_values("round")
    if rounds is not None:
        races = races[races["round"].isin(rounds)]
    round_list = [int(r["round"]) for _, r in races.iterrows()]
    circuits = {int(r["round"]): str(r["circuit"]) for _, r in races.iterrows()}

    all_rows, t0 = [], _time.time()
    n_workers = max(1, min(int(workers), len(round_list)))
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_strategy_race_job, test_year, rnd, n_sims, cfg): rnd
                    for rnd in round_list}
            for fut in as_completed(futs):
                rnd = futs[fut]
                rows = fut.result()
                all_rows.extend(rows)
                print(f"{test_year}R{rnd} {circuits[rnd]}: {len(rows)} drivers "
                      f"[{_time.time()-t0:.0f}s]", flush=True)
    else:
        for rnd in round_list:
            rows = _strategy_race_job(test_year, rnd, n_sims, cfg)
            all_rows.extend(rows)
            print(f"{test_year}R{rnd} {circuits[rnd]}: {len(rows)} drivers "
                  f"[{_time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(all_rows)
    if len(df):
        print(f"\n[{n_workers} workers, {_time.time()-t0:.0f}s total]")
        summarize_strategy(df, exclude=cfg.get("validation", {}).get("strategy_exclude_circuits"))
    if out_csv and len(df):
        from pathlib import Path
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"wrote {out_csv}")
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
    ap.add_argument("--workers", type=int, default=5, help="parallel rounds (strategy mode)")
    args = ap.parse_args()
    if args.strategy:
        run_strategy_backtest(args.test_year, rounds=args.rounds, n_sims=args.sims,
                              out_csv=args.out_csv, workers=args.workers)
    else:
        run_backtest(args.test_year, max_races=args.max_races, n_sims=args.sims)
