"""Backtest metrics — how well predicted strategies match observed references.

Hard metrics (human-readable, used for cross-validation reporting):
  * stop accuracy  — model's modal stop count == observed modal stop count
  * pit error      — |predicted pit lap - observed| (laps), at the observed stop count
  * compound match — predicted compound multiset == observed multiset
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from .config import GlobalParams, SimConfig
from . import optimizer


def case_metrics(case, params: GlobalParams, config: SimConfig) -> dict:
    obs = case.obs
    res = optimizer.optimize(case.ctx, params, config)
    pred_stops = max(res.p_by_stops, key=res.p_by_stops.get)
    stop_ok = int(pred_stops == obs.ref_stops)

    at = [r for r in res.ranked if r.n_stops == obs.ref_stops]
    pit_err = np.nan
    if at and obs.ref_pit_laps and len(at[0].pit_laps) == len(obs.ref_pit_laps):
        pit_err = float(np.mean([abs(a - b) for a, b in zip(at[0].pit_laps, obs.ref_pit_laps)]))

    comp_ok = int(Counter(res.optimal.compounds) == Counter(obs.ref_sequence))
    return {
        "track": case.ctx.track, "year": case.ctx.year, "event": case.ctx.event_name,
        "stop_ok": stop_ok, "pit_err": pit_err, "comp_ok": comp_ok,
        "pred": res.optimal, "pred_stops": pred_stops,
        "p_by_stops": res.p_by_stops, "obs": obs, "res": res,
    }


def summarize(cases, params: GlobalParams, config: SimConfig) -> dict:
    rows = [case_metrics(c, params, config) for c in cases if c.usable]
    if not rows:
        return {"n": 0}
    pit_errs = [r["pit_err"] for r in rows if np.isfinite(r["pit_err"])]
    return {
        "n": len(rows),
        "stop_acc": float(np.mean([r["stop_ok"] for r in rows])),
        "comp_acc": float(np.mean([r["comp_ok"] for r in rows])),
        "pit_mae": float(np.median(pit_errs)) if pit_errs else float("nan"),
        "rows": rows,
    }


def report(cases, params: GlobalParams, config: SimConfig, title: str = "") -> dict:
    """Print a per-race predicted-vs-observed table and return the summary."""
    s = summarize(cases, params, config)
    if s["n"] == 0:
        print(f"{title}: no usable cases"); return s
    print(f"\n{title}  ({s['n']} races)   "
          f"stop-acc {s['stop_acc']:.0%} | compound-acc {s['comp_acc']:.0%} | "
          f"pit MAE {s['pit_mae']:.1f} laps")
    print(f"  {'race':30s} {'predicted':24s} {'observed':22s} {'stop':>4s} {'pit':>4s}")
    for r in sorted(s["rows"], key=lambda x: (x["track"], x["year"])):
        pe = f"{r['pit_err']:.0f}" if np.isfinite(r["pit_err"]) else "-"
        mark = "OK" if r["stop_ok"] else "x"
        print(f"  {r['event'][:28]+' '+str(r['year']):30s} "
              f"{r['pred'].label():24s} {r['obs'].label():22s} {mark:>4s} {pe:>4s}")
    return s
