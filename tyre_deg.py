#!/usr/bin/env python3
"""Per-track, per-compound tyre degradation model (s/lap), validated vs Pirelli.

Predictive by construction: a target year's degradation is estimated only from
*prior* seasons at the same circuit (no look-ahead), so it works for races that
haven't happened yet.

Two independent estimators, both measured on the SAME clean green-flag laps:

  1. pooled joint OLS (per circuit, all prior seasons at once)
         LapTime = Σ_y α_y[year==y] + φ·LapNumber + Σ_c δ_c·(TyreLife where Compound==c)
     α_y  per-season intercepts (absorb car-pace evolution between seasons)
     φ    fuel-burn + track-evolution rate (negative; car speeds up over a race)
     δ_c  per-compound degradation  ->  the number we want, in s/lap

  2. robust per-stint slope: fuel-correct each stint, OLS slope of lap time vs
     tyre age, then take the lap-count-weighted MEDIAN slope per compound.

Unlike the old predict_strategy.py, degradation is read from ALL clean green laps
(not just the fastest 35%), so real pace loss isn't flattened, and there are NO
per-compound fudge multipliers.

Race data alone measures HARD well (run to the end of its stints, fully observed)
but under-reads MEDIUM/SOFT, which teams pit before the cliff — that truncated
phase simply isn't in race stints. So with --practice we add a third estimator
from FP1/FP2/FP3 long runs (where teams push those compounds further), and the
RECOMMENDED per-compound number is a hybrid: race for HARD, practice for MED/SOFT.
Validated vs the three Pirelli 2026 references, that hybrid reaches ~0.030 s/lap MAE.

Usage:
  python tyre_deg.py --validate                    # race-only estimators vs Pirelli
  python tyre_deg.py --validate --practice         # add practice + hybrid recommend
  python tyre_deg.py "Austria" 2026 --practice     # one circuit, full breakdown
  python tyre_deg.py "Canada" 2026 --target-practice  # also fold in 2026's own FP
  python tyre_deg.py "Spain" 2026 --seasons 5
"""

import sys
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import fastf1 as f1
from scipy.stats import theilslopes

f1.Cache.enable_cache(os.environ.get("FASTF1_CACHE_DIR", "./f1_cache"))
try:
    f1.set_log_level("ERROR")
except Exception:  # noqa: BLE001
    pass

SLOTS = ("SOFT", "MEDIUM", "HARD")   # print order: softest first (matches Pirelli tuples)
_ORD = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}

MAX_HISTORY = 4          # prior seasons to pool
DROP_STINT_START = 2     # ignore the first N flying laps of a stint (out-lap + warmup)
MIN_STINT_LAPS = 8       # min clean laps for a stint to yield a slope
FUEL_BOUNDS = (-0.14, -0.02)  # sane clamp for the fuel+evo rate (s/lap)

PRACTICE_SESSIONS = ("FP1", "FP2", "FP3")  # not every team long-runs in FP2, so scan all
MIN_LONGRUN = 6          # min representative laps for a stint to count as a race-sim run
FUEL_ADD_FALLBACK = -0.05  # fuel+evo add-back if no race fuel rate is available

# Pirelli 2026 pre-race degradation (s/lap), given as (soft, medium, hard).
PIRELLI_2026 = {
    "Barcelona": {"query": "Spain",   "ref": {"SOFT": 0.25, "MEDIUM": 0.18, "HARD": 0.10}},
    "Austria":   {"query": "Austria", "ref": {"SOFT": 0.13, "MEDIUM": 0.10, "HARD": 0.07}},
    "Canada":    {"query": "Canada",  "ref": {"SOFT": 0.08, "MEDIUM": 0.06, "HARD": 0.04}},
}


# --------------------------------------------------------------------------- data


def load_race(track, year):
    s = f1.get_session(year, track, "R")
    s.load(laps=True, telemetry=False, weather=False, messages=False)
    return s


def clean_laps(session):
    """Repair FastF1's missing-compound issue and assign per-driver stint ids."""
    d = session.laps.copy()
    d["Compound"] = d["Compound"].replace("nan", np.nan)
    d = d.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    d["_pp"] = d.groupby("Driver")["PitInTime"].shift().notna()   # prev lap pitted -> new stint
    d["stint_id"] = d.groupby("Driver")["_pp"].cumsum()
    d["Compound"] = d.groupby(["Driver", "stint_id"])["Compound"].transform(
        lambda x: x.fillna(x.dropna().mode().iloc[0]) if x.notna().any() else x
    )
    return d


def green_laps(laps, drop_start=DROP_STINT_START):
    """Clean green-flag racing laps, on dry slots only, past the stint warm-up."""
    d = laps.copy()
    d["LapTime_s"] = d["LapTime"].dt.total_seconds()
    mask = (
        d["LapTime_s"].notna()
        & d["PitInTime"].isna()
        & d["PitOutTime"].isna()
        & ~d["TrackStatus"].astype(str).str.contains("[4567]", regex=True)  # SC/red/VSC
        & d["Compound"].isin(SLOTS)
        & (d["TyreLife"] > drop_start)
    )
    if "Deleted" in d.columns:
        mask &= ~(d["Deleted"] == True)  # noqa: E712  (NaN -> kept)
    d = d[mask].copy()
    # drop gross outliers (traffic/lockups/errors): > 107% of the field median
    if len(d):
        cut = d["LapTime_s"].median() * 1.07
        d = d[d["LapTime_s"] <= cut].copy()
    return d


# ------------------------------------------------------------------- estimators


def pooled_ols(pool):
    """Joint regression over all pooled green laps -> (fuel_rate, {compound: deg})."""
    g = pool.dropna(subset=["LapNumber", "TyreLife", "Compound", "LapTime_s"])
    if len(g) < 50:
        return np.nan, {}
    years = sorted(g["year"].unique())
    cols = [(g["year"] == y).astype(float).values for y in years]   # season intercepts
    cols.append(g["LapNumber"].astype(float).values)               # fuel + track evo
    present = [c for c in SLOTS if (g["Compound"] == c).sum() >= MIN_STINT_LAPS]
    for c in present:
        cols.append(((g["Compound"] == c) * g["TyreLife"]).astype(float).values)
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, g["LapTime_s"].values, rcond=None)
    fuel = float(np.clip(beta[len(years)], *FUEL_BOUNDS))
    deg = {c: float(beta[len(years) + 1 + i]) for i, c in enumerate(present)}
    return fuel, deg


def _wmedian(vals, weights):
    order = np.argsort(vals)
    vals, weights = np.asarray(vals)[order], np.asarray(weights)[order]
    cw = np.cumsum(weights)
    return float(vals[np.searchsorted(cw, cw[-1] / 2.0)])


def stint_slopes(pool, fuel_rate):
    """Fuel-correct each stint, OLS slope vs tyre age, weighted-median per compound."""
    g = pool.dropna(subset=["LapNumber", "TyreLife", "Compound", "LapTime_s"]).copy()
    g["fc"] = g["LapTime_s"] - fuel_rate * g["LapNumber"]
    acc = defaultdict(list)   # compound -> [(slope, n_laps), ...]
    for (_, _, _), st in g.groupby(["year", "Driver", "stint_id"]):
        if len(st) < MIN_STINT_LAPS:
            continue
        comp = st["Compound"].mode().iloc[0]
        if comp not in SLOTS:
            continue
        x = st["TyreLife"].astype(float).values
        if x.std() < 1e-6:
            continue
        slope = float(np.polyfit(x, st["fc"].values, 1)[0])
        acc[comp].append((slope, len(st)))
    out = {}
    for c, arr in acc.items():
        out[c] = _wmedian([s for s, _ in arr], [n for _, n in arr])
    return out, {c: len(arr) for c, arr in acc.items()}


# ---------------------------------------------------------------- practice long runs


def load_practice(track, year):
    """Load every available practice session (FP1/FP2/FP3) for a weekend."""
    out = []
    for name in PRACTICE_SESSIONS:
        try:
            s = f1.get_session(year, track, name)
            s.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception:  # noqa: BLE001  (sprint weekends lack FP2/FP3, etc.)
            continue
        if s.laps is not None and len(s.laps):
            out.append((name, s))
    return out


def _run_slope(run):
    """Robust deg slope (s/lap) of a single practice run vs tyre age.

    Drops within-run cool-down/backing-off laps (slower than 103% of the run's
    own median), then Theil-Sen slope (resistant to the remaining junk laps).
    Returns (slope, n_used) or None if the run isn't a usable long run.
    """
    r = run.copy()
    med = r["LapTime_s"].median()
    r = r[r["LapTime_s"] <= med * 1.03]
    if len(r) < MIN_LONGRUN:
        return None
    x = r["TyreLife"].astype(float).values
    if x.max() - x.min() < MIN_LONGRUN - 1:   # need real spread in tyre age
        return None
    slope = float(theilslopes(r["LapTime_s"].values, x)[0])
    return slope, len(r)


def practice_longrun_deg(track, years, fuel_rate):
    """Per-compound deg from practice race-sim long runs, pooled over `years`.

    Each run's raw slope = deg + fuel + track-evo; we add back `fuel_rate`
    (negative) to recover intrinsic degradation, then take the lap-weighted
    median across runs. Practice exposes the degraded phase teams skip in races.
    """
    add_back = fuel_rate if np.isfinite(fuel_rate) else FUEL_ADD_FALLBACK
    acc = defaultdict(list)   # compound -> [(deg, n_used), ...]
    used_years = []
    for y in years:
        sessions = load_practice(track, y)
        if sessions:
            used_years.append(y)
        for _, s in sessions:
            g = green_laps(clean_laps(s), drop_start=1)  # keep more: practice runs are short
            for (_, _), run in g.groupby(["Driver", "stint_id"]):
                if len(run) < MIN_LONGRUN or run["Compound"].mode().iloc[0] not in SLOTS:
                    continue
                res = _run_slope(run)
                if res is None:
                    continue
                slope, n = res
                acc[run["Compound"].mode().iloc[0]].append((slope - add_back, n))
    deg = {c: _wmedian([d for d, _ in a], [n for _, n in a]) for c, a in acc.items()}
    return deg, {c: len(a) for c, a in acc.items()}, used_years


# ----------------------------------------------------------------------- circuit


def circuit_degradation(track, target_year, max_back=MAX_HISTORY,
                        practice=False, target_practice=False):
    """Pool prior seasons at `track` and return the estimators + diagnostics.

    practice=True adds a practice-long-run estimate; target_practice=True also
    folds in the target year's own FP sessions (available from Friday of the
    race weekend) for a sharper, near-race estimate.
    """
    pool, seasons = [], []
    for y in range(target_year - 1, target_year - 1 - max_back, -1):
        try:
            s = load_race(track, y)
        except Exception:  # noqa: BLE001  (race may not exist that year)
            continue
        if s.laps is None or len(s.laps) == 0:
            continue
        g = green_laps(clean_laps(s))
        if not len(g):
            continue
        g["year"] = y
        pool.append(g)
        seasons.append(y)
    if not pool:
        return None
    allg = pd.concat(pool, ignore_index=True)
    fuel, ols = pooled_ols(allg)
    stint, stint_n = stint_slopes(allg, fuel)
    laps_per = {c: int((allg["Compound"] == c).sum()) for c in SLOTS}
    res = {
        "seasons": seasons, "fuel": fuel, "ols": ols, "stint": stint,
        "stint_n": stint_n, "laps": laps_per, "n_total": len(allg),
    }
    if practice or target_practice:
        pyears = list(range(target_year - 1, target_year - 1 - max_back, -1))
        if target_practice:
            pyears = [target_year] + pyears
        pdeg, pn, py = practice_longrun_deg(track, pyears, fuel)
        res.update({"practice": pdeg, "practice_n": pn, "practice_years": py})
        res["combined"] = combine(res)
    return res


def combine(res):
    """Recommended per-compound degradation.

    HARD is run to the end of its stints in races, so race data observes its full
    (gentle) degradation -> use the race OLS estimate. MEDIUM/SOFT are pitted before
    their cliff in races, truncating what race data can see, so use practice
    long-runs which push them further. Each falls back to the other if unavailable.
    """
    prac = res.get("practice", {})
    out = {}
    for c in SLOTS:
        race = res["ols"].get(c, res["stint"].get(c, np.nan))
        p = prac.get(c, np.nan)
        primary, backup = (race, p) if c == "HARD" else (p, race)
        out[c] = primary if np.isfinite(primary) else backup
    return out


# ------------------------------------------------------------------------ report


def print_circuit(label, res):
    has_p = "practice" in res
    print(f"\n{label}  |  seasons {res['seasons']}  |  {res['n_total']} clean laps  "
          f"|  fuel+evo {res['fuel']:+.3f} s/lap")
    if has_p:
        print(f"  practice long runs from {res['practice_years']}")
    cols = f"  {'compound':8s} {'deg (OLS)':>10s} {'deg (stint)':>12s}"
    if has_p:
        cols += f" {'deg (prac)':>11s} {'runs':>5s} {'RECOMMEND':>10s}"
    print(cols + f" {'laps':>6s} {'stints':>7s}")
    for c in SLOTS:
        ols = res["ols"].get(c, np.nan)
        stt = res["stint"].get(c, np.nan)
        row = f"  {c:8s} {ols:>10.3f} {stt:>12.3f}"
        if has_p:
            row += (f" {res['practice'].get(c, np.nan):>11.3f} "
                    f"{res['practice_n'].get(c, 0):>5d} {res['combined'].get(c, np.nan):>10.3f}")
        print(row + f" {res['laps'][c]:>6d} {res['stint_n'].get(c, 0):>7d}")


def validate(target_year=2026, practice=False, target_practice=False):
    src = "prior seasons + practice long runs" if practice else "prior seasons"
    if target_practice:
        src += f" (incl. {target_year} FP)"
    print(f"Validating tyre-degradation model against Pirelli {target_year} "
          f"pre-race numbers\n  source: {src} (up to {MAX_HISTORY} seasons back)\n")
    methods = ["OLS", "stint"] + (["prac", "COMB"] if practice else [])
    keys = {"OLS": "ols", "stint": "stint", "prac": "practice", "COMB": "combined"}
    errs = {m: [] for m in methods}
    head = f"  {'track':11s} {'cmpd':7s} {'Pirelli':>8s}"
    for m in methods:
        head += f" {m:>8s}"
    for m in methods:
        head += f" {'err' + m:>8s}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for label, cfg in PIRELLI_2026.items():
        res = circuit_degradation(cfg["query"], target_year,
                                  practice=practice, target_practice=target_practice)
        if res is None:
            print(f"  {label:11s}  (no history found)")
            continue
        for c in SLOTS:
            ref = cfg["ref"][c]
            vals = {m: res.get(keys[m], {}).get(c, np.nan) for m in methods}
            row = f"  {label:11s} {c:7s} {ref:>8.3f}"
            for m in methods:
                row += f" {vals[m]:>8.3f}"
            for m in methods:
                e = abs(vals[m] - ref)
                if np.isfinite(e):
                    errs[m].append(e)
                row += f" {e:>8.3f}"
            print(row)
        print(f"  {'':11s} fuel+evo {res['fuel']:+.3f}  seasons {res['seasons']}")
    print("\n  mean abs error (s/lap):")
    for m in methods:
        v = errs[m]
        if v:
            print(f"    {m:6s} MAE {np.mean(v):.3f}   (max {np.max(v):.3f}, n={len(v)})")


def main():
    args = list(sys.argv[1:])
    practice = "--practice" in args or "--target-practice" in args
    target_practice = "--target-practice" in args
    args = [a for a in args if a not in ("--practice", "--target-practice")]
    if not args or args[0].lstrip("-").lower() == "validate":
        validate(practice=practice, target_practice=target_practice)
        return
    track = args[0]
    year, max_back = 2026, MAX_HISTORY
    rest = args[1:]
    i = 0
    while i < len(rest):
        if rest[i] == "--seasons" and i + 1 < len(rest):
            max_back = int(rest[i + 1]); i += 2
        else:
            year = int(rest[i]); i += 1
    res = circuit_degradation(track, year, max_back, practice, target_practice)
    if res is None:
        raise SystemExit(f"No history found for '{track}' before {year}.")
    print_circuit(f"{track.title()} {year}", res)


if __name__ == "__main__":
    main()
