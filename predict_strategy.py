#!/usr/bin/env python3
"""Predict the optimal race strategy for an UPCOMING (or any) race.

Standalone version of the methodology developed in
formation_lap/analysis/strategy_methodology.ipynb, calibrated toward the
official Pirelli pre-race numbers.

It builds a per-compound lap-time model from the circuit's history (seasons
strictly *before* the target year — so there is no look-ahead and it works for
races that haven't happened yet), then simulates every sensible 1- and 2-stop
strategy and prints them ranked by modelled race time.

Pipeline (per circuit, pooled over recent seasons):
  1. fuel_rate is *estimated* per circuit (joint regression), not hardcoded;
  2. degradation is read from the PUSHING-pace envelope (fastest laps per tyre
     age) so tyre-management doesn't flatten it;
  3. a per-compound CALIBRATION (learned from Pirelli reference numbers) corrects
     the residual gap — race telemetry structurally under-reads degradation,
     worst for soft, whose cliff teams never expose (they pit it off early);
  4. compound pace offsets blend the measured fresh-lap gap with a softness ladder.

Usage:
    python predict_strategy.py "Spain" 2026     # predict an upcoming race
    python predict_strategy.py "Bahrain"        # year defaults to current
    python predict_strategy.py "Monza" 2024     # also a valid pre-race backtest
    python predict_strategy.py --calibrate      # recompute DEG_CALIBRATION from refs

Limitations: pure pace model — no safety cars, weather, or track position. The
calibration is only as good as the Pirelli references on file (see PIRELLI_REFERENCE).
"""

import sys
import os
from datetime import date
from itertools import product
from collections import defaultdict

import numpy as np
import pandas as pd
import fastf1 as f1

f1.Cache.enable_cache(os.environ.get("FASTF1_CACHE_DIR", "./f1_cache"))
try:
    f1.set_log_level("ERROR")  # quiet FastF1's load/network chatter
except Exception:  # noqa: BLE001
    pass

FUEL_PRIOR = -0.055         # fallback if the per-circuit fuel fit is unstable
FUEL_BOUNDS = (-0.12, -0.03)  # clamp the estimated fuel rate to a sane range
PUSH_QUANTILE = 0.35        # keep the fastest 35% of laps per tyre-age (pushing pace)
LADDER_STEP = 0.7           # s of fresh pace per compound step (softer = faster), prior
OFFSET_PRIOR_WEIGHT = 0.6   # how much the offset leans on the ladder vs measured gap
MIN_FIT_LAPS = 10           # min clean laps to fit a compound's degradation in one race
MIN_ELIGIBLE_LAPS = 40      # min pooled clean laps for a compound to be *recommendable*
MIN_STINT = 6               # min laps per stint (avoids degenerate token stints)
MAX_HISTORY = 4             # how many prior seasons to pool

# (hard, medium, soft) Pirelli nominations. UNVERIFIED — extend/correct as needed.
# Keyed by (year, lowercased track substring). Filling this in upgrades a track
# from relative-slot pooling to actual-Cx pooling.
CX_NOMINATIONS = {
    (2024, "bahrain"): ("C1", "C2", "C3"),
    (2024, "miami"): ("C2", "C3", "C4"),
    (2025, "miami"): ("C3", "C4", "C5"),
    (2026, "miami"): ("C3", "C4", "C5"),
    (2025, "italy"): ("C3", "C4", "C5"),
    (2025, "monza"): ("C3", "C4", "C5"),
    (2023, "spain"): ("C1", "C2", "C3"),
    (2024, "spain"): ("C1", "C2", "C3"),
    (2025, "spain"): ("C1", "C2", "C3"),
    (2026, "spain"): ("C2", "C3", "C4"),
}

# Official Pirelli pre-race degradation (s/lap) per relative compound, for
# calibration. Add a row each race weekend to sharpen DEG_CALIBRATION.
PIRELLI_REFERENCE = {
    (2026, "spain"): {"HARD": 0.10, "MEDIUM": 0.18, "SOFT": 0.25},
}

# Per-slot degradation calibration multipliers, learned from PIRELLI_REFERENCE
# via `--calibrate`. >1 because race telemetry under-reads degradation; the gap
# grows with softness (softer = more of its life unobserved). Re-run --calibrate
# after editing PIRELLI_REFERENCE and paste the result here.
DEG_CALIBRATION = {"HARD": 1.47, "MEDIUM": 2.23, "SOFT": 2.58}

_REL = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}  # index into a (hard, med, soft) tuple
_SOFTNESS = {"HARD": 1, "MEDIUM": 2, "SOFT": 3}  # fallback ladder when Cx unknown


# --------------------------------------------------------------------------- data


def load_race(track, year):
    s = f1.get_session(year, track, "R")
    s.load(laps=True, telemetry=False, weather=False, messages=False)
    return s


def clean_laps(session):
    """Repair FastF1's missing-compound issue (stored as the literal string 'nan')."""
    d = session.laps.copy()
    d["Compound"] = d["Compound"].replace("nan", np.nan)
    d = d.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    d["_pp"] = (
        d.groupby("Driver")["PitInTime"].shift().notna()
    )  # prev lap pitted -> new stint
    d["stint_id"] = d.groupby("Driver")["_pp"].cumsum()
    d["Compound"] = d.groupby(["Driver", "stint_id"])["Compound"].transform(
        lambda x: x.fillna(x.dropna().mode().iloc[0]) if x.notna().any() else x
    )
    for sid in d["stint_id"].unique():
        known = d.loc[d["stint_id"] == sid, "Compound"].dropna()
        fillc = known.mode().iloc[0] if len(known) else "MEDIUM"
        d.loc[(d["stint_id"] == sid) & (d["Compound"].isna()), "Compound"] = fillc
    return d


def green_flying(laps):
    d = laps.copy()
    d["LapTime_s"] = d["LapTime"].dt.total_seconds()
    return d[
        d["LapTime_s"].notna()
        & d["PitInTime"].isna()
        & d["PitOutTime"].isna()
        & ~d["TrackStatus"].astype(str).str.contains("[4567]", regex=True)
        & (d["TyreLife"] > 1)
    ]


# ----------------------------------------------------------------------- compounds


def nominations(track, year):
    key = (year, track.lower())
    if key in CX_NOMINATIONS:
        return CX_NOMINATIONS[key]
    for (y, t), nom in CX_NOMINATIONS.items():  # loose substring match
        if y == year and t in track.lower():
            return nom
    return None


def cx_label(slot, cx):
    return cx if cx else slot


def softness(slot, cx):
    """Higher = softer/faster-fresh. Cx number when known, else relative ladder."""
    if cx:
        return int(cx[1:])
    return _SOFTNESS.get(slot)


# ----------------------------------------------------------------------- model fit


def estimate_fuel_rate(pool):
    """Per-circuit fuel/track-evo rate from the pooled green laps.

    Joint regression with per-season intercepts (so car evolution doesn't leak
    into the fuel term): LapTime ~ season_dummies + fuel*LapNumber + sum_c deg_c*age.
    Clamped to a sane range; falls back to FUEL_PRIOR if degenerate.
    """
    g = pool.dropna(subset=["LapNumber", "TyreLife", "Compound", "LapTime_s"])
    if len(g) < 50:
        return FUEL_PRIOR
    years = sorted(g["year"].unique())
    cols = [(g["year"] == y).astype(float).values for y in years]  # season intercepts
    cols.append(g["LapNumber"].astype(float).values)  # fuel
    for c in ("HARD", "MEDIUM", "SOFT"):
        cols.append(((g["Compound"] == c) * g["TyreLife"]).values)
    X = np.column_stack(cols)
    try:
        beta, *_ = np.linalg.lstsq(X, g["LapTime_s"].values, rcond=None)
        fuel = float(beta[len(years)])
    except Exception:  # noqa: BLE001
        return FUEL_PRIOR
    if not np.isfinite(fuel):
        return FUEL_PRIOR
    return float(np.clip(fuel, *FUEL_BOUNDS))


def fit_deg(green, fuel_rate, push_quantile=PUSH_QUANTILE, min_laps=MIN_FIT_LAPS):
    """Per-compound degradation from the pushing-pace envelope of one race.

    `green` is the cleaned green-flag laps for one session. For each tyre age we
    keep only the fastest `push_quantile` of fuel-corrected laps (stripping tyre
    management), then fit a line. Returns {slot: {deg, base, n}}.
    """
    g = green.dropna(subset=["LapNumber", "TyreLife", "Compound", "LapTime_s"]).copy()
    g["fc"] = g["LapTime_s"] - fuel_rate * g["LapNumber"]  # normalise to lap-0 fuel
    out = {}
    for c, gc in g.groupby("Compound"):
        if c not in _REL or len(gc) < min_laps:
            continue
        d = pd.DataFrame({"x": gc["TyreLife"].astype(float).values, "y": gc["fc"].values})
        kept = [sub[sub["y"] <= sub["y"].quantile(push_quantile)]
                for _, sub in d.groupby("x") if len(sub) >= 2]
        dd = pd.concat(kept) if kept else d
        if len(dd) < min_laps:
            continue
        deg, base = np.polyfit(dd["x"].values, dd["y"].values, 1)
        out[c] = {"deg": deg, "base": base, "n": len(gc)}
    return out


def pit_loss(session):
    d = clean_laps(session).copy()
    d["LapTime_s"] = d["LapTime"].dt.total_seconds()
    d["ts"] = d["TrackStatus"].astype(str)
    losses = []
    for _, dd in d.groupby("Driver"):
        dd = dd.sort_values("LapNumber")
        base = green_flying(dd)["LapTime_s"].median()
        if not np.isfinite(base):
            continue
        for lap in dd.loc[dd["PitInTime"].notna(), "LapNumber"]:
            ri = dd[dd["LapNumber"] == lap]
            ro = dd[dd["LapNumber"] == lap + 1]
            if not len(ri) or not len(ro):
                continue
            if (
                ri["ts"].str.contains("[4567]").any()
                or ro["ts"].str.contains("[4567]").any()
            ):
                continue
            il, ol = ri["LapTime_s"].iloc[0], ro["LapTime_s"].iloc[0]
            if np.isfinite(il) and np.isfinite(ol):
                losses.append((il - base) + (ol - base))
    losses = np.array([x for x in losses if np.isfinite(x)])
    if not len(losses):
        return np.nan
    return float(np.median(losses[losses < np.median(losses) * 1.3]))


# ------------------------------------------------------------------ circuit history


def build_circuit_model(track, target_year, max_back=MAX_HISTORY):
    """Pool prior seasons at the circuit (strictly before target_year).

    Returns (observations, pit_losses, n_laps, seasons_used, event_name, fuel_rate).
    Each observation is one compound from one prior race:
        {slot, cx, deg, gap (fresh pace vs HARD that race), n, year}
    """
    per_season, pool, pit_losses, lap_counts, seasons, event_name = [], [], [], [], [], None
    for y in range(target_year - 1, target_year - 1 - max_back, -1):
        try:
            s = load_race(track, y)
        except Exception:  # noqa: BLE001  (race may not exist / not be cached)
            continue
        if s.laps is None or len(s.laps) == 0:
            continue
        g = green_flying(clean_laps(s)).copy()
        g["year"] = y
        per_season.append((y, g))
        pool.append(g)
        seasons.append(y)
        if event_name is None:
            event_name = s.event["EventName"]
        lap_counts.append(int(clean_laps(s)["LapNumber"].max()))
        pl = pit_loss(s)
        if np.isfinite(pl):
            pit_losses.append(pl)
    if not seasons:
        return [], [], None, [], None, FUEL_PRIOR

    fuel_rate = estimate_fuel_rate(pd.concat(pool, ignore_index=True))
    obs = []
    for y, g in per_season:
        model = fit_deg(g, fuel_rate)
        nom = nominations(track, y)
        hard_base = model.get("HARD", {}).get("base", np.nan)
        for slot, v in model.items():
            gap = (
                v["base"] - hard_base
                if np.isfinite(hard_base) and np.isfinite(v["base"])
                else np.nan
            )
            obs.append({"slot": slot, "cx": (nom[_REL[slot]] if nom else None),
                        "deg": v["deg"], "gap": gap, "n": v["n"], "year": y})
    n_laps = max(lap_counts) if lap_counts else None
    return obs, pit_losses, n_laps, seasons, event_name, fuel_rate


def pool_for_target(obs, track, target_year, apply_cal=True):
    """Pool observations onto the upcoming race's compounds (Cx if known, else slot).

    Degradation is calibrated per slot (DEG_CALIBRATION) unless apply_cal=False.
    """
    nom = nominations(track, target_year)
    compounds = {}
    for slot in ("HARD", "MEDIUM", "SOFT"):
        target_cx = nom[_REL[slot]] if nom else None
        sel = [o for o in obs if target_cx and o["cx"] == target_cx]  # exact-Cx pool
        source = "Cx " + target_cx if sel else "slot"
        if not sel:
            sel = [o for o in obs if o["slot"] == slot]  # slot fallback
        if not sel:
            continue
        n_tot = sum(o["n"] for o in sel)
        raw_deg = sum(o["deg"] * o["n"] for o in sel) / n_tot
        cal = DEG_CALIBRATION.get(slot, 1.0) if apply_cal else 1.0
        gaps = [(o["gap"], o["n"]) for o in sel if np.isfinite(o["gap"])]
        gap = (
            (sum(g * n for g, n in gaps) / sum(n for _, n in gaps)) if gaps else np.nan
        )
        compounds[slot] = {
            "deg": raw_deg * cal,
            "raw_deg": raw_deg,
            "cal": cal,
            "gap": gap,
            "n": n_tot,
            "cx": target_cx,
            "source": source,
        }
    return compounds, nom


# ----------------------------------------------------------------------- simulator


def stint_cost(slot, length, model, offsets, a0=1):
    ages = np.arange(a0, a0 + length)
    return length * offsets[slot] + model[slot]["deg"] * ages.sum()


def race_time_rel(seq, pit_laps, n_laps, model, offsets, pl):
    bounds = [0] + list(pit_laps) + [n_laps]
    total = 0.0
    for i, slot in enumerate(seq):
        length = bounds[i + 1] - bounds[i]
        if length <= 0:
            return np.inf
        total += stint_cost(slot, length, model, offsets)
    return total + (len(seq) - 1) * pl


def best_pitlaps(seq, n_laps, model, offsets, pl):
    nstop = len(seq) - 1
    if nstop == 0:
        return (), race_time_rel(seq, (), n_laps, model, offsets, pl)
    best = (None, np.inf)
    rng = range(MIN_STINT, n_laps - MIN_STINT + 1)
    if nstop == 1:
        for p in rng:
            t = race_time_rel(seq, (p,), n_laps, model, offsets, pl)
            if t < best[1]:
                best = ((p,), t)
    else:
        for p1 in rng:
            for p2 in range(p1 + MIN_STINT, n_laps - MIN_STINT + 1):
                t = race_time_rel(seq, (p1, p2), n_laps, model, offsets, pl)
                if t < best[1]:
                    best = ((p1, p2), t)
    return best


def candidate_sequences(slots, max_stops=2):
    seqs = []
    for nstop in range(1, max_stops + 1):
        for seq in product(slots, repeat=nstop + 1):
            if len(set(seq)) < 2:  # dry two-compound rule
                continue
            if any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
                continue
            seqs.append(seq)
    return seqs


def compute_offsets(compounds, slots):
    """Blend the measured fresh-lap gap with the softness ladder; fastest = 0."""
    if not slots:
        return {}
    sof = {s: softness(s, compounds[s]["cx"]) or _SOFTNESS[s] for s in slots}
    ref = max(slots, key=lambda s: sof[s])
    prior = {s: (sof[ref] - sof[s]) * LADDER_STEP for s in slots}
    finite = [compounds[s]["gap"] for s in slots if np.isfinite(compounds[s]["gap"])]
    if finite:
        base = min(finite)
        meas = {s: (compounds[s]["gap"] - base if np.isfinite(compounds[s]["gap"])
                    else prior[s]) for s in slots}
    else:
        meas = dict(prior)
    return {s: OFFSET_PRIOR_WEIGHT * prior[s] + (1 - OFFSET_PRIOR_WEIGHT) * meas[s]
            for s in slots}


# --------------------------------------------------------------------- calibration


def recompute_calibration():
    """Learn per-slot deg multipliers from PIRELLI_REFERENCE."""
    acc = defaultdict(list)
    for (year, track), ref in PIRELLI_REFERENCE.items():
        obs, _, _, seasons, _, _ = build_circuit_model(track, year)
        if not seasons:
            print(f"  (skipped {track} {year}: no history)")
            continue
        comp, _ = pool_for_target(obs, track, year, apply_cal=False)
        for slot, target in ref.items():
            if slot in comp and comp[slot]["raw_deg"] > 0:
                acc[slot].append(target / comp[slot]["raw_deg"])
    mult = {s: (float(np.mean(v)) if v else 1.0) for s in ("HARD", "MEDIUM", "SOFT")
            for v in [acc.get(s, [])]}
    return mult, {s: len(acc.get(s, [])) for s in ("HARD", "MEDIUM", "SOFT")}


# --------------------------------------------------------------------------- report


def run_calibration():
    print(f"Recomputing DEG_CALIBRATION from {len(PIRELLI_REFERENCE)} reference race(s)...")
    mult, counts = recompute_calibration()
    print("\nPaste this into the DEG_CALIBRATION constant:\n")
    print("DEG_CALIBRATION = {")
    for s in ("HARD", "MEDIUM", "SOFT"):
        print(f'    "{s}": {mult[s]:.2f},   # mean of {counts[s]} reference race(s)')
    print("}")
    if max(counts.values()) < 2:
        print("\nNOTE: only one reference race — these multipliers are an exact fit to "
              "it and may not generalise. Add more PIRELLI_REFERENCE rows.")


def predict(track, year):
    obs, pit_losses, n_laps, seasons, hist_name, fuel = build_circuit_model(track, year)
    if not seasons:
        raise SystemExit(
            f"No historical races found for '{track}' before {year} "
            f"(tried {year - 1}..{year - MAX_HISTORY}). Check the track name, the cache, "
            f"or your network connection."
        )
    pl = float(np.median(pit_losses)) if pit_losses else np.nan
    if not np.isfinite(pl) or n_laps is None:
        raise SystemExit("Not enough clean green-flag history to model this circuit.")

    event_name = hist_name
    try:
        event_name = f1.get_event(year, track)["EventName"]
    except Exception:  # noqa: BLE001
        pass

    compounds, nom = pool_for_target(obs, track, year)
    eligible = [s for s, v in compounds.items()
                if v["n"] >= MIN_ELIGIBLE_LAPS and np.isfinite(v["deg"])]
    offsets = compute_offsets(compounds, eligible)
    calibrated = any(abs(v["cal"] - 1.0) > 1e-9 for v in compounds.values())

    print(f"\n{event_name} {year}  —  predicted strategy")
    print(f"model pooled from {track.title()} {seasons} | ~{n_laps} laps | "
          f"pit loss ~ {pl:.1f}s | fuel ~ {fuel:+.3f}s/lap"
          f" | calibration {'ON' if calibrated else 'OFF'}")
    if not nom:
        print(f"(no Cx nominations on file for {year} — pooling by tyre slot and "
              f"showing relative compounds; add a CX_NOMINATIONS row to refine)")
    if not calibrated:
        print("(DEG_CALIBRATION is all 1.0 — run `--calibrate` to fit it to Pirelli refs)")

    print("\nPer-compound model (pushing-pace, calibrated):")
    print(f"  {'compound':9s} {'shown':>6s} {'deg s/lap':>10s} {'cal×':>5s} "
          f"{'fresh Δ':>8s} {'laps':>6s} {'from':>9s} {'use?':>5s}")
    for slot in ("HARD", "MEDIUM", "SOFT"):
        if slot not in compounds:
            continue
        v = compounds[slot]
        off = offsets.get(slot, np.nan)
        off_s = f"{off:+.2f}" if np.isfinite(off) else "  -"
        print(f"  {slot:9s} {cx_label(slot, v['cx']):>6s} {v['deg']:+10.3f} "
              f"{v['cal']:>5.2f} {off_s:>8s} {v['n']:6d} {v['source']:>9s} "
              f"{'yes' if slot in eligible else 'no':>5s}")

    if len(eligible) < 2:
        print("\nFewer than two well-sampled dry compounds in this circuit's history "
              "— can't build a two-compound strategy.")
        return

    rows = []
    for seq in candidate_sequences(eligible):
        plaps, t = best_pitlaps(seq, n_laps, compounds, offsets, pl)
        rows.append((seq, plaps, t))
    rows.sort(key=lambda r: r[2])
    best = rows[0][2]

    print("\nPredicted strategies (best first):")
    print(f"  {'#':>2s}  {'stops':>5s}  {'compounds':18s} {'pit lap(s)':>12s} {'Δ vs best':>10s}")
    for i, (seq, plaps, t) in enumerate(rows[:6], 1):
        shown = "→".join(cx_label(s, compounds[s]["cx"]) for s in seq)
        rel = "/".join(s[0] for s in seq)
        pits = ",".join(map(str, plaps)) if plaps else "-"
        tag = "  <-- recommended" if i == 1 else ""
        print(f"  {i:>2d}  {len(seq) - 1:>5d}  {shown:8s} ({rel:7s}) {pits:>12s} "
              f"{t - best:>9.1f}s{tag}")
    print()


def main():
    if len(sys.argv) >= 2 and sys.argv[1].lstrip("-").lower() == "calibrate":
        run_calibration()
        return
    if len(sys.argv) < 2:
        print('Usage: python predict_strategy.py "<track>" [year]   (or --calibrate)')
        raise SystemExit(1)
    track = sys.argv[1]
    year = int(sys.argv[2]) if len(sys.argv) > 2 else date.today().year
    predict(track, year)


if __name__ == "__main__":
    main()
