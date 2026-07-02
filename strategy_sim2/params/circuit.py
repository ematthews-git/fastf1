"""Circuit profiles derived from historical dry races (nothing hardcoded).

Each profile captures the track characteristics a strategist reasons about:
race length, base lap time, pit-lane time loss, safety-car / VSC likelihood,
on-track overtaking difficulty, and tyre-degradation severity (from the lap model).
Circuits with thin history fall back to field-wide medians. Cached to JSON.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from strategy_sim2.data import clean, session_filter
from strategy_sim2.params.lapmodel import LapModel
from strategy_sim2.settings import load_settings, resolve_path


@dataclass
class CircuitProfile:
    circuit: str
    n_laps: int
    base_lap_time: float        # median clean green lap (s)
    pit_loss: float             # time lost making a pit stop (s)
    sc_prob: float              # P(full safety car occurs in the race)
    vsc_prob: float             # P(virtual safety car occurs)
    sc_expected_laps: float     # mean SC/VSC-affected laps per race
    passes_per_race: float      # on-track position gains per race (overtaking proxy)
    overtaking_difficulty: float  # 0 easy .. 1 hard (relative across circuits)
    deg_severity: float         # mean tyre-deg slope (s/lap) from the lap model
    fuel_coef: float            # fuel effect (s/lap) from the lap model
    n_races: int
    fallback: bool = False


def _pit_loss_for_race(raw: pd.DataFrame) -> list[float]:
    """Estimate per-stop time loss = (in-lap delta) + (out-lap delta) vs green pace."""
    losses = []
    for _, g in raw.groupby("driver"):
        g = g.sort_values("lap_number")
        green_med = g.loc[g["is_clean"], "lap_time_s"].median()
        if not np.isfinite(green_med):
            continue
        inlaps = g[g["is_inlap"] & ~g["is_sc"] & ~g["is_vsc"]]
        for _, il in inlaps.iterrows():
            out = g[(g["lap_number"] == il["lap_number"] + 1) & g["is_outlap"]]
            if not len(out):
                continue
            ol = out.iloc[0]
            if ol["is_sc"] or ol["is_vsc"]:
                continue
            it, ot = il["lap_time_s"], ol["lap_time_s"]
            if pd.notna(it) and pd.notna(ot):
                loss = (it - green_med) + (ot - green_med)
                if 5.0 < loss < 60.0:  # guard against garbage
                    losses.append(float(loss))
    return losses


def _passes_for_race(raw: pd.DataFrame) -> float:
    """Sum of on-track positions gained on green racing laps (overtaking proxy)."""
    rac = raw[raw["is_green"] & ~raw["is_inlap"] & ~raw["is_outlap"]].copy()
    if not len(rac):
        return 0.0
    rac = rac.sort_values(["driver", "lap_number"])
    rac["prev_pos"] = rac.groupby("driver")["position"].shift(1)
    gained = (rac["prev_pos"] - rac["position"]).clip(lower=0)
    return float(gained.sum())


def build_circuit_profiles(lap_model: LapModel, cfg: dict | None = None,
                           save: bool = True,
                           years: list[int] | None = None) -> dict[str, CircuitProfile]:
    cfg = cfg or load_settings()
    races = session_filter.included_races(cfg)
    if years is not None:
        races = races[races["year"].isin(years)]

    per_circuit: dict[str, dict] = {}
    for _, r in races.iterrows():
        raw = clean.get_clean_race(int(r["year"]), int(r["round"]), cfg)
        if raw is None or not len(raw):
            continue
        circ = str(r["circuit"])
        d = per_circuit.setdefault(circ, {"n_laps": [], "base": [], "pit": [],
                                          "sc": [], "vsc": [], "sc_laps": [], "passes": []})
        d["n_laps"].append(int(raw["total_laps"].iloc[0]))
        d["base"].append(float(raw.loc[raw["is_clean"], "lap_time_s"].median()))
        d["pit"].extend(_pit_loss_for_race(raw))
        d["sc"].append(bool(raw["is_sc"].any()))
        d["vsc"].append(bool(raw["is_vsc"].any()))
        d["sc_laps"].append(int((raw.drop_duplicates("lap_number")["is_sc"]
                                 | raw.drop_duplicates("lap_number")["is_vsc"]).sum()))
        d["passes"].append(_passes_for_race(raw))

    # Field-wide fallbacks.
    all_pit = [p for d in per_circuit.values() for p in d["pit"]]
    global_pit = float(np.median(all_pit)) if all_pit else 22.0
    global_base = float(np.median([np.median(d["base"]) for d in per_circuit.values()])) if per_circuit else 90.0

    profiles: dict[str, CircuitProfile] = {}
    passes_vals = {c: float(np.mean(d["passes"])) for c, d in per_circuit.items() if d["passes"]}
    pmin = min(passes_vals.values()) if passes_vals else 0.0
    pmax = max(passes_vals.values()) if passes_vals else 1.0

    for circ, d in per_circuit.items():
        passes = float(np.mean(d["passes"])) if d["passes"] else 0.0
        difficulty = 1.0 - (passes - pmin) / (pmax - pmin) if pmax > pmin else 0.5
        profiles[circ] = CircuitProfile(
            circuit=circ,
            n_laps=int(np.median(d["n_laps"])),
            base_lap_time=float(np.median(d["base"])),
            pit_loss=float(np.median(d["pit"])) if d["pit"] else global_pit,
            sc_prob=float(np.mean(d["sc"])),
            vsc_prob=float(np.mean(d["vsc"])),
            sc_expected_laps=float(np.mean(d["sc_laps"])),
            passes_per_race=passes,
            overtaking_difficulty=float(np.clip(difficulty, 0.0, 1.0)),
            deg_severity=float(lap_model.deg_severity(circ)),
            fuel_coef=float(lap_model.fuel_coef(circ)),
            n_races=len(d["n_laps"]),
        )

    if save and profiles:
        path = resolve_path(cfg["data"]["circuit_profiles_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({c: asdict(p) for c, p in profiles.items()}, f, indent=2)
    return profiles


def load_circuit_profiles(cfg: dict | None = None) -> dict[str, CircuitProfile]:
    cfg = cfg or load_settings()
    path = resolve_path(cfg["data"]["circuit_profiles_path"])
    with open(path) as f:
        raw = json.load(f)
    return {c: CircuitProfile(**d) for c, d in raw.items()}


def get_profile(circuit: str, profiles: dict[str, CircuitProfile],
                lap_model: LapModel | None = None) -> CircuitProfile:
    """Return the circuit's profile, or a field-median fallback profile."""
    if circuit in profiles:
        return profiles[circuit]
    vals = list(profiles.values())
    med = lambda attr: float(np.median([getattr(p, attr) for p in vals])) if vals else 0.0
    return CircuitProfile(
        circuit=circuit,
        n_laps=int(med("n_laps")) if vals else 57,
        base_lap_time=med("base_lap_time") if vals else 90.0,
        pit_loss=med("pit_loss") if vals else 22.0,
        sc_prob=med("sc_prob") if vals else 0.5,
        vsc_prob=med("vsc_prob") if vals else 0.4,
        sc_expected_laps=med("sc_expected_laps") if vals else 4.0,
        passes_per_race=med("passes_per_race") if vals else 30.0,
        overtaking_difficulty=0.5,
        deg_severity=lap_model.deg_severity(circuit) if lap_model else med("deg_severity"),
        fuel_coef=lap_model.fuel_coef(circuit) if lap_model else med("fuel_coef"),
        n_races=0, fallback=True,
    )
