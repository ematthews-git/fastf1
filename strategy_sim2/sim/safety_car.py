"""Probabilistic safety-car / VSC generation for a single race instance.

Full SC and VSC are modelled separately (they differ strategically): a full SC
bunches the field and makes a pit stop much cheaper, while a VSC only slows the pace.
Occurrence and duration are drawn from the circuit's historical rates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GREEN, VSC, SC = 0, 1, 2


@dataclass
class SafetyCarPlan:
    status: np.ndarray  # per-lap status code (len = n_laps): GREEN / VSC / SC

    def is_sc(self, lap: int) -> bool:
        return 0 < lap <= len(self.status) and self.status[lap - 1] == SC

    def is_vsc(self, lap: int) -> bool:
        return 0 < lap <= len(self.status) and self.status[lap - 1] == VSC

    def any_neutralised(self, lap: int) -> bool:
        return self.is_sc(lap) or self.is_vsc(lap)


def sample_plan(profile, n_laps: int, cfg: dict, rng: np.random.Generator) -> SafetyCarPlan:
    sc = cfg["safety_car"]
    status = np.zeros(n_laps, dtype=int)

    def place(kind: int, lo: int, hi: int):
        dur = int(rng.integers(lo, hi + 1))
        # SCs rarely fall on lap 1; allow anywhere from lap 1..n-1 otherwise.
        start = int(rng.integers(1, max(2, n_laps - dur)))
        status[start - 1: start - 1 + dur] = np.where(
            status[start - 1: start - 1 + dur] == GREEN, kind,
            status[start - 1: start - 1 + dur])

    if rng.random() < float(profile.sc_prob):
        place(SC, int(sc["sc_min_laps"]), int(sc["sc_max_laps"]))
    if rng.random() < float(profile.vsc_prob):
        place(VSC, int(sc["vsc_min_laps"]), int(sc["vsc_max_laps"]))
    return SafetyCarPlan(status=status)


def pace_factor(plan: SafetyCarPlan, lap: int, cfg: dict) -> float:
    sc = cfg["safety_car"]
    if plan.is_sc(lap):
        return float(sc["sc_factor"])
    if plan.is_vsc(lap):
        return float(sc["vsc_factor"])
    return 1.0
