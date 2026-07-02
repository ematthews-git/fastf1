"""Post-qualifying race context: the known facts going into a Grand Prix.

Bundles the target weekend's entry list, starting grid and base (qualifying) pace with
the historical parameter set, circuit profile and plausibility prior. This is the main
operating mode; the preliminary (previous-year) mode lives in ``context/prelim.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from strategy_sim2.data import collector
from strategy_sim2.data.schema import DRY_COMPOUNDS
from strategy_sim2.generation.plausibility import StrategyPrior, build_strategy_prior
from strategy_sim2.params import pace
from strategy_sim2.params.circuit import CircuitProfile, get_profile
from strategy_sim2.params.estimate import ParameterSet
from strategy_sim2.settings import load_settings


@dataclass
class WeekendContext:
    year: int
    round: int
    circuit: str
    profile: CircuitProfile
    params: ParameterSet
    prior: StrategyPrior
    grid: dict[str, int]
    base_pace: dict[str, float]
    teams: dict[str, str]
    allocation: tuple[str, ...]
    cfg: dict
    mode: str = "postquali"

    def drivers(self) -> list[str]:
        return sorted(self.grid, key=lambda d: self.grid[d])


def build_postquali_context(year: int, rnd: int, params: ParameterSet,
                            profiles: dict[str, CircuitProfile],
                            cfg: dict | None = None,
                            allocation: tuple[str, ...] | None = None) -> WeekendContext:
    cfg = cfg or load_settings()

    # Starting grid: prefer the actual grid (race session, incl. penalties); if the
    # race hasn't run, fall back to the qualifying classification order.
    race = collector.load_session(year, rnd, "R", weather=False)
    grid, teams, circuit = {}, {}, None
    if race is not None:
        res = collector.session_results(race)
        circuit = str(res["circuit"].iloc[0])
        for _, r in res.iterrows():
            d = str(r["driver"])
            g = int(r["grid"]) if r["grid"] and r["grid"] > 0 else len(res)
            grid[d] = g
            teams[d] = str(r["team"])
    else:
        q = collector.load_session(year, rnd, "Q", weather=False)
        res = collector.session_results(q)
        circuit = str(res["circuit"].iloc[0])
        for _, r in res.iterrows():
            d = str(r["driver"])
            grid[d] = int(r["finish_position"]) if r["finish_position"] > 0 else len(res)
            teams[d] = str(r["team"])

    base_pace = pace.quali_pace(year, rnd)
    if base_pace:
        med = float(np.median(list(base_pace.values())))
    else:
        med = get_profile(circuit, profiles, params.lap).base_lap_time
    for d in grid:  # fill any missing quali time with a back-of-grid estimate
        base_pace.setdefault(d, med + 0.8)

    profile = get_profile(circuit, profiles, params.lap)
    prior = build_strategy_prior(circuit, cfg)
    return WeekendContext(
        year=year, round=rnd, circuit=circuit, profile=profile, params=params,
        prior=prior, grid=grid, base_pace=base_pace, teams=teams,
        allocation=tuple(allocation or DRY_COMPOUNDS), cfg=cfg,
    )
