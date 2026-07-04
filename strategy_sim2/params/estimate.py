"""Fit and bundle all historical parameters into a single ParameterSet.

The ParameterSet is what the simulator and context builders consume. It is cached to
disk (pickle) so repeated runs don't refit; pass ``rebuild=True`` after new data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pickle

from strategy_sim2.params import dataset
from strategy_sim2.params.dnf import DNFModel, fit_dnf
from strategy_sim2.params.lapmodel import LapModel, fit_lap_model
from strategy_sim2.params.startline import StartModel, fit_start
from strategy_sim2.settings import load_settings, resolve_path


@dataclass
class ParameterSet:
    lap: LapModel
    dnf: DNFModel
    start: StartModel
    n_laps_rows: int = 0
    n_result_rows: int = 0


def _cache_path(cfg: dict) -> Path:
    return resolve_path(cfg["data"]["derived_dir"]) / "paramset.pkl"


def fit_all(cfg: dict | None = None, use_cache: bool = True, rebuild: bool = False,
            years: list[int] | None = None,
            before: tuple[int, int] | None = None) -> ParameterSet:
    """Fit all parameters. ``years`` restricts to seasons; ``before=(year, round)``
    applies the expanding-window cutoff (train only on races strictly before the
    target). Restricted fits are never cached."""
    cfg = cfg or load_settings()
    path = _cache_path(cfg)
    restricted = years is not None or before is not None
    if not restricted and use_cache and not rebuild and path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    laps = dataset.training_laps(cfg, years=years, before=before)
    results = dataset.training_results(cfg, years=years, before=before)
    lap1 = dataset.training_lap1(cfg, years=years, before=before)
    if not len(laps) or not len(results):
        raise ValueError("no training data available; build the manifest/cache first")

    ps = ParameterSet(
        lap=fit_lap_model(laps, cfg),
        dnf=fit_dnf(results, cfg),
        start=fit_start(results, lap1, cfg),
        n_laps_rows=int(len(laps)),
        n_result_rows=int(len(results)),
    )
    if not restricted:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(ps, f)
    return ps
