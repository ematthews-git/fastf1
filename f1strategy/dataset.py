"""Build (and cache) the per-race inputs+targets the calibrator consumes.

A :class:`RaceCase` bundles the measured :class:`TrackContext` (race-only tyre
inputs — always available, no look-ahead) with the :class:`RaceObservation`
(the front-runner reference strategy = the label). FastF1 I/O is the expensive
part, so a whole dataset is built once and pickled; Optuna trials then run pure
CPU over the cached cases.
"""

from __future__ import annotations

import os
import json
import pickle
import hashlib
from dataclasses import dataclass
from typing import Callable, Optional

from .config import TrackContext
from .data.loaders import load_race, event_name
from .data.laps import clean_laps, race_lap_count
from .tyre import build_tyre_model
from .pitloss import pit_loss
from .overtaking import overtake_ease
from .field import field_from_session
from .observations import race_observation, RaceObservation

CACHE_DIR = ".f1strategy_cache"

# Default calibration universe: dry, ground-effect-era (2022-2025) races across a
# spread of circuits with varied degradation / SC / track-position character.
# Names chosen to resolve unambiguously in FastF1 (see the 'Great Britain' trap).
# Non-representative entries (wet / red-flag) are dropped automatically downstream.
# Targets are 2023-2025 so tyre history stays inside the ground-effect era
# (the 2022 floor means a 2022 target would have no same-era history).
DEFAULT_RACES: list[tuple[int, str]] = [
    (2023, "Spain"), (2024, "Spain"), (2025, "Spain"),
    (2023, "Italy"), (2024, "Italy"), (2025, "Italy"),
    (2023, "Austria"), (2024, "Austria"), (2025, "Austria"),
    (2023, "Silverstone"), (2025, "Silverstone"),
    (2023, "Miami"), (2024, "Miami"), (2025, "Miami"),
    (2023, "Canada"), (2024, "Canada"), (2025, "Canada"),
    (2024, "Belgium"), (2025, "Belgium"),
    (2024, "Bahrain"), (2025, "Australia"),
    # low overtaking-ease circuits, where track position / traffic drives strategy
    (2023, "Hungary"), (2024, "Hungary"), (2025, "Hungary"),
    (2024, "Netherlands"), (2023, "Monaco"),
]


@dataclass
class RaceCase:
    ctx: TrackContext
    obs: RaceObservation

    @property
    def track(self) -> str:
        return self.ctx.track

    @property
    def usable(self) -> bool:
        """Has a reference strategy and enough compounds to build one."""
        return (self.obs.representative and self.obs.weight > 0
                and len(self.ctx.compounds) >= 2)


def expected_lap_count(track: str, year: int, loader: Callable) -> Optional[int]:
    """Scheduled race distance from the most recent prior season (no look-ahead)."""
    for y in range(year - 1, year - 5, -1):
        try:
            s = loader(y, track)
        except Exception:  # noqa: BLE001
            continue
        if getattr(s, "laps", None) is not None and len(s.laps):
            return race_lap_count(clean_laps(s))
    return None


def build_case(year: int, track: str, seasons_back: int = 3,
               use_practice: bool = False, loader: Callable = load_race) -> Optional[RaceCase]:
    """Build one RaceCase, or None if the circuit has too little history."""
    tm = build_tyre_model(track, year, seasons_back=seasons_back, use_practice=use_practice)
    if len(tm["compounds"]) < 2:
        return None
    race = loader(year, track)               # target race: the label + true distance
    obs = race_observation(race, track, year)
    n_laps = obs.n_laps or expected_lap_count(track, year, loader)
    if not n_laps:
        return None
    ctx = TrackContext(
        track=track, year=year, event_name=event_name(year, track, fallback=obs.event_name),
        n_laps=n_laps, pit_loss=pit_loss(track), fuel_rate=tm["fuel_rate"],
        compounds=tm["compounds"], overtake_ease=overtake_ease(track),
        seasons_used=tuple(tm["seasons_used"]), notes=f"tyre sources: {tm['sources']}",
    )
    # Attach the real field, focal = the top reference front-runner (its actual grid).
    drv = obs.drivers
    if drv is not None and len(drv):
        elig = drv[drv["eligible"]]
        if len(elig):
            focal = elig.sort_values("freedom", ascending=False).iloc[0]["Driver"]
            ctx.field = field_from_session(race, ctx, focal)
    return RaceCase(ctx=ctx, obs=obs)


def _cache_key(races, seasons_back, use_practice) -> str:
    payload = json.dumps({"races": races, "sb": seasons_back, "prac": use_practice},
                         sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def build_dataset(races: Optional[list] = None, seasons_back: int = 3,
                  use_practice: bool = False, cache: bool = True,
                  verbose: bool = True) -> list[RaceCase]:
    """Build (and pickle-cache) RaceCases for a list of (year, track)."""
    races = races or DEFAULT_RACES
    path = None
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f"dataset_{_cache_key(races, seasons_back, use_practice)}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                cases = pickle.load(f)
            if verbose:
                print(f"[dataset] loaded {len(cases)} cached cases from {path}")
            return cases

    cases: list[RaceCase] = []
    for year, track in races:
        try:
            case = build_case(year, track, seasons_back, use_practice)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[dataset] {track} {year}: FAILED ({type(e).__name__}: {e})")
            continue
        if case is None:
            if verbose:
                print(f"[dataset] {track} {year}: skipped (insufficient history)")
            continue
        cases.append(case)
        if verbose:
            o = case.obs
            tag = o.label() if o.representative else f"excluded: {o.reason}"
            print(f"[dataset] {case.ctx.event_name:26s} {year}: {tag}  "
                  f"[{'USE' if case.usable else 'skip'}]")
    if cache and path:
        with open(path, "wb") as f:
            pickle.dump(cases, f)
        if verbose:
            print(f"[dataset] cached {len(cases)} cases -> {path}")
    return cases
