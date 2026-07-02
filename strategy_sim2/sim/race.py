"""Discrete-event, lap-by-lap, full-field race simulator.

One call to :func:`simulate_race` plays out a single stochastic race: opening-lap
mixing, per-lap fuel/tyre/noise pace, pit stops, safety cars, overtaking with track
position, and retirements — returning each driver's finishing position and race time.
Randomness lives entirely in the passed ``rng`` so Monte-Carlo callers can use common
random numbers across a driver's candidate strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from strategy_sim2.params.circuit import CircuitProfile
from strategy_sim2.params.dnf import DNFModel
from strategy_sim2.params.lapmodel import LapModel
from strategy_sim2.params.startline import StartModel
from strategy_sim2.sim import safety_car
from strategy_sim2.sim.overtaking import OvertakeParams, resolve_lap
from strategy_sim2.sim.pitstops import pit_loss_sample


@dataclass
class Entry:
    driver: str
    team: str
    grid: int
    base_pace: float               # qualifying lap time (s)
    compounds: tuple[str, ...]     # compound per stint
    pit_laps: tuple[int, ...]      # planned in-laps (len = n_stops)


@dataclass
class RaceContext:
    circuit: CircuitProfile
    lap_model: LapModel
    dnf_model: DNFModel
    start_model: StartModel
    entries: list[Entry]
    cfg: dict
    # cached per-driver deg/offset closures are built lazily in simulate_race


@dataclass
class RaceResult:
    finish_position: dict[str, int]
    race_time: dict[str, float]
    classified: dict[str, bool]
    n_passes: int = 0


def _jitter_pits(planned, n_laps, rng, sd, min_stint):
    out, prev = [], 0
    for pl in planned:
        j = int(round(rng.normal(pl, sd)))
        j = max(prev + min_stint, min(j, n_laps - min_stint))
        out.append(j)
        prev = j
    return out


def _stint_schedule(pit_laps, compounds, n_laps):
    comp = [compounds[-1]] * n_laps
    age = [1] * n_laps
    is_pit = [False] * n_laps
    bounds = [0] + list(pit_laps) + [n_laps]
    for s in range(len(compounds)):
        start, end = bounds[s] + 1, bounds[min(s + 1, len(bounds) - 1)]
        for lap in range(start, end + 1):
            if 1 <= lap <= n_laps:
                comp[lap - 1] = compounds[s]
                age[lap - 1] = lap - start + 1
    for pl in pit_laps:
        if 1 <= pl <= n_laps:
            is_pit[pl - 1] = True
    return comp, age, is_pit


def simulate_race(ctx: RaceContext, rng: np.random.Generator) -> RaceResult:
    prof, cfg, lm = ctx.circuit, ctx.cfg, ctx.lap_model
    n = prof.n_laps
    ents = ctx.entries
    N = len(ents)
    sim_cfg = cfg["simulation"]
    lap1_pen = float(sim_cfg["lap1_penalty"])
    gap_scale = float(sim_cfg["start_gap_scale"])
    jitter_sd = float(sim_cfg["pit_jitter_sd"])
    min_stint = int(cfg["generation"]["min_stint"])
    fuel = lm.fuel_coef(prof.circuit)

    plan = safety_car.sample_plan(prof, n, cfg, rng)
    op = OvertakeParams.for_circuit(prof.overtaking_difficulty, cfg)

    # --- per-driver pre-race draws ---
    noise = np.array([lm.noise_std(e.driver) for e in ents])
    retire_lap = np.full(N, n + 1)
    for i, e in enumerate(ents):
        if rng.random() < ctx.dnf_model.dnf_prob(e.driver):
            retire_lap[i] = ctx.dnf_model.sample_retire_lap(n, rng)
    start_gain = np.array([ctx.start_model.sample_gain(e.driver, rng) for e in ents])
    schedules = []
    pit_cost = []  # per-driver dict lap-> loss
    for e in ents:
        pits = _jitter_pits(e.pit_laps, n, rng, jitter_sd, min_stint)
        comp, age, is_pit = _stint_schedule(pits, e.compounds, n)
        schedules.append((comp, age, is_pit))

    # --- initial order from grid ---
    order = sorted(range(N), key=lambda i: ents[i].grid)
    cum = np.zeros(N)
    within = np.zeros(N, dtype=bool)
    alive = np.ones(N, dtype=bool)
    laps_done = np.zeros(N, dtype=int)

    for lap in range(1, n + 1):
        # retirements at the start of the lap
        for i in range(N):
            if alive[i] and retire_lap[i] == lap:
                alive[i] = False
        order = [i for i in order if alive[i]]
        if not order:
            break

        factor = safety_car.pace_factor(plan, lap, cfg)
        neutralised = factor > 1.0
        drs_enabled = (lap > 2) and not neutralised

        raw = np.zeros(N)
        for i in order:
            comp, age, is_pit = schedules[i]
            c, a = comp[lap - 1], age[lap - 1]
            drive = (ents[i].base_pace + fuel * (n - lap)
                     + lm.pace_offset(c, prof.circuit)
                     + lm.deg(c, a, prof.circuit, ents[i].driver))
            if not neutralised:
                drive += noise[i] * rng.normal()
            if lap == 1:
                drive += lap1_pen - start_gain[i] * gap_scale
            drive *= factor
            if is_pit[lap - 1]:
                drive += pit_loss_sample(prof, cfg, rng,
                                         under_sc=plan.is_sc(lap), under_vsc=plan.is_vsc(lap))
            raw[i] = drive
            laps_done[i] = lap

        if lap == 1 or neutralised:
            # free mixing at the start / no passing under neutralisation:
            # order by total time (captures pit drops), then bunch under full SC.
            new_cum = cum.copy()
            for i in order:
                new_cum[i] = cum[i] + raw[i]
            order = sorted(order, key=lambda i: new_cum[i])
            if plan.is_sc(lap):
                lead = new_cum[order[0]]
                for r, i in enumerate(order):
                    new_cum[i] = lead + r * op.min_gap
            cum = new_cum
            within = np.zeros(N, dtype=bool)
        else:
            order, cum, within, _ = resolve_lap(order, cum, raw, within, op, drs_enabled, rng)

    # --- classification ---
    finishers = [i for i in range(N) if alive[i]]
    finishers.sort(key=lambda i: cum[i])
    dnfs = [i for i in range(N) if not alive[i]]
    dnfs.sort(key=lambda i: (-laps_done[i], ents[i].grid))

    finish_position, race_time, classified = {}, {}, {}
    pos = 1
    for i in finishers:
        d = ents[i].driver
        finish_position[d] = pos
        race_time[d] = float(cum[i])
        classified[d] = True
        pos += 1
    for i in dnfs:
        d = ents[i].driver
        finish_position[d] = pos
        race_time[d] = float("nan")
        classified[d] = False
        pos += 1
    return RaceResult(finish_position, race_time, classified)
