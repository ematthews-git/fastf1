"""Circuit-aware overtaking model.

Track position is sticky: a faster car only clears the car ahead if its pace advantage
beats a circuit-dependent threshold *and* it wins a circuit-dependent probability roll
(boosted by DRS when it was within range). If it fails, it is held in dirty air and
loses time. This is what makes track position, traffic, undercuts and overcuts matter
rather than clean-air pace alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OvertakeParams:
    min_gap: float
    drs_gap: float
    drs_bonus: float
    penalty: float
    pass_prob: float      # blended for this circuit's difficulty
    threshold: float      # pace advantage (s) needed to attempt, this circuit

    @classmethod
    def for_circuit(cls, difficulty: float, cfg: dict) -> "OvertakeParams":
        o = cfg["overtaking"]
        d = float(np.clip(difficulty, 0.0, 1.0))
        lerp = lambda a, b: float(a) * (1 - d) + float(b) * d
        return cls(
            min_gap=float(o["min_gap"]), drs_gap=float(o["drs_gap"]),
            drs_bonus=float(o["drs_bonus"]), penalty=float(o["penalty"]),
            pass_prob=lerp(o["pass_prob_easy"], o["pass_prob_hard"]),
            threshold=lerp(o["threshold_easy"], o["threshold_hard"]),
        )


def resolve_lap(order, cum, raw_time, within_range, p: OvertakeParams,
                drs_enabled: bool, rng: np.random.Generator):
    """Advance one lap with overtaking. Operates front-to-back on the current order.

    Args:
        order: driver indices in current track order (front first).
        cum: cumulative time array (indexed by driver id); updated in place copy.
        raw_time: clean-air lap time array (indexed by driver id).
        within_range: bool array, True if driver was within DRS range last lap.
        drs_enabled: False on the first two laps and under neutralisation.
    Returns (new_order, new_cum, new_within_range, n_passes).
    """
    new_cum = cum.copy()
    new_within = np.zeros_like(within_range)
    order = list(order)
    passes = 0

    for k in range(len(order)):
        me = order[k]
        if k == 0:
            new_cum[me] = cum[me] + raw_time[me]
            continue
        ahead = order[k - 1]
        drs = p.drs_bonus if (drs_enabled and within_range[me]) else 0.0
        tent = cum[me] + raw_time[me] - drs
        ahead_cum = new_cum[ahead]
        gap = tent - ahead_cum

        if gap >= p.min_gap:
            new_cum[me] = tent
            new_within[me] = gap <= p.drs_gap
            continue

        # Caught the car ahead this lap.
        pace_adv = ahead_cum - tent  # >0 means genuinely faster
        if pace_adv >= p.threshold and rng.random() < p.pass_prob:
            new_cum[me] = min(tent + p.penalty, ahead_cum - p.min_gap)
            new_cum[ahead] = new_cum[ahead] + p.penalty
            order[k - 1], order[k] = me, ahead  # swap track positions
            passes += 1
        else:
            new_cum[me] = ahead_cum + p.min_gap  # stuck in dirty air, loses time
            new_within[me] = True
    return order, new_cum, new_within, passes
