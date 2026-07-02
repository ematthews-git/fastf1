"""Pit-stop time loss.

The time lost making a stop is the circuit's pit loss plus small stochastic variation,
discounted when the stop is taken under a safety car or VSC (the field is slow, so
relatively little time is conceded).
"""
from __future__ import annotations

import numpy as np


def pit_loss_sample(profile, cfg: dict, rng: np.random.Generator,
                    under_sc: bool = False, under_vsc: bool = False) -> float:
    base = float(profile.pit_loss)
    loss = base + rng.normal(0.0, 0.8)  # crew/lane variation
    sc = cfg["safety_car"]
    if under_sc:
        loss *= float(sc["sc_pit_discount"])
    elif under_vsc:
        loss *= float(sc["vsc_pit_discount"])
    return float(max(loss, 5.0))
