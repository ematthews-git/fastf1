"""Extract each driver's *strategic* stint sequence from race laps.

Raw stint data is noisy: red-flag stoppages and late safety cars produce clusters of
near-free tyre changes (e.g. Monaco 2026 pits on laps 58,59,65,67,68) that are not
strategy. We merge sub-minimum stints into their neighbour and collapse consecutive
same-compound stints, recovering the intended compound sequence and pit laps. Shared by
the plausibility prior and the backtest so both see the same clean picture.
"""
from __future__ import annotations

import pandas as pd

from strategy_sim2.data.schema import DRY_COMPOUNDS


def extract_driver_strategy(g: pd.DataFrame, min_stint: int = 5) -> dict | None:
    """Return {compounds, n_stops, pit_laps, family} for one driver, or None if wet.

    Real strategic stints (length >= min_stint) are kept in order. Sub-minimum stints
    (red-flag / SC free changes, 1-2 lap blips) are dropped. Two same-compound real
    stints are collapsed ONLY when a dropped stint sat between them (a red-flag artifact);
    directly-adjacent same-compound real stints are a genuine repeat-compound stop
    (e.g. MEDIUM-HARD-HARD) and are preserved.
    """
    g = g.dropna(subset=["stint"]).sort_values("lap_number")
    stints = []  # [compound, start_lap, end_lap, length]
    for _, sg in g.groupby("stint"):
        mode = sg["compound"].mode()
        c = str(mode.iloc[0]) if len(mode) else None
        stints.append([c, int(sg["lap_number"].min()), int(sg["lap_number"].max()), len(sg)])
    if not stints or any(s[0] not in DRY_COMPOUNDS for s in stints):
        return None

    result: list[list] = []
    dropped_between = False
    for c, s, e, L in stints:
        if L >= min_stint:
            if result and result[-1][0] == c and dropped_between:
                result[-1][2] = e            # merge across a dropped (red-flag) stint
                result[-1][3] += L
            else:
                result.append([c, s, e, L])   # new real stint (keeps M-H-H distinct)
            dropped_between = False
        else:
            dropped_between = True             # sub-minimum stint -> noise, drop it
    if not result:                             # wholly chaotic race: use the dominant stint
        result = [max(stints, key=lambda x: x[3])]

    comps = tuple(s[0] for s in result)
    pit_laps = [s[2] for s in result[:-1]]
    return {"compounds": comps, "n_stops": len(comps) - 1, "pit_laps": pit_laps,
            "family": (len(comps) - 1, tuple(sorted(comps)))}


def extract_all(raw: pd.DataFrame, min_stint: int = 5) -> dict[str, dict]:
    out = {}
    for drv, g in raw.groupby("driver"):
        strat = extract_driver_strategy(g, min_stint)
        if strat:
            out[str(drv)] = strat
    return out
