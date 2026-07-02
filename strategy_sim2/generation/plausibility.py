"""Historical plausibility priors for strategies.

History *weights* candidates, it never *gates* them: every rule-legal strategy keeps
a non-zero prior (add-alpha smoothing), so a strategy nobody has tried at this circuit
— perhaps made competitive by new compounds or regulations — remains eligible and can
still win once the simulator evaluates it.

A strategy's "pattern" is (n_stops, multiset of compounds), which generalises across
stint order (H-M-H and M-H-H share a pattern); ordering is left to the simulator.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from strategy_sim2.data import clean, session_filter
from strategy_sim2.data.schema import DRY_COMPOUNDS
from strategy_sim2.settings import load_settings


def _compound_pattern(compounds: tuple[str, ...]) -> tuple:
    return tuple(sorted(compounds))


def derive_race_strategies(raw: "pd.DataFrame") -> list[tuple[str, ...]]:
    """Recover each driver's dry stint-compound sequence from a race's laps."""
    out = []
    for _, g in raw.groupby("driver"):
        g = g.dropna(subset=["stint"])
        comps = []
        for _, sg in g.groupby("stint"):
            mode = sg["compound"].mode()
            if len(mode) and mode.iloc[0] in DRY_COMPOUNDS:
                comps.append(str(mode.iloc[0]))
        if comps:
            out.append(tuple(comps))
    return out


@dataclass
class StrategyPrior:
    circuit: str
    stop_counts: Counter = field(default_factory=Counter)
    patterns: Counter = field(default_factory=Counter)
    n: int = 0
    alpha: float = 1.0
    max_stops: int = 3

    def prior(self, compounds: tuple[str, ...]) -> float:
        """Smoothed plausibility in (0, 1]: P(n_stops) * P(compound multiset)."""
        n_stops = len(compounds) - 1
        n_stop_options = self.max_stops + 1
        p_stops = (self.stop_counts.get(n_stops, 0) + self.alpha) / (self.n + self.alpha * n_stop_options)
        n_pat_options = max(len(self.patterns), 1) + 6  # headroom for unseen patterns
        p_pat = (self.patterns.get(_compound_pattern(compounds), 0) + self.alpha) / (
            self.n + self.alpha * n_pat_options)
        return float(p_stops * p_pat)


def build_strategy_prior(circuit: str, cfg: dict | None = None,
                         alpha: float = 1.0, max_stops: int = 3) -> StrategyPrior:
    cfg = cfg or load_settings()
    races = session_filter.included_races(cfg)
    races = races[races["circuit"] == circuit]

    sp = StrategyPrior(circuit=circuit, alpha=alpha, max_stops=max_stops)
    for _, r in races.iterrows():
        raw = clean.get_clean_race(int(r["year"]), int(r["round"]), cfg)
        if raw is None:
            continue
        for comps in derive_race_strategies(raw):
            n_stops = len(comps) - 1
            if n_stops < 0:
                continue
            sp.stop_counts[n_stops] += 1
            sp.patterns[_compound_pattern(comps)] += 1
            sp.n += 1
    return sp
