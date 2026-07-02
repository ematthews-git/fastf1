"""Candidate strategy generation.

Generation *proposes*, it does not *select*. It enumerates the entire rule-legal
candidate space (all stop counts in range, all compound sequences over the weekend
allocation satisfying the >=2-compound rule), computes each one's optimal pit windows
from tyre economics, and attaches a historical plausibility prior. Nothing rule-legal
is excluded. A competitiveness-based ``shortlist`` trims the set for expensive Monte
Carlo — that is pruning by analytic race time, not by historical filtering, so novel
strong strategies survive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from strategy_sim2.data.schema import DRY_COMPOUNDS
from strategy_sim2.generation.plausibility import StrategyPrior
from strategy_sim2.params.circuit import CircuitProfile
from strategy_sim2.params.lapmodel import LapModel
from strategy_sim2.settings import load_settings


@dataclass
class Candidate:
    compounds: tuple[str, ...]      # compound per stint (len = n_stops + 1)
    pit_laps: tuple[int, ...]       # planned pit laps (len = n_stops)
    n_stops: int
    stint_lengths: tuple[int, ...]
    analytic_cost: float            # strategy-dependent race-time proxy (s), lower is better
    prior: float                    # historical plausibility (0, 1]
    start_compound: str = ""

    def __post_init__(self):
        if not self.start_compound and self.compounds:
            self.start_compound = self.compounds[0]


def _optimal_stint_lengths(compounds, n_laps, lap_model, circuit, min_stint, max_frac):
    """Distribute n_laps over stints to minimise analytic strategy cost.

    The per-stint cost sum_(age=1..L)[offset_c + deg_c*age] is separable and convex,
    so greedily assigning each lap to the stint with the lowest marginal cost is exact.
    Per-compound caps (fraction of race distance) bound physical tyre life so that a
    thin-data / under-estimated deg curve can't produce an absurd 45-lap soft stint.
    """
    n_stints = len(compounds)
    if min_stint * n_stints > n_laps:
        return None
    degs = [lap_model.deg_slope(c, circuit) for c in compounds]
    offs = [lap_model.pace_offset(c, circuit) for c in compounds]
    caps = [max(min_stint, int(round(max_frac.get(c, 1.0) * n_laps))) for c in compounds]
    if sum(caps) < n_laps:
        return None  # these compounds can't legally cover the distance
    lengths = [min_stint] * n_stints
    for _ in range(n_laps - min_stint * n_stints):
        marg = [offs[s] + degs[s] * (lengths[s] + 1) if lengths[s] < caps[s] else float("inf")
                for s in range(n_stints)]
        s = marg.index(min(marg))
        if marg[s] == float("inf"):
            return None
        lengths[s] += 1
    return lengths


def _analytic_cost(compounds, lengths, lap_model, circuit, pit_loss):
    cost = 0.0
    for c, L in zip(compounds, lengths):
        deg, off = lap_model.deg_slope(c, circuit), lap_model.pace_offset(c, circuit)
        cost += off * L + deg * (L * (L + 1) / 2.0)
    return cost + (len(compounds) - 1) * pit_loss


def _enumerate_sequences(allocation, min_stops, max_stops, min_distinct):
    for n_stops in range(min_stops, max_stops + 1):
        for seq in product(allocation, repeat=n_stops + 1):
            if len(set(seq)) >= min_distinct:
                yield seq


def generate_candidates(circuit_profile: CircuitProfile, lap_model: LapModel,
                        prior: StrategyPrior, allocation: tuple[str, ...] | None = None,
                        cfg: dict | None = None) -> list[Candidate]:
    cfg = cfg or load_settings()
    allocation = tuple(allocation or DRY_COMPOUNDS)
    gcfg = cfg.get("generation", {})
    min_stops = int(gcfg.get("min_stops", 1))
    max_stops = int(gcfg.get("max_stops", 3))
    min_stint = int(gcfg.get("min_stint", 6))
    max_frac = {k.upper(): float(v) for k, v in
                gcfg.get("max_stint_frac", {"SOFT": 0.5, "MEDIUM": 0.7, "HARD": 1.0}).items()}
    min_distinct = int(cfg["compounds"]["min_distinct_dry"])

    n_laps = circuit_profile.n_laps
    pit_loss = circuit_profile.pit_loss
    candidates: list[Candidate] = []
    for seq in _enumerate_sequences(allocation, min_stops, max_stops, min_distinct):
        lengths = _optimal_stint_lengths(seq, n_laps, lap_model, circuit_profile.circuit,
                                         min_stint, max_frac)
        if lengths is None:
            continue
        pit_laps = tuple(int(sum(lengths[:i + 1])) for i in range(len(lengths) - 1))
        cost = _analytic_cost(seq, lengths, lap_model, circuit_profile.circuit, pit_loss)
        candidates.append(Candidate(
            compounds=tuple(seq), pit_laps=pit_laps, n_stops=len(seq) - 1,
            stint_lengths=tuple(lengths), analytic_cost=cost, prior=prior.prior(seq),
        ))
    return candidates


def shortlist(candidates: list[Candidate], k: int = 12, w_prior: float = 6.0,
              ensure_stop_diversity: bool = True) -> list[Candidate]:
    """Trim to the k most promising candidates by competitiveness + a plausibility nudge.

    Score blends analytic race time (primary) with the plausibility prior (secondary),
    so novel-but-fast strategies survive while implausible-and-slow ones drop out.
    """
    if not candidates:
        return []
    import math

    best_cost = min(c.analytic_cost for c in candidates)

    def score(c: Candidate) -> float:
        # seconds lost vs the best analytic strategy, minus a small plausibility bonus
        return (c.analytic_cost - best_cost) - w_prior * math.log(max(c.prior, 1e-9))

    ranked = sorted(candidates, key=score)
    if not ensure_stop_diversity:
        return ranked[:k]

    chosen, seen_stops = [], set()
    for c in ranked:  # guarantee the best of each stop-count appears
        if c.n_stops not in seen_stops:
            chosen.append(c); seen_stops.add(c.n_stops)
    for c in ranked:
        if len(chosen) >= k:
            break
        if c not in chosen:
            chosen.append(c)
    return sorted(chosen, key=score)[:k]
