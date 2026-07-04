"""Candidate strategy generation.

Generation *proposes*, it does not *select*. It enumerates the entire rule-legal
candidate space (stop counts in the circuit's range, all compound sequences over the
weekend allocation satisfying the >=2-compound rule), computes each one's pit windows
from tyre economics, then **blends those windows toward the circuit's historically
observed windows** (capturing the undercut / real-world timing), and attaches a
historical plausibility prior. Nothing rule-legal is excluded. A competitiveness-based
``shortlist`` trims for Monte Carlo — pruning by analytic race time, not history.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

from strategy_sim2.data.schema import DRY_COMPOUNDS
from strategy_sim2.generation.plausibility import StrategyPrior
from strategy_sim2.params.circuit import CircuitProfile
from strategy_sim2.params.lapmodel import LapModel
from strategy_sim2.settings import load_settings


@dataclass
class Candidate:
    compounds: tuple[str, ...]
    pit_laps: tuple[int, ...]
    n_stops: int
    stint_lengths: tuple[int, ...]
    analytic_cost: float
    prior: float
    start_compound: str = ""
    pit_windows: tuple[tuple[int, int], ...] = ()  # [lo, hi] lap range per stop

    def __post_init__(self):
        if not self.start_compound and self.compounds:
            self.start_compound = self.compounds[0]


def _optimal_stint_lengths(compounds, n_laps, lap_model, circuit, min_stint, max_frac):
    """Distribute n_laps over stints to minimise analytic strategy cost (convex, so a
    greedy marginal allocation is exact). Uses the nonlinear (cliff) degradation."""
    n = len(compounds)
    if min_stint * n > n_laps:
        return None
    offs = [lap_model.pace_offset(c, circuit) for c in compounds]
    caps = [max(min_stint, int(round(max_frac.get(c, 1.0) * n_laps))) for c in compounds]
    if sum(caps) < n_laps:
        return None
    lengths = [min_stint] * n
    for _ in range(n_laps - min_stint * n):
        marg = [offs[s] + lap_model.deg(compounds[s], lengths[s] + 1, circuit)
                if lengths[s] < caps[s] else math.inf for s in range(n)]
        s = marg.index(min(marg))
        if marg[s] == math.inf:
            return None
        lengths[s] += 1
    return lengths


def _analytic_cost(compounds, lengths, lap_model, circuit, pit_loss):
    cost = 0.0
    for c, L in zip(compounds, lengths):
        cost += lap_model.pace_offset(c, circuit) * L
        cost += sum(lap_model.deg(c, a, circuit) for a in range(1, L + 1))
    return cost + (len(compounds) - 1) * pit_loss


def _sanitize_pits(pits, n_laps, min_stint):
    out, prev = [], 0
    for p in sorted(pits):
        p = max(prev + min_stint, min(int(p), n_laps - min_stint))
        out.append(p)
        prev = p
    return out


def _lengths_from_pits(pits, n_laps):
    bounds = [0] + list(pits) + [n_laps]
    return tuple(bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1))


def _enumerate_sequences(allocation, min_stops, max_stops, min_distinct, allocation_sets):
    """All rule-legal sequences: >=2 distinct compounds AND no compound used more times
    than the physical race allocation supports (e.g. only ~2 fresh sets of each)."""
    for n_stops in range(min_stops, max_stops + 1):
        for seq in product(allocation, repeat=n_stops + 1):
            if len(set(seq)) < min_distinct:
                continue
            if any(seq.count(c) > allocation_sets.get(c, 2) for c in set(seq)):
                continue
            yield seq


def generate_candidates(circuit_profile: CircuitProfile, lap_model: LapModel,
                        prior: StrategyPrior, allocation: tuple[str, ...] | None = None,
                        cfg: dict | None = None) -> list[Candidate]:
    cfg = cfg or load_settings()
    allocation = tuple(allocation or DRY_COMPOUNDS)
    g = cfg.get("generation", {})
    circuit = circuit_profile.circuit
    rules = cfg.get("circuit_rules", {}).get(circuit, {})
    min_stops = int(rules.get("min_stops", g.get("min_stops", 1)))
    max_stops = int(rules.get("max_stops", g.get("max_stops", 3)))
    min_stint = int(g.get("min_stint", 6))
    win_hist_max = float(g.get("window_history_weight", 0.8))  # max weight on history
    win_k = float(g.get("window_history_k", 4.0))    # observations to reach half-weight
    undercut = float(g.get("undercut_shift_laps", 2.0))  # analytic optimum is 'clean-air
    # late'; real stops come earlier under undercut threat (bias measured on train data)
    max_frac = {k.upper(): float(v) for k, v in
                g.get("max_stint_frac", {"SOFT": 0.5, "MEDIUM": 0.7, "HARD": 1.0}).items()}
    allocation_sets = {k.upper(): int(v) for k, v in
                       g.get("allocation_sets", {"SOFT": 2, "MEDIUM": 2, "HARD": 2}).items()}
    min_distinct = int(cfg["compounds"]["min_distinct_dry"])

    n_laps, pit_loss = circuit_profile.n_laps, circuit_profile.pit_loss
    candidates: list[Candidate] = []
    for seq in _enumerate_sequences(allocation, min_stops, max_stops, min_distinct,
                                    allocation_sets):
        lengths = _optimal_stint_lengths(seq, n_laps, lap_model, circuit, min_stint, max_frac)
        if lengths is None:
            continue
        n_stops = len(seq) - 1
        # Analytic optimum is 'clean-air late': shift for undercut threat, then blend
        # toward observed windows with weight growing in history richness (observed
        # windows already embed the real undercut timing).
        pit_analytic = [max(min_stint, sum(lengths[:i + 1]) - undercut) for i in range(n_stops)]
        hist = prior.median_pit_laps(n_stops, n_laps)
        if hist and len(hist) == n_stops:
            w = win_hist_max * prior.n_pit_obs(n_stops) / (prior.n_pit_obs(n_stops) + win_k)
            pits = [(1 - w) * a + w * h for a, h in zip(pit_analytic, hist)]
        else:
            pits = pit_analytic
        pits = tuple(_sanitize_pits(pits, n_laps, min_stint))
        lengths = _lengths_from_pits(pits, n_laps)
        cost = _analytic_cost(seq, lengths, lap_model, circuit, pit_loss)
        ranges = prior.pit_lap_quantiles(n_stops, n_laps)
        if not ranges or len(ranges) != n_stops:
            ranges = [(max(min_stint, p - 4), min(n_laps - min_stint, p + 4)) for p in pits]
        candidates.append(Candidate(tuple(seq), pits, n_stops, lengths, cost,
                                    prior.prior(seq), pit_windows=tuple(ranges)))
    return candidates


def shortlist(candidates: list[Candidate], k: int = 16, w_prior: float = 6.0,
              rep_prior_weight: float = 1.0) -> list[Candidate]:
    """Trim to <=k candidates while guaranteeing FAMILY coverage.

    A family is (n_stops, compound multiset). Diagnosis showed hard-heavy families
    (e.g. H-H-M) being crowded out of the shortlist by analytic cost even when history
    strongly supports them — capping the Monte-Carlo pool's recall. We therefore keep
    the best variant of EVERY rule-legal family (the allocation rule keeps the family
    count ~16), ranked by blended cost+prior score; only if families exceed k do the
    lowest-scoring drop.

    Two *different* prior weights are used on purpose:
      * ``w_prior`` (heavy) ranks WHICH families make the shortlist — history should keep
        off-meta-but-real families in.
      * ``rep_prior_weight`` (light) picks the representative ORDER *within* a family. The
        heavy prior over-favours a MEDIUM start even when that means ENDING ON SOFT (e.g.
        MEDIUM-SOFT over SOFT-MEDIUM), and the sim then rates that ends-on-soft order badly,
        burying an otherwise-realistic family. Letting economics (the cliff already punishes
        ending on soft) lead the order choice yields the order teams actually run.
    """
    if not candidates:
        return []
    best_cost = min(c.analytic_cost for c in candidates)

    def fam_score(c: Candidate) -> float:  # ranks families for inclusion (history-heavy)
        return (c.analytic_cost - best_cost) - w_prior * math.log(max(c.prior, 1e-9))

    def rep_score(c: Candidate) -> float:  # picks the order within a family (economics-led)
        return (c.analytic_cost - best_cost) - rep_prior_weight * math.log(max(c.prior, 1e-9))

    families: dict[tuple, Candidate] = {}
    for c in candidates:
        f = (c.n_stops, tuple(sorted(c.compounds)))
        if f not in families or rep_score(c) < rep_score(families[f]):
            families[f] = c  # most plausible ORDER of this family (economics-led)
    return sorted(families.values(), key=fam_score)[:k]
