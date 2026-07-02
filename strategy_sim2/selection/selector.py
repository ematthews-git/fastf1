"""Strategy selection (independent policy).

Consumes evaluated outcome matrices plus the generation plausibility priors and returns
the ranked 2-5 candidates a strategist would actually consider. The default policy ranks
by expected finishing position with a small plausibility tiebreak, reports each option's
probability of being optimal (from the common-random-number pairing), and enforces
diversity so the shortlist shows genuinely different strategies rather than near-clones.

Because selection is isolated, this policy can later become risk-averse, points-weighted
or team-oriented without touching generation or evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strategy_sim2.evaluation.outcomes import Outcome
from strategy_sim2.generation.generator import Candidate


@dataclass
class SelectedStrategy:
    candidate: Candidate
    outcome: Outcome
    p_optimal: float
    rank: int


def _family(c: Candidate) -> tuple:
    return (c.n_stops, tuple(sorted(c.compounds)))


def select(pool: list[Candidate], finish: np.ndarray, rtime: np.ndarray,
           cfg: dict, n_positions: int) -> list[SelectedStrategy]:
    K, S = finish.shape
    sel = cfg.get("selection", {})
    k_min = int(sel.get("min_candidates", 2))
    k_max = int(sel.get("max_candidates", 5))
    w_prior = float(sel.get("prior_weight", 0.3))

    outcomes = [Outcome(finish[k], rtime[k]) for k in range(K)]
    # Rank on expected finish given the driver finishes: DNF risk is ~strategy-independent
    # here, so excluding it isolates the strategy effect and de-compresses the front.
    mean_fin = np.array([o.mean_finish_classified for o in outcomes])
    priors = np.array([max(c.prior, 1e-9) for c in pool])
    score = mean_fin - w_prior * np.log(priors)  # lower is better

    # P(optimal) via CRN pairing: per sim, the candidate giving the best finish.
    f = np.where(np.isnan(finish), n_positions + 1, finish)
    best_per_sim = f.argmin(axis=0)
    p_optimal = np.bincount(best_per_sim, minlength=K) / S

    order = list(np.argsort(score))

    # Keep the best representative of each (n_stops, compound-multiset) family.
    reps: dict[tuple, int] = {}
    for i in order:
        reps.setdefault(_family(pool[i]), i)
    ranked = sorted(reps.values(), key=lambda i: score[i])

    chosen = ranked[:k_max]
    # Guarantee at least two distinct stop counts if the field offers them.
    stop_counts = {pool[i].n_stops for i in chosen}
    if len(stop_counts) < 2:
        for i in ranked:
            if pool[i].n_stops not in stop_counts:
                chosen[-1] = i
                break
    chosen = sorted(set(chosen), key=lambda i: score[i])[:k_max]
    if len(chosen) < k_min:
        for i in ranked:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= k_min:
                break

    return [SelectedStrategy(pool[i], outcomes[i], float(p_optimal[i]), rank + 1)
            for rank, i in enumerate(sorted(chosen, key=lambda i: score[i]))]
