"""Strategy selection (independent policy).

The selection stage feeds a fan-facing page: it shows the ~5 strategies most likely to
actually be *run* at the race, not the single race-time optimum. With a fixed display
budget (the site has room for 5), the job is coverage of the plausible strategy space —
so this policy chooses the set that captures the most **plausibility mass** rather than
ranking near-identical optima.

Plausibility of a candidate ``q_i`` blends the historical prior (what gets run at this
circuit) with the driver-conditioned Monte-Carlo competitiveness (what is actually fast
from this grid slot). The 5 slots are then filled by a greedy submodular set-cover: each
pick adds the most plausibility mass not already represented, so distinct tyre-set /
stop-count strategies are preferred and near-clones are suppressed. The trade between
showing a second *ordering* of a likely set versus a fresh distinct strategy is decided
automatically per race by whether that alternate order genuinely carries mass.

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
    tier: str = ""            # coarse fan-facing label: Most likely / Alternative / Long-shot
    plausibility: float = 0.0  # normalised q_i (share of plausibility mass)


def _family(c: Candidate) -> tuple:
    return (c.n_stops, tuple(sorted(c.compounds)))


def plausibility_mass(pool: list[Candidate], finish: np.ndarray, rtime: np.ndarray,
                      cfg: dict, n_positions: int):
    """Normalised per-candidate plausibility mass q, plus outcomes and p_optimal.

    q_i = prior^a * comp^b: history-led membership, sim competitiveness modulates.
    (An additive prior+comp hedge was tried and measured WORSE — it starved
    historically-normal-but-sim-mediocre strategies e.g. Mexico soft-medium, without fixing
    the sim-diluted cases; the product's prior floor keeps such strategies in the shown 5.)
    A soft gate down-weights strategies far slower than the driver's own best without
    hard-pruning, so recall of the realised strategy is preserved.
    """
    K, S = finish.shape
    scfg = cfg.get("selection", {})
    a = float(scfg.get("plausibility_prior_exp", 1.0))   # weight on historical prior
    b = float(scfg.get("plausibility_comp_exp", 0.6))    # weight on sim competitiveness
    tau = float(scfg.get("comp_temperature", 2.5))       # positions -> competitiveness scale
    gate = float(scfg.get("comp_gate_positions", 8.0))   # soft-drop strategies this much slower

    outcomes = [Outcome(finish[k], rtime[k]) for k in range(K)]
    mean_fin = np.array([o.mean_finish_classified for o in outcomes])
    priors = np.array([max(c.prior, 1e-12) for c in pool])
    e_min = float(np.min(mean_fin))

    comp = np.exp(-(mean_fin - e_min) / max(tau, 1e-6))
    pen = np.where((mean_fin - e_min) <= gate, 1.0, 1e-4)
    q = (priors ** a) * (comp ** b) * pen
    tot = float(q.sum())
    q = q / tot if tot > 0 else np.full(K, 1.0 / K)

    # P(optimal) via CRN pairing: per sim, the candidate giving the best finish. Kept as a
    # secondary, fan-facing "which would have won" signal (no longer the ranking key).
    f = np.where(np.isnan(finish), n_positions + 1, finish)
    p_optimal = np.bincount(f.argmin(axis=0), minlength=K) / S
    return q, outcomes, p_optimal


def greedy_cover(pool: list[Candidate], q: np.ndarray, k_min: int, k_max: int,
                 lam_order: float, lam_clone: float) -> list[int]:
    """Greedy submodular coverage under the fixed k_max budget. gain(c|S) = q_c * novelty:
    new (stop-count, tyre-set) cell -> 1.0 ; a 2nd ordering of a shown set -> lam_order ;
    a near-clone (same sequence) -> lam_clone. Monotone submodular => greedy is near-optimal."""
    K = len(pool)
    chosen: list[int] = []
    cells: set[tuple] = set()
    seqs: set[tuple] = set()
    remaining = list(range(K))
    while len(chosen) < min(k_max, K) and remaining:
        best_i, best_gain = remaining[0], -1.0
        for i in remaining:
            if pool[i].compounds in seqs:
                nov = lam_clone
            elif _family(pool[i]) in cells:
                nov = lam_order
            else:
                nov = 1.0
            g = q[i] * nov
            if g > best_gain:
                best_gain, best_i = g, i
        chosen.append(best_i)
        cells.add(_family(pool[best_i]))
        seqs.add(pool[best_i].compounds)
        remaining.remove(best_i)

    # Guarantee the display minimum even in a degenerate pool.
    for i in remaining:
        if len(chosen) >= k_min:
            break
        chosen.append(i)
    return chosen


def field_display(pool: list[Candidate], q_field: np.ndarray, cfg: dict) -> list[int]:
    """Race-level shown-k: the same greedy set-cover run on FIELD-aggregated plausibility
    mass (mean of the drivers' normalised q). This is what a per-race fan page shows, and
    what the race-level modal metrics score. Returns pool indices in display (q) order."""
    scfg = cfg.get("selection", {})
    chosen = greedy_cover(pool, q_field,
                          int(scfg.get("min_candidates", 2)),
                          int(scfg.get("max_candidates", 5)),
                          float(scfg.get("order_novelty", 0.4)),
                          float(scfg.get("clone_novelty", 0.05)))
    return sorted(chosen, key=lambda i: -q_field[i])


def select(pool: list[Candidate], finish: np.ndarray, rtime: np.ndarray,
           cfg: dict, n_positions: int, prior=None) -> list[SelectedStrategy]:
    scfg = cfg.get("selection", {})
    k_min = int(scfg.get("min_candidates", 2))
    k_max = int(scfg.get("max_candidates", 5))
    lam_order = float(scfg.get("order_novelty", 0.4))    # mass kept for a 2nd order of a set
    lam_clone = float(scfg.get("clone_novelty", 0.05))   # mass kept for a near-clone
    thr = list(scfg.get("tier_thresholds", [0.6, 0.2]))  # q-ratio cuts for the tier labels
    t_hi = float(thr[0])
    t_lo = float(thr[1]) if len(thr) > 1 else t_hi

    q, outcomes, p_optimal = plausibility_mass(pool, finish, rtime, cfg, n_positions)
    chosen = greedy_cover(pool, q, k_min, k_max, lam_order, lam_clone)

    # Display order + coarse plausibility tiers (relative to the most plausible shown).
    chosen.sort(key=lambda i: -q[i])
    q_top = q[chosen[0]] if chosen else 1.0

    def tier(i: int) -> str:
        r = q[i] / q_top if q_top > 0 else 0.0
        return "Most likely" if r >= t_hi else "Alternative" if r >= t_lo else "Long-shot"

    return [SelectedStrategy(pool[i], outcomes[i], float(p_optimal[i]), rank + 1,
                             tier(i), float(q[i]))
            for rank, i in enumerate(chosen)]
