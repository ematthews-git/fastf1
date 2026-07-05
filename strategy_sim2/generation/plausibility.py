"""Historical plausibility priors + observed pit-window distributions.

History *weights* candidates, it never *gates* them: every rule-legal strategy keeps a
non-zero prior (add-alpha smoothing), so a strategy nobody has tried can still win once
simulated. Strategies are extracted with the shared cleaner (red-flag / SC stint flurries
merged), so anomalous races don't pollute the prior. We also record observed pit-lap
fractions per stop count, used to calibrate generated pit windows toward reality.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from strategy_sim2.data import clean, session_filter
from strategy_sim2.data.strategy import extract_all
from strategy_sim2.params.nominations import sequence_relabeler
from strategy_sim2.settings import load_settings


def _pattern(compounds: tuple[str, ...]) -> tuple:
    return tuple(sorted(compounds))


def _perm_count(pattern: tuple[str, ...]) -> int:
    """Number of distinct orderings of a compound multiset."""
    from math import factorial
    n = factorial(len(pattern))
    for c in set(pattern):
        n //= factorial(pattern.count(c))
    return n


@dataclass
class StrategyPrior:
    circuit: str
    stop_counts: Counter = field(default_factory=Counter)
    patterns: Counter = field(default_factory=Counter)
    sequences: Counter = field(default_factory=Counter)       # circuit ORDERED sequences
    start_counts: Counter = field(default_factory=Counter)   # circuit start compounds
    global_start: Counter = field(default_factory=Counter)   # all-circuit start compounds
    global_patterns: Counter = field(default_factory=Counter)  # all-circuit multisets
    global_sequences: Counter = field(default_factory=Counter)  # all-circuit ORDERED sequences
    global_stops: Counter = field(default_factory=Counter)     # all-circuit stop counts
    global_n: float = 0.0
    pit_fracs: dict = field(default_factory=lambda: defaultdict(list))  # n_stops -> [frac tuples]
    n: float = 0.0
    alpha: float = 1.0
    max_stops: int = 3
    start_blend_k: float = 12.0  # laps of circuit evidence to outweigh the global start prior
    pattern_blend_k: float = 10.0  # circuit strategies needed to outweigh the global meta
    # Weekend practice long-run laps per compound (teams rehearse their race tyres).
    weekend_usage: dict = field(default_factory=dict)
    usage_alpha: float = 8.0
    usage_weight: float = 1.0
    order_weight: float = 1.0  # exponent on P(order | multiset) in the sequence prior

    def usage_prior(self, compound: str) -> float:
        """P(compound raced | weekend practice usage); neutral 1.0 when no weekend data."""
        if not self.weekend_usage:
            return 1.0
        total = sum(self.weekend_usage.values())
        return float((self.weekend_usage.get(compound, 0) + self.usage_alpha)
                     / (total + 3 * self.usage_alpha))

    def start_prior(self, compound: str) -> float:
        """P(start compound): circuit evidence shrunk toward the global distribution
        (reality is strongly MEDIUM-start; the model must not invent SOFT starts)."""
        gn = sum(self.global_start.values())
        g = (self.global_start.get(compound, 0) + self.alpha) / (gn + 3 * self.alpha)
        cn = sum(self.start_counts.values())
        if cn == 0:
            return float(g)
        c = (self.start_counts.get(compound, 0) + self.alpha) / (cn + 3 * self.alpha)
        w = cn / (cn + self.start_blend_k)
        return float(w * c + (1 - w) * g)

    def _blend(self, circ_count: float, circ_n: float, glob_count: float,
               glob_n: float, n_options: float) -> float:
        """Hierarchical smoothing: circuit evidence shrunk toward the recency-weighted
        global meta (a new-regulation strategy fashion shows up globally seasons before
        a single circuit accumulates local evidence of it)."""
        g = (glob_count + self.alpha) / (glob_n + self.alpha * n_options)
        if circ_n <= 0:
            return float(g)
        c = (circ_count + self.alpha) / (circ_n + self.alpha * n_options)
        w = circ_n / (circ_n + self.pattern_blend_k)
        return float(w * c + (1 - w) * g)

    def pattern_prior(self, compounds: tuple[str, ...]) -> float:
        n_pat = max(len(self.global_patterns), len(self.patterns), 1) + 6
        pat = _pattern(compounds)
        return self._blend(self.patterns.get(pat, 0), self.n,
                           self.global_patterns.get(pat, 0), self.global_n, n_pat)

    def order_prior(self, compounds: tuple[str, ...]) -> float:
        """P(this ordering | its multiset), circuit evidence shrunk toward the global
        ordering convention (which is near-universal: run the softer tyre earlier, save
        the durable one for the final stint). This is what ranks MEDIUM-MEDIUM-HARD above
        MEDIUM-HARD-MEDIUM — the multiset pattern prior cannot see the difference."""
        seq = tuple(compounds)
        pat = _pattern(seq)
        n_perm = _perm_count(pat)
        if n_perm <= 1:
            return 1.0
        # Totals over all observed orderings of this multiset (counters are small).
        circ_tot = sum(v for s, v in self.sequences.items() if _pattern(s) == pat)
        glob_tot = sum(v for s, v in self.global_sequences.items() if _pattern(s) == pat)
        g = (self.global_sequences.get(seq, 0) + self.alpha) / (glob_tot + self.alpha * n_perm)
        if circ_tot <= 0:
            return float(g)
        c = (self.sequences.get(seq, 0) + self.alpha) / (circ_tot + self.alpha * n_perm)
        w = circ_tot / (circ_tot + self.pattern_blend_k)
        return float(w * c + (1 - w) * g)

    def prior(self, compounds: tuple[str, ...]) -> float:
        p_stops = self.stop_prior(len(compounds) - 1)
        p_pat = self.pattern_prior(compounds)
        # Weekend usage: geometric mean over stints (scale-free across stop counts).
        p_use = float(np.exp(np.mean([np.log(self.usage_prior(c)) for c in compounds])))
        return float(p_stops * p_pat * self.start_prior(compounds[0])
                     * p_use ** self.usage_weight
                     * self.order_prior(compounds) ** self.order_weight)

    def stop_prior(self, n_stops: int) -> float:
        return self._blend(self.stop_counts.get(n_stops, 0), self.n,
                           self.global_stops.get(n_stops, 0), self.global_n,
                           self.max_stops + 1)

    def observed_stop_counts(self) -> list[int]:
        return sorted(self.stop_counts)

    def median_pit_laps(self, n_stops: int, n_laps: int, min_samples: int = 4):
        """Observed median pit laps for this stop count, or None if too few samples."""
        fracs = self.pit_fracs.get(n_stops, [])
        if len(fracs) < min_samples:
            return None
        arr = np.array(fracs, dtype=float)  # shape [samples, n_stops]
        return [int(round(float(np.median(arr[:, i])) * n_laps)) for i in range(n_stops)]

    def n_pit_obs(self, n_stops: int) -> int:
        return len(self.pit_fracs.get(n_stops, []))

    def pit_lap_quantiles(self, n_stops: int, n_laps: int, lo: float = 25.0,
                          hi: float = 75.0, min_samples: int = 4):
        """Observed [lo, hi] percentile pit-lap window per stop, or None if thin."""
        fracs = self.pit_fracs.get(n_stops, [])
        if len(fracs) < min_samples:
            return None
        arr = np.array(fracs, dtype=float)
        return [(int(round(float(np.percentile(arr[:, i], lo)) * n_laps)),
                 int(round(float(np.percentile(arr[:, i], hi)) * n_laps)))
                for i in range(n_stops)]


def build_strategy_prior(circuit: str, cfg: dict | None = None,
                         alpha: float = 1.0, max_stops: int = 3,
                         before: tuple[int, int] | None = None) -> StrategyPrior:
    """Build the prior from races strictly before ``before`` (no target-race leakage).

    The circuit's own races feed stop counts / patterns / pit windows; ALL races in the
    window feed the global start-compound distribution used for shrinkage.
    """
    from strategy_sim2.params.dataset import filter_window

    cfg = cfg or load_settings()
    pcfg = cfg.get("prior", {})
    alpha = float(pcfg.get("alpha", alpha))
    start_k = float(pcfg.get("start_blend_k", 12.0))
    pattern_k = float(pcfg.get("pattern_blend_k", 10.0))
    order_w = float(pcfg.get("order_weight", 1.0))
    season_boost = float(pcfg.get("same_season_boost", 1.0))
    relabel_gamma = float(pcfg.get("relabel_blend", 0.5))  # 0=raw labels, 1=pure relabel
    min_stint = int(cfg["cleaning"].get("min_strategic_stint", 5))
    decay = float(cfg["training"].get("recency_decay", 0.7))
    races = filter_window(session_filter.included_races(cfg), None, before)
    sp = StrategyPrior(circuit=circuit, alpha=alpha, max_stops=max_stops,
                       start_blend_k=start_k, pattern_blend_k=pattern_k,
                       order_weight=order_w)
    if not len(races):
        return sp
    ref_year = before[0] if before else int(races["year"].max())
    for _, r in races.iterrows():
        raw = clean.get_clean_race(int(r["year"]), int(r["round"]), cfg)
        if raw is None or not len(raw):
            continue
        # Recency weight: strategy fashions shift with regs/tyres; recent seasons dominate.
        # same_season_boost > 1 lets an emerging in-season meta outweigh stale history
        # at circuits whose own record predates it.
        w = decay ** (ref_year - int(r["year"]))
        if int(r["year"]) == ref_year:
            w *= season_boost
        here = str(r["circuit"]) == circuit
        # Nomination-aware: circuit-LOCAL evidence is counted BOTH raw and translated
        # into the target year's label space (C-numbers). Strategy choice is partly
        # PHYSICAL (what a compound can do transfers via C-numbers — Barcelona) and
        # partly LABEL-RELATIVE (teams structure the race around whatever triple is
        # nominated — Monaco, where any compound survives); a pure relabel measurably
        # broke the label-relative races, a pure raw count breaks the physical ones.
        # Global counters stay label-keyed — cross-circuit conventions ("start the
        # medium, save the hard") are label-relative by nature.
        relabel = (sequence_relabeler(int(r["year"]), circuit, ref_year, cfg)
                   if here else None)
        n_laps = int(raw["total_laps"].iloc[0])
        for strat in extract_all(raw, min_stint).values():
            ns = strat["n_stops"]
            if ns < 0:
                continue
            sp.global_start[strat["compounds"][0]] += w
            sp.global_patterns[_pattern(strat["compounds"])] += w
            sp.global_sequences[tuple(strat["compounds"])] += w
            sp.global_stops[ns] += w
            sp.global_n += w
            if not here:
                continue
            raw_seq = tuple(strat["compounds"])
            sp.stop_counts[ns] += w
            for seq, ws in (((raw_seq, w),) if relabel is None else
                            ((raw_seq, w * (1 - relabel_gamma)),
                             (tuple(relabel(raw_seq)), w * relabel_gamma))):
                if ws <= 0:
                    continue
                sp.patterns[_pattern(seq)] += ws
                sp.sequences[seq] += ws
                sp.start_counts[seq[0]] += ws
            sp.n += w
            if ns >= 1 and n_laps > 0 and len(strat["pit_laps"]) == ns:
                sp.pit_fracs[ns].append(tuple(pl / n_laps for pl in strat["pit_laps"]))
    return sp
