"""Typed configuration and result schemas shared across the package.

Everything the simulator/optimizer/calibrator exchange is a plain dataclass so
the whole thing is trivial to serialise (JSON) and drop into a backend pipeline
with no notebook or global state.

Four groups:
  * :class:`GlobalParams` — the *calibrated* behavioural parameters (theta). A
    deliberately small, interpretable, **global** (not per-track) vector, so the
    inverse problem stays low-dimensional and resistant to overfitting.
  * :class:`SimConfig` — non-calibrated toggles/knobs (SC on/off, practice on/off,
    stop/stint limits, softmax temperature, ...).
  * :class:`TrackContext` / :class:`CompoundModel` — the per-race *measured*
    inputs the simulator consumes (tyre curves, fuel rate, pit loss, SC prior).
  * :class:`StrategyResult` / :class:`StrategyPrediction` — outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# Relative dry compounds, softest first. We model by relative slot (not absolute
# Cx) in v1: within one circuit the nomination is ~stable across seasons, and the
# post-practice update uses the weekend's own compounds directly. Cx pooling is a
# noted refinement.
COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")
SOFTNESS: dict[str, int] = {"HARD": 1, "MEDIUM": 2, "SOFT": 3}  # higher = softer/faster-fresh


# --------------------------------------------------------------------------- theta


@dataclass(frozen=True)
class GlobalParams:
    """Calibrated global behavioural parameters (theta).

    These are the *only* things the inverse problem tunes. Per-track tyre/fuel/pit
    inputs stay measured from data — theta just encodes the handful of behavioural
    truths a pure lap-time minimiser misses, shared across every circuit.

    Attributes
    ----------
    deg_scale:
        Multiplier on measured degradation. Residual correction only (the
        race+practice hybrid does the real work), so it is bounded tight near 1.
        >1 shortens optimal stints -> more/earlier stops.
    pit_stop_penalty:
        Effective seconds added *per stop* on top of the raw pit-lane loss, for
        the track-position / dirty-air cost of rejoining. The main lever pulling
        the optimum away from "too many stops" toward what teams actually do.
    stint_risk:
        Convex penalty (s per lap) on tyre age beyond ``risk_free_life`` laps.
        Encodes risk aversion / the tendency to pit before the cliff; biases pit
        laps earlier and shortens long stints.
    risk_free_life:
        Tyre age (laps) below which ``stint_risk`` does not bite.
    sc_influence:
        Weight on the safety-car expected-value term. Only used when
        ``SimConfig.use_sc`` is True; higher -> stops delayed to hedge for a cheap
        SC pit. Ignored (and not searched) in the SC-off calibration.
    """

    deg_scale: float = 1.0
    pit_stop_penalty: float = 2.0
    stint_risk: float = 0.05
    risk_free_life: float = 20.0
    sc_influence: float = 0.0

    # (low, high, prior) search bounds used by the calibrator. Keeping this next to
    # the fields keeps the search space and the schema from drifting apart.
    SEARCH_SPACE: dict[str, tuple[float, float, float]] = field(
        default=None, init=False, repr=False, compare=False,
    )

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("SEARCH_SPACE", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalParams":
        allowed = {f for f in cls.__dataclass_fields__ if f != "SEARCH_SPACE"}
        return cls(**{k: v for k, v in d.items() if k in allowed})


# Search bounds live at module level (a frozen dataclass field default can't hold a
# mutable dict cleanly). low, high, prior. sc_influence is only searched SC-on.
SEARCH_SPACE: dict[str, tuple[float, float, float]] = {
    # deg_scale is a *measurement* correction on race-sourced degradation (HARD, which
    # is the best-observed compound), so it stays a tight near-1 nudge — it must not
    # become a behavioural lever, or inflating HARD deg makes the model shun the long
    # hard stints that real 1-stops use.
    # deg_scale is a *measurement* correction on race-sourced degradation (HARD, the
    # best-observed compound), so it stays a tight near-1 nudge.
    "deg_scale": (0.85, 1.40, 1.00),
    "pit_stop_penalty": (0.0, 14.0, 2.0),
    # stint_risk (steepness) + risk_free_life (knee) are the behavioural pit-earlier
    # lever. NB: a per-compound quadratic "cliff" was tried and cross-validation
    # rejected it (it induced token min-length soft stints); see README notes.
    "stint_risk": (0.0, 0.60, 0.05),
    "risk_free_life": (10.0, 34.0, 20.0),
    "sc_influence": (0.0, 1.0, 0.0),
}
# Parameters not tuned when SC hedging is off (kept at their defaults).
SC_ONLY_PARAMS: frozenset[str] = frozenset({"sc_influence"})


# ------------------------------------------------------------------- sim config


@dataclass
class SimConfig:
    """Non-calibrated simulation settings and toggles."""

    use_sc: bool = True          # include safety-car expected-value hedging
    use_practice: bool = False   # fold the target weekend's FP long-runs into tyre inputs
    max_stops: int = 3           # search 1..max_stops
    min_stint: int = 7           # minimum laps per stint (avoids degenerate token stints)
    pit_lap_step: int = 1        # granularity of the pit-lap grid search
    softmax_temp: float = 1.5    # temperature (s) for turning race-time gaps into probabilities
    sc_scenarios: int = 12       # discrete SC-window scenarios for the EV integral
    require_two_compounds: bool = True  # dry-race two-compound rule


# ------------------------------------------------------------------ track inputs


@dataclass
class CompoundModel:
    """Measured pace/degradation model for one relative compound at one circuit."""

    slot: str                    # SOFT | MEDIUM | HARD
    deg: float                   # degradation, s/lap (already hybrid-selected)
    base_offset: float           # fresh-lap pace vs the fastest available compound, s (>=0)
    n: int = 0                   # laps of data behind the estimate
    source: str = ""             # "race" | "practice" | "hybrid" | "prior"
    cx: Optional[str] = None     # absolute Pirelli compound if known (else None)


@dataclass
class SCModel:
    """Per-track safety-car prior used by the EV term (only when use_sc)."""

    p_race: float                # P(at least one SC during the race)
    exp_count: float             # expected number of SC periods
    pit_loss_under_sc: float     # effective pit loss when stopping under SC (< green pit loss)


@dataclass
class TrackContext:
    """All measured, per-race inputs the simulator needs for one prediction."""

    track: str
    year: int
    event_name: str
    n_laps: int
    pit_loss: float                       # green-flag pit-lane time loss, s
    fuel_rate: float                      # fuel+track-evo lap-time gain, s/lap (negative)
    compounds: dict[str, CompoundModel]   # slot -> model (only well-sampled slots)
    sc: Optional[SCModel] = None
    seasons_used: tuple[int, ...] = ()
    notes: str = ""

    @property
    def available(self) -> list[str]:
        """Compound slots with a usable model, softest first."""
        return [c for c in COMPOUNDS if c in self.compounds]


# ---------------------------------------------------------------------- outputs


@dataclass
class StrategyResult:
    """One evaluated strategy."""

    compounds: tuple[str, ...]   # e.g. ("SOFT", "MEDIUM")
    pit_laps: tuple[int, ...]    # laps at which stops happen, e.g. (24,)
    race_time: float             # modelled race time (relative units), lower is better
    n_stops: int = 0
    delta_to_best: float = 0.0   # seconds behind the optimum
    prob: float = 0.0            # softmax probability mass on this strategy

    def label(self) -> str:
        seq = "→".join(s[0] for s in self.compounds)     # S→M
        pits = ",".join(map(str, self.pit_laps)) or "-"
        return f"{self.n_stops}-stop {seq} @ {pits}"


@dataclass
class StrategyPrediction:
    """Full prediction returned by :func:`predict_strategy`."""

    track: str
    year: int
    event_name: str
    optimal: StrategyResult
    ranked: list[StrategyResult]              # best-first
    p_by_stops: dict[int, float]             # {1: .., 2: .., 3: ..}
    pit_windows: dict[int, tuple[int, int]]  # stop index -> (lap_lo, lap_hi) central window
    context: TrackContext
    used_practice: bool = False
    used_sc: bool = True

    def summary(self) -> str:
        opt = self.optimal
        dist = " ".join(f"{k}-stop {v:.0%}" for k, v in sorted(self.p_by_stops.items()))
        return (f"{self.event_name} {self.year}: {opt.label()}  "
                f"(P: {dist})")
