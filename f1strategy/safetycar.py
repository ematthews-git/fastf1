"""Safety-car hedging — a per-track SC prior and an expected-value pit adjustment.

Entirely optional: everything here is only consulted when ``SimConfig.use_sc`` is
True and a :class:`SCModel` is attached to the context, so the user can compare the
model's behaviour with and without SC hedging by flipping one flag.

Behavioural target (user's words): "higher SC probability encourages teams to pit
later in case a SC comes." We encode this as an expected-value penalty on *early*
stops: pitting at lap L forgoes the option of a cheap SC stop over the remaining
race, and that forgone option is worth more the earlier you stop and the more
SC-prone the track is. It is a closed-form collapse of "expected saving integrated
over the lap at which an SC might arrive," and its strength is the calibrated
``sc_influence`` parameter.
"""

from __future__ import annotations

from typing import Callable, Optional

from .config import SCModel

# Effective pit loss when stopping under SC/VSC, as a fraction of the green pit loss
# (the field is bunched and slow, so a stop costs far less track time).
SC_PIT_LOSS_FRAC = 0.45

# Hand-curated P(>=1 SC or VSC during the race) by circuit. GENERIC / FOR TESTING;
# a data-driven estimate is available via measure_sc_probability. Keyed by lowercase
# substrings (same convention as pitloss).
SC_PROB: dict[str, float] = {
    "singapore": 0.80, "marina bay": 0.80,
    "azerbaijan": 0.70, "baku": 0.70,
    "saudi": 0.65, "jeddah": 0.65,
    "monaco": 0.60,
    "australia": 0.55, "melbourne": 0.55,
    "brazil": 0.50, "interlagos": 0.50, "paulo": 0.50,
    "canada": 0.50, "montreal": 0.50,
    "vegas": 0.50,
    "qatar": 0.40, "losail": 0.40, "lusail": 0.40,
    "miami": 0.40,
    "britain": 0.40, "silverstone": 0.40,
    "belgium": 0.40, "spa": 0.40,
    "abu dhabi": 0.38, "yas": 0.38,
    "japan": 0.35, "suzuka": 0.35,
    "united states": 0.35, "austin": 0.35, "cota": 0.35,
    "netherlands": 0.32, "zandvoort": 0.32,
    "austria": 0.30, "spielberg": 0.30,
    "italy": 0.30, "monza": 0.30,
    "mexico": 0.30, "hermanos": 0.30,
    "china": 0.30, "shanghai": 0.30,
    "imola": 0.30, "emilia": 0.30,
    "bahrain": 0.28,
    "hungar": 0.22, "budapest": 0.22,
    "spain": 0.15, "barcelona": 0.15, "catal": 0.15,   # notably clean
}
DEFAULT_SC_PROB = 0.35


def sc_probability(track: str) -> float:
    key = str(track).lower()
    for sub, val in SC_PROB.items():
        if sub in key:
            return val
    return DEFAULT_SC_PROB


def sc_model(track: str, pit_loss: float) -> SCModel:
    """Build the per-track SC prior consumed by :func:`sc_adjustment`."""
    p = sc_probability(track)
    return SCModel(p_race=p, exp_count=p * 1.2,
                   pit_loss_under_sc=SC_PIT_LOSS_FRAC * pit_loss)


def sc_adjustment(seq, pit_laps, ctx, params, config) -> float:
    """Expected-value penalty (s) for stopping early given SC risk.

    penalty = sc_influence * (green_pit_loss - sc_pit_loss) * p_race *
              Σ_stops (laps_remaining_after_stop / n_laps)

    Zero when hedging is off, no SC model is attached, or the strategy makes no
    stops. Larger sc_influence / p_race pushes the optimum toward later (and fewer)
    stops, matching real hedging behaviour.
    """
    sc = ctx.sc
    if sc is None or not config.use_sc or not pit_laps:
        return 0.0
    saving = max(0.0, ctx.pit_loss - sc.pit_loss_under_sc)
    n = ctx.n_laps
    infl = params.sc_influence
    pen = 0.0
    for lap in pit_laps:
        pen += infl * saving * sc.p_race * max(0.0, (n - lap) / n)
    return pen


# ---------------------------------------------------------------- data-driven prior


def measure_sc_probability(track: str, years, loader: Optional[Callable] = None) -> Optional[float]:
    """Fraction of past races at a circuit with >=1 SC/VSC (from race-control msgs).

    Returns None if no races could be inspected. Loads messages, so it is slower;
    use it to refresh :data:`SC_PROB`, not on the hot path.
    """
    from .data import loaders
    loader = loader or (lambda y, t: loaders.load_race(y, t, messages=True))
    seen = hits = 0
    for y in years:
        try:
            s = loader(y, track)
            rc = s.race_control_messages
        except Exception:  # noqa: BLE001
            continue
        if rc is None or not len(rc):
            continue
        seen += 1
        msg = rc["Message"].astype(str).str.upper()
        if msg.str.contains("SAFETY CAR").any() or msg.str.contains("VIRTUAL SAFETY CAR").any():
            hits += 1
    return (hits / seen) if seen else None
