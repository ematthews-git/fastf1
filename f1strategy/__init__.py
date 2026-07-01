"""f1strategy — inverse-calibration F1 race-strategy predictor.

A forward race simulator scores any strategy (compound sequence + pit laps); a
strategy search returns the optimum and a probability distribution; a small set
of *global* behavioural parameters is calibrated so the simulator's optimum
matches the strategies real front-runners actually ran.

Public entrypoint: :func:`f1strategy.predict.predict_strategy`.
"""

from .config import (
    GlobalParams,
    SimConfig,
    TrackContext,
    CompoundModel,
    StrategyResult,
    StrategyPrediction,
)

__all__ = [
    "GlobalParams",
    "SimConfig",
    "TrackContext",
    "CompoundModel",
    "StrategyResult",
    "StrategyPrediction",
]

__version__ = "0.1.0"
