"""Joint fuel + tyre lap-time model.

Fuel and tyre-degradation effects are *jointly* identified from race laps, which is
more correct than the paper's two-stage (fuel-then-tyre-on-residuals) approach. We
use a fixed-effects "within" regression: a per (driver, race) intercept is absorbed
by demeaning each group, so base pace / circuit length / driver quality drop out and
we never need qualifying times for training.

Per-lap green-flag model (after removing the driver-race intercept):

    lap_time = fuel * laps_remaining
             + offset_c                      # compound pace offset (vs MEDIUM)
             + deg_c * tyre_life             # linear degradation per compound
             + noise

Parameters are estimated globally, then per circuit and per driver with empirical-
Bayes shrinkage toward the global estimate so sparse cells stay stable (requirement:
hierarchical / pooled rather than noisy individual fits).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy_sim2.settings import load_settings

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
_REF = "MEDIUM"  # reference compound: offset_MEDIUM == 0
_MIN_ROWS = 40


@dataclass
class _Fit:
    fuel: float
    offset: dict            # compound -> pace offset vs MEDIUM (s); NaN if unidentified
    deg: dict               # compound -> linear deg slope (s / lap of tyre age)
    resid_std: float
    n: int


def _demean(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract per-group means (fixed-effects within transform)."""
    tmp = pd.DataFrame(np.column_stack([y, X]))
    means = tmp.groupby(groups).transform("mean").to_numpy()
    d = tmp.to_numpy() - means
    return d[:, 1:], d[:, 0]


def _fit_within(laps: pd.DataFrame) -> _Fit | None:
    need = ["lap_time_s", "laps_remaining", "tyre_life", "compound", "year", "round", "driver"]
    df = laps.dropna(subset=need)
    if len(df) < _MIN_ROWS:
        return None

    comp = df["compound"].to_numpy()
    age = df["tyre_life"].to_numpy(float)
    cols: dict[str, np.ndarray] = {"fuel": df["laps_remaining"].to_numpy(float)}
    for c in COMPOUNDS:
        if c != _REF:
            cols[f"off_{c}"] = (comp == c).astype(float)
    for c in COMPOUNDS:
        cols[f"deg_{c}"] = age * (comp == c).astype(float)

    names = list(cols.keys())
    X = np.column_stack([cols[n] for n in names])
    y = df["lap_time_s"].to_numpy(float)
    groups = (df["year"].astype(str) + "_" + df["round"].astype(str) + "_"
              + df["driver"].astype(str)).to_numpy()
    Xd, yd = _demean(X, y, groups)

    # Only fit columns that retain variation after demeaning (others unidentified).
    good = Xd.std(axis=0) > 1e-9
    beta = np.full(len(names), np.nan)
    if good.any():
        b, *_ = np.linalg.lstsq(Xd[:, good], yd, rcond=None)
        beta[good] = b
    resid = yd - np.nan_to_num(Xd @ np.nan_to_num(beta))

    idx = {n: i for i, n in enumerate(names)}
    offset = {_REF: 0.0}
    for c in COMPOUNDS:
        if c != _REF:
            offset[c] = float(beta[idx[f"off_{c}"]])
    deg = {c: float(beta[idx[f"deg_{c}"]]) for c in COMPOUNDS}
    return _Fit(fuel=float(beta[idx["fuel"]]), offset=offset, deg=deg,
                resid_std=float(np.nanstd(resid)), n=int(len(df)))


def _shrink(local: float, glob: float, n: int, k: float) -> float:
    if local is None or not np.isfinite(local):
        return glob
    w = n / (n + k)
    return w * local + (1.0 - w) * glob


@dataclass
class LapModel:
    glob: _Fit
    by_circuit: dict[str, _Fit] = field(default_factory=dict)
    deg_dev_by_driver: dict[str, dict] = field(default_factory=dict)  # additive dev vs global
    noise_by_driver: dict[str, float] = field(default_factory=dict)

    # --- accessors used by the simulator ---
    def fuel_coef(self, circuit: str | None = None) -> float:
        f = self.by_circuit.get(circuit)
        return f.fuel if f else self.glob.fuel

    def pace_offset(self, compound: str, circuit: str | None = None) -> float:
        f = self.by_circuit.get(circuit, self.glob)
        v = f.offset.get(compound)
        if v is None or not np.isfinite(v):
            v = self.glob.offset.get(compound, 0.0)
        return float(v)

    def deg_slope(self, compound: str, circuit: str | None = None,
                  driver: str | None = None) -> float:
        f = self.by_circuit.get(circuit, self.glob)
        base = f.deg.get(compound)
        if base is None or not np.isfinite(base):
            base = self.glob.deg[compound]
        dev = self.deg_dev_by_driver.get(driver, {}).get(compound, 0.0) if driver else 0.0
        return max(0.0, float(base + dev))  # tyres degrade: slope >= 0

    def deg(self, compound: str, age: float, circuit: str | None = None,
            driver: str | None = None) -> float:
        return self.deg_slope(compound, circuit, driver) * float(age)

    def noise_std(self, driver: str | None = None) -> float:
        return float(self.noise_by_driver.get(driver, self.glob.resid_std))

    def deg_severity(self, circuit: str | None = None) -> float:
        """Representative degradation (s/lap) across compounds — feeds strategy logic."""
        return float(np.mean([self.deg_slope(c, circuit) for c in COMPOUNDS]))


def fit_lap_model(laps: pd.DataFrame, cfg: dict | None = None) -> LapModel:
    cfg = cfg or load_settings()
    k_circuit = float(cfg.get("params", {}).get("k_circuit", 500))
    k_driver = float(cfg.get("params", {}).get("k_driver", 800))

    glob = _fit_within(laps)
    if glob is None:
        raise ValueError("insufficient clean laps to fit the lap model")

    model = LapModel(glob=glob)

    # per-circuit params, shrunk toward global
    for circuit, sub in laps.groupby("circuit"):
        fit = _fit_within(sub)
        if fit is None:
            continue
        model.by_circuit[str(circuit)] = _Fit(
            fuel=_shrink(fit.fuel, glob.fuel, fit.n, k_circuit),
            offset={c: _shrink(fit.offset.get(c, np.nan), glob.offset[c], fit.n, k_circuit)
                    for c in COMPOUNDS},
            deg={c: _shrink(fit.deg.get(c, np.nan), glob.deg[c], fit.n, k_circuit)
                 for c in COMPOUNDS},
            resid_std=fit.resid_std, n=fit.n,
        )

    # per-driver additive deg deviation + noise, shrunk toward global
    for driver, sub in laps.groupby("driver"):
        fit = _fit_within(sub)
        if fit is None:
            model.noise_by_driver[str(driver)] = glob.resid_std
            continue
        model.deg_dev_by_driver[str(driver)] = {
            c: _shrink((fit.deg[c] - glob.deg[c]) if np.isfinite(fit.deg[c]) else np.nan,
                       0.0, fit.n, k_driver)
            for c in COMPOUNDS
        }
        model.noise_by_driver[str(driver)] = _shrink(fit.resid_std, glob.resid_std,
                                                      fit.n, k_driver)
    return model


def describe(model: LapModel) -> str:
    g = model.glob
    lines = [f"global: fuel={g.fuel:.4f} s/lap  n={g.n}  noise_std={g.resid_std:.3f}"]
    lines.append("  offsets vs MEDIUM: " + ", ".join(f"{c}={model.glob.offset[c]:+.3f}" for c in COMPOUNDS))
    lines.append("  deg slopes: " + ", ".join(f"{c}={model.glob.deg[c]:.4f}" for c in COMPOUNDS))
    return "\n".join(lines)
