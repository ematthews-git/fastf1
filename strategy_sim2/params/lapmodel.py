"""Joint fuel + tyre lap-time model with a degradation cliff.

Fuel and tyre degradation are jointly identified from race laps via a fixed-effects
"within" regression (per driver-race intercept absorbed by demeaning, so no quali data
is needed). The linear degradation slope per compound is well identified.

Two effects can't be read straight off the regression and are handled explicitly:

* **Degradation cliff.** Teams pit *before* the cliff, so it is censored from the data
  and a quadratic term estimates as ~0. We instead place a knee at the observed stint
  length (data-driven) and add an accelerating penalty beyond it (config rate), which
  shortens optimal stints (more stops) and discourages ending a race on a soft tyre.
* **Compound pace offsets.** Softs run early at high fuel confound with the fuel term, so
  the regression indicator is compressed/noisy. We estimate a direct fresh-tyre offset and
  regularise it toward the known SOFT<MEDIUM<HARD ordering (empirical-Bayes, not a fudge).

Per-lap model: lap = fuel*laps_remaining + offset_c + slope_c*age + cliff_c(age) + noise
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy_sim2.settings import load_settings

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
_REF = "MEDIUM"
_MIN_ROWS = 40


@dataclass
class _Fit:
    fuel: float
    offset: dict            # compound -> regression pace offset vs MEDIUM (fallback only)
    deg: dict               # compound -> linear deg slope (s / lap)
    resid_std: float
    n: int


def _demean(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    deg_dev_by_driver: dict[str, dict] = field(default_factory=dict)
    noise_by_driver: dict[str, float] = field(default_factory=dict)
    offsets_global: dict = field(default_factory=dict)   # regularised fresh-tyre offsets
    knee: dict = field(default_factory=dict)             # compound -> cliff onset (laps)
    knee_by_circuit: dict = field(default_factory=dict)  # circuit -> {compound -> knee}
    cliff_rate: float = 0.02                             # s / lap^2 beyond the knee

    def fuel_coef(self, circuit: str | None = None) -> float:
        f = self.by_circuit.get(circuit)
        return f.fuel if f else self.glob.fuel

    def pace_offset(self, compound: str, circuit: str | None = None) -> float:
        v = self.offsets_global.get(compound)
        if v is not None and np.isfinite(v):
            return float(v)
        f = self.by_circuit.get(circuit, self.glob)
        v = f.offset.get(compound)
        return float(v) if v is not None and np.isfinite(v) else 0.0

    def _slope(self, compound, circuit, driver):
        f = self.by_circuit.get(circuit, self.glob)
        base = f.deg.get(compound)
        if base is None or not np.isfinite(base):
            base = self.glob.deg[compound]
        dev = self.deg_dev_by_driver.get(driver, {}).get(compound, 0.0) if driver else 0.0
        return max(0.0, float(base + dev))

    def _cliff(self, compound, age, circuit=None):
        knee = self.knee_by_circuit.get(circuit, {}).get(compound)
        if knee is None or not np.isfinite(knee):
            knee = self.knee.get(compound, 1e9)
        over = max(0.0, float(age) - knee)
        return self.cliff_rate * over * over

    def deg_slope(self, compound, circuit=None, driver=None) -> float:
        return self._slope(compound, circuit, driver)

    def deg(self, compound, age, circuit=None, driver=None) -> float:
        return self._slope(compound, circuit, driver) * float(age) + self._cliff(compound, age, circuit)

    def noise_std(self, driver: str | None = None) -> float:
        return float(self.noise_by_driver.get(driver, self.glob.resid_std))

    def deg_severity(self, circuit: str | None = None) -> float:
        return float(np.mean([self.deg(c, 20, circuit) / 20.0 for c in COMPOUNDS]))


def _direct_offsets(laps: pd.DataFrame, model: LapModel) -> dict:
    """Fresh-tyre (age 2-4) fuel/age-corrected pace per compound, relative to MEDIUM."""
    df = laps.dropna(subset=["lap_time_s", "laps_remaining", "tyre_life", "compound"])
    fresh = df[(df["tyre_life"] >= 2) & (df["tyre_life"] <= 4) & df["compound"].isin(COMPOUNDS)]
    if len(fresh) < 30:
        return {}
    rows = []
    for r in fresh.itertuples():
        c = r.lap_time_s - model.fuel_coef(r.circuit) * r.laps_remaining \
            - model.deg(r.compound, r.tyre_life, r.circuit)
        rows.append((r.year, r.round, r.driver, r.compound, c))
    cd = pd.DataFrame(rows, columns=["year", "round", "driver", "compound", "corr"])
    piv = cd.groupby(["year", "round", "driver", "compound"])["corr"].median().unstack("compound")
    if _REF not in piv.columns:
        return {}
    out = {_REF: 0.0}
    for c in COMPOUNDS:
        if c != _REF and c in piv.columns:
            dev = (piv[c] - piv[_REF]).dropna()
            out[c] = float(dev.mean()) if len(dev) >= 5 else np.nan
    return out


def _stint_table(laps: pd.DataFrame) -> pd.DataFrame:
    df = laps.dropna(subset=["stint", "tyre_life", "compound"]).copy()
    df["compound"] = df["compound"].astype(str)
    return df.groupby(["circuit", "year", "round", "driver", "stint"]).agg(
        comp=("compound", "first"), age=("tyre_life", "max")).reset_index()


def _knees_from(st: pd.DataFrame, pct: float, min_n: int, default) -> dict:
    out = {}
    for c in COMPOUNDS:
        vals = pd.to_numeric(st.loc[st["comp"] == c, "age"], errors="coerce").dropna().to_numpy()
        out[c] = float(np.percentile(vals, pct)) if len(vals) >= min_n else default
    return out


def fit_lap_model(laps: pd.DataFrame, cfg: dict | None = None) -> LapModel:
    cfg = cfg or load_settings()
    p = cfg.get("params", {})
    k_circuit = float(p.get("k_circuit", 500))
    k_driver = float(p.get("k_driver", 800))
    prior = {k.upper(): float(v) for k, v in
             p.get("compound_offset_prior", {"SOFT": -0.35, "MEDIUM": 0.0, "HARD": 0.25}).items()}
    w_prior = float(p.get("offset_prior_weight", 0.5))

    glob = _fit_within(laps)
    if glob is None:
        raise ValueError("insufficient clean laps to fit the lap model")
    model = LapModel(glob=glob, cliff_rate=float(p.get("cliff_rate", 0.02)))
    pct = float(p.get("knee_percentile", 80))
    st = _stint_table(laps)
    model.knee = _knees_from(st, pct, min_n=10, default=40.0)
    # Per-circuit tyre life: softs die sooner at abrasive tracks (blend toward global).
    for circ, sub in st.groupby("circuit"):
        kc = _knees_from(sub, pct, min_n=8, default=None)
        model.knee_by_circuit[str(circ)] = {
            c: (0.7 * kc[c] + 0.3 * model.knee[c]) if kc[c] is not None else model.knee[c]
            for c in COMPOUNDS}

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
        model.noise_by_driver[str(driver)] = _shrink(fit.resid_std, glob.resid_std, fit.n, k_driver)

    # Direct fresh-tyre offsets, regularised toward the known compound ordering.
    direct = _direct_offsets(laps, model)
    model.offsets_global = {
        c: (1 - w_prior) * direct.get(c, prior.get(c, 0.0)) + w_prior * prior.get(c, 0.0)
        if np.isfinite(direct.get(c, np.nan)) else prior.get(c, 0.0)
        for c in COMPOUNDS
    }
    return model


def describe(model: LapModel) -> str:
    g = model.glob
    return "\n".join([
        f"global: fuel={g.fuel:.4f} s/lap  n={g.n}  noise_std={g.resid_std:.3f}",
        "  offsets vs MEDIUM: " + ", ".join(f"{c}={model.pace_offset(c):+.3f}" for c in COMPOUNDS),
        "  deg slope: " + ", ".join(f"{c}={g.deg[c]:.4f}" for c in COMPOUNDS),
        "  knee (laps): " + ", ".join(f"{c}={model.knee.get(c, 0):.0f}" for c in COMPOUNDS),
        "  deg @30 (cliff): " + ", ".join(f"{c}={model.deg(c, 30):.2f}s" for c in COMPOUNDS),
    ])
