"""Derived fan-facing statistics for the pre-race page.

Everything here is a PURE READ over objects the pipeline has already produced — the
per-driver Monte-Carlo outcomes, the fitted parameter set (lap/DNF models), the circuit
profile and the plausibility prior. Nothing in this module re-simulates, re-selects or
re-fits, so surfacing these numbers cannot change the model's actual predictions; it only
repackages what selection/evaluation already computed.

Two groups:
  * per-driver   -> plausibility-weighted win/podium/points, projected finish + grid mover,
                    reliability, tyre-management index
  * race-level   -> tyre life & compound deltas, undercut power, degradation rank, SC/VSC,
                    overtaking, stop-count split, chaos index, pole-to-win
"""
from __future__ import annotations

import numpy as np

from strategy_sim2.params.lapmodel import COMPOUNDS


def _r(x, n=3):
    return None if x is None or (isinstance(x, float) and x != x) else round(float(x), n)


# --------------------------------------------------------------------------- driver-level
def _weights(sel) -> np.ndarray:
    """Plausibility shares across a driver's shown strategies (equal if all zero)."""
    w = np.array([max(getattr(s, "plausibility", 0.0), 0.0) for s in sel], dtype=float)
    return w / w.sum() if w.sum() > 0 else np.full(len(sel), 1.0 / len(sel))


def _tyre_management(driver: str, lap_model, circuit: str) -> float:
    """Signed index: how much kinder (+) or harsher (-) this driver is on tyres than the
    field, as a fraction of the circuit's degradation slope. From the fitted per-driver
    deg deviation — 0 when the driver has no reliable deviation estimate."""
    dev = getattr(lap_model, "deg_dev_by_driver", {}).get(driver, {})
    if not dev:
        return 0.0
    slope_mean = float(np.mean([lap_model.deg_slope(c, circuit) for c in COMPOUNDS]))
    if slope_mean <= 1e-6:
        return 0.0
    dev_mean = float(np.mean([dev.get(c, 0.0) for c in COMPOUNDS]))
    return float(np.clip(-dev_mean / slope_mean, -1.0, 1.0))


def aggregate_driver(driver: str, sel, grid: int, dnf_model, lap_model,
                     circuit: str) -> dict | None:
    """Plausibility-weighted driver-level summary over the shown strategies. ``None`` when
    the driver has no candidates. Keys prefixed ``_`` are internal (not serialised)."""
    if not sel:
        return None
    w = _weights(sel)
    outs = [s.outcome for s in sel]

    p_win = float(sum(wi * o.p_win for wi, o in zip(w, outs)))
    p_podium = float(sum(wi * o.p_podium for wi, o in zip(w, outs)))
    p_points = float(sum(wi * o.p_points for wi, o in zip(w, outs)))
    exp_finish = float(sum(wi * o.mean_finish_classified for wi, o in zip(w, outs)))

    # Spread of the plausibility-mixture over ALL sims (incl. DNFs) via law of total
    # variance — the per-driver ingredient of the race chaos index.
    m_all = np.array([float(np.mean(o.finishes)) for o in outs])
    v_all = np.array([float(np.var(o.finishes)) for o in outs])
    mix_mean = float((w * m_all).sum())
    finish_std = float(np.sqrt(max((w * (v_all + m_all ** 2)).sum() - mix_mean ** 2, 0.0)))

    projected = int(round(exp_finish))
    return {
        "p_win": _r(p_win, 4),
        "p_podium": _r(p_podium, 4),
        "p_points": _r(p_points, 4),
        "expected_finish": _r(exp_finish, 2),
        "projected_finish": projected,
        "grid_to_finish_delta": int(grid - projected),  # +ve = gains places vs grid
        "dnf_prob": _r(dnf_model.dnf_prob(driver), 3),
        "tyre_management_vs_field": _r(_tyre_management(driver, lap_model, circuit), 3),
        "_finish_std": finish_std,
        "_p_win": p_win,
    }


# ----------------------------------------------------------------------------- race-level
def _tyre_life(lap_model, circuit: str, compound: str, n_laps: int, min_stint: int):
    """Usable stint laps before the degradation cliff (the data-driven knee)."""
    knee = getattr(lap_model, "knee_by_circuit", {}).get(circuit, {}).get(compound)
    if knee is None or not np.isfinite(knee):
        knee = getattr(lap_model, "knee", {}).get(compound)
    if knee is None or not np.isfinite(knee):
        return None
    return int(round(float(np.clip(knee, min_stint, n_laps))))


def _undercut_power(lap_model, prior, circuit: str, n_laps: int, min_stint: int):
    """Per-lap pace advantage of a fresh MEDIUM over a worn one at a representative
    first-stop age — the essence of undercut strength at this track."""
    worn_age = None
    mp = prior.median_pit_laps(1, n_laps)
    if mp:
        worn_age = mp[0]
    if not worn_age:
        worn_age = max(min_stint, n_laps // 2)
    gain = lap_model.deg("MEDIUM", worn_age, circuit) - lap_model.deg("MEDIUM", 2, circuit)
    return _r(max(gain, 0.0), 2)


def _deg_rank(profiles, circuit: str) -> dict | None:
    """Where this circuit's degradation severity ranks among all known circuits
    (rank 1 = highest deg)."""
    if not profiles or circuit not in profiles:
        return None
    sev = {c: p.deg_severity for c, p in profiles.items()}
    rank = 1 + sum(1 for v in sev.values() if v > sev[circuit])
    return {"rank": rank, "of": len(sev)}


def race_stats(wctx, driver_agg: dict, profiles, cfg: dict) -> dict:
    p = wctx.profile
    lm = wctx.params.lap
    prior = wctx.prior
    circuit = wctx.circuit
    n_laps = int(p.n_laps)
    n_pos = len(wctx.drivers())
    min_stint = int(cfg["generation"].get("min_stint", 6))

    stops = {n: prior.stop_prior(n) for n in range(1, int(prior.max_stops) + 1)}
    tot = sum(stops.values()) or 1.0

    aggs = [a for a in driver_agg.values() if a]
    chaos = None
    if aggs:
        uniform_std = n_pos / (12 ** 0.5)  # spread of a uniform finish over the field
        if uniform_std > 0:
            mean_std = float(np.mean([a["_finish_std"] for a in aggs]))
            chaos = int(round(100 * min(max(mean_std / uniform_std, 0.0), 1.0)))

    pole = min(wctx.drivers(), key=lambda d: wctx.grid[d])
    pole_agg = driver_agg.get(pole)

    return {
        "tyre_life_laps": {c: _tyre_life(lm, circuit, c, n_laps, min_stint) for c in COMPOUNDS},
        "compound_pace_s_vs_medium": {c: _r(lm.pace_offset(c, circuit), 2) for c in COMPOUNDS},
        "undercut_s_per_lap": _undercut_power(lm, prior, circuit, n_laps, min_stint),
        "pit_loss_s": _r(p.pit_loss, 2),
        "degradation": {"severity": _r(p.deg_severity, 4), **( _deg_rank(profiles, circuit) or {})},
        "safety_car_prob": _r(p.sc_prob, 2),
        "vsc_prob": _r(p.vsc_prob, 2),
        "expected_sc_vsc_laps": _r(p.sc_expected_laps, 1),
        "overtaking_difficulty_0to100": int(round(100 * p.overtaking_difficulty)),
        "expected_on_track_passes": _r(p.passes_per_race, 1),
        "stop_count_distribution": {str(n): _r(v / tot, 3) for n, v in stops.items()},
        "most_likely_stops": int(max(stops, key=stops.get)),
        "chaos_index_0to100": chaos,
        "pole_to_win_prob": (_r(pole_agg["_p_win"], 4) if pole_agg else None),
    }
