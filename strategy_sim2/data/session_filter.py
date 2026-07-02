"""Dry / wet / mixed session detection and a transparent inclusion manifest.

This simulator models dry races only. A session is dry iff no rainfall was recorded
*and* no driver ran wet or intermediate tyres. Every session we consider is written
to ``data/manifest.json`` with the inclusion decision and its reason, so the training
set is fully auditable and reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from strategy_sim2.data import collector
from strategy_sim2.data.schema import WET_COMPOUNDS
from strategy_sim2.settings import load_settings, resolve_path


def classify_dry(ses) -> tuple[bool, str]:
    """Return (is_dry, reason) for a loaded session."""
    weather = collector.weather_summary(ses)
    if weather["rainfall_any"]:
        return False, "rainfall recorded"
    laps = collector.session_laps(ses)
    wet_laps = int(laps["compound"].isin(WET_COMPOUNDS).sum())
    if wet_laps > 0:
        return False, f"wet/intermediate tyres used ({wet_laps} laps)"
    return True, "dry"


def _manifest_entry(ses) -> dict:
    meta = collector.session_meta(ses)
    is_dry, reason = classify_dry(ses)
    # A race must have enough laps/drivers to be usable for parameter fitting.
    usable = is_dry and meta.n_laps >= 20 and meta.n_drivers >= 10
    if is_dry and not usable:
        reason = f"dry but unusable (laps={meta.n_laps}, drivers={meta.n_drivers})"
    return {
        "year": meta.year, "round": meta.round, "session": "R",
        "circuit": meta.circuit, "event_name": meta.event_name,
        "event_format": meta.event_format, "date": meta.event_date,
        "n_laps": meta.n_laps, "n_drivers": meta.n_drivers,
        "is_dry": is_dry, "included": bool(usable), "reason": reason,
    }


def build_manifest(cfg: dict | None = None, years: list[int] | None = None,
                   save: bool = True) -> pd.DataFrame:
    """Scan the training window's races and record inclusion decisions.

    First run downloads every race (slow, then cached). Rounds with no data yet
    (future races) are skipped silently.
    """
    cfg = cfg or load_settings()
    start = cfg["training"]["start_year"]
    end = cfg["training"]["end_year"]
    last_round_end = cfg["target"]["last_completed_round"]
    years = years or list(range(start, end + 1))

    entries: list[dict] = []
    for year in years:
        sched = collector.get_schedule(year)
        for rnd in sched["RoundNumber"].tolist():
            rnd = int(rnd)
            if rnd < 1:
                continue
            if year == end and rnd > last_round_end:
                continue  # not yet run
            ses = collector.load_session(year, rnd, "R", weather=True, messages=False)
            if ses is None:
                continue
            entries.append(_manifest_entry(ses))

    manifest = pd.DataFrame(entries)
    if save and len(manifest):
        path = resolve_path(cfg["data"]["manifest_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "training_window": {"start_year": start, "end_year": end,
                                "end_last_round": last_round_end},
            "n_total": int(len(manifest)),
            "n_included": int(manifest["included"].sum()),
            "n_excluded": int((~manifest["included"]).sum()),
            "sessions": entries,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    return manifest


def load_manifest(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_settings()
    path = resolve_path(cfg["data"]["manifest_path"])
    if not Path(path).exists():
        return pd.DataFrame()
    with open(path) as f:
        payload = json.load(f)
    return pd.DataFrame(payload["sessions"])


def included_races(cfg: dict | None = None) -> pd.DataFrame:
    m = load_manifest(cfg)
    return m[m["included"]].reset_index(drop=True) if len(m) else m
