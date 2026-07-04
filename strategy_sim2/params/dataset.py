"""Assemble training tables from cached sessions.

Different parameters use different windows (a per-parameter-window policy):
  * lap model  -> dry included races only (clean laps)
  * DNF/start  -> every scanned race in the manifest (dry + wet), since reliability
                  and opening-lap chaos are not dry-specific.
Per-race results and lap-1 positions are cached to disk to avoid reloading sessions.
"""
from __future__ import annotations

import pandas as pd

from strategy_sim2.data import clean, collector, session_filter
from strategy_sim2.settings import load_settings, resolve_path


def _meta_path(cfg, kind: str, year: int, rnd: int):
    d = resolve_path(cfg["data"]["derived_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{kind}_{year}_{rnd:02d}.pkl"


def get_race_meta(year: int, rnd: int, cfg: dict | None = None,
                  use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return (results, lap1_positions) for one race, cached. None if unavailable."""
    cfg = cfg or load_settings()
    rpath, lpath = _meta_path(cfg, "results", year, rnd), _meta_path(cfg, "lap1", year, rnd)
    if use_cache and rpath.exists() and lpath.exists():
        return pd.read_pickle(rpath), pd.read_pickle(lpath)
    ses = collector.load_session(year, rnd, "R", weather=False, messages=False)
    if ses is None:
        return None
    results = collector.session_results(ses)
    laps = collector.session_laps(ses)
    lap1 = (laps[laps["lap_number"] == 1][["year", "round", "driver", "position"]]
            .dropna(subset=["position"]).reset_index(drop=True))
    results.to_pickle(rpath)
    lap1.to_pickle(lpath)
    return results, lap1


def filter_window(df: pd.DataFrame, years: list[int] | None = None,
                  before: tuple[int, int] | None = None) -> pd.DataFrame:
    """Restrict a races frame by season list and/or a strict (year, round) cutoff.

    ``before=(y, r)`` keeps only races that happened strictly before round r of year y —
    the expanding-window rule used by backtests so training matches what production
    would have known pre-race (including completed current-season rounds).
    """
    if not len(df):
        return df
    if years is not None:
        df = df[df["year"].isin(years)]
    if before is not None:
        y, r = before
        df = df[(df["year"] < y) | ((df["year"] == y) & (df["round"] < r))]
    return df


def training_laps(cfg: dict | None = None, years: list[int] | None = None,
                  before: tuple[int, int] | None = None) -> pd.DataFrame:
    cfg = cfg or load_settings()
    races = filter_window(session_filter.included_races(cfg), years, before)
    frames = []
    for _, r in races.iterrows():
        df = clean.get_clean_race(int(r["year"]), int(r["round"]), cfg)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    alllaps = pd.concat(frames, ignore_index=True)
    return alllaps[alllaps["is_clean"]].reset_index(drop=True)


def _all_manifest_races(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_settings()
    m = session_filter.load_manifest(cfg)
    return m if len(m) else session_filter.build_manifest(cfg)


def training_results(cfg: dict | None = None, years: list[int] | None = None,
                     before: tuple[int, int] | None = None) -> pd.DataFrame:
    cfg = cfg or load_settings()
    frames = []
    for _, r in filter_window(_all_manifest_races(cfg), years, before).iterrows():
        meta = get_race_meta(int(r["year"]), int(r["round"]), cfg)
        if meta is not None:
            frames.append(meta[0])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def training_lap1(cfg: dict | None = None, years: list[int] | None = None,
                  before: tuple[int, int] | None = None) -> pd.DataFrame:
    cfg = cfg or load_settings()
    frames = []
    for _, r in filter_window(_all_manifest_races(cfg), years, before).iterrows():
        meta = get_race_meta(int(r["year"]), int(r["round"]), cfg)
        if meta is not None:
            frames.append(meta[1])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
