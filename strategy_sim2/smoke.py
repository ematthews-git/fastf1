"""Phase-0 smoke test: prove FastF1 data access and confirm the schema we build on.

Run:  venv/bin/python -m strategy_sim2.smoke
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import fastf1

from strategy_sim2.settings import cache_dir, load_settings

REQUIRED_LAP_COLS = [
    "Driver", "DriverNumber", "Team", "LapNumber", "LapTime", "Stint",
    "Compound", "TyreLife", "FreshTyre", "PitInTime", "PitOutTime",
    "TrackStatus", "Position", "IsAccurate", "Deleted",
]
REQUIRED_RESULT_COLS = ["Abbreviation", "GridPosition", "Position", "Status", "Q1", "Q2", "Q3"]


def main() -> None:
    cfg = load_settings()
    fastf1.Cache.enable_cache(str(cache_dir(cfg)))

    year = cfg["target"]["year"]
    upcoming = cfg["target"]["upcoming_round"]
    last_done = cfg["target"]["last_completed_round"]

    sch = fastf1.get_event_schedule(year, include_testing=False)
    print(f"{year} schedule: {len(sch)} rounds")
    ev = sch.loc[sch["RoundNumber"] == upcoming].iloc[0]
    print(f"  upcoming round {upcoming}: {ev['EventName']} @ {ev['Location']} "
          f"({ev['EventDate'].date()}, {ev['EventFormat']})  <- preliminary mode target")

    ses = fastf1.get_session(year, last_done, "R")
    ses.load(telemetry=False, weather=True, messages=False)
    laps = ses.laps
    res = ses.results

    print(f"\nLoaded {year} R{last_done} {ses.event['EventName']} (R): "
          f"laps={laps.shape}, drivers={laps['Driver'].nunique()}")

    missing_laps = [c for c in REQUIRED_LAP_COLS if c not in laps.columns]
    missing_res = [c for c in REQUIRED_RESULT_COLS if c not in res.columns]
    print("  required lap columns present:", not missing_laps, missing_laps or "")
    print("  required result columns present:", not missing_res, missing_res or "")
    print("  compounds:", laps["Compound"].value_counts(dropna=False).to_dict())
    print("  track-status codes seen:", sorted(laps["TrackStatus"].dropna().unique().tolist()))
    print("  weather rainfall any:", bool(ses.weather_data["Rainfall"].any()))

    ok = not missing_laps and not missing_res
    print("\nSMOKE", "OK" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
