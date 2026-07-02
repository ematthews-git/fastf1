import fastf1 as f1
from fastf1 import plotting as f1plt
import requests
import pandas as pd

TRACK = "silverstone"
YEAR = 2025

f1.Cache.enable_cache("./f1_cache")


def get_sc_prob(track: str):

    session = f1.get_session(YEAR, 1, "R")
    session.load(messages=True, weather=False, telemetry=False)

    rc = session.race_control_messages
    sc_messages = rc[
        rc["Message"].str.contains("SAFETY CAR", na=False)
        & rc["Status"].str.contains("DEPLOYED|IN THIS LAP", na=False)
    ]

    print(sc_messages)


def get_red_prob():
    session = f1.get_session(2021, "jeddah", "R")
    session.load(messages=True, weather=False, telemetry=False)

    ver = session.laps.pick_drivers("VER")

    print(session.laps)
    # for lap in session.laps:
    #     # print(lap)
    #     print(lap)

    print(ver["Time"])

    # rc = session.race_control_messages
    # print(rc)

    # red = rc[rc["Message"] == "RED FLAG"]
    # print(red)
    # print(len(red))


# What info do I want?
# - lap record
# - weather (for upcoming) + historical
def get_upcoming(track: str) -> str:
    race = f1.get_event(2026, track)
    return race
    # provides date, name, format, and dates of specific events
    # whats in the API?


def lap_record(track: str):
    """Finds track record. is slow."""
    rows = []
    offset = 0
    limit = 100

    # Paginate through all results
    while True:
        data = requests.get(
            f"https://api.jolpi.ca/ergast/f1/circuits/{track}/qualifying.json",
            params={"limit": limit, "offset": offset},
        ).json()

        mrdata = data["MRData"]
        races = mrdata["RaceTable"]["Races"]
        if not races:
            break

        for race in races:
            for result in race["QualifyingResults"]:
                best_time = result.get("Q3") or result.get("Q2") or result.get("Q1")
                if not best_time:
                    continue
                rows.append(
                    {
                        "season": race["season"],
                        "race": race["raceName"],
                        "driver": result["Driver"]["familyName"],
                        "best_time": best_time,
                    }
                )

        offset += limit
        if offset >= int(mrdata["total"]):
            break

    df = pd.DataFrame(rows)
    df["q_time"] = pd.to_timedelta("00:0" + df["best_time"])

    record = df.loc[df["q_time"].idxmin()]
    print(
        f"Track record (qualifying): {record['driver']} — {record['q_time']} ({record['race']}, {record['season']})"
    )


def get_strategy(year, track):
    # two years ?
    session = f1.get_session(year, track, "R")
    session.load()
    print(session)

    laps = session.laps
    print(laps)

    tyre_data = laps[["Driver", "Stint", "Compound", "TyreLife", "LapNumber"]].copy()
    # this object has an entry for every lap and every driver

    pit_stops = _get_pit_stops(laps)

    stints = (
        laps.groupby(["Driver", "Stint", "Compound"])
        .agg(
            StartLap=("LapNumber", "min"),
            EndLap=("LapNumber", "max"),
            Laps=("LapNumber", "count"),
        )
        .reset_index()
    )

    print(stints)

    strategy_per_driver = (
        stints.groupby("Driver")["Compound"]
        .apply(lambda x: "->".join(x))
        .reset_index()
        .rename(columns={"Compound": "Strategy"})
    )
    print(strategy_per_driver)

    # equivalent strategies can be defined as any strategy that goes compound -> compound -> compound ?
    # lets say yes, but identify anomolous examples by looking at pit window

    # ok we have a clean strategy object - now we need pit window
    # pit window cant just be the entire window -
    # it must group by equivelent strategies - create an array of all laps pitted - pit window can be the middle 70% ?

    strategy_counts = (
        strategy_per_driver.groupby("Strategy")
        .agg(Count=("Driver", "count"), Drivers=("Driver", list))
        .sort_values("Count", ascending=False)
    )
    print(strategy_counts)


def calculate_pit_window(sessions: list, stint_number: int = 1):
    """Calculates pint window based on previous (dry) year.

    Args:
        sessions (list): List of sessions for the same circuit
        stint_number (int, optional): Which stop to calculate the window for. Defaults to 1.
    """


def _get_pit_stops(laps):
    """Gets a neatly organised dataframe of pit stops.

    (Driver, StintIn, LapNumber_x, PitInTime, LapNumber_y, PitOutTime, PitDuration)
    """
    pit_in = laps[laps["PitInTime"].notna()][
        ["Driver", "Stint", "LapNumber", "PitInTime"]
    ]
    pit_out = laps[laps["PitOutTime"].notna()][
        ["Driver", "Stint", "LapNumber", "PitOutTime"]
    ]

    # PitOut belongs to the *next* stint, so match Stint N in with Stint N+1 out
    pit_in = pit_in.rename(columns={"Stint": "StintIn"})
    pit_out = pit_out.rename(columns={"Stint": "StintOut"})
    pit_out["StintIn"] = pit_out["StintOut"] - 1

    pit_stops = pit_in.merge(pit_out, on=["Driver", "StintIn"])

    pit_stops["PitDuration"] = (
        pit_stops["PitOutTime"] - pit_stops["PitInTime"]
    ).dt.total_seconds()

    return pit_stops


def _pit_loss_normal():
    # use sessions, get median pit time under exclusively NORMAL conditions

    sessions = []

    s = f1.get_session(YEAR - 1, TRACK, "R")
    s.load()

    sessions.append(s)

    pit_times = []

    for session in sessions:
        laps = session.laps
        print(len(laps))
        relevant_laps = laps[
            ~(laps["TrackStatus"].str.contains("[4567]", regex=True))
            & (laps["PitInTime"].notna() | laps["PitOutTime"].notna())
        ]


def main():
    # run functions to test
    print("Hello World")
    # print(get_upcoming(TRACK))
    # lap_record(TRACK)
    get_strategy(2026, "barcelona")


if __name__ == "__main__":
    main()
