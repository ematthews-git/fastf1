"""Rate-limited, resumable backfill of the training window into the local cache.

FastF1's public API allows ~500 calls/hour. Each race load costs several calls, so
the full 2021-2026 window can't be fetched at once. This tool paces network loads,
sleeps and resumes when the hourly budget is exhausted, and skips races already
processed (their derived pickles exist), so it can be re-run any time.

Run (background):  venv/bin/python -m strategy_sim2.data.backfill --delay 20
"""
from __future__ import annotations

import argparse
import time

from strategy_sim2.data import clean, collector
from strategy_sim2.params import dataset
from strategy_sim2.settings import load_settings


def _processed(cfg, year: int, rnd: int) -> bool:
    dpath = clean._derived_path(cfg, year, rnd)
    rpath, lpath = dataset._meta_path(cfg, "results", year, rnd), dataset._meta_path(cfg, "lap1", year, rnd)
    return dpath.exists() and rpath.exists() and lpath.exists()


def backfill(cfg: dict | None = None, delay: float = 20.0,
             cooldown: float = 3660.0, years: list[int] | None = None) -> None:
    cfg = cfg or load_settings()
    start, end = cfg["training"]["start_year"], cfg["training"]["end_year"]
    last_round_end = cfg["target"]["last_completed_round"]
    years = years or list(range(start, end + 1))

    done = fetched = skipped = 0
    for year in years:
        try:
            sched = collector.get_schedule(year)
        except Exception as e:
            print(f"[backfill] {year} schedule unavailable ({e}); skipping", flush=True)
            continue
        for rnd in [int(r) for r in sched["RoundNumber"] if int(r) >= 1]:
            if year == end and rnd > last_round_end:
                continue
            if _processed(cfg, year, rnd):
                done += 1
                continue

            # Respect the hourly budget: wait for reset if we've been limited.
            if collector.rate_limited():
                print(f"[backfill] rate limited; sleeping {cooldown:.0f}s", flush=True)
                time.sleep(cooldown)
                collector._RATE_LIMITED = False

            ses = collector.load_session(year, rnd, "R", weather=True, messages=False)
            if ses is None:
                if collector.rate_limited():
                    print(f"[backfill] limit hit at {year} R{rnd}; sleeping {cooldown:.0f}s", flush=True)
                    time.sleep(cooldown)
                    collector._RATE_LIMITED = False
                else:
                    skipped += 1
                continue

            clean.get_clean_race(year, rnd, cfg)   # writes laps pkl (reads cache)
            dataset.get_race_meta(year, rnd, cfg)  # writes results/lap1 pkl (reads cache)
            fetched += 1
            print(f"[backfill] fetched {year} R{rnd} ({ses.event['EventName']})  "
                  f"[done={done} fetched={fetched} skipped={skipped}]", flush=True)
            time.sleep(delay)

    print(f"[backfill] complete: already_done={done} fetched={fetched} "
          f"unavailable={skipped}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=20.0, help="seconds between network loads")
    ap.add_argument("--cooldown", type=float, default=3660.0, help="sleep on rate limit")
    ap.add_argument("--years", type=int, nargs="*", default=None)
    args = ap.parse_args()
    backfill(delay=args.delay, cooldown=args.cooldown, years=args.years)
