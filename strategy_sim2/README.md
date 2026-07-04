# strategy_sim2 — F1 pre-race strategy simulator

Generates the ~2–5 most **plausible** strategy candidates per driver before a Grand Prix,
ranked by **expected race outcome** (finishing-position distribution) — not clean-air race
time. Built independently, using [FastF1](https://github.com/theOehrly/Fast-F1) data and the
discrete-event framework of Sulsters (2018) (`../paper-sulsters.md`) as a conceptual
reference, adapted and improved for modern (3-compound) F1.

## Philosophy

Real teams optimise around track position, traffic, overtaking difficulty, degradation,
safety cars and risk — not lap-time alone. So this engine:

1. **Generates** the full *rule-legal* candidate space and weights each with a historical
   **plausibility prior** (history informs, never excludes — a novel strategy made
   competitive by new compounds/regs can still win).
2. **Evaluates** each candidate with a full-field Monte-Carlo race simulation where track
   position and traffic actually cost time.
3. **Selects** the ranked 2–5 by expected finishing outcome, with P(optimal) and diversity.

Generation, evaluation and selection are independent components.

## Pipeline

```
data/        FastF1 collection, dry/wet filtering (manifest), clean-lap building,
             strategy extraction (red-flag/SC stint flurries merged), rate-limited backfill
params/      lap model (joint fuel+tyre, deg cliff w/ data-driven knees, hierarchical/shrunk),
             weekend practice/sprint long-run model (relative deg, offsets, usage),
             DNF (Beta-Bernoulli), start-line, circuit profiles
generation/  ALL rule-legal candidates (>=2 compounds + physical set allocation) with
             recency-weighted priors (stop count, pattern, start compound, weekend usage);
             family-coverage shortlist; history-calibrated pit windows (+undercut shift)
sim/         discrete-event lap-by-lap simulator: overtaking(+DRS), safety car/VSC, pits
context/     postquali (grid+quali+practice) and prelim (pre-weekend, current-season form);
             expanding-window rule: priors/params only see races strictly before the target
evaluation/  Monte-Carlo with common random numbers; per-candidate outcome distributions
selection/   rank 2-5 by expected finish (given finished) + priors, P(optimal), diversity
report/      JSON output for the strategy page (incl. pit-window ranges)
validation/  finish backtest + strategy-accuracy backtest (--strategy, expanding window)
tuning/      Optuna OOS tuning (2024/25 train folds -> 2026 holdout)
```

## Usage

```bash
# one-time: cache FastF1 data (rate-limited, resumable) and fit parameters
venv/bin/python -m strategy_sim2.smoke                         # verify data access
venv/bin/python -m strategy_sim2.data.backfill --delay 20     # fill training window (repeatable)

# main mode: after qualifying (a completed race for validation, or a run weekend)
venv/bin/python -m strategy_sim2.run --mode postquali --year 2026 --round 8 --sims 1000

# preliminary mode: upcoming race, no sessions yet (current-season form + prev year)
venv/bin/python -m strategy_sim2.run --mode prelim --year 2026 --round 9

# out-of-sample validation and tuning
venv/bin/python -m strategy_sim2.validation.backtest --test-year 2024 --max-races 5
venv/bin/python -m strategy_sim2.validation.backtest --strategy --test-year 2026 --sims 200
venv/bin/python -m strategy_sim2.data.backfill --practice --years 2026 --rounds 1 2 3   # FP/Sprint cache
venv/bin/python -m strategy_sim2.tuning.optuna_tune --trials 25 --sims 80
```

Output: `strategy_sim2/output/<year>_<round>_<mode>.json` — per driver, ranked candidates
with `expected_finish` (given finished), full `finish_distribution`, `p_win/p_podium/p_points`,
`p_dnf`, `p_optimal`, CI, planned pit laps and plausibility prior; plus run metadata and the
derived circuit profile. Config lives in `config/settings.yaml` (nothing hardcoded).

## Data & configuration

- Training window **2021–2026** (set in `config/settings.yaml`); estimators are
  recency-weighted and hierarchically pooled because 2026 is a new regulation set.
- Dry-only: wet/mixed sessions are auto-detected and excluded; every decision is logged to
  `data/manifest.json`.
- FastF1 enforces ~500 API calls/hour — `data.backfill` paces and resumes accordingly.

## Validated results

- Simulator: Spearman(grid, finish) ≈ 0.99; ~3 DNFs/race tracking fitted per-driver rates.
- Parameters (90 dry races): fuel ≈ 0.05 s/lap; deg SOFT>MEDIUM>HARD; DNF ≈ 12% with prior
  Beta(4.1, 30.2) (cf. paper's Beta(2.9, 13.1)); circuit deg ranks Bahrain/Spa/Barcelona high,
  Monaco/Jeddah low; Monaco hardest to overtake, Interlagos easiest.
- In-sample: Spearman(predicted E[finish], actual) = 0.95 (Austria 2026).
- Out-of-sample (train 2021–23 → test 2024): finish Spearman ≈ 0.74, actual strategy
  generated ~70% and ranked in the top-5 ~68%.

## Known limitations / future work

- Compound pace offsets are modest (identification limited by within-race demeaning).
- Circuits with few dry races (e.g. Silverstone) have noisier degradation → shrunk to global.
- Extensible (unused yet): practice-session ingestion, weather, live in-race Bayesian updates,
  team-specific pit times, tyre-temperature and track-evolution effects.
```
