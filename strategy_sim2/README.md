# strategy_sim2 — F1 pre-race strategy simulator

Generates the ~2–5 most **plausible** strategy candidates before a Grand Prix, ranked so the
**modal strategy the field actually runs is surfaced in the shown five**, with each candidate's
**expected race outcome** (finishing-position distribution) attached — not clean-air race time.
Built independently, using [FastF1](https://github.com/theOehrly/Fast-F1) data and the
discrete-event framework of Sulsters (2018) (`../paper-sulsters.md`) as a conceptual reference,
adapted for modern (3-compound) F1.

## Philosophy

Real teams optimise around track position, traffic, overtaking difficulty, degradation, safety
cars and risk — not lap-time alone. So this engine:

1. **Generates** the full *rule-legal* candidate space and weights each with a historical
   **plausibility prior** (history informs, never excludes — a novel strategy made competitive by
   new compounds/regs can still win).
2. **Evaluates** each candidate with a full-field Monte-Carlo race simulation where track position
   and traffic actually cost time.
3. **Selects** the shown 2–5 as a *set-cover* of plausibility mass, so the strategies most likely
   to be run — in the right compound order — are surfaced, each with its outcome distribution.

Generation, evaluation and selection are independent components.

## Inputs and outputs

```
                          ┌──────────────────────────────────────────────────┐
  EXTERNAL INPUTS         │                 strategy_sim2                     │        OUTPUT
                          │                                                  │
FastF1 API ───────────────┤  data/    collect · dry/wet filter (manifest)   │
  race/quali/practice     │           clean laps · realised-strategy extract │
  laps, results, weather  │                          │                       │
                          │                          ▼                       │
config/settings.yaml ─────┤  params/  lap model (fuel+tyre deg, cliff knees) │
  training window,        │           weekend long-runs · DNF · start-line   │
  allocation, blend knobs │           circuit profiles                       │
                          │                          │                       │
config/nominations.yaml ──┤  ── Pirelli C-numbers translate history into ─   │
  Pirelli C1–C6 per       │     the target race's SOFT/MEDIUM/HARD labels    │
  (year, circuit)         │                          │                       │
                          │                          ▼                       │
circuit_rules ────────────┤  generation/  ALL rule-legal candidates          │
  min_stops, max_stint,   │               + sequence-aware plausibility prior │
  per-year (Monaco/Lusail)│               + history-calibrated pit windows    │
                          │                          │  ~20-family shortlist  │
Race context ─────────────┤                          ▼                       │
  grid, quali pace,       │  sim/         discrete-event race × N (overtake, │
  entry list, teams       │               DRS, SC/VSC, pit loss, traffic)     │
  (postquali) OR          │                          │                       │
  season form (prelim)    │                          ▼                       │
                          │  evaluation/  Monte-Carlo (common random numbers)│──▶ output/<year>_<round>_
                          │               → per-candidate finish distribution │    <mode>.json
                          │                          │                       │
                          │                          ▼                       │    per driver: ranked 2–5
                          │  selection/   plausibility-mass set-cover → 2–5  │    candidates + derived
                          │               tiers, P(optimal), order diversity  │    stats (each candidate:
                          │                          │                       │     compounds+order, pit
                          │                          ▼                       │     laps+windows, expected
                          │  report/      JSON: strategies + race_stats +     │     finish, distribution,
                          │               per-driver derived numbers          │     CI, p_win/podium/points
                          └──────────────────────────────────────────────────┘     /dnf, p_optimal, tier)
                                                                              + race_stats (tyre life,
  expanding-window rule: params + priors only ever see races STRICTLY BEFORE    compound deltas, undercut,
  the target race — the backtest has no leakage.                                deg rank, SC/VSC, overtaking,
                                                                                stop split, chaos, pole-win)
  Derived stats are a pure read over the sim outputs + fitted params —        + per-driver win/podium/points,
  they never re-simulate or re-select, so they cannot change predictions.       projected finish + grid mover,
                                                                                reliability, tyre management
                                                                              + meta and circuit_profile
```

**Inputs, in full**

| Source | What it provides |
|---|---|
| **FastF1 API** | Race/qualifying/practice laps, classifications, weather (cached locally, rate-limited) |
| **`config/settings.yaml`** | Training window, recency decay, generation/allocation rules, prior blend knobs, selection knobs, sim parameters |
| **`config/nominations.yaml`** | Pirelli compound nominations (C-numbers, hard/medium/soft) per `(year, circuit)`, 2021–2026 — lets label-keyed history translate across a nomination shift |
| **`circuit_rules`** (in settings) | Public pre-race regulations: mandatory stop counts and mandated max stint lengths, year-scoped (Monaco 2-stop from 2025, Lusail stint caps) |
| **Race context** | *postquali*: grid, qualifying pace, entry list, teams, this weekend's practice long-runs. *prelim* (pre-weekend): current-season form, grid = rank of predicted pace |

**Output** — one JSON file per race (`output/<year>_<round>_<mode>.json`): run `meta`, the derived
`circuit_profile`, a `race_stats` block, and per driver a ranked list of 2–5 candidates plus
driver-level derived numbers. Each candidate carries the compound sequence and start compound,
planned pit laps and observed-window ranges, stint lengths, `expected_finish` (given the driver
finishes) and `expected_finish_all` (incl. DNF sims), the full `finish_distribution` with CI,
`p_win/p_podium/p_points/p_dnf`, `p_optimal`, `tier` (Most likely / Alternative / Long-shot) and
its `plausibility` share.

**Derived stats** (a pure, read-only repackaging of the sim outputs and fitted parameters — they
never re-simulate, re-select or re-fit, so they cannot change any prediction):

- **`race_stats`** — `tyre_life_laps` (usable laps to the cliff, per compound), `compound_pace_s_vs_medium`
  (negative = faster), `undercut_s_per_lap` (fresh-vs-worn medium), `pit_loss_s`, `degradation`
  (severity + rank among all circuits), `safety_car_prob` / `vsc_prob` / `expected_sc_vsc_laps`,
  `overtaking_difficulty_0to100`, `expected_on_track_passes`, `stop_count_distribution` +
  `most_likely_stops`, `chaos_index_0to100` (field-wide finish-spread), `pole_to_win_prob`.
- **Per driver** — plausibility-weighted `p_win` / `p_podium` / `p_points`, `expected_finish`,
  `projected_finish` and `grid_to_finish_delta` (the grid "mover"), `dnf_prob` (reliability), and
  `tyre_management_vs_field` (+ = kinder on tyres than the field).

Ordinal outputs (projected order, movers, head-to-heads) track well (OOS finish Spearman ≈ 0.74);
absolute probabilities inherit the sim's hand-set overtaking/SC calibration, so treat `p_win`,
`pole_to_win_prob` and `expected_on_track_passes` as directional until those params are fit.

## Pipeline (module by module)

```
data/        FastF1 collection, dry/wet filtering (manifest), clean-lap building, circuit-name
             normalisation (Monaco≡Monte Carlo), strategy extraction (SC/red-flag stint flurries
             merged), rate-limited backfill
params/      lap model (joint fuel+tyre, deg cliff w/ data-driven knees, hierarchical/shrunk),
             weekend practice/sprint long-run model (relative deg, offsets, usage), DNF
             (Beta-Bernoulli), start-line, circuit profiles;
             nominations.py translates historical compounds into the target year's label space
generation/  ALL rule-legal candidates (>=2 compounds + physical set allocation) with a
             recency-weighted, SEQUENCE-aware prior (stop count, pattern, order|multiset, start
             compound, weekend usage); family-coverage shortlist keeping second orderings where
             history supports them; history-calibrated pit windows (+undercut shift)
sim/         discrete-event lap-by-lap simulator: overtaking(+DRS), safety car/VSC, pits
context/     postquali (grid+quali+practice) and prelim (pre-weekend, current-season form);
             expanding-window rule: priors/params only see races strictly before the target
evaluation/  Monte-Carlo with common random numbers; per-candidate outcome distributions
selection/   race-level field display (plausibility-mass set-cover) + per-driver shown 2–5,
             tiers, P(optimal), order/clone diversity
report/      JSON output for the strategy page (incl. pit-window ranges)
validation/  finish backtest + strategy-accuracy backtest (--strategy, expanding window);
             scores the RACE-level modal metric (see below)
tuning/      fast selection tuner (cache sims once, re-run select() per trial) + full Optuna refit
```

## Usage

```bash
# one-time: cache FastF1 data (rate-limited, resumable) and fit parameters
venv/bin/python -m strategy_sim2.smoke                         # verify data access
venv/bin/python -m strategy_sim2.data.backfill --delay 20      # fill training window (repeatable)

# main mode: after qualifying (a completed race for validation, or a live weekend)
venv/bin/python -m strategy_sim2.run --mode postquali --year 2026 --round 8 --sims 1000

# preliminary mode: upcoming race, no sessions yet (current-season form + prev year)
venv/bin/python -m strategy_sim2.run --mode prelim --year 2026 --round 9

# out-of-sample validation (expanding window, no leakage) and tuning
venv/bin/python -m strategy_sim2.validation.backtest --strategy --test-year 2025 --sims 200 --workers 6
venv/bin/python -m strategy_sim2.validation.backtest --strategy --test-year 2026 --sims 200 --workers 6
venv/bin/python -m strategy_sim2.tuning.tune_selection --year 2025 --rounds 2 3 7 11 15 20 21 23
```

Config lives in `config/settings.yaml` and `config/nominations.yaml` (nothing hardcoded).

## The accuracy metric

The strategy page shows **one list of up to five strategies per race**. Success is judged at the
**race level**: the *modal* ordered strategy the field actually ran — or a near-tied second (within
~2 runners) — must appear in that shown five. The backtest reports this as `RACE modal-order in 5`
(and `modal-set in 5` for the compound multiset), aggregating each driver's plausibility mass into
one field-level display. Per-driver "was this driver's own strategy in their own top-5" is also
reported but is a much harsher proxy, **not** the product goal, and is not the tuning objective.

## Validated results

Expanding-window strategy backtest (params + priors trained only on races before each target race):

| | RACE modal-order in 5 | RACE modal-set in 5 | generation recall |
|---|---|---|---|
| **2025** (18 dry races) | **83.3%** (15/18) | 83.3% | 93.9% |
| **2026** (6 dry races) | **100%** (6/6) | 100% | 97.1% |

Barcelona is included in both aggregates (the nomination table removed the need to excuse it).
Finish-order correlation from earlier runs: OOS (train 2021–23 → test 2024) Spearman ≈ 0.74;
in-sample ≈ 0.95. Simulator sanity: Spearman(grid, finish) ≈ 0.99, ~3 DNFs/race.

## Design notes

- **Nomination-aware history.** Deg/knee/offsets and the strategy prior are keyed to the relative
  labels SOFT/MEDIUM/HARD, but Pirelli nominates a different physical triple each weekend. When a
  circuit's nomination shifts between seasons, `config/nominations.yaml` translates history through
  the C-numbers. The prior blends raw and translated circuit history 50/50 (`prior.relabel_blend`):
  strategy choice is part physical (transfers via C-numbers — Barcelona) and part label-relative
  (teams plan around whatever triple is nominated — Monaco); both extremes lose races.
- **Order matters.** The prior is sequence-aware (`P(order | multiset)`), the shortlist can keep a
  family's second ordering, and selection rewards order diversity — so the *order* teams run, not
  just the compound set, is surfaced.
- **Known regulations, not hindsight.** `circuit_rules` encodes only publicly-announced pre-race
  rules (mandatory stop counts, mandated stint-length caps), year-scoped.

## Known limitations / future work

- Compound pace offsets are modest (identification limited by within-race demeaning).
- Circuits with few dry races (e.g. Silverstone) have noisier degradation → shrunk to global.
- Remaining backtest misses are prior-mass, not selection-layer: late-race soft gambles
  (Zandvoort 2025), sprint-weekend anomalies (São Paulo 2025), and rare 4-stop plans
  (Barcelona 2025) — in-weekend evidence (sprint/FP soft usage) is the likeliest next lever.
- `config/nominations.yaml` must be extended with each new season's Pirelli nominations.
- Extensible (unused yet): weather, live in-race Bayesian updates, team-specific pit times,
  tyre-temperature and track-evolution effects.
