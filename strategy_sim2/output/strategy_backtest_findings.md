# Strategy-accuracy backtest — dry 2026 races (post-quali)

**Method.** Parameters trained on **2021–2025 only** (test year excluded → no leakage),
evaluated on all 6 dry 2026 races (Melbourne, Shanghai, Suzuka, Monaco, Barcelona,
Spielberg), post-qualifying mode, **200 sims/driver**, 102 classified driver-races. For
each driver we compare the model's predictions to the realised strategy on: compound
choice (multiset), compound order (exact sequence), stop count, and pit-window laps.
Baseline = "predict this circuit's historically most common strategy for everyone."

## Headline results

| Metric | All races | **Excl. Monaco** | Baseline (excl. Monaco) |
|---|---|---|---|
| Stop-count acc (top pick) | 58.8% | **69.8%** | 73.3% |
| Compound-set acc (top pick) | 46.1% | **54.7%** | 45.3% |
| Compound-order acc (top pick) | 28.4% | **33.7%** | — |
| Compound-set in top-5 | 67.6% | **80.2%** | — |
| Compound-order in top-5 | 36.3% | **43.0%** | — |
| Actual strategy generated (recall) | 68.6% | **81.4%** | — |
| First-stop lap MAE | 7.3 laps | 7.3 laps | — |
| First stop within ±5 laps | 48.3% | 48.3% | — |

By actual stop count (all races):

| Actual stops | n | stop-count acc | compound-set acc | set in top-5 |
|---|---|---|---|---|
| 1-stop | 39 | **92%** | **90%** | **100%** |
| 2-stop | 35 | 69% | 34% | 86% |
| 3-stop | 11 | **0%** | 0% | 0% |
| 4–7 (all Monaco) | 17 | 0% | 0% | 0% |

## What worked

- **One-stop races are nailed:** 92% stop-count, 90% compound-set, 100% set-in-top-5.
  The engine reliably identifies the classic MEDIUM–HARD one-stopper.
- **It surfaces the realistic compound choice:** on normal dry races the actual compound
  set is generated 81% of the time and sits in the selected top-5 80% of the time — the
  core objective ("rank realistic strategies highly") is being met.
- **It beats the pure-history baseline on compound choice** (54.7% vs 45.3%), i.e. the
  simulation adds value over "just copy the circuit's usual strategy."
- **Per-circuit:** Suzuka (90% stop / 85% set) and Shanghai (73% / 67%) are excellent;
  Spielberg and Barcelona get the stop count right but the exact order less so.
- Finishing-order correlation remains strong (0.74 OOS / 0.95 in-sample from earlier runs).

## Sources of inaccuracy

1. **Anomalous races (Monaco).** Monaco 2026 was **red-flagged** and, with the 2025+
   mandatory-two-stop rule, drivers took 4–7 "stops" — clusters of near-free end-of-race
   tyre changes (e.g. pits on laps 58,59,65,67,68). These aren't pace-driven strategy and
   are unpredictable by any pace model; the generator also caps at 3 stops. Monaco alone
   pulls stop-count accuracy from ~70% down to 59%.

2. **A one-stop bias / stop-count under-prediction.** The confusion matrix shows the model
   turns 2-stops into 1-stops (11 cases) and 3-stops into 2-stops (10 cases) and **never
   predicts 3+**. Roots: (a) tyre degradation is modelled as **linear with no cliff**, so
   long stints aren't punished enough; (b) degradation is **under-estimated at thin-data
   circuits** (shrunk toward a low global); (c) the naive baseline beats the model on stop
   count precisely because history encodes the modal stop count that the economics miss.

3. **Compound order is weak even when the set is right (34% vs 55%).** The model
   over-predicts HARD starts and SOFT/short final stints (predicted HARD-SOFT ×16,
   HARD-MEDIUM ×16) while reality is dominated by **MEDIUM-HARD** (×35, start medium / end
   hard). Cause: the estimated **compound pace offsets are too flat** (SOFT only ~0.1 s
   faster than MEDIUM) because the within-race fixed-effects regression compresses them,
   and nothing discourages **ending on the soft** (no cliff, no safety-car-risk term).

4. **Pit-window timing (first-stop MAE ≈ 7 laps, only 48% within ±5).** The model plans the
   *pace-optimal* lap, but real stops are pulled earlier by **undercut pressure**, or moved
   by safety cars and traffic — none of which shift the analytic optimum.

5. **Untuned simulator parameters.** Overtaking probabilities/thresholds, pit-jitter and the
   selection plausibility weight are still at hand-set defaults, not fit by the Optuna
   harness.

## Actionable improvements (priority order)

1. **Add a degradation cliff and fix compound offsets.** Replace linear deg with a
   convex/piecewise term that steepens past each compound's usable window, and estimate
   fresh-tyre compound offsets from stint-start pace (or a dedicated model) instead of the
   compression-prone within-race regression. This is the single highest-leverage change: it
   simultaneously (a) shortens stints → raises stop counts (fixes the 1-stop bias) and
   (b) discourages ending on soft / starting hard (fixes compound order). Expected to move
   both stop-count and order accuracy up materially.

2. **Make stop count history-aware.** Blend the plausibility prior into stop-count selection
   more strongly (the baseline beats us here), and validate/raise per-circuit deg severity
   with more dry data. Quick win: increase `k_circuit` shrinkage only where data is thick.

3. **Model the undercut in pit-window placement.** Bias planned stops earlier when a car is
   in traffic/close behind, and calibrate windows to observed historical stint-length
   distributions per circuit×compound rather than the pure analytic optimum. Target
   first-stop MAE ≤ 4 laps.

4. **Handle circuit rules & red flags.** Detect red-flag races and exclude their inflated
   stint counts from "actual" evaluation (or model free red-flag stops); add circuit-specific
   `min_stops` (Monaco = 2 from 2025) and make `max_stops` circuit-aware.

5. **Run a full Optuna study** (train fold → held-out) to set overtaking/threshold/jitter and
   the selection prior weight; the harness exists (`tuning/optuna_tune.py`) but hasn't been
   run at scale.

6. **De-prioritise ending on the softest compound** in generation/selection unless deg
   clearly supports it — a cheap heuristic that would fix many order mispredictions now.

7. **More data.** Backfill any remaining dry races and refit; several circuits (e.g.
   Silverstone) still rest on only 2 dry races.
