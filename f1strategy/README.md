# f1strategy

Predicts the pit-stop strategy F1 teams will actually run at a circuit, and updates
that prediction once free-practice data is in. Built as an **inverse-calibration**
problem, not a regression:

1. a **forward simulator** scores any strategy (compound sequence + pit laps) from
   physically-motivated inputs (tyre curves, fuel, pit loss, track-position, optional
   safety-car EV);
2. a **strategy search** returns the optimum and a probability distribution;
3. a small set of **global behavioural parameters** is tuned (Optuna) so the
   simulator's optimum matches the strategies real front-runners chose, across many
   historical races.

Per-track tyre/fuel/pit inputs stay measured from data; only a handful of *global*
(not per-track) behavioural scalars are calibrated, so the model doesn't overfit and
degradation numbers stay data-driven.

## Layout

| module | role |
|---|---|
| `config.py` | dataclasses: `GlobalParams` (θ), `SimConfig` (toggles), `TrackContext`, `StrategyPrediction` |
| `data/loaders.py`, `data/laps.py` | FastF1 loading; lap cleaning, stint reconstruction, green-flag filtering |
| `observations.py` | reference strategy per race (the calibration target) + strategic-freedom weighting |
| `tyre.py` | per-compound pace + degradation, race+practice **hybrid**, no look-ahead |
| `pitloss.py` | per-track pit loss (hand-curated placeholder; swap in your own function) |
| `safetycar.py` | per-track SC prior + expected-value hedging term (only when `use_sc`) |
| `simulator.py` | forward model: `race_time(strategy | θ, context)` |
| `optimizer.py` | candidate enumeration, DP pit-lap placement, ranking, distribution |
| `dataset.py` | build + cache (`ctx`, `obs`) cases for calibration |
| `calibrate.py` | the inverse problem: loss, Optuna study, leave-one-track-out CV, persistence |
| `predict.py` | **online entrypoint** `predict_strategy(...)` |
| `backtest.py` | predicted-vs-observed metrics |
| `cli.py` | `python -m f1strategy.cli predict/calibrate/backtest` |

## Usage

```python
from f1strategy.predict import predict_strategy

# base prediction (pre-weekend, prior seasons only)
pred = predict_strategy("Monza", 2025)

# updated prediction (after free practice)
pred = predict_strategy("Monza", 2025, use_practice=True)

# compare safety-car hedging on/off
pred = predict_strategy("Monza", 2025, use_sc=False)

print(pred.summary())            # optimal strategy + P(stops)
pred.optimal                     # StrategyResult(compounds, pit_laps, ...)
pred.p_by_stops, pred.pit_windows
```

CLI:

```
python -m f1strategy.cli predict "Monza" 2025 --practice
python -m f1strategy.cli calibrate --trials 250 --cv --practice   # writes params/calibrated_sc_off.json
python -m f1strategy.cli calibrate --sc-on --practice             # writes params/calibrated_sc_on.json
python -m f1strategy.cli backtest
```

## Global parameters (θ)

Calibrated, global, interpretable:

- `deg_scale` — corrects the race-measurement degradation under-read; applied **only to
  race-sourced compounds** (practice-sourced degradation is already unbiased).
- `pit_stop_penalty` — effective seconds added per stop for track-position / dirty-air
  cost; the main lever pulling the optimum toward realistic stop counts.
- `stint_risk` + `risk_free_life` — linear penalty on tyre age past a knee; risk
  aversion / the tendency to pit before the cliff (biases pit laps earlier).
- `sc_influence` — strength of safety-car hedging (SC-on profile only).

Two profiles are persisted — `params/calibrated_sc_{on,off}.json` — selected by `use_sc`.

## Porting into a backend pipeline

- Pure functions + dataclasses, no notebook/global state; θ is plain JSON.
- Inject your own data loading anywhere a FastF1 `Session` is accepted, or replace
  `pitloss.pit_loss(track, year)` with your real pit-loss function (same signature).
- One entrypoint: `predict_strategy(track, year, use_practice=, use_sc=) -> StrategyPrediction`.

## Modelling notes / not-yet-v1

- Compound modelled by relative slot (SOFT/MEDIUM/HARD); Cx-nomination pooling is a
  refinement. FastF1 track-name matching is fuzzy — prefer canonical names
  (e.g. "Silverstone", not "Great Britain", which mis-resolves).
- 2026 is a new regulation era: per-track tyre inputs for 2026 should lean on practice
  (`use_practice=True`); the global θ transfers across eras by design.
- Degradation is linear (with a non-negative floor). A per-compound quadratic **cliff**
  (knee = `cliff_budget/deg`) was implemented and **rejected by cross-validation**: it
  induced token minimum-length soft stints (dispatch the soft in ~8 laps, then run
  durable rubber), dropping out-of-sample stop-acc 73%→60% and compound 40%→13%. A
  workable cliff needs a token-stint guard (min stint as a fraction of the race, or a
  balanced-split prior) — noted for v2, along with an explicit track-evolution term and
  a per-team pace factor.
- Biggest remaining error is compound choice at low-deg tracks (Monza: model starts
  SOFT, reality MEDIUM→HARD) — the linear model under-costs a long soft stint. This is
  the main target for the cliff-with-guard work.
