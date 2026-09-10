---
name: time-series-forecasting-models
description: >-
  Score a forecast against the baseline it has to beat - naive, seasonal-naive, drift, mean -
  with MASE, rolling-origin evaluation and a Diebold-Mariano test, instead of an R^2 on a price
  level. TRIGGER - forecasting, ARIMA, SARIMA, SARIMAX, auto_arima, ETS, Holt-Winters,
  exponential smoothing, statsmodels ARIMA trend, walk-forward, rolling origin, expanding
  window; naive forecast, seasonal naive, drift method, MASE, sMAPE, MAPE, OWA,
  Diebold-Mariano, DM test; LSTM stock price prediction, N-BEATS, TFT, "my model predicts
  prices with 99% R^2", "is my forecast better than the naive one", M4 competition, M5
  competition. SKIP for volatility forecasting and HAR-RV (volatility-models), for choosing a
  forecasting library (factor-and-timeseries-research), for cointegration and unit roots
  (stat-arb-cointegration), for correcting across many models (backtest-validation), and for
  turning a forecast into positions (signal-construction).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Time-series forecasting models

**Almost every impressive forecasting result on a price series is a lag detector.** Predict
tomorrow's price with today's and you get an R^2 of 0.99; the model has learned nothing, and its
implied return forecast is exactly zero. This skill supplies the four baselines a forecast has to
beat, the scale-free score that makes them comparable, the walk-forward that stops the future
leaking in, and the test that says whether a difference is real.

Every number below is printed by `scripts/forecast_baselines.py` (numpy / scipy, seed 0,
`statsmodels` optional). Two seeded DGPs run through the same machinery: **A**, a daily log-price
random walk with drift (1,300 days, drift 3 bp/day, vol 1 %/day), and **B**, a monthly series
with a linear trend and a 12-period seasonal (240 points).

## 1. 🚨 The random-walk illusion, measured on one series

| quantity, same 1,300-day price series | value |
|---|---|
| R^2 of `price[t]` on `price[t-1]` | **0.987032** |
| R^2 of `return[t]` on `return[t-1]` | **0.000133** |
| R^2 of the naive price forecast against a constant-mean benchmark | **0.996700** |
| R^2 of the *same* forecast against the naive benchmark | **0.000000** |

The first three numbers describe a forecast that contains no information; the fourth is its
honest score. **An out-of-sample R^2 only means something relative to a benchmark you name**, and
for a price series that benchmark is the naive forecast, not the sample mean.

🚨 **The lag signature.** For the naive price forecast, `corr(forecast[t], actual[t-k])` is

| k | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| corr | 0.9891 | **1.0000** | 0.9892 | 0.9780 | 0.9675 | 0.9530 |

The maximum is at **k = 1**, not k = 0. Any "price forecast" whose cross-correlation with the
truth peaks one step behind is reproducing the last observation. Plot that correlogram before you
believe a price-prediction chart; it is one line of numpy and it settles the question.

And the same model as a **return** forecast predicts exactly zero every day: out-of-sample R^2
against the training mean return **0.002195**, and no direction at all. On this path 51.7 % of
days were up, so "long every day" is the accuracy any directional claim has to beat — not 50 %.

> **Forecast returns, not levels. Report the benchmark next to every R^2. If the benchmark is
> not stated, the R^2 is not a result.**

## 2. Four baselines and one scale-free score

| baseline | forecast for step `i` ahead from `y[1..T]` |
|---|---|
| naive | `y[T]` |
| seasonal naive | `y[T + i - m*ceil(i/m)]` — the same phase of the last complete cycle |
| drift | `y[T] + i * (y[T] - y[1]) / (T - 1)` — the line through the first and last points |
| mean | `mean(y[1..T])` |

**MASE** (Hyndman-Koehler 2006) divides the forecast's MAE by the **in-sample** naive MAE,
`mean(|y[t] - y[t-m]|)` over the *training* set. That denominator is what makes it comparable
across series and readable without context: MASE > 1 is worse than the training-set naive.

✅ The definition's own identity, checked rather than asserted: the MASE of the in-sample naive
forecast is **1.0000000000**, at both `m = 1` and `m = 12`. On the held-out days the naive
forecast scores **0.9054** — not 1, because the scale of the moves changed, which is the one
thing MASE does not normalise away.

⚠️ MAPE and sMAPE are the alternatives you will meet: MAPE is undefined at zero and asymmetric
(it punishes over-forecasts more), sMAPE is bounded but still unstable near zero. Neither was
measured here. Use MASE for anything that crosses zero, which includes every return series.

## 3. Rolling-origin walk-forward: what beats the naive forecast, and where

One origin per step; at origin `T` the model sees `y[:T]` and nothing else, and is scored on
`y[T]`. ✅ Measured (MASE, one-step-ahead):

| forecast | **A: daily price**, 300 origins | **B: monthly seasonal**, 72 origins |
|---|---|---|
| **naive** | **0.9054** | 1.0995 |
| seasonal naive | 2.1403 | **0.8534** |
| drift | 0.9067 | 1.1034 |
| mean | 15.2999 | 5.4063 |
| ARIMA | 0.9051 *(1,1,1)* | **0.8385** *(1,0,0)x(1,0,0,12)* |
| ETS (Holt-Winters) | 0.9067 | **0.4846** |

Diebold-Mariano against the naive forecast, squared loss, h = 1, HLN-corrected (negative = the
model wins):

| forecast | A: DM / p | B: DM / p |
|---|---|---|
| seasonal naive | +10.942 / 0.0000 | **-2.283 / 0.0255** |
| drift | +1.460 / 0.1453 | +0.345 / 0.7313 |
| mean | +17.819 / 0.0000 | +11.727 / 0.0000 |
| ARIMA | -1.408 / **0.1601** | **-2.679 / 0.0092** |
| ETS | +1.478 / 0.1405 | **-5.936 / 0.0000** |

**The contrast is the point.** On the random walk, ARIMA's MASE (0.9051) is *below* the naive's
(0.9054) and its DM statistic is negative — and the p-value is **0.16**, so the difference is
noise. Nothing beats the naive forecast on a series with nothing in it, and a table of MASE
values alone would have let you claim otherwise. On the seasonal series three methods beat it and
the DM test agrees at 5 %, with ETS at less than half the naive's error.

⚠️ **Refit cadence is a modelling choice, not an implementation detail** — though it happened to
be small here: refitting the ARIMA 6 times instead of 2 on DGP A moved MASE from 0.9051 to
0.9050, and 18 times instead of 3 on DGP B moved 0.8385 to 0.8380. State the cadence; a result
that only survives at one cadence is a tuned result. ✅ Between refits the demo extends the model
with `res.append(new_obs, refit=False)`, statsmodels' documented way to add observations without
re-estimating.

🚨 **`ExponentialSmoothing` has no `append`,** so there is no cheap honest way to carry a fit
forward. Freezing the estimated smoothing parameters between refits and re-fitting with
`optimized=False` is a *different* estimator — the initial states come from the heuristic rather
than the likelihood — and it is not a small difference. ✅ Measured on DGP B: **MASE 4.0784**
with the parameters frozen between 24-origin refits against **0.4846** refit at every origin, an
8.4x gap produced entirely by how the walk-forward was wired (on DGP A, 0.9294 against 0.9067).
Refit ETS at every origin and pay the time.

## 4. Diebold-Mariano: the difference between "smaller MSE" and "better"

`d[t] = L(e1[t]) - L(e2[t])`; the statistic is `mean(d) / sqrt(V(mean(d)))` with `V` a
Newey-West long-run variance truncated at `h - 1` lags. `harvey=True` applies the
Harvey-Leybourne-Newbold (1997) small-sample factor `sqrt((T + 1 - 2h + h(h-1)/T)/T)` and refers
the result to `t(T-1)`.

✅ Three checks on the implementation, all printed by the script:

- **At h = 1 without the Harvey factor it must be a one-sample t-test on `d`** using the
  population variance, so DM / `scipy.stats.ttest_1samp` must equal `sqrt(T/(T-1))`. Measured:
  DM -1.925657, scipy -1.923248, ratio **1.0012523486** against an expected 1.0012523486 —
  agreement to **0.0e+00**.
- **Antisymmetry:** DM(e1, e2) = -2.759902 and DM(e2, e1) = +2.759902, same p-value 0.006139.
- **Size under the null:** two forecasts with equal expected loss, 1,000 runs of T = 250 — it
  rejects **5.8 %** at the 5 % level, against a Monte Carlo standard error of 0.7 %.

⚠️ **What it does not do.** The null is *equal expected loss*, not "the models are the same". It
assumes the loss differential is covariance-stationary, which nested models estimated on the same
data can violate. And **it compares two forecasts.** A table of pairwise DM tests over ten models
is ten trials: use `arch.bootstrap`'s SPA, StepM or MCS instead — those take **losses**, and
`../../../fin-libraries/skills/lib-arch/SKILL.md` leads with the sign trap that inverts them.
`../../../fin-core/skills/backtest-validation/SKILL.md` owns the correction.

## 5. statsmodels: two defaults that change the forecast

✅ Source-verified against the installed statsmodels 0.15.0.

- 🚨 **An integrated ARIMA has no drift term by default.**
  `statsmodels/tsa/arima/model.py` sets `trend = "c"` when `trend is None and not integrated`,
  and `trend = "n"` otherwise — so `ARIMA(y, order=(1,1,0))` estimates only `['ar.L1',
  'sigma2']`. ✅ Measured on a series drifting +0.04634 per step in sample, the 50-step forecast
  slope is **+0.00007 — a flat line**; passing `trend="t"` recovers **+0.04629**. On a price
  series with a real drift, the default silently forecasts no drift at all.
- ✅ **`ARIMA(y, order=(0,1,0))` IS the naive forecast.** With `trend="n"` it has no mean
  parameters at all, and its one-step forecast equals the last observation to **0.0e+00** across
  four origins. Useful as a sanity anchor: if your ARIMA pipeline does not reproduce the naive
  forecast at order (0,1,0), the pipeline is wrong, not the data.
- ✅ `ARIMA(endog, exog=None, order=(0,0,0), seasonal_order=(0,0,0,0), trend=None,
  enforce_stationarity=True, enforce_invertibility=True, concentrate_scale=False,
  trend_offset=1, ...)`, and `MLEResults.append(endog, exog=None, refit=False,
  fit_kwargs=None)` is the walk-forward extension used above.
- ✅ `ExponentialSmoothing(endog, trend=None, damped_trend=False, seasonal=None, *,
  seasonal_periods=None, initialization_method="estimated", ...)` — `trend` and `seasonal` are
  `None` by default, so a bare call is simple exponential smoothing with neither.
  `statsmodels.tsa.exponential_smoothing.ets.ETSModel` is the newer state-space ETS
  (`error="add"` by default); the two are different implementations of overlapping models.
- ⚠️ `sktime`, `statsforecast`, `pmdarima`, `darts`, `prophet` and `neuralforecast` are **not
  installed here** and nothing about them was checked.
  `../../../fin-models/skills/factor-and-timeseries-research/SKILL.md` chooses between them and
  carries the verified leakage findings in their CV splitters.

## 6. What the M4 and M5 competitions actually say

These are the largest public forecasting evaluations, and they are routinely cited for claims
they do not support. ⚠️ Read at the sources below on 2026-09-09; this repo did not re-run them.

- ⚠️ **M4 (2018, 100,000 series) was won by a hybrid, not by deep learning.** Slawek Smyl's
  ES-RNN mixes exponential smoothing with LSTM layers (Smyl, *IJF* 36(1), 2020, 75-85). The
  second-place method used XGBoost to *weight* forecasts from standard statistical methods.
- ⚠️ **M4's own Finding 5: the six pure ML methods performed poorly** — none was more accurate
  than the `Comb` benchmark, and only one beat `Naive2` (Makridakis, Spiliotis &
  Assimakopoulos, *IJF* 34(4), 2018, 802-808; abstract read at
  https://ideas.repec.org/a/eee/intfor/v34y2018i4p802-808.html). ✅ From the organizers' own
  evaluation code, `Comb` is the equal average of SES, Holt and Damped exponential smoothing,
  and `Naive2` is the naive forecast on a seasonally adjusted series
  (https://github.com/Mcompetitions/M4-methods, `Benchmarks and Evaluation.R`).
- ⚠️ **OWA is normalised by Naive2, not by Comb**: `OWA = 0.5 * [sMAPE/sMAPE(Naive2) +
  MASE/MASE(Naive2)]`, straight from that same evaluation code. The two benchmarks get conflated
  constantly — Comb is what the ML methods failed to beat; Naive2 is the OWA denominator.
- ⚠️ **M5 (2020) is Walmart unit sales, and gradient boosting won it.** The Accuracy track was
  won by an equal-weighted ensemble of 220 LightGBM models, 22.4 % better than the `ES_bu`
  benchmark; only 7.5 % of teams' final submissions beat that benchmark at all. **Deep learning
  did not lose M5**: third place was a pure ensemble of 43 LSTM networks and second place used
  N-BEATS forecasts to adjust LightGBM (Makridakis, Spiliotis & Assimakopoulos, M5 Accuracy
  paper, *IJF* 38(4), 2022, 1346-1364; authors' preprint at
  https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf).
  ⚠️ The same paper reports the gains concentrating at the top of the hierarchy: ~40 % over the
  benchmark at the most aggregated level and ~3 % at the product-store level, which is where
  most of the series are.
- 🚨 **Neither competition tells you anything about daily equity returns.** M5 is retail unit
  sales. M4 contains a "Finance" domain, but it is scored as a *level* extrapolation with
  sMAPE/MASE/OWA, not as return prediction, and there is no cross-section and no trading.
  ⚠️ Rob Hyndman, writing in the M4 special issue itself: "I know of no large-scale forecasting
  competition for finance data" (*A brief history of forecasting competitions*, *IJF* 36(1),
  2020, 7-14; author's copy at
  https://robjhyndman.com/papers/forecasting-competitions.pdf). ⚠️ The M4 Finance domain is
  reported as 24,534 of the 100,000 series, of which 1,559 are daily — secondhand, from Table 8
  of the sktime M4 replication (arXiv 2005.08067); the primary table was behind a paywall, and
  **what those series contain could not be established.**

**On N-BEATS, TFT and LSTM for financial returns, this skill carries no verified claim that any
of them beats the naive forecast on daily equity returns.** ⚠️ The nearest thing to a positive
claim found (arXiv 2408.12408, an evaluation of TCN / N-BEATS / TFT / N-HiTS / TiDE / xLSTM on
the S&P 500 and EWZ) reports that on the original undenoised data the models sit near random
guessing, and that only three of the six beat the naive forecast even after wavelet denoising —
N-BEATS and TFT were not among them, and the paper does not say whether the denoising was fitted
before or after the split. ⚠️ Goyal and Welch's out-of-sample study of the equity premium (NBER
w10483; *RFS* 21(4), 2008) found no standard predictor helped an investor outpredict the
prevailing historical mean — monthly and annual data with economic predictors, so it is evidence
about the difficulty of the target rather than about any architecture.
`../../../fin-llm/skills/rl-and-ml-trading/SKILL.md` owns deep-learning trading claims.

## 7. Traps

- 🚨 **R^2 on a level, or any metric on a level.** Section 1. Forecast the difference.
- 🚨 **No baseline in the table.** If naive, seasonal-naive, drift and mean are not in the
  comparison, the result is unscored. MASE puts them there by construction.
- 🚨 **A single train/test split.** One split is one draw. Roll the origin, and report the number
  of origins next to the score.
- 🚨 **Fitting anything on the full series before splitting** — scaling, denoising, differencing
  parameters, seasonal decomposition, a stationarity test used to choose `d`. Every one of those
  belongs inside the fold.
  `../../../fin-core/skills/signal-construction/scripts/assert_causal.py` checks the result.
- 🚨 **`trend=None` on an integrated ARIMA.** Section 5: a flat forecast on a drifting series.
- 🚨 **A p-value from a table of models.** DM compares two. More than two needs SPA / StepM / MCS
  and a trial ledger (`../../../fin-core/skills/backtest-validation/SKILL.md`).
- ⚠️ **Accuracy is not profit.** A forecast that beats the naive one on MASE can still lose money
  after costs, and a directional accuracy of 51.7 % on this seed is exactly the base rate of up
  days. `../../../fin-core/skills/signal-construction/SKILL.md` turns a forecast into a position;
  `../../../fin-core/skills/execution-cost-analysis/SKILL.md` prices the trading.

## 8. Scripts

- `scripts/forecast_baselines.py` — the two framings of one series and the lag signature, MASE
  and its identity, the rolling-origin tables for both DGPs, the ARIMA and ETS walk-forwards,
  the Diebold-Mariano implementation with its three verification checks, and the statsmodels
  trend defaults. `statsmodels` is imported inside four functions; without it the baselines, MASE
  and the DM test all still run. Seed 0, **41 s** of compute (36-48 s wall clock across runs; the slowest script in
  this plugin).

## 9. Where this sits

`../volatility-models/SKILL.md` owns forecasting *variance* (and HAR-RV, which is the baseline
there). `../stat-arb-cointegration/SKILL.md` owns unit roots and stationarity testing between
two series. `../state-space-and-kalman/SKILL.md` owns latent-state models and their
filtered-vs-smoothed rule. `../../../fin-models/skills/factor-and-timeseries-research/SKILL.md`
chooses the library and carries the foundation-model evidence.
`../../../fin-core/skills/backtest-validation/SKILL.md` corrects across many models.
`../../../fin-libraries/skills/lib-arch/SKILL.md` owns `arch.bootstrap`'s SPA / StepM / MCS.
`../../../fin-llm/skills/rl-and-ml-trading/SKILL.md` owns deep-learning trading claims.
