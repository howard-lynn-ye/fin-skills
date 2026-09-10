---
name: state-space-and-kalman
description: >-
  Estimate a time-varying hedge ratio or beta with a Kalman filter, and know which of its three
  state series you are allowed to trade - the smoothed one has read the whole sample. TRIGGER -
  Kalman filter, Kalman smoother, RTS smoother, state space model, local level, local linear
  trend, time-varying beta, dynamic hedge ratio, dynamic linear model, DLM, process noise Q,
  observation noise R, signal-to-noise ratio; statsmodels.tsa.statespace, KalmanSmoother,
  MLEModel, UnobservedComponents, RecursiveLS, recursive_coefficients, smoothed_state,
  filtered_state, predicted_state, states.smoothed, pykalman, filterpy, simdkalman; "my Kalman
  beta is beautifully smooth", "how do I pick Q", "the beta path looks flat". SKIP for discrete
  regime labels and Markov switching (regime-detection), for the cointegrating hedge ratio and
  spread z-scores (stat-arb-cointegration), for GARCH conditional variance (volatility-models),
  and for ARIMA/ETS point forecasts (time-series-forecasting-models).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# State space and Kalman filters

**A Kalman filter returns three estimates of the same state, and only one of them is yours to
trade.** The smoothed path is the one every tutorial plots, because it is the cleanest — and it
is conditioned on the whole sample, so on this skill's seeded series the smoothed beta had
already completed **49 % of a break** on the day *before* the break happened.

Every number below is printed by `scripts/kalman_models.py` (numpy / scipy, seed 0, 17 s,
`statsmodels` optional). The filter and the RTS smoother are re-implemented in numpy so the
skill works without `statsmodels`; ✅ where it is installed they agree with
`statsmodels.tsa.statespace.kalman_smoother.KalmanSmoother` on the same system to **4.4e-16
(predicted), 4.4e-16 (filtered), 3.6e-15 (smoothed)** and **9.1e-13** in log-likelihood
(statsmodels 0.15.0, two-state model; the three-state trend model agrees to 1.8e-12).

The running example is `y_t = alpha + beta_t x_t + e_t` — a hedge of one asset on another — with
`beta_t` a random walk that steps from 1.0 to 0.5 at t = 750 of 1,500. `alpha` = 3 bp/day and
`e_t` has sd 0.5 %/day, so a perfect hedge earns a Sharpe of about 0.95 in expectation.

## 1. 🚨 Three state series; one of them is a signal

| series | statsmodels attribute | conditions on | usable to hedge day t? |
|---|---|---|---|
| predicted `a_{t\|t-1}` | `predicted_state` | y_1..y_{t-1} | **yes** |
| filtered `a_{t\|t}` | `filtered_state` | y_1..y_t | no — it needs today's return, which is what you are hedging |
| smoothed `a_{t\|N}` | `smoothed_state` | y_1..y_N, the whole sample | no — a backward pass over the future |

✅ Measured, on the 300 days around the break, as `corr(beta_est[t], beta_true[t+k])` for
`k` in ±40 (a positive best `k` means the estimate at t matches the truth `k` days *later*):

| estimate | best k | corr at k=+10 | corr at k=-10 | crosses the midpoint | had moved, day before the break |
|---|---|---|---|---|---|
| smoothed | **+0 d** (0.940) | 0.931 | 0.933 | **+0 d** | **49 %** |
| filtered | -15 d (0.928) | 0.845 | 0.926 | +21 d | -3 % |
| predicted | -15 d (0.929) | 0.841 | 0.926 | +22 d | -3 % |

The smoothed series is symmetric about zero — it is as correlated with beta ten days ahead as
ten days behind, which is what "it saw the future" looks like in a cross-correlogram. The two
causal series peak 15 days behind and cross the level midpoint three weeks after the break. As
beta's own random walk gets faster the anticipation grows: the share of the step the smoothed
estimate had already taken by the day before the break runs **42 % / 49 % / 59 % / 76 %** for
daily beta innovations of 0.000 / 0.002 / 0.005 / 0.010.

⚠️ Over the *whole* 1,500 days the cross-correlogram is flat (every lag within ±40 correlates
0.94-0.97) and its argmax is noise. Score the lead on a window around the move, as the table
does; a lead-lag number computed over a long flat stretch means nothing.

`../../../fin-core/skills/regime-detection/SKILL.md` measures the same
smoothed-versus-filtered distinction for Markov-switching regime *labels*, including what the
leak is worth in Sharpe and why "I used the filtered probabilities" is not a defence. Read it
there; it is not repeated here.

## 2. The tell that needs no truth: a residual below the floor

A hedge on the true beta leaves `alpha + e_t`, so its residual variance is `var(e)` — a **floor**
no beta estimate can beat, because every estimate can only add mis-hedge. ✅ Measured over
t >= 250:

| beta used | Sharpe | resid var | beta RMSE | resid vs the true-beta floor |
|---|---|---|---|---|
| oracle: true beta_t | 1.43 | 2.496e-05 | 0 | — |
| **smoothed (t\|N)** | **1.55** | 2.439e-05 | 0.0693 | **-2.3 %** |
| **filtered (t\|t)** | **1.53** | 2.406e-05 | 0.0914 | **-3.6 %** |
| predicted (t\|t-1), causal | 1.47 | 2.553e-05 | 0.0926 | +2.3 % |
| rolling 250-day OLS, causal | 1.52 | 2.705e-05 | 0.1563 | +8.4 % |
| full-sample OLS (constant) | 1.45 | 3.259e-05 | 0.2829 | +30.6 % |

🚨 **Two hedges beat a floor set by the truth.** That is arithmetically impossible for an
estimate built without today's data, and it is the cheapest look-ahead detector there is. The
decomposition is exact — with `mis_t = (beta_true - beta_used) x_t`,

```
var(resid)/var(e) - 1  =  var(mis)/var(e)   +   2 cov(mis, e)/var(e)
                          ---- mis-hedge --      -- noise absorbed --
```

| beta used | mis-hedge | noise absorbed | total | corr(mis, e) |
|---|---|---|---|---|
| smoothed | +2.07 % | **-4.36 %** | -2.29 % | -0.152 (t **-5.4**) |
| filtered | +3.92 % | **-7.52 %** | -3.60 % | -0.190 (t **-6.8**) |
| predicted | +4.05 % | -1.78 % | +2.27 % | -0.044 (t -1.6) |
| rolling 250-day OLS | +9.77 % | -1.42 % | +8.35 % | -0.023 (t -0.8) |

The middle column is zero in expectation for any beta chosen without seeing `e_t`; the causal
rows reach |t| = 1.6, the two that used `y_t` reach 5.4 and 6.8. **The filtered series is the
worse offender**, not the smoothed one: `beta_{t|t}` is updated by today's own innovation, so it
regresses away part of today's idiosyncratic return. Live, that update lands after the close.

**The practical version, when there is no true beta to compare with:** recompute your result on
`predicted` and on `filtered`. If the answer moves, the difference is same-day information you
will not have. Run the beta series through
`../../../fin-core/skills/signal-construction/scripts/assert_causal.py` before it reaches a
backtest.

## 3. What the look-ahead is actually worth

🚨 **Not much on one break — and that is the trap, not the reassurance.** Smoothed beats causal
by **+0.08 Sharpe** (1.55 vs 1.47) here, well inside single-path noise. The look-ahead is
visible in the *shape* — the anticipated break, the floor violation — long before it is visible
in the headline number, so "my Sharpe barely changed" is not evidence that the leak is harmless.

It becomes large exactly when the state moves faster than the filter can follow. ✅ Measured with
beta flipping 1.0 <-> 0.5 every `p` days (seed 0, t >= 250; the causal filter needs about 20 days
per flip):

| p (days) | q_beta fitted | beta RMSE sm / pred | resid vs floor sm / pred | Sharpe oracle / smoothed / filtered / **predicted** / rolling-250 |
|---|---|---|---|---|
| 250 | 7.5e-04 | 0.094 / 0.122 | -3.6 % / +5.5 % | 1.43 / 1.54 / 1.46 / **1.36** / 1.43 |
| 120 | 2.1e-03 | 0.109 / 0.163 | -4.0 % / +10.2 % | 1.43 / 1.60 / 1.56 / **1.39** / 1.60 |
| 60 | 4.3e-03 | 0.141 / 0.200 | -7.0 % / +12.7 % | 1.43 / 1.62 / 1.65 / **1.39** / 1.24 |
| 30 | 9.0e-03 | 0.163 / 0.225 | -9.5 % / +21.2 % | 1.43 / 1.57 / 1.64 / **1.33** / 1.30 |

The smoothed-minus-causal gap grows from +0.18 to +0.24 Sharpe, the causal hedge falls further
below the oracle as it spends more of the sample mid-transition, and the floor violation deepens
monotonically to -9.5 %. This is the same shape as the regime-detection result: **look-ahead
flatters most where a live detector performs worst.**

## 4. 🚨 Choosing Q: the profile likelihood is not unimodal

With the observation variance `h` concentrated out, a local-level fit is a one-dimensional search
over the signal-to-noise ratio `q/h`. That surface has a second optimum at `q -> 0` — "beta is
constant" — whenever beta moves in steps rather than smoothly.

✅ Measured on the flipping-beta series (`p` = 60, seed 0), same data, same model, two searches:

| search | fitted q_beta | log-likelihood | `converged` | predicted beta sd (true 0.2519) | RMSE |
|---|---|---|---|---|---|
| `scipy.optimize.minimize_scalar(method="bounded")` over the whole interval | **3.12e-13** | 5638.87 | **True** | **0.0096** | 0.257 |
| coarse log grid (41 points) + a bounded refine | 4.28e-03 | **5713.53** | True | 0.2378 | 0.200 |

The naive search reports success on a likelihood **74.7 nats worse**, with a `q` eleven orders of
magnitude too small and a beta that has stopped tracking altogether. Nothing warns. Grid the
likelihood first, refine around the best cell, and **print the fitted `q` and the standard
deviation of the state path** — a state path that does not move is the symptom.

⚠️ The fitted `q_beta` is not the DGP's random-walk variance and should not be read as one: on
the main series it comes out at 2.23e-04 against a random-walk part of 4.00e-06, because the
one-off step at t = 750 has to be absorbed by the only mechanism the model has. The observation
variance is recovered accurately (h = 2.51e-05 against a true 2.50e-05).

**Local level vs local linear trend.** ✅ RMSE of the causal predicted beta, t >= 250: on the
break DGP 0.0926 (level) vs 0.0888 (trend); on a linearly drifting beta 0.0640 vs **0.0435**. The
trend model pays for itself when beta drifts and costs little when it does not, but its fitted
`q_slope` collapses to ~1e-13 (a deterministic drift), and the two likelihoods are not comparable
because the slope state carries its own prior — do not select between them on `llf`.

## 5. statsmodels: what the attributes actually contain

✅ All source-verified against the installed statsmodels 0.15.0 (BSD-3-Clause).

- **`res.predicted_state` has `nobs + 1` columns**, `filtered_state` and `smoothed_state` have
  `nobs`. The extra column is the one-step forecast past the end of the sample. Verified:
  `res.states.predicted` has **1,501 rows for 1,500 observations**, while `.filtered` has 1,500.
  Assigning it straight into a DataFrame column is an off-by-one that shifts the whole series.
- **`res.states`** (`statsmodels/tsa/statespace/mlemodel.py`, built around line 2876) is a
  namespace with `.predicted`, `.filtered`, `.smoothed` and their `_cov` — all three are there,
  and nothing marks which is safe.
- **`res.fittedvalues` is `res.forecasts`**, the one-step-ahead prediction `Z a_{t|t-1}`
  (`mlemodel.py`, `def fittedvalues`); verified equal to `predicted_state[0, :nobs]` for a local
  level. It is the *predicted* series, not the smoothed one — a rare default that is on your side.
- 🚨 **`RecursiveLS(...).fit().recursive_coefficients.smoothed` is not time-varying.**
  `RecursiveLS` models the coefficients as **fixed**, so smoothing a constant returns the
  full-sample OLS answer, painted across every date. ✅ Measured on the main series: the
  `smoothed` beta has a spread of **2.4e-15** and sits within **1.8e-15** of the full-sample OLS
  slope, while `filtered` — the genuine expanding-window estimate — has a spread of 0.2616. A
  plot of `.smoothed` is a flat line at the answer computed from all the data.
- The low-level `KalmanSmoother(k_endog, k_states, k_posdef)` takes `bind(y[None, :])`,
  `["design"]` as `(k_endog, k_states, nobs)`, `["transition"]`, `["selection"]`,
  `["state_cov"]`, `["obs_cov"]`, then `initialize_known(a0, P0)` and `.smooth()`. `a0, P0` are
  the mean and covariance of the state **before** the first observation — the same convention the
  script's numpy filter uses, which is why they agree to 4e-16.
- ⚠️ `pykalman` and `filterpy` are the other names people reach for; neither is installed here
  and nothing about them was verified. `statsmodels` is the one this skill checked.

## 6. Traps

- 🚨 **Smoothed states in a backtest.** Section 1. The clean chart is the symptom.
- 🚨 **"Filtered is past-only".** It uses `y_t` to estimate the state at `t`, and section 2
  measures it absorbing more of the day's noise than the smoother does.
- 🚨 **A flat beta path from a converged fit.** Section 4: `q -> 0` is a real local optimum and
  `converged` is `True` there.
- 🚨 **Fitting Q and R on the full sample, then filtering the same sample.** Even with the
  predicted series, the *parameters* were chosen with the future. Refit on a rolling window if
  the result is sensitive to them; on this data the label leak dwarfs the parameter leak, as it
  does in regime-detection.
- 🚨 **A diffuse prior is not free.** This script uses `P0 = 1e5 h` on alpha and beta; the first
  few dozen observations of any state series are prior, not data. Drop a burn-in — the demo
  scores everything from t >= 250 — and check it with
  `../../../fin-core/skills/signal-construction/scripts/warmup_probe.py`.
- 🚨 **`q_beta` is a researcher degree of freedom.** Hand-tuning `q` until the beta path "looks
  right" is fitting the answer; every value tried is a trial for
  `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`. Estimate it by maximum
  likelihood, or fix it in advance and say so.
- ⚠️ **A time-varying beta is not automatically better.** On this series the causal Kalman hedge
  (Sharpe 1.47, beta RMSE 0.093) beat a rolling 250-day OLS (1.52, RMSE 0.156) on tracking error
  but not on Sharpe, and both beat a constant full-sample beta (1.45, RMSE 0.283) on residual
  variance. Report the constant-beta baseline next to the state-space one.

## 7. Scripts

- `scripts/kalman_models.py` — the numpy Kalman filter and RTS smoother, the local-level and
  local-linear-trend fits with `h` concentrated out, the statsmodels cross-check, the lead-lag
  and floor measurements, the `q -> 0` optimiser failure, and the `RecursiveLS` finding.
  `statsmodels` is imported inside two functions; without it the demo prints the same tables and
  says the recursions are unchecked. Seed 0, **17 s**.

## 8. Where this sits

`../../../fin-core/skills/regime-detection/SKILL.md` owns the smoothed-vs-filtered lesson for
discrete regime labels and measures what it is worth. `../stat-arb-cointegration/SKILL.md` is
where a time-varying hedge ratio usually ends up — as the spread of a pair — and owns the
cointegration tests and the z-score rules. `../volatility-models/SKILL.md` is the other latent
state in this plugin. `../../../fin-core/skills/signal-construction/SKILL.md` owns
`assert_causal` and `warmup_probe`. `../../../fin-core/skills/backtest-validation/SKILL.md`
counts `q` as a trial. `../../../fin-models/skills/factor-and-timeseries-research/SKILL.md` owns
factor betas estimated by regression rather than by filter.
