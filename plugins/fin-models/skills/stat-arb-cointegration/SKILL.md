---
name: stat-arb-cointegration
description: >-
  Screen, test and trade a cointegrated pair without counting the trials wrong, applying the
  single-series ADF table to a fitted residual, or estimating the hedge ratio on the window you
  score it in. TRIGGER - cointegration, Engle-Granger, statsmodels coint, coint_johansen,
  Johansen, trace statistic, max eigenvalue, VECM, cointegrating vector; pairs trading,
  statistical arbitrage, stat arb, spread, hedge ratio, spread z-score, entry and exit
  thresholds, half-life, Ornstein-Uhlenbeck, OU mean reversion; adfuller on the residual,
  CollinearityWarning, "my screen found 30 cointegrated pairs", "the spread stopped mean
  reverting", "in-sample Sharpe 2 and it lost money live". SKIP for a time-varying hedge ratio
  by Kalman filter (state-space-and-kalman), for the multiple-testing machinery itself and
  PSR/DSR (backtest-validation), for regime labels (regime-detection), and for the cost of
  trading two legs (execution-cost-analysis).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Statistical arbitrage and cointegration

**A pairs screen is a multiple-testing machine wrapped around a test most people run with the
wrong table.** Among 20 independent random walks there are 190 pairs; the residual of a fitted
cointegrating regression oscillates around zero *by construction*, so the in-sample spread trade
looks profitable whether or not the pair is real. This skill measures both, with seeds.

Every number below is printed by `scripts/cointegration.py` (numpy, seed 0, 2.9 s,
`statsmodels` optional). The Engle-Granger statistic is implemented here without any published
table — ✅ it reproduces `statsmodels.tsa.stattools.coint` to **4.0e-14** on the demo pair and
**9.1e-14** as a maximum over all 190 screened pairs — and its critical values are simulated
under the null rather than looked up.

## 1. 🚨 The residual has been fitted, so the ADF table does not apply

`coint` runs a two-step test: OLS of `y` on `x` with a constant, then an ADF-type regression on
the residual. ✅ Source-verified in statsmodels 0.15.0
(`statsmodels/tsa/stattools/_stattools.py`, `def coint`): the second stage is
`adfuller(res_co.resid, maxlag=maxlag, autolag=autolag, regression="n")` — **no constant** — and
the critical values come from `mackinnoncrit(N=k_vars, regression=trend, nobs=nobs - 1)` with
`k_vars = 2` for a pair, not `N = 1`.

Running `adfuller(resid)` yourself and reading its p-value uses **both** the wrong regression
(its default is `regression="c"`) and the wrong table (single-series, `N = 1`). ✅ Measured on
2,000 simulated pairs of independent random walks, T = 500, one augmentation lag:

| statistic | 1 % | 5 % | 10 % |
|---|---|---|---|
| Engle-Granger, simulated here under the null | -3.955 | **-3.404** | -3.102 |
| what `coint` returns (MacKinnon 2010, N=2) | -3.919 | **-3.348** | -3.053 |
| `adfuller`'s single-series value (N=1, constant) | — | **-2.867** | — |
| the same fitted residuals, ADF-with-constant statistic, simulated | — | **-3.401** | — |

- ✅ MacKinnon's 5 % value is well calibrated here: it rejects **5.1 %** of the null pairs.
- 🚨 The single-series value rejects **16.5 %** of independent random-walk pairs instead of 5 %.
  Fitting the hedge ratio first shifts the null distribution by **-0.534**, and the ADF table
  knows nothing about it.
- ✅ On the actual screen: 190 pairs of independent random walks, **6 pass `coint` at p < 0.05**
  and **17 pass `adfuller`-on-the-residual at p < 0.05** — the wrong table nearly triples the
  false-positive count on one seed.

**Three more `coint` behaviours, all source-verified and all measured on the demo pair:**

- 🚨 **`autolag` defaults to `"aic"`**, and its own docstring warns this changed from `None` in
  statsmodels 0.8. Same data, same call, one keyword: `autolag="aic"` gives **t = -3.7514,
  p = 0.0157**; `autolag=None, maxlag=1` gives **t = -3.5767, p = 0.0262**. Fix the lag order and
  record it, or the test is a lag search you did not count.
- 🚨 **`trend="n"` returns critical values that are all `NaN`** (the source comments that the
  2010 values are not available) while still returning a p-value — 0.0175 here. Measured: the
  returned `crit` array is all-NaN, `True`.
- 🚨 **Near-collinear series always "cointegrate".** When `res_co.rsquared >= 1 - 100 * SQRTEPS`
  the function skips the ADF entirely, warns `CollinearityWarning`, and returns
  **t = -inf, p = 0.0**. Measured against `1.0001 * y` plus 1e-9 noise. Two share classes, a
  stock and its own ADR, or an index and a near-replicating basket land here, and p = 0.0 is not
  evidence of anything.

## 2. 🚨 190 pairs is 190 trials, and the survivors trade beautifully in sample

✅ 20 independent random walks, T = 500, screened on all 190 pairs at the simulated 5 % value
(-3.404): **6 pairs pass** (3.2 %; 9.5 expected — the 190 tests share series, so they are not
190 independent draws and the count is noisier than a binomial). Those six are then traded with
a z-score on the spread, entry at ±2, exit at ±0.5, hedge ratio and z-score mean/sd all from the
screening window:

| pair | EG t | in-sample Sharpe (trades) | next 250 days, same beta/mean/sd |
|---|---|---|---|
| (4, 12) | -3.47 | **1.48** (5) | -0.41 |
| (4, 15) | -3.67 | **1.92** (10) | -0.94 |
| (8, 19) | -3.99 | **1.66** (6) | -3.25 |
| (9, 11) | -3.41 | **0.97** (4) | -0.48 |
| (14, 18) | -4.48 | **2.28** (10) | -1.31 |
| (15, 17) | -3.63 | **2.26** (10) | -0.16 |

Mean in-sample Sharpe **1.76**, mean out-of-sample **-1.09**, and **0 of 6** positive out of
sample. There is nothing in this data — the series are independent random walks by construction.

🚨 **Look at the trade counts out of sample: one each, 100 % of days in a position.** The frozen
z-score entered on the first day and never came back inside the band; `|z|` never fell below
**11.78** in any of the six windows, reaching 44.6 in one. That is the whole failure mode in one
number: a spread of two random walks has no level to revert to, so the z-score computed from a
past window walks away and the "market-neutral, mean-reverting" position becomes a naked bet
held for a year.

**Count the pairs.** A universe of `n` names is `n(n-1)/2` trials before any parameter is chosen;
1,000 names is 499,500. Record them in
`../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py` and apply the deflation
there — `../../../fin-core/skills/backtest-validation/SKILL.md` owns the correction, this skill
just supplies the trial count. Reducing the count is a modelling decision (screen within a
sector, require an economic link, pre-register the universe), not a p-value adjustment.

## 3. 🚨 A finite half-life is not evidence of anything

The OU half-life comes from an AR(1) on the spread: `ds_t = a + b s_{t-1} + e_t`, half-life
`-ln 2 / ln(1 + b)` (exact for the discrete AR(1); `-ln 2 / b` is the continuous-time
approximation).

✅ Measured on the genuine pair (spread AR(1) with `phi` = 0.95, true half-life **13.5 d**): the
AR(1) on the *true* spread gives `phi` = 0.955, half-life **15.2 d** (OU approximation 15.5 d),
t = -4.81. On the OLS spread built with the estimated hedge ratio 0.714 instead of the true 0.8,
half-life **18.8 d** — the hedge-ratio error lengthens the apparent half-life by a quarter.

🚨 **And on the 190 random-walk pairs the half-life is finite in 100 % of them, median 48 days**
(median 14 days among the six that passed the test). `b` is negative in every single fitted
residual, because a residual is orthogonal to its own regressor in-sample. A half-life you can
compute is not a half-life that exists. Report it with the AR(1) t-statistic next to it, and
treat "the half-life is 20 days" on its own as a description of the fit, not a finding.

## 4. Estimating the hedge ratio on the window you trade

✅ A genuinely cointegrated pair (true beta 0.8, spread `phi` 0.95, T = 1,000), scored on days
500-999 six ways:

| what the strategy knew | Sharpe, days 500-999 |
|---|---|
| **hedge ratio and z-score from days 500-999 themselves** | **2.49** |
| first window's beta, first window's z mean/sd, frozen | 0.07 |
| first window's beta, **rolling 60-day z** | 2.13 |
| **rolling 250-day beta, rolling 60-day z** (fully causal) | **1.20** |
| the TRUE beta, rolling 60-day z | 1.86 |
| (for reference: same-window fit on days 0-499) | 1.87 |

🚨 **In-sample minus causal: +1.29 Sharpe**, on a pair that really is cointegrated. The
in-sample fit is not inventing the edge here — it is inflating a real one by more than half.

Two things worth reading off the ladder rather than the headline:

- **Rolling everything is not automatically the right causal answer.** The fixed beta estimated
  on the *prior* window with a rolling z-score scored 2.13, better than re-estimating beta on a
  rolling 250-day window (1.20) and better than the true beta (1.86), because on this DGP the
  true beta is constant and re-estimating it just adds noise. ✅ The rolling beta ranged
  **0.318 to 1.191** around a true 0.8. If the relationship is stable, a beta fixed before the
  test window is both causal and better; if it is not, see `../state-space-and-kalman/SKILL.md`,
  and note that its predicted (not smoothed) state is the one you may trade.
- **The frozen z-score is the failure case again** (0.07): freezing beta is fine, freezing the
  spread's *mean and standard deviation* is not, because the spread's level drifts.

## 5. Johansen: what it returns, and what it misses

✅ Source-verified: `coint_johansen(endog, det_order, k_ar_diff)`
(`statsmodels/tsa/vector_ar/vecm.py`). `det_order` is -1 (none) / 0 (constant) / 1 (linear
trend) and warns outside that; more than 12 variables warns too. The result exposes
`lr1`/`trace_stat` and `lr2`/`max_eig_stat`, with `cvt`/`trace_stat_crit_vals` and
`cvm`/`max_eig_stat_crit_vals`, plus `eig`, `evec` and `ind`. The tables come from
`statsmodels/tsa/coint_tables.py` (`c_sjt`, `c_sja`), whose docstrings cite **MacKinnon, Haug,
Michelis (1996)** and note they were generated with MacKinnon's `johdist.f`; they return NaN
outside 1 ≤ n ≤ 12.

🚨 **The critical-value columns are 90 %, 95 %, 99 % — the reverse orientation of `coint`'s
1 %, 5 %, 10 %.** `crit[1]` means the 5 % level in one and the 95 % level in the other, and
`crit[0]` means 1 % in one and 90 % in the other. ✅ Measured, 2 series, `det_order=0`:

```
trace, r=0    [13.4294  15.4943  19.9349]      max-eig, r=0    [12.2971  14.2639  18.52  ]
trace, r<=1   [ 2.7055   3.8415   6.6349]      max-eig, r<=1   [ 2.7055   3.8415   6.6349]
```

Reject `r = 0` when the trace statistic **exceeds** its critical value (the opposite direction to
Engle-Granger's, which rejects on a sufficiently negative t).

✅ **Measured power, same 500 days:** on the genuinely cointegrated pair Johansen reports trace
14.59 against a 95 % value of 15.49 → **rank 0, no cointegration found**, while Engle-Granger
rejects at p = 0.0262. On the *best false* random-walk pair Johansen reports trace 22.24 → rank 1.
And across the six Engle-Granger passes, Johansen agrees on **4 of 6**. Two tests agreeing is not
independent confirmation — they are computed from the same 500 observations.

The cointegrating vector's implied hedge ratio was **0.702** against OLS 0.714 and a true 0.8:
Johansen normalises `evec[:, 0]` however the eigen-decomposition lands, so take
`-evec[1, 0] / evec[0, 0]` for `y - beta x` and check the sign against a scatter plot before
trading it.

## 6. Traps

- 🚨 **`adfuller` on a fitted residual** — section 1. Use `coint`, or simulate your own null.
- 🚨 **Every pair screened is a trial** — section 2, and the count is `n(n-1)/2`.
- 🚨 **A frozen z-score** — sections 2 and 4. Recompute the spread mean and sd on a trailing
  window; a level estimated once will drift away from the spread.
- 🚨 **A finite half-life on a random walk** — section 3, 100 % of 190 pairs.
- 🚨 **`autolag`, `trend` and the collinearity short circuit** — section 1. Pin the keywords.
- 🚨 **Prices, not returns, and no adjustment mismatch.** Cointegration is a statement about
  *levels*, so the two price series must be on the same adjustment convention over the whole
  window. A dividend adjustment applied to one leg and not the other manufactures a trend in the
  spread — see `../../../fin-core/skills/market-data-engineering/SKILL.md`.
- ⚠️ **Two legs, two spreads, two borrow costs.** Everything above is gross. A 13-day half-life
  with entry at 2 sigma trades often enough that costs decide the answer;
  `../../../fin-core/skills/execution-cost-analysis/SKILL.md` owns that, and the pair's short
  leg has a borrow rate that is not in any price file.
- ⚠️ **Survivorship in the universe.** Screening today's listed names over five years of history
  finds pairs among the companies that survived. `../../../fin-core/skills/market-data-sourcing/SKILL.md`
  and `../../../fin-core/skills/research-integrity-guards/SKILL.md` own that.
- 🚨 **Causality of the signal itself.** `z[t]` must be computed from data through `t` and earn
  `t -> t+1`; the script's `spread_pnl` enforces exactly that. Check yours with
  `../../../fin-core/skills/signal-construction/scripts/assert_causal.py`.

## 7. Scripts

- `scripts/cointegration.py` — the Engle-Granger statistic and its simulated null, the
  190-pair screen with both the right and the wrong table, the six false pairs traded, the
  in-sample-vs-rolling ladder on a real pair, Johansen and its tables, and the OU half-life.
  `statsmodels` is imported inside each cross-check function; without it the script still
  simulates its own critical values, screens, and trades. Seed 0, **2.9 s**.

## 8. Where this sits

`../../../fin-core/skills/backtest-validation/SKILL.md` owns the multiple-testing correction
that the trial count from section 2 feeds. `../state-space-and-kalman/SKILL.md` owns a
time-varying hedge ratio and the filtered-vs-smoothed rule that governs it.
`../time-series-forecasting-models/SKILL.md` owns the unit-root and stationarity testing that
comes before any of this. `../../../fin-core/skills/signal-construction/SKILL.md` owns
`assert_causal` and `warmup_probe`.
`../../../fin-core/skills/execution-cost-analysis/SKILL.md` owns the two-leg cost.
`../../../fin-core/skills/research-integrity-guards/SKILL.md` owns the pre-report audit.
