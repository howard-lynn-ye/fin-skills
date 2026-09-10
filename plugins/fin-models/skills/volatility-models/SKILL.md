---
name: volatility-models
description: >-
  Fit and forecast volatility - GARCH, range-based realized variance, HAR-RV - without the two
  errors that silently move the answer: the units `arch` expects, and a range estimator used on
  bars that gap. TRIGGER - GARCH, GARCH(1,1), EGARCH, GJR-GARCH, arch_model, conditional_volatility,
  DataScaleWarning, "y is poorly scaled", rescale=True, res.scale, alpha + beta near 1, IGARCH,
  persistence; realized volatility, realized variance, Parkinson, Garman-Klass, Rogers-Satchell,
  Yang-Zhang, close-to-close, high-low volatility estimator; HAR-RV, Corsi; "my GARCH forecast says
  1200% annualised vol", "which volatility estimator should I use", vol targeting input. SKIP for
  the arch package's API and its SPA/StepM/MCS bootstrap (lib-arch), for choosing a forecasting
  library (factor-and-timeseries-research), for discrete high-vol/low-vol state labels
  (regime-detection), for implied volatility (derivatives-pricing), and for turning a vol forecast
  into position size (portfolio-and-risk).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Volatility models

**A volatility number is meaningless without its units, and every library disagrees about them.**
`arch` was written for returns in percent; the range estimators are written for a continuous
price path with no overnight gap; HAR-RV is written for realized variance, not squared returns.
Each mismatch is silent — nothing raises, and `converged=True` is printed either way.

Every number below is printed by `scripts/vol_models.py` (numpy / scipy, seed 0, 3 s, `arch`
optional). ✅ Measured against `arch` 8.0.0 where marked; the GARCH(1,1) estimator is
re-implemented in numpy so the skill's claims hold on a machine without `arch`, and the
re-implementation is checked against the library rather than assumed to match it.

## 1. 🚨 `arch` is a percent library, and the default is to warn and carry on

✅ Source-verified in the installed `arch` 8.0.0. `arch_model`'s signature
(`arch/univariate/mean.py`, `def arch_model`) is
`arch_model(y, x=None, mean="Constant", lags=0, vol="GARCH", p=1, o=0, q=1, power=2.0,
dist="normal", hold_back=None, rescale=None)` — so the bare call is a **constant-mean
GARCH(1,1) with normal innovations and no rescaling**.

✅ `ARCHModel._check_scale` (`arch/univariate/base.py`, lines 305-326) is the whole mechanism:

- it runs only when `self.rescale in (None, True)`;
- it computes `scale = float(np.var(resids))` and multiplies by powers of ten until
  `0.1 <= scale < 10000`;
- if `rescale is None` (**the default**) it emits `DataScaleWarning` and **returns without
  rescaling** — `res.scale` stays `1.0`;
- only `rescale=True` sets `self.scale`, and then `fit` replaces `y` with `scale * y`.

✅ The warning text (`arch/utility/exceptions.py`, `data_scale_warning`) says "y is poorly scaled,
which may affect convergence of the optimizer", names the recommended multiplier, and ends by
telling you the warning can be disabled with `rescale=False` — which silences the message and
keeps the bad fit.

**What that costs, measured.** Seeded GARCH(1,1), T = 3,000, generated in percent
(`omega` 0.02, `alpha` 0.08, `beta` 0.90, so daily vol 1 % = 15.9 % annualised). The same series
is handed to `arch` four ways:

| call | warns | `res.scale` | fitted alpha / beta | llf | SLSQP iters / evals |
|---|---|---|---|---|---|
| returns in percent | no | 1 | 0.08830 / 0.89609 | -4112.801 | 10 / 62 |
| decimals, `rescale=None` (default) | **yes** | **1** | **0.10000 / 0.88000** | 9701.794 | **6 / 18** |
| decimals, `rescale=True` | no | **100** | 0.08830 / 0.89609 | -4112.801 | 10 |
| decimals, `rescale=False` | **no** | 1 | 0.10000 / 0.88000 | 9701.794 | 6 / 18 |

🚨 **The decimal fit did not fit anything.** `alpha = 0.10000, beta = 0.88000` is exactly arch's
starting grid point; the optimiser moved `mu` by -5.0e-05, `omega` by **-5.1e-14**, `alpha` by
**-2.5e-09** and `beta` by **-2.2e-08** and stopped after 6 iterations, reporting
`convergence_flag == 0`. Put the two likelihoods in comparable units — a decimal fit of the same
model should reach `llf_percent + T ln 100`, the Jacobian of `y / 100` — and the decimal run is
**0.916 nats short**. It is not a different parameterisation of the same optimum; it is a worse
optimum, flagged converged.

The damage on this path is quiet rather than dramatic, which is why it survives review: the
conditional-variance path from the decimal fit, put back into percent², differs from the percent
fit by **11.7 % at its worst and 3.4 % RMS**, and reported persistence is `0.9800` (the starting
value) against `0.9844`. Nobody looks at a persistence of 0.98 and suspects the optimiser never
ran.

🚨 **The 100x error is downstream, in the units of the forecast.** `res.forecast().variance` is in
the units the model was *estimated* in, which after `rescale=True` is (100 x return)². Measured
1-step-ahead annualised vol from the same four fits, each divided by its own `res.scale`:
**0.1222 / 0.1215 / 0.1222 / 0.1215** against a true 0.1286. Skip the division and the
`rescale=True` fit reports **12.22, i.e. 1222 % annualised vol**. Same for a percent fit read as
decimals.

> **Fit on `returns * 100`. Divide every variance by `res.scale**2` (and every volatility by
> `res.scale`) before annualising. `forecast()` returns variance, not volatility.**

`../../../fin-libraries/skills/lib-arch/SKILL.md` owns the rest of the package's API — `o=1` for
the leverage effect, `reindex=False`, `dist="skewt"`, and the `arch.bootstrap` half whose SPA /
StepM / MCS take **losses**.

## 2. GARCH(1,1) by hand: what it recovers and what it does not

`scripts/vol_models.py` implements the estimator in ~40 lines of numpy/scipy: arch's backcast
(✅ `VolatilityProcess.backcast`, `arch/univariate/volatility.py`: `tau = min(75, n)`,
`w = 0.94 ** arange(tau)` normalised, `sum(resids[:tau]**2 * w)`), the recursion
`sigma2[t] = omega + alpha e[t-1]^2 + beta sigma2[t-1]` seeded at
`sigma2[0] = omega + (alpha + beta) * backcast`, the Gaussian likelihood, and SLSQP under arch's
own bounds (`(1e-8 v, 10 v)` for omega with `v = mean(e^2)`, `(0, 1)` for alpha and beta) plus
`alpha + beta < 1`.

✅ Measured agreement with `arch` 8.0.0 on the seeded series:

| check | result |
|---|---|
| parameters, own minus arch | mu -3.4e-08, omega +2.9e-07, alpha +8.2e-07, beta -1.1e-06 |
| log-likelihood, own minus arch | **+4.0e-09** |
| our recursion evaluated at arch's parameters vs `res.conditional_volatility**2` | max abs diff **8.9e-16** |

✅ **Parameter recovery, T = 3,000** (true `mu` 0.03, `omega` 0.02, `alpha` 0.08, `beta` 0.90):
fitted `mu` 0.00943, `omega` 0.01732, `alpha` 0.08830, `beta` 0.89608, persistence 0.984 against
a true 0.980. The conditional-variance path tracks the truth with **corr 0.9997** and RMSE 0.047
on a mean variance of 1.029.

🚨 **The mean is the parameter that does not converge.** `mu` is off by -0.02057 — **1.11 standard
errors of the sample mean** (se 0.01858) — while `alpha` and `beta` land within **0.8 %** of the
true persistence. Three thousand days is a lot of information about variance dynamics and almost
none about drift. A GARCH fit is not a return forecast, and `res.params["mu"]` is not an expected
return.

**What GARCH(1,1) is and is not for.** The `h`-step forecast is
`sigma2[t+h] = omega + (alpha + beta) sigma2[t+h-1]`, which converges geometrically to
`omega / (1 - alpha - beta)`; at a persistence of 0.984 the half-life is `ln 0.5 / ln 0.984` ≈ 43
days, so a one-year forecast is the unconditional variance with extra steps. Use it for
next-day-to-next-month conditional variance (vol targeting, VaR scaling, option-free risk), not
for horizons past a quarter.

## 3. Range-based realized variance: five estimators, three assumptions

Formulas as implemented, with `u = ln(H/O)`, `d = ln(L/O)`, `c = ln(C/O)`, `o = ln(O/C_prev)`,
`r = ln(C/C_prev)`, averaged over an `n`-day window:

| estimator | formula | assumes |
|---|---|---|
| close-to-close | `var(r, ddof=1)` | nothing beyond i.i.d. returns |
| Parkinson (1980) | `mean((u - d)^2) / (4 ln 2)` | zero drift, no gap, continuous H/L |
| Garman-Klass (1980) | `mean(0.5 (u - d)^2 - (2 ln 2 - 1) c^2)` | zero drift, no gap, continuous H/L |
| Rogers-Satchell (1991) | `mean(u(u - c) + d(d - c))` | **any drift**, no gap, continuous H/L |
| Yang-Zhang (2000) | `var(o) + k var(c) + (1 - k) RS`, `k = 0.34 / (1.34 + (n+1)/(n-1))` | any drift, **gaps handled**, continuous H/L |

⚠️ The formulas and the constant `k` are as published in the four papers; this skill did not read
the papers, so what is verified here is **the property each paper claims**, by simulation on a
GBM with a known variance — which is the claim that matters and the one that can be wrong in your
code. 12,600 days, 390 intraday steps per day, 21-day windows (600 windows), annual vol 20 %.
`bias` is mean estimate / true variance with its Monte Carlo standard error; `efficiency` is
`Var(close-to-close) / Var(estimator)`:

| estimator | bias, zero drift | efficiency | bias, drift 200 %/yr | bias, 30 % of variance overnight | efficiency there |
|---|---|---|---|---|---|
| close-to-close | 0.976 +/- 0.013 | 1.00 | 0.976 | 0.975 | 1.00 |
| Parkinson | 0.929 +/- 0.005 | **5.38** | **1.074** | **0.650** | 10.13 |
| Garman-Klass | 0.907 +/- 0.004 | **8.31** | 0.955 | **0.635** | 15.65 |
| Rogers-Satchell | 0.909 +/- 0.005 | **6.23** | **0.902** | **0.636** | 11.72 |
| Yang-Zhang | 0.918 +/- 0.005 | **7.41** | 0.912 | **0.940** | 6.43 |

Three results, each measured, each matching what the corresponding paper claims:

- ✅ **The efficiency gain is real and large.** Five to eight close-to-close observations' worth
  of information per bar, and RMSE against the true variance is roughly halved (0.45-0.50 x).
  If you have OHLC and no intraday data, this is the cheapest accuracy in the toolbox.
- ✅ **Rogers-Satchell is the drift-independent one, and it is the only one.** Take the drift from
  0 to 200 %/yr and RS moves 0.909 -> 0.902 (inside one standard error) while Parkinson goes
  0.929 -> **1.074**: at a strong trend Parkinson *over*-states variance by 7 %, because a
  trending day's high-low range is wide for a reason that is not volatility. Garman-Klass moves
  0.907 -> 0.955. At a plausible equity drift of 50 %/yr none of them moves measurably — the
  drift term is second order, and it only bites on strongly trending instruments.
- 🚨 **An overnight gap breaks three of the five.** Move 30 % of the daily variance into the
  close-to-open jump and Parkinson, Garman-Klass and Rogers-Satchell read **0.650, 0.635, 0.636**
  — they measure only the part of the day they can see, and they do not warn. Yang-Zhang, which
  adds `var(o)` back, reads 0.940. **On daily equity bars, which always gap, the choice is
  Yang-Zhang or close-to-close.** Their efficiency *rises* to 10-16 in that column, which is the
  trap in one number: a tighter estimate of the wrong quantity.

🚨 **Every range estimator is biased low on a discretely observed path**, because the observed
high and low of `k` samples are inside the continuous path's high and low. Same GBM, 4,200 days,
zero drift, no gap — the whole column has to walk to 1.000 as sampling gets finer, and it does:

| steps/day | close-to-close | Parkinson | Garman-Klass | Rogers-Satchell | Yang-Zhang |
|---|---|---|---|---|---|
| 26 (15-min) | 0.998 | 0.771 | 0.683 | 0.678 | 0.723 |
| 78 (5-min) | 0.999 | 0.866 | 0.815 | 0.814 | 0.840 |
| 390 (1-min) | 0.968 | 0.929 | 0.909 | 0.911 | 0.919 |
| 3,900 (6-sec) | 1.021 | 0.981 | 0.966 | 0.958 | 0.967 |

That convergence to 1.000 **is** the verification of the four constants (`1/(4 ln 2)`,
`2 ln 2 - 1`, the RS cross-products, the Yang-Zhang `k`): get any of them wrong and the column
converges to something else. Read the other direction, it is a warning: on a real daily bar built
from minute data, Parkinson reads about **7 % low** on variance (3.5 % on vol) purely from
discretisation, before any market microstructure. Close-to-close has no such bias — its wobble
here (0.968, 1.021) is sampling noise on 200 windows, standard error 0.021-0.025.

## 4. HAR-RV (Corsi 2009)

An OLS of realized variance on its own daily, weekly and monthly averages —
`RV[t] ~ c + b1 mean(RV[t-1]) + b5 mean(RV[t-5..t-1]) + b22 mean(RV[t-22..t-1])`. Three regressors,
no optimiser, and it is the baseline that volatility papers have to beat.

✅ On seeded realized variance (3,000 days, 78 five-minute returns/day from a log-vol AR(1),
`phi` 0.98), fit on the first 2,000 days: `c` 6.89254e-06, daily **0.354747**, weekly
**0.591848**, monthly **0.0118399** (they sum to 0.9584 on a mean RV of 1.658e-04).
✅ `arch.univariate.HARX(rv, lags=[1, 5, 22], volatility=ConstantVariance())` returns the same
four numbers to **max abs diff 3.9e-15** — the design matrix convention matches, which is the
thing worth checking, since an off-by-one in the lag windows is a look-ahead.

Scored one step ahead on the held-out 1,000 days:

| forecast | MSE | QLIKE |
|---|---|---|
| **HAR** | **1.024e-09** | **0.0259** |
| naive (yesterday's RV) | 1.322e-09 | 0.0326 |
| trailing 22-day mean | 2.172e-09 | 0.0591 |

HAR / naive = **0.775 on MSE, 0.794 on QLIKE**. ⚠️ On this DGP — a persistent log-vol AR(1), not
a true cascade — the monthly term carries almost no weight (0.012); do not read the coefficient
split as evidence for the three-component story. Report QLIKE next to MSE: MSE on variance is
dominated by the few largest days, and a forecast can win it by being conservative everywhere.

## 5. Traps

- 🚨 **Decimals into `arch`.** Section 1. The default warns and does not fix it, `rescale=False`
  removes the warning and keeps the problem, and `rescale=True` changes the units of everything
  that comes back.
- 🚨 **Reading `forecast().variance` as volatility, or in the wrong scale.** It is variance, in
  estimation units. `sqrt(variance) / res.scale * sqrt(252)` is the annualised number.
- 🚨 **A range estimator on daily bars.** Parkinson/Garman-Klass/Rogers-Satchell read 35 % low
  when 30 % of the variance is overnight. Use Yang-Zhang, or accept close-to-close.
- 🚨 **`alpha + beta` at 0.999+ is a diagnosis, not a fit.** Persistence pinned at the constraint
  boundary means the unconditional variance `omega / (1 - alpha - beta)` explodes and long-horizon
  forecasts are meaningless. It is usually a structural break or an outlier inside the sample,
  not evidence of true near-unit-root volatility.
- 🚨 **Fitting on the whole sample and calling the fitted `conditional_volatility` a forecast.**
  Every point in that series was produced with parameters estimated from the *entire* series,
  including its own future. It is an in-sample filter. For a backtest, refit on a rolling window
  and use `forecast(horizon=1)` from the end of each window — the same
  smoothed-vs-filtered distinction that
  `../../../fin-core/skills/regime-detection/SKILL.md` measures for regime labels, where a
  full-sample fit anticipated 47 % of regime switches *before they started*. Do not re-derive it
  here; that skill has the numbers.
- 🚨 **A volatility model is a trial like any other.** GARCH vs GJR vs EGARCH, normal vs t vs
  skew-t, window length, the estimator in section 3, the annualisation factor — each choice is a
  degree of freedom, and a strategy tuned on the winner is a best-of-N result. Record them in
  `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`.
- ⚠️ **252 is a convention, not a fact.** Annualising with 252 vs 260 vs the actual trading-day
  count moves a volatility number by ~1.5 %. Pick one, state it, and use the same one for the
  Sharpe ratio (`../../../fin-core/skills/portfolio-and-risk/SKILL.md`).
- 🚨 **Volatility as a signal is a causal-timing question.** A vol estimate that uses today's
  close cannot size today's position. Run the series through
  `../../../fin-core/skills/signal-construction/scripts/assert_causal.py` and check the burn-in
  with `warmup_probe.py` — a 22-day HAR regressor needs 22 days of history before its first
  honest value.

## 6. Scripts

- `scripts/vol_models.py` — all four sections: the numpy GARCH(1,1) MLE and its agreement with
  `arch`, the four scaling paths, the range-estimator bias/efficiency/discretisation tables, and
  HAR-RV against `arch`'s `HARX`. numpy + scipy; `arch` is imported inside two functions and the
  demo degrades to a numpy-only comparison without it. Seed 0, **3 s**.

## 7. Where this sits

`../../../fin-libraries/skills/lib-arch/SKILL.md` is the deep dive on the package (API, licence,
the bootstrap half). `../../../fin-models/skills/factor-and-timeseries-research/SKILL.md` chooses
between forecasting libraries and carries the foundation-model evidence.
`../../../fin-core/skills/regime-detection/SKILL.md` owns discrete vol *states* and the
smoothed-vs-filtered timing lesson. `../state-space-and-kalman/SKILL.md` is the other
latent-state estimator in this plugin, and its filtered-vs-smoothed trap is the same one.
`../../../fin-core/skills/derivatives-pricing/SKILL.md` owns implied volatility.
`../../../fin-core/skills/portfolio-and-risk/SKILL.md` turns a variance forecast into position
size and owns the annualisation conventions.
