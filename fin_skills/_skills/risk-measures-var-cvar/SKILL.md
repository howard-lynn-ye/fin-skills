---
name: risk-measures-var-cvar
description: >-
  Compute Value-at-Risk and Expected Shortfall by the four estimators that disagree in the tail,
  and backtest them properly. TRIGGER - VaR, value at risk, CVaR, expected shortfall, ES, tail
  risk, 99% VaR, 95% VaR, historical simulation VaR, parametric normal VaR, Cornish-Fisher
  expansion, EVT, peaks over threshold, generalized Pareto, scipy genpareto, tail index xi,
  Kupiec proportion of failures, Christoffersen independence, conditional coverage, VaR
  exceptions or breaches, traffic light test, square root of time scaling, 10-day VaR, Basel,
  filtered historical simulation, quantstats value_at_risk sign; "how many exceptions should I
  see", "is my VaR model backtesting ok". SKIP for estimating the covariance matrix a
  parametric VaR needs (covariance-and-risk-models), for minimising CVaR to choose weights
  (portfolio-optimizers), for GARCH fitting itself (factor-and-timeseries-research), and for
  Sharpe, drawdown and tearsheet conventions (portfolio-and-risk).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Risk measures: VaR and CVaR

**"The 99 % VaR is 2.3 %" is not a fact until three things are stated: the estimator, the
horizon, and how the number was scaled to that horizon.** All three are choices, and on the same
data they move the answer by more than the risk does.

Every number below is printed by `scripts/var_cvar.py` (numpy / scipy, seed 0, about 6 s). The
DGP is a GARCH(1,1) with alpha 0.08, beta 0.90 and unit-variance Student-t(5) innovations at
1 %/day unconditional volatility, so the truth is known: the unconditional quantile from a
1,000,000-day path, and the conditional quantile in closed form.

🔑 **Sign convention here: every VaR and ES is a POSITIVE LOSS**, `losses = -returns`.

## 1. Four estimators, one tail, and they do not agree

✅ Measured on a 2,500-day sample, 99 % 1-day, against the 1,000,000-day truth:

| estimator | VaR % | ES % | VaR error | ES error |
|---|---|---|---|---|
| historical (empirical quantile) | 2.404 | 3.306 | -9.8 % | -12.7 % |
| 🚨 parametric normal | 2.144 | 2.459 | **-19.6 %** | **-35.0 %** |
| 🚨 Cornish-Fisher | 4.164 | 7.080 | **+56.2 %** | **+87.0 %** |
| **EVT peaks-over-threshold**, u = 90th pct | 2.479 | 3.343 | **-7.0 %** | **-11.7 %** |
| TRUE (1e6-day path of the same DGP) | 2.666 | 3.786 | - | - |

- The **normal** estimator is not slightly wrong, it is wrong in the direction that matters: it
  understates ES by 35 % on fat-tailed data. ✅ The true Student-t(5) conditional quantile at
  99 % is **2.606 sigma** against the normal's 2.326, and the conditional ES is **3.449 sigma**
  against 2.665.
- 🚨 **Cornish-Fisher stops being a quantile function.** ✅ At skew 0 the expansion's derivative
  is `1 + (3z^2 - 3)K/24`, minimised at z = 0 where it equals `1 - K/8`: the break is at
  **excess kurtosis 8 exactly** (a 0.05-step search reports 8.05). This sample has skew -0.75
  and excess kurtosis **12.59**, and ✅ its expansion is **decreasing over z in [-0.50, +0.67]**,
  which is to say the "74.9 % VaR" it returns (-0.114 sigma) is *smaller* than the "30.9 %
  VaR" (+0.280 sigma). A function that goes down as the confidence level goes up is not a
  quantile function. The +56 % at 99 % is not conservatism, it is a series evaluated outside
  its domain. **Call `cornish_fisher_is_monotone(skew, exkurt)` before using it** - a daily
  equity index is routinely past the threshold.
- **EVT is the best of the four** and still 7 % low, because 2,500 days contains 25 tail
  observations. ✅ The fit: tail index xi = **0.115** (a genuinely heavy tail), scale 0.00586,
  250 exceedances over the 90th percentile. Its closed-form ES agrees with numerical
  integration of the fitted Pareto tail to four decimals: **3.3428 %** both ways.
- ✅ Source-verified numerically in **scipy 1.13.0**: `genpareto.pdf(x, c)` equals
  `(1 + c x)^(-1 - 1/c)` for c != 0 and `exp(-x)` at c = 0, so scipy's shape parameter `c`
  **is** the tail index xi with no reparameterization. Fit the exceedances with `floc=0`;
  leaving `loc` free re-estimates the threshold you already chose.

⚠️ Threshold choice is a researcher degree of freedom: the 90th percentile is a convention, not
a result. Report the threshold, the exceedance count and xi alongside the number, and record the
choice as a trial in `../../../fin-core/skills/backtest-validation/SKILL.md`.

## 2. The rolling backtest: count and timing are different failures

✅ Measured: 500-day rolling window, 2,000 test days, 99 % 1-day VaR. `pi11` is
P(exception | exception yesterday); the unconditional rate is 1 %.

| method | exceptions | expected | Kupiec p | indep. p | cc p | n11 | pi11 | mean VaR % |
|---|---|---|---|---|---|---|---|---|
| historical | 28 | 20 | 0.090 | 0.408 | 0.169 | 1 | 0.04 | 2.396 |
| 🚨 parametric normal | **36** | 20 | **0.001** | 0.167 | **0.002** | 2 | 0.06 | 2.147 |
| Cornish-Fisher | 15 | 20 | 0.240 | 0.100 | 0.130 | 1 | 0.07 | 3.180 |
| EVT POT (refit every 10 d) | 25 | 20 | 0.279 | 0.320 | 0.340 | 1 | 0.04 | 2.500 |
| 🚨 EWMA(0.94) + normal quantile | **32** | 20 | **0.013** | 0.106 | 0.012 | 2 | 0.06 | 2.019 |
| **EWMA(0.94) filtered historical** | **21** | 20 | **0.824** | 0.217 | **0.455** | 1 | 0.05 | 2.325 |
| oracle: true sigma_t x t(5) quantile | 13 | 20 | 0.093 | 0.680 | 0.224 | 0 | 0.00 | 2.346 |

**The two halves of the problem are separable.** The EWMA filter supplies the right *scale* day
by day but a normal quantile still misses the *shape*, so `EWMA + normal` fails the count at
p = 0.013 while carrying the lowest mean VaR of the lot (2.019 % - it is cheap and wrong).
Filtered historical simulation keeps the EWMA scale and takes the **empirical** quantile of the
standardized residuals, and it passes everything with 21 exceptions against 20 expected.
Cornish-Fisher errs the other way: 15 exceptions, and ✅ **36 % more capital than the oracle's
mean VaR** (3.180 % against 2.346 %) for the privilege.

🚨 **The independence test rejects nothing here - and that is not evidence of independence.**
`pi11` runs 4 to 7 times the unconditional 1 % in every unconditional method, yet no p-value is
below 0.10, because 20 exceptions in 2,000 days produce one or two consecutive pairs to test on.
Section 3 measures what that is worth.

## 3. The backtests themselves, and their real size

✅ The statistic, computed two ways: `kupiec_pof(28, 2000, 0.01)` returns **2.8748121042** and
`-2 * (binom.logpmf(28, 2000, 0.01) - binom.logpmf(28, 2000, 28/2000))` returns **2.8748121042**
(difference 2.1e-14). The binomial coefficient cancels between the two hypotheses, which is why
the likelihood ratio may be written either way.

- **Kupiec proportion-of-failures**:
  `LR_uc = -2 ln[(1-p)^(n-x) p^x / ((1-x/n)^(n-x) (x/n)^x)]`, chi2(1). ✅ It is exactly 0 when
  x = p n.
- **Christoffersen independence**: a first-order Markov chain on the hit sequence against a
  constant hit probability, `LR_ind`, chi2(1). ✅ It is 0 when `pi01 = pi11`.
- **Conditional coverage**: `LR_cc = LR_uc + LR_ind`, chi2(2).

🚨 **Their nominal size is not their real size.** ✅ Measured on 10,000 iid Bernoulli(0.01) hit
sequences of 500 days, nominal 5 %:

| test | rejection rate under the null |
|---|---|
| Kupiec | **6.9 %** (oversized: it rejects correct models too often) |
| Christoffersen independence | **1.4 %** (undersized: almost no power) |
| conditional coverage | **1.8 %** |

The hit count is a discrete random variable with about 5 expected exceptions in 500 days, so the
chi2 approximation is coarse in both directions. **A passed independence test at the 99 % level
is close to no evidence at all**; at the 95 % level there are five times as many exceptions and
the test has something to work with.

🚨 **And "independence rejected" does not mean clustering.** ✅ Measured: a perfectly *regular*
pattern - exactly one exception every fifth day - gives `LR_ind` = **50.1**, p = **0.000**. The
test compares `pi01` with `pi11` and fires just as loudly on under-dispersion as on clustering.

🚨 **Whenever `pi01 = pi11` it sees nothing, however predictable the sequence.** ✅ Measured: the
block pattern `1,1,0,0` repeated has `pi01 = pi11 = 0.50`, so `LR_ind` = **2.5e-03**, p =
**0.960** - a sequence you could forecast perfectly passes the independence test. The Markov
alternative only looks one day back; anything structured at a longer lag is invisible to it by
construction.

## 4. 🚨 sqrt(h) scaling is wrong in both directions

✅ Source-verified in the **RiskMetrics Technical Document, Fourth Edition (December 1996)**,
Sec. 5.2, Eq. [5.21]: the square-root-of-time relation is derived *inside* the EWMA/IGARCH model,
and the document itself warns that the rule normally "results from the assumption that variances
are constant" while in its own derivation they vary. It names three cases where scaling up is
problematic: mean-reverting rates or prices, boundaries that limit movements, and using a
volatility estimate optimised for one horizon at another.

✅ Measured, 99 %, loss = the sum of h daily returns:

| returns | 1-day VaR % | sqrt(10) x | true 10-day % | error |
|---|---|---|---|---|
| iid normal, 1 %/day | 2.321 | 7.341 | 7.289 | **+0.7 %** (the assumption holds) |
| iid Student-t(5), fat tails only | 2.615 | 8.270 | 7.590 | **+9.0 %** |
| GARCH-t, unconditional | 2.666 | 8.431 | 8.172 | +3.2 % |
| GARCH-t, sigma_t = 0.5 x long-run | 1.303 | 4.121 | 4.461 | **-7.6 %** |
| GARCH-t, sigma_t = 1.0 x long-run | 2.606 | 8.242 | 8.024 | +2.7 % |
| GARCH-t, sigma_t = 2.0 x long-run | 5.213 | 16.485 | 15.528 | +6.2 % |

**The sign of the error is not fixed.** Fat tails alone make sqrt(h) *overstate*, because a sum
of h draws is closer to normal than one draw is. Starting from a quiet day makes it
*understate*, because volatility mean-reverts upward over the horizon. "sqrt(h) is conservative"
is false as often as it is true, and the direction depends on the state you are in when you
compute it - which is exactly when you care.

✅ And the error grows with the horizon:

| state | h = 5 | h = 10 | h = 25 | h = 60 |
|---|---|---|---|---|
| GARCH-t, sigma_t = 0.5 x long-run | -1.1 % | -5.8 % | **-16.9 %** | **-29.3 %** |
| GARCH-t, sigma_t = 2.0 x long-run | +5.8 % | +8.4 % | +13.2 % | +25.6 % |
| iid Student-t(5), no clustering | +7.1 % | +9.0 % | +9.7 % | +11.5 % |

A quarterly VaR scaled from a quiet day understates the truth by 29 %. **The guard.**
`scale_var_sqrt_time(var, h)` refuses unless you pass `iid_normal_zero_mean=True`, so the
assumption appears in the code that relies on it. Simulate the horizon from the current
conditional variance instead (`horizon_var_from_state`).

## 5. VaR is not subadditive; ES is

✅ Measured, exactly: two independent positions, each losing 1.0 with probability 2 %, at the
97.5 % level.

| measure | each | sum of the two | the combined book |
|---|---|---|---|
| 🚨 VaR | 0.000 | 0.000 | **1.000** |
| ES | 0.800 | 1.600 | **1.016** |

`VaR(A+B) > VaR(A) + VaR(B)`: **diversification makes the VaR go up**, which is not a paradox to
be explained away - it is why VaR is not a coherent risk measure and why an optimizer that
minimises VaR can be told to concentrate. ES obeys the inequality. Optimize and report ES; quote
VaR when a regulator asks for it. `../portfolio-optimizers/SKILL.md` has the linear program that
minimises ES directly.

## 6. Traps

- 🚨 **A VaR without its estimator, horizon and scaling stated.** The four estimators here span
  2.14 % to 4.16 % on one sample.
- 🚨 **Cornish-Fisher past excess kurtosis 8.** It is no longer a quantile function.
- 🚨 **A normal quantile on a fat-tailed series.** -35 % on ES here.
- 🚨 **Passing Kupiec and declaring victory.** Count and timing are different failures, and at
  99 % the timing test has 1.4 % power against a nominal 5 %.
- 🚨 **sqrt(h) scaling.** Section 4.
- 🚨 **Minimising VaR.** Section 5.
- 🚨 **Sign and definition mismatches across libraries.** ✅ Source-verified in **quantstats
  0.0.81**: `qs.stats.value_at_risk(returns, sigma=1, confidence=0.95)` returns
  `norm.ppf(1 - confidence, mu, sigma * std)` - a **parametric normal** VaR as a **negative
  return**, not an empirical quantile and not a positive loss (measured: **-0.016236** on a
  seeded normal series). `conditional_value_at_risk` is the mean of `returns[returns < var]`,
  also negative (**-0.020824**), and it uses a strict `<`. Its `confidence` is a confidence
  level, not a tail probability, and a value above 1 is silently divided by 100. See
  `../../../fin-libraries/skills/lib-quantstats/SKILL.md`.
- ⚠️ **Two historical-ES conventions.** The mean of losses at or beyond the interpolated
  quantile, and the Rockafellar-Uryasev form with the atom at VaR split fractionally. ✅ On this
  data they agree to four decimals at n = 2,500, 500 and 100 and the VaR differs only in the
  fourth decimal - but two libraries reporting the two conventions will not tie out to the digit,
  so say which you used.
- ⚠️ The EVT threshold, the window length, the decay factor and the refit cadence are all
  trials.

## 7. Scripts

- `scripts/var_cvar.py` - all five sections. `var_es_historical`, `var_es_normal`,
  `var_es_cornish_fisher`, `cornish_fisher_is_monotone`, `var_es_evt` with `gpd_es_numeric` as
  its independent check, `kupiec_pof` with `kupiec_via_binomial`,
  `christoffersen_independence`, `conditional_coverage`, `backtest_size_under_null`,
  `simulate_garch_t`, `rolling_var` (seven methods including an oracle),
  `horizon_var_from_state`, `scale_var_sqrt_time` (the guard) and `discrete_var_es`. numpy and
  scipy only; no optional dependency. About 6 s.

## 8. Where this sits

`../covariance-and-risk-models/SKILL.md` supplies the covariance a parametric or portfolio VaR
needs, and owns the EWMA recursion used here; `../portfolio-optimizers/SKILL.md` minimises CVaR
as a linear program; `../../../fin-core/skills/portfolio-and-risk/SKILL.md` owns the Sharpe,
drawdown and annualization conventions and the wider risk-metric audit;
`../../../fin-core/skills/backtest-validation/SKILL.md` counts the estimator, threshold and
window choices as trials; `../../../fin-libraries/skills/lib-arch/SKILL.md` is the deep dive on
fitting the GARCH model whose conditional variance the good rows above depend on; and
`../../../fin-libraries/skills/lib-quantstats/SKILL.md` owns the reporting library whose sign
convention is the opposite of this skill's.
