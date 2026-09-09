---
name: covariance-and-risk-models
description: >-
  Estimate a covariance matrix an optimizer can actually invert, and report how much variance it
  hides. TRIGGER - covariance matrix estimation, sample covariance singular, "matrix is not
  positive definite", np.cov more assets than observations, N > T, condition number, Ledoit-Wolf
  shrinkage, sklearn LedoitWolf, CovarianceShrinkage, shrinkage intensity or delta, RiskMetrics
  EWMA, lambda 0.94 or 0.97, exponentially weighted covariance, exp_cov span, PCA or statistical
  factor risk model, Marchenko-Pastur, Barra fundamental factor model, specific risk, predicted
  vs realized volatility, risk model bias test; "my minimum-variance portfolio has 90x leverage",
  "the optimizer says 0% risk". SKIP for turning a covariance into weights and the optimizers
  themselves (portfolio-optimizers), for VaR, Expected Shortfall and their backtests
  (risk-measures-var-cvar), for GARCH and univariate volatility forecasting (volatility-models),
  and for Sharpe and drawdown conventions (portfolio-and-risk).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Covariance and risk models

**`np.cov(returns.T)` is an unbiased estimate of every entry and a bad estimate of the matrix.**
The optimizer does not consume entries; it consumes the inverse, and the inverse is where the
estimation error concentrates. Two failures follow, and both look like success in-sample: with
N > T the matrix is singular and the optimizer finds its null space, and even with N < T the
variance it predicts for its own portfolio is biased low.

Every number below is printed by `scripts/risk_model.py` (numpy / pandas, fixed seeds, about
10 s). The data is a k-factor DGP whose true covariance is known, so "predicted", "true" and
"out-of-sample realized" are three separate columns rather than two.

## 1. 🚨 N > T: the optimizer reports 0.00 % risk and delivers 243 %

✅ Measured: 100 assets from a 3-factor DGP, minimum-variance weights `Sigma^-1 1 / 1'Sigma^-1 1`,
volatilities annualized at sqrt(252), `true` from the DGP covariance:

| T | estimator | rank | cond | gross lev | max abs w | pred vol | true vol | true/pred var |
|---|---|---|---|---|---|---|---|---|
| **60** | 🚨 sample | **59** | 2.25e+20 | **92.5** | 2.92 | **0.00 %** | **243.44 %** | **-9.0e+15** |
| 60 | Ledoit-Wolf identity | 100 | 1.82e+02 | 3.2 | 0.09 | 4.92 % | 10.67 % | 4.69 |
| 250 | sample | 100 | 5.39e+02 | 3.5 | 0.16 | 5.13 % | 9.05 % | 3.11 |
| 250 | Ledoit-Wolf identity | 100 | 3.30e+02 | 3.2 | 0.15 | 5.59 % | 8.61 % | 2.38 |
| 1000 | sample | 100 | 2.19e+02 | 2.6 | 0.20 | 6.52 % | 7.57 % | 1.35 |
| 1000 | Ledoit-Wolf identity | 100 | 2.06e+02 | 2.6 | 0.20 | 6.58 % | 7.55 % | 1.32 |

The T = 60 row is not "optimistic", it is **meaningless**: the predicted variance is negative,
which is why the bias ratio is -9.0e+15 and the predicted volatility rounds to zero. A sample
covariance from T observations has rank at most T - 1, the minimum-variance solution puts its
weight in the null space, and the answer is gross leverage 92.5 with a single 292 % position.
Nothing in the output says "singular"; `np.linalg.solve` returns a vector.

**Shrinkage does not merely improve it, it changes the kind of object.** At T = 60 the same
sample gives a full-rank matrix with condition number 182 instead of 2.3e+20, leverage 3.2
instead of 92.5, and a true/predicted variance ratio of 4.69 instead of nonsense. It is still
badly wrong (4.69x), which is the honest reading: **shrinkage restores arithmetic, it does not
manufacture data.** T = 60 for 100 assets is not enough for a minimum-variance portfolio at any
estimator.

**The guard.** ✅ `check_invertible(Sigma, n_obs)` fires on the sample covariance and clears on
the shrunk one from the *same* sample:

```
check_invertible(sample, n_obs=60)      -> ValueError: covariance is rank 59 < 100: singular
   (a sample covariance from 60 observations has rank at most 59); shrink it or impose a
   factor structure before inverting
check_invertible(ledoit_wolf, n_obs=60) -> ok, rank 100, cond 1.82e+02, short_sample=True
```

🔑 The test is **rank and condition number, not N vs T**. A guard that refuses on `n_obs <= N`
alone also refuses the fix. `short_sample` is returned as a flag so a caller can insist the
matrix was regularized without blocking it.

## 2. The number to report: realized / predicted variance

An optimizer chooses the weights that best exploit whatever errors are in the matrix, so the
variance it predicts for *its own* portfolio is biased low. ✅ Measured: 50 assets, 250 in-sample
days, minimum-variance weights, held over the next 250 days; medians over 20 seeds, `[min, max]`
across seeds:

| estimator | pred vol | true vol | OOS vol | true/pred | OOS/pred |
|---|---|---|---|---|---|
| sample | 7.93 % | 9.91 % | 9.64 % | **1.53** [1.29, 1.99] | **1.53** [1.16, 1.91] |
| Ledoit-Wolf, identity target | 8.16 % | 9.79 % | 9.50 % | 1.36 [1.18, 1.77] | 1.38 [1.10, 1.66] |
| **Ledoit-Wolf, constant correlation** | 9.31 % | 9.80 % | 9.52 % | **1.08** [0.90, 1.19] | **1.06** [0.88, 1.33] |
| 🚨 EWMA, lambda = 0.94 | 3.47 % | 16.23 % | 16.70 % | **20.60** [12.33, 37.64] | 22.11 [12.93, 39.53] |
| PCA, 3 statistical factors | 8.42 % | 9.26 % | 9.05 % | 1.18 [1.10, 1.35] | 1.19 [1.04, 1.33] |
| Barra-style, exposures known | 8.69 % | 8.87 % | 8.75 % | **1.04** [0.97, 1.09] | 1.03 [0.85, 1.24] |

- At N/T = 1/5 with no singularity in sight, the **sample-covariance portfolio still realizes
  53 % more variance than it promised**, and the worst seed realizes 99 % more. This is the
  Michaud bias, and it is why an in-sample efficient frontier is not evidence.
- Shrinking to a **constant correlation** beat shrinking to a scaled identity here (1.08 against
  1.36), because the DGP has a market factor and the identity target denies it. The median
  intensities differ accordingly: **0.038** for the identity target, **0.226** for constant
  correlation. A shrinkage intensity near zero means the target was uninformative, not that the
  sample covariance was good.
- The **Barra-style row is the ceiling, and it is cheating**: it is handed the true exposures.
  1.04 is what a correct factor structure buys; the PCA row at 1.18 is what estimating that
  structure from returns costs.
- ⚠️ These ratios are DGP-specific. What transfers is the *procedure*: fit on one window, hold
  the weights, compare realized to predicted variance, and report that ratio next to the Sharpe.

## 3. RiskMetrics EWMA: the recursion, the decay factor, and what it is not for

✅ Source-verified by reading the **RiskMetrics Technical Document, Fourth Edition (December
1996)**, chapter 5:

- Sec. 5.3.2 and Table 5.9: **lambda = 0.94 for the daily data set, 0.97 for the monthly**.
  Chosen by minimising the RMSE of the *variance* forecast (Eq. [5.30]) for each of 480+ series
  and then taking a weighted average of the per-series optima, the weights being a measure of
  individual forecast accuracy (Eqs. [5.32]-[5.35]). **One lambda is applied to the entire
  covariance matrix** - per-element decay factors can be PSD but the document cites Crnkovic and
  Drachman (1995) that they are then subject to substantial bias.
- Eq. [5.23] fixes the recursion: `sigma2[t+1|t] = 0.94 * sigma2[t|t-1] + 0.06 * r[t]^2`.
- Sec. 5.3.1.1: RiskMetrics **assumes the mean of daily returns is zero** - standard deviations
  are centred on zero rather than the sample mean, and covariances take deviations around zero.
  `ewma_cov` therefore does not demean unless you pass `demean=True`.
- Eq. [5.26]: the effective number of days is `K = ln(tolerance) / ln(lambda)`. ✅ Reproduced:
  at the 1 % tolerance level K = **74.4** for lambda 0.94 and **151.2** for 0.97, against the
  74 and 151 printed in Table 5.7 of the document. Table 5.9 lists 550 daily returns used in
  production against an effective 75 (daily) and 150 (monthly).
- ✅ Measured: half-life `ln(0.5)/ln(lambda)` is **11.20** periods at 0.94 and **22.76** at 0.97.

🚨 **An EWMA covariance is a forecast, not a matrix to invert.** ✅ Measured at N = 50, T = 250:
the EWMA(0.94) matrix has condition number **2.03e+03** against **1.02e+02** for the equally
weighted sample covariance, because it carries the estimation error of a 74-day sample -
`T_eff/N = 1.5` against 5.0. That is the whole story of the EWMA row in section 2: predicted
3.47 % volatility, realized 16.70 %. Use it to *forecast* tomorrow's risk; use a longer lambda,
shrinkage or a factor structure for anything you invert.

🚨 **`adjust=True` is pandas' default and is not the RiskMetrics recursion.** ✅ Measured over
2,000 days: the recursion matches `ewm(alpha=0.06, adjust=False)` to **3.25e-19**, while
`adjust=True` differs by **86.7 %** at day 10, **4.44 %** at day 50, and 1.8e-15 by day 2,000.
The two agree eventually; they disagree exactly over the warm-up window a rolling backtest keeps
re-entering.

✅ Source-verified in **PyPortfolioOpt 1.6.0** (`pypfopt/risk_models.py`): `exp_cov` defaults to
`span=180`, which is lambda = **0.98895** and a half-life of **62.4** days - nothing like
RiskMetrics. It calls `covariation.ewm(span=span).mean()`, so pandas' `adjust=True` default
applies; it demeans each series with its **full-sample** mean (a look-ahead if you slide the
window); it fills the matrix with an N(N+1)/2 loop of pairwise pandas calls; and it multiplies
by `frequency` (252) before returning. See `../../../fin-libraries/skills/lib-pyportfolioopt/SKILL.md`.

## 4. PCA factors and how many are real

`n_factors_above_mp` counts correlation eigenvalues above the Marchenko-Pastur edge
`(1 + sqrt(N/T))^2`, the largest eigenvalue pure noise can produce. ✅ Measured on the 3-factor
DGP:

| N | T | MP edge | eigenvalues above it | top four eigenvalues |
|---|---|---|---|---|
| 100 | 1000 | 1.73 | **3** (DGP has 3) | 35.22, 3.54, 1.89, 1.22 |
| 50 | 250 | 2.09 | **2** (DGP has 3) | 15.97, 2.45, 1.50, 1.41 |

The count is a **lower bound and it shrinks as T shrinks**: the same DGP loses a factor when the
sample is short, because the edge rises with N/T. Use it to reject an over-parameterized
statistical model (10 "factors" from 250 days of 50 assets are mostly noise), not to prove a
factor is absent. ⚠️ The edge is computed with residual variance 1; with real factors present the
residual variance is below 1, so the true edge is lower and the test is conservative.

## 5. Cross-checks, and one import order that kills the interpreter

✅ Measured, each in a fresh interpreter:

- **scikit-learn 1.4.2** `sklearn.covariance.LedoitWolf`: max absolute covariance difference
  **1.08e-19**, shrinkage **0.039930** against this module's 0.039930 (difference 6.9e-18).
  ✅ Source-verified: it shrinks toward `mu * I` with `mu = trace(cov)/n_features`, subtracts the
  sample mean (`assume_centered=False`), and divides by `n_samples` - so it is the **biased**
  (ddof = 0) covariance, which is why `ledoit_wolf(X, "identity", ddof=0)` is what matches.
- **PyPortfolioOpt 1.6.0** `CovarianceShrinkage(...).ledoit_wolf("constant_correlation")`: max
  absolute difference **1.08e-19**, `delta` **0.189378** against 0.189378 (difference 0.0e+00),
  with `frequency=1` to defeat the annualization. ✅ Source-verified: its default target is
  `"constant_variance"` (which just calls sklearn's estimator), and `_format_and_annualize`
  **multiplies the result by `frequency`, default 252**, then runs
  `fix_nonpositive_semidefinite(fix_method="spectral")`. A pypfopt covariance is annualized and
  a sklearn one is not; comparing them without setting `frequency=1` is a factor-of-252 error.

🚨 **`import sklearn` followed by `import osqp` terminates the interpreter.** ✅ Measured on this
machine (Windows 11, Python 3.11.3, scikit-learn 1.4.2, osqp 1.1.3, cvxpy 1.9.2, numpy 2.2.6):

| order | return code |
|---|---|
| `import sklearn.covariance` then `import osqp` | **3221225477** (0xC0000005, access violation) |
| `import osqp` then `import sklearn.covariance` | **0** |

The minimal reproduction is those two imports and nothing else; no computation is involved. osqp
is a cvxpy dependency, so **PyPortfolioOpt, Riskfolio-Lib and skfolio all inherit it**. A
process crash cannot be caught by `try/except`, so a notebook that imports sklearn first and
PyPortfolioOpt second dies with no traceback and looks like a kernel bug. Either import the
cvxpy side first, or do what this script does and give each library its own process
(`probe()`). ⚠️ This is one machine's toolchain; check the order on yours before concluding it
is universal - the check is two imports.

## 6. Traps

- 🚨 **Inverting a covariance without looking at its rank.** Section 1.
- 🚨 **Reporting the in-sample frontier.** Report realized/predicted variance (section 2).
- 🚨 **An EWMA matrix in an optimizer.** 74 effective days is not a sample for 50 assets.
- 🚨 **`ewm(adjust=True)` for a RiskMetrics recursion.** 86.7 % off at day 10.
- 🚨 **Mixing annualized and per-period covariances.** pypfopt annualizes by default (x252),
  sklearn does not, and `np.cov` does not.
- 🚨 **Demeaning when RiskMetrics does not.** The document sets the mean to zero deliberately;
  and demeaning with a full-sample mean inside a rolling window is a look-ahead.
- 🚨 **A shrinkage intensity near 0 read as "the sample covariance was fine".** It means the
  target was uninformative - 0.038 for the identity target against 0.226 for constant
  correlation on the same data.
- 🚨 **`import sklearn` before `import cvxpy`/PyPortfolioOpt.** Section 5.
- ⚠️ Riskfolio-Lib and skfolio are **not installed in this environment** and nothing about them
  was verified here; see `../../../fin-libraries/skills/lib-riskfolio/SKILL.md` and
  `../../../fin-libraries/skills/lib-skfolio/SKILL.md`.

## 7. Scripts

- `scripts/risk_model.py` - all five sections. `sample_cov`, `ledoit_wolf` (identity and
  constant-correlation targets), `ewma_cov`, `ewma_variance_path`, `effective_days`,
  `half_life`, `span_to_lambda`, `pca_factor_cov`, `n_factors_above_mp`, `barra_cov`,
  `gmv_weights`, `bias_ratio`, `condition_report`, `check_invertible`, and `probe`, which runs
  one library check in a fresh interpreter. Without scikit-learn or PyPortfolioOpt the probes
  print that the library is missing and the script still exits 0. About 10 s.

## 8. Where this sits

`../portfolio-optimizers/SKILL.md` consumes the matrix this skill produces and owns the
constraints; `../risk-measures-var-cvar/SKILL.md` turns a variance into a VaR and backtests it;
`../factor-models/SKILL.md` builds the factor returns a fundamental risk model regresses on;
`../../../fin-core/skills/portfolio-and-risk/SKILL.md` owns the Sharpe, drawdown and annualization
conventions used above; `../../../fin-libraries/skills/lib-arch/SKILL.md` owns GARCH, the
conditional-variance alternative to EWMA; and
`../../../fin-libraries/skills/lib-skfolio/SKILL.md` and
`../../../fin-libraries/skills/lib-riskfolio/SKILL.md` are the per-library deep dives for the two
estimator zoos this skill does not have installed.
