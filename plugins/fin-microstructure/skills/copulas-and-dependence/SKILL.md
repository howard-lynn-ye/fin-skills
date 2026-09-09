---
name: copulas-and-dependence
description: >-
  Separate the marginals from the dependence - Gaussian, Student t, Clayton and Gumbel
  copulas, Kendall's tau, tail dependence coefficients, and what fitting the wrong family
  costs in the joint tail. TRIGGER - copula, Gaussian copula, t copula, Student t copula,
  Clayton copula, Gumbel copula, Archimedean copula, Sklar's theorem; tail dependence, lower
  tail dependence, upper tail dependence, lambda_U, "correlation is not dependence",
  "correlations go to one in a crisis", joint tail probability, joint exceedance; Kendall's
  tau, Spearman rho, rank correlation, pseudo-observations, inversion of Kendall's tau, copula
  MLE, copulas python, copulae, statsmodels copula; diversification benefit, "my VaR says the
  portfolio is safe". SKIP for estimating a covariance matrix and shrinkage
  (covariance-and-risk-models), for VaR/CVaR methods and their backtests
  (risk-measures-var-cvar), and for GARCH marginals (volatility-models).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Copulas and dependence

**Correlation pins down the middle of a joint distribution and says nothing at all about the
corner.** Four copulas can agree on Kendall's tau, on Spearman's rho, on the linear correlation
and on both marginals, and disagree by a factor of five about how often two assets break their
1% quantile on the same day.

Every number marked ✅ Measured is printed by `scripts/copulas.py` (numpy 2.2.6 + scipy 1.13.0,
seed 20260909, **32 s**). ✅ source-verified means read in the installed scipy source.

## 1. The closed forms

| family | Kendall's tau | lower tail dep. | upper tail dep. |
|---|---|---|---|
| Gaussian(ρ) | `(2/π)·arcsin(ρ)` | **0** for every ρ < 1 | **0** |
| t(ρ, ν) | `(2/π)·arcsin(ρ)` | `2·t_{ν+1}(−sqrt((ν+1)(1−ρ)/(1+ρ)))` | same (radially symmetric) |
| Clayton(θ) | `θ/(θ+2)` | `2^(−1/θ)` | 0 |
| Gumbel(θ) | `1 − 1/θ` | 0 | `2 − 2^(1/θ)` |

🔑 **The tau column is why this skill exists**: `tau` for the t copula does not contain `ν`, so
a t copula and a Gaussian copula at the same `ρ` have *identical* Kendall's tau, *identical*
Spearman's rho and *identical* linear correlation. Every dependence summary a spreadsheet
computes is blind to the difference.

At **ρ = 0.5**, every family below is set to the same **τ = 1/3**: Gaussian ρ=0.5, t ρ=0.5 ν=4,
Clayton θ=1, Gumbel θ=1.5.

✅ Measured — the samplers, against those taus, on 200,000 draws (the Gumbel sampler goes
through a positive-stable variable and Clayton through a gamma; an error in either shows up
here in the third decimal and nowhere else):

| family | τ closed form | τ empirical | diff |
|---|---|---|---|
| gaussian | 0.333333 | 0.333465 | +0.00013 |
| t | 0.333333 | 0.333796 | +0.00046 |
| clayton | 0.333333 | 0.334431 | +0.00110 |
| gumbel | 0.333333 | 0.332777 | −0.00056 |

✅ Measured — each analytic log-density against a central finite difference of its own CDF, at
(0.3, 0.7) and (0.9, 0.85): worst relative difference **6.0e-08** across all four families.

## 2. ✅ Tail dependence is a limit — so evaluate it, do not estimate it

`lambda_L = lim_{u→0} C(u,u)/u`. **A sample cannot verify this**, because the whole difficulty
is that there are no observations at `u = 1e-6`. So `scripts/copulas.py` evaluates `C(u,u)`
*exactly* — a deterministic bivariate normal CDF by Plackett's identity, and the bivariate t on
top of it as a chi-square mixture — and marches `u` down:

| family | closed form | u=1e-2 | u=1e-3 | u=1e-4 | u=1e-5 | u=1e-6 |
|---|---|---|---|---|---|---|
| gaussian | **0.000000** | 0.129392 | 0.054259 | 0.023311 | 0.010164 | 0.004476 |
| t (ν=4) | **0.253170** | 0.287678 | 0.263493 | 0.256379 | 0.254179 | **0.253489** |
| clayton | **0.500000** | 0.502513 | 0.500250 | 0.500025 | 0.500003 | **0.500000** |
| gumbel (upper) | **0.412599** | 0.417268 | 0.413065 | 0.412646 | 0.412604 | **0.412599** |

🚨 **Read the Gaussian row.** "Zero tail dependence" does not mean "no joint tail mass". At
`u = 1%` — a quantile you can actually estimate — the Gaussian copula still puts **0.129** of
the mass of one tail into the joint tail. It only reaches 0.004 at `u = 1e-6`. **The limit is
zero; the decay is slow, and it is slow exactly over the range where your data live.** The
Gaussian copula is not "obviously" wrong on a scatterplot. It is wrong in the corner, by an
amount that grows as you go further in.

🔑 **Same τ on every row. Tail dependence 0.000 / 0.253 / 0.500 / 0.413.** The family is a
choice you make, not a parameter the data give you.

## 3. 🚨 `scipy.stats.multivariate_t.cdf` is randomised

Found while building §2. ✅ source-verified in the installed scipy 1.13.0 source — the signature
is `cdf(x, loc, shape, df, allow_singular, *, maxpts=None, lower_limit=None, random_state=None)`
and its own docstring says *"maxpts: Maximum number of points to use for integration. The
default is 1000 times the number of dimensions"*. It is a randomised quadrature.

✅ Measured — four calls, one argument, at `u = 1e-4` where the exact value is **2.563785e-05**:

| | call 1 | call 2 | call 3 | call 4 | spread |
|---|---|---|---|---|---|
| `multivariate_t.cdf` | 2.2648e-05 | **1.4657e-04** | 2.8716e-05 | 1.3373e-05 | **11.0×** |
| `multivariate_normal.cdf` | 2.3311e-06 | 2.3311e-06 | 2.3311e-06 | 2.3311e-06 | 1.00× |

The bivariate normal returns the identical value every time and matches the Plackett quadrature
to **1.5e-15 relative**. The t does not, and the spread is different on every run (6.7× and
11.0× on two runs of this demo). 🚨 **Deep in the tail, where you actually need it, the default
`maxpts` is nowhere near enough.** Pass `random_state`, raise `maxpts`, or use a deterministic
rule — but do not put its output in a table and call the table reproducible.

## 4. ✅ Fitting: inversion of Kendall's tau, and MLE

✅ Measured — 15 seeded samples of 4,000 per family:

| family | true | τ-inversion mean | sd | MLE mean | sd |
|---|---|---|---|---|---|
| gaussian | 0.5000 | 0.5025 | 0.0117 | 0.5021 | **0.0099** |
| t | 0.5000 | 0.5031 | 0.0130 | 0.5036 | 0.0110 |
| clayton | 1.0000 | 1.0079 | 0.0450 | 1.0092 | **0.0298** |
| gumbel | 1.5000 | 1.5072 | 0.0195 | 1.5057 | 0.0170 |

Both are consistent. **Inversion of tau is a closed form for all four families** — it costs
nothing, cannot fail to converge, and is invariant to the marginals. MLE is 10–35% sharper and
needs the family to be right.

🚨 **Neither estimator can tell you which family.** They estimate a parameter *given* a family.
That is the entire cost, and §5 is the size of it.

## 5. 🚨 The trap: a Gaussian copula fitted to tail-dependent data

✅ Measured — 400,000 draws from a t copula (ρ=0.5, ν=4); a Gaussian copula fitted to them by
Kendall's tau returns **ρ̂ = 0.4989**. Identical correlation, identical rank correlation,
identical marginals. `P[both below the q quantile]`:

| q | empirical | ± | t copula (exact) | **Gaussian (exact)** | emp / Gaussian | independent |
|---|---|---|---|---|---|---|
| 5% | 0.016645 | 0.000202 | 0.016937 | 0.012156 | **1.37×** | 0.002500 |
| 1% | 0.002860 | 0.000084 | 0.002877 | 0.001288 | **2.22×** | 0.000100 |
| 0.5% | 0.001385 | 0.000059 | 0.001385 | 0.000494 | **2.80×** | 0.000025 |
| **0.1%** | 0.000253 | 0.000025 | 0.000263 | **0.000054** | **4.68×** | 0.000001 |

🚨 **In ten years of daily data (2,520 days), "both assets below their 1% quantile" happens 7.2
times under the truth and 3.2 times under the fitted Gaussian copula.** Both models agree the
correlation is 0.5. The error is not in the correlation and no correlation-based diagnostic
will find it.

🔑 The `emp / Gaussian` column **grows monotonically as you go further into the tail** — 1.37×,
2.22×, 2.80×, 4.68×. Backtesting the fit at 5% would look almost acceptable. That is the shape
of every zero-tail-dependence misspecification: harmless where you have data, unbounded where
you do not.

## 6. 🚨 What it does to VaR and to the diversification number

✅ Measured — equally weighted, **standard normal marginals under both copulas**, equicorrelation
ρ = 0.5, 250,000 draws. Under the Gaussian copula the portfolio is exactly normal, so the
comparison has an exact leg. Only the copula differs:

| d | level | VaR Gauss | exact | VaR t-cop | ratio | ES Gauss | ES t-cop | ratio | div. benefit G / t |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 99.0% | 2.0056 | 2.0147 | 2.0764 | 1.035 | 2.3007 | 2.4262 | 1.055 | 13.8% / 10.7% |
| 2 | 99.9% | 2.6750 | 2.6762 | 2.8732 | **1.074** | 2.8969 | 3.1417 | 1.085 | 13.4% / 7.0% |
| 10 | 99.9% | 2.2849 | 2.2918 | 2.6033 | **1.139** | 2.4785 | 2.9235 | 1.180 | 26.1% / 15.8% |
| **50** | 99.9% | 2.1893 | 2.2069 | 2.4880 | **1.136** | 2.3796 | 2.7732 | **1.165** | **29.2% / 19.5%** |

- 🔑 **Read the ratio column down, not across.** At `d = 2` the copula moves the 99.9% VaR by
  **+7.4%** — a two-asset portfolio's tail is mostly its *marginals'* tails. At `d = 50` it
  moves it by **+13.6%** and the expected shortfall by **+16.5%**, because **diversification is
  exactly what tail dependence destroys**: the Gaussian copula lets 50 assets average their bad
  days away; the t copula makes them share one.
- 🚨 **The "diversification benefit" column is the one that goes in a memo.** At `d = 50` and
  99.9% the Gaussian copula claims **29.2%** and the t copula, *same correlation, same
  marginals*, delivers **19.5%** — the benefit is overstated by nearly a half.
- ⚠️ These marginals are normal by construction, so the whole difference is dependence. On real
  data the marginals are fat-tailed too and the two effects compound;
  `../../../fin-models/skills/risk-measures-var-cvar/SKILL.md` owns the marginal side.

## 7. What the script gives you

`scripts/copulas.py` — numpy + scipy only, importable, no file writes, no network:

| Function | Does |
|---|---|
| `kendall_tau(family, param)` / `tau_to_param(family, tau)` | the τ column of §1, both ways |
| `tail_dependence(family, param, df)` | `(lambda_L, lambda_U)` — exact zeros where there are none |
| `bvn_cdf(a, b, rho)` | deterministic bivariate normal CDF, Plackett's identity |
| `bvt_cdf(a, b, rho, df)` | deterministic bivariate t CDF, as a chi-square mixture |
| `copula_cdf(family, u, v, ...)` / `copula_logpdf(...)` | `C` and `log c` for all four |
| `tail_ratio(family, u, ...)` | the finite-`u` ratios of §2 |
| `sample(family, n, ...)` | Gaussian, t, Clayton (gamma) and Gumbel (positive stable) |
| `sample_elliptical(d, n, rho, df)` | the d-dimensional equicorrelation version — §6 |
| `fit_by_tau` / `fit_mle` | §4 |
| `joint_tail_study(...)` | §5 |
| `portfolio_tail(..., d)` | §6, with the exact Gaussian leg |
| `scipy_mvt_reproducibility()` | §3 |

⚠️ **What is not here**: vine copulas and anything above two dimensions for the Archimedean
families (the d-dimensional code is elliptical only), time-varying and regime-switching
copulas, empirical/non-parametric copulas, and goodness-of-fit tests for the family choice
(Cramér-von Mises on the empirical copula is the usual route, and it is weak in exactly the
region §5 is about).

## Where this sits

- `../../../fin-models/skills/risk-measures-var-cvar/SKILL.md` — VaR and ES estimation and
  their backtests. §6's numbers are the dependence half of that story; the marginal half is
  there.
- `../../../fin-models/skills/covariance-and-risk-models/SKILL.md` — estimating the correlation
  matrix in the first place, and shrinkage. A copula takes ρ as given; that skill is where ρ
  comes from, and its estimation error is on top of everything here.
- `../../../fin-models/skills/volatility-models/SKILL.md` — the standard pipeline is GARCH
  marginals plus a copula. That skill owns the marginals.
- `../monte-carlo-methods/SKILL.md` — §5 and §6 are simulated, so their `±` columns are
  Monte Carlo standard errors and inherit everything that skill says about them.
- `../hawkes-processes/SKILL.md` — dependence across *time* rather than across assets, and the
  same lesson: a summary statistic that matches is not a model that matches.
