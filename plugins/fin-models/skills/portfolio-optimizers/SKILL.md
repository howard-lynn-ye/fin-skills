---
name: portfolio-optimizers
description: >-
  Turn expected returns and a covariance matrix into weights, and measure what the optimizer did
  to your estimation error on the way. TRIGGER - portfolio optimization, mean-variance,
  Markowitz, efficient frontier, tangency or max-Sharpe portfolio, minimum-variance portfolio,
  long-only and budget constraints, scipy SLSQP or linprog for weights, Black-Litterman,
  BlackLittermanModel, tau, Omega, market-implied prior, views matrix P and Q, risk parity, equal
  risk contribution, ERC, inverse volatility, minimum CVaR, Rockafellar-Uryasev linear program,
  1/N benchmark, DeMiguel Garlappi Uppal, weight turnover; "my optimizer puts 90% in one asset",
  "the weights change completely every month". SKIP for choosing between optimizer libraries and
  reporting the result (portfolio-and-risk), for the covariance matrix and its N > T failure
  (covariance-and-risk-models), for expected returns (factor-models), for VaR and ES
  (risk-measures-var-cvar), and for HRP and PyPortfolioOpt's API traps (lib-pyportfolioopt).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Portfolio optimizers

**An optimizer returns the weights that are best for the inputs. Nobody has the inputs.** Two
measurements should travel with every weight vector, and a backtest shows neither: how much the
book turns over when the expected returns move by a basis point, and what the rule earns out of
sample against 1/N. This skill measures both, on a universe where 1/N is genuinely suboptimal so
the comparison is not rigged.

Every number below is printed by `scripts/optimizers.py` (numpy / scipy, seeds fixed, about 9 s;
PyPortfolioOpt optional and cross-checked when present).

## 1. 🚨 A one-basis-point change in mu turns over 7.5 % of the book

✅ Measured: 25 assets, 120 months of estimates, `mu` perturbed by +-1 bp per asset with random
signs, 20 draws. Turnover is `0.5 * sum |w_new - w_old|`. The sample covariance has condition
number 66.0; Ledoit-Wolf shrinkage (intensity 0.067) brings it to 44.6.

| solver | gross leverage | turnover per bp | max | sign flips |
|---|---|---|---|---|
| 🚨 tangency, sample Sigma | 5.87 | **7.54 %** | 10.30 % | 0.6 |
| 🚨 tangency, Ledoit-Wolf Sigma | 5.85 | **7.43 %** | 9.85 % | 0.0 |
| mean-variance long-only, gamma = 3, sample | 1.00 | **1.38 %** | 1.76 % | 0.0 |
| mean-variance long-only, gamma = 3, Ledoit-Wolf | 1.00 | 1.35 % | 1.71 % | 0.0 |
| **min-variance long-only (mu-free)** | 1.00 | **0.00 %** | 0.00 % | 0.0 |

✅ For scale: the standard error of each of these 120-month means is **44 to 91 bp a month**
(median 69). **The perturbation that moved 7.5 % of the book is 69 times smaller than the noise
already in `mu_hat`.** That is what "the optimizer is an error maximizer" means in numbers - not
that it is sensitive, but that it is sensitive far below the resolution of its own inputs.

- **Constraints do more than shrinkage here.** Long-only plus a budget cuts turnover from 7.5 %
  to 1.4 % and gross leverage from 5.9 to 1.0; shrinking the covariance alone moves turnover by
  0.11 pp. Shrink `mu`, not only `Sigma` - or constrain.
- **A mu-free rule has no mu sensitivity at all**, by construction. If you cannot defend your
  expected returns, that is not a rhetorical point, it is a solver choice.

**The guard.** `assert_stable_weights(mu, Sigma, solver, max_turnover=0.05)` raises:

```
ValueError: a 1 bp perturbation of mu turns over 7.5% of the book (limit 5%): the weights are
estimation error, not a view. Shrink mu and Sigma, add constraints, or use a mu-free rule
(min-variance, risk parity, 1/N).
```

## 2. 🚨 Out of sample, the optimizer gives back an edge it really had

✅ Measured: 25 assets from a one-factor DGP with per-asset alpha, 120-month rolling window, 360
test months, 6 seeds. **At the true moments the tangency portfolio's Sharpe is 0.708 against
1/N's 0.376 - a real 1.88x edge**, so this universe is not one where 1/N wins by construction.

| strategy | OOS Sharpe (mean of 6 seeds) | worst | best |
|---|---|---|---|
| oracle: tangency weights at the TRUE moments | **0.747** | 0.62 | 0.84 |
| **mean-variance long-only (sample)** | **0.387** | 0.23 | 0.51 |
| **1/N** | **0.382** | 0.29 | 0.52 |
| min-variance (Ledoit-Wolf) | 0.327 | 0.10 | 0.61 |
| min-variance (sample) | 0.292 | 0.04 | 0.60 |
| 🚨 mean-variance (sample, unconstrained) | **-0.028** | -0.25 | 0.17 |
| 🚨 mean-variance (Ledoit-Wolf, unconstrained) | -0.068 | -0.19 | 0.14 |

**Read the gap between the top and bottom rows.** The oracle earns 0.747. The same rule fed
120 months of its own data earns **-0.028** - it does not merely fail to capture the edge, it
delivers a negative Sharpe on a universe with a positive one. 1/N, which knows nothing, earns
0.382. Shrinking the covariance does not rescue it (-0.068), because the damage is in `mu`;
adding long-only constraints does (0.387, marginally the best rule here), which is the
Jagannathan-Ma effect - a constraint is an implicit shrinkage of the inputs it binds on.

✅ The estimation window it takes for the unconstrained sample rule to catch 1/N, same test
months, mean over seeds:

| window M (months) | 60 | 120 | 240 | 480 | **960** | 1920 |
|---|---|---|---|---|---|---|
| sample mean-variance | 0.03 | -0.03 | 0.09 | 0.36 | **0.45** | 0.60 |
| 1/N | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 |

**960 months - 80 years - is the first window at which it wins on this DGP.** ✅ DeMiguel,
Garlappi and Uppal's own abstract, read at the working-paper PDF (NBER SI 2006 draft of what
became *Optimal Versus Naive Diversification*, RFS 22(5), 1915-1953, 2009): for parameters
calibrated to U.S. stock market data, a 25-asset portfolio needs an estimation window of **more
than 3,000 months**, and a 50-asset portfolio **more than 6,000**, "although in practice these
parameters are estimated using 120 months of data". They also report that across fourteen
optimal-portfolio models and seven datasets, **none** is consistently better than 1/N on Sharpe
ratio, certainty-equivalent return or turnover.

🔑 **1/N is the benchmark, not the punchline.** Report your rule's out-of-sample Sharpe next to
1/N's on the same window. A rule that cannot beat it has not earned its complexity.

## 3. Black-Litterman: what the formula does and what the library defaults are

`black_litterman()` implements the posterior in two algebraically equivalent forms and prints
their gap, so an implementation error cannot hide:

```
mu_bl = pi + tau*S P' (P tau*S P' + Omega)^-1 (Q - P pi)                     (solve form)
      = [(tau*S)^-1 + P' Omega^-1 P]^-1 [(tau*S)^-1 pi + P' Omega^-1 Q]      (precision form)
Sigma_bl = Sigma + M,   M = [(tau*S)^-1 + P' Omega^-1 P]^-1
```

✅ Measured: the two mean forms agree to **2.08e-17** and the two covariance forms to
**3.47e-18**.

✅ Reproduced against **PyPortfolioOpt 1.6.0** `BlackLittermanModel` on a 5-asset example
(delta 2.5, tau 0.05, two views): prior **4.34e-19**, posterior mean **4.34e-19**, `bl_cov()`
**8.47e-21**, `omega` **2.17e-19**.

| asset | w_mkt | prior pi | posterior | change | prior vol | post vol |
|---|---|---|---|---|---|---|
| A bonds | 40.0 % | 0.31 % | 0.33 % | +0.01 % | 5.00 % | 5.12 % |
| B US equity | 30.0 % | 2.83 % | 4.06 % | +1.23 % | 15.00 % | 15.26 % |
| C intl equity | 15.0 % | 3.01 % | 5.06 % | +2.05 % | 17.00 % | 17.21 % |
| D EM equity | 5.0 % | 3.23 % | 5.44 % | +2.20 % | 22.00 % | 22.30 % |
| E commodities | 10.0 % | 1.11 % | 1.61 % | +0.50 % | 14.00 % | 14.33 % |

The views were "C returns 7 %" and "D beats B by 2 %". **A and E were never named and both
moved**, through the covariance - that is the point of the model, and it is also why a single
careless view rearranges a book. Note the posterior volatility is *higher* than the prior on
every asset: `Sigma_bl = Sigma + M` adds parameter uncertainty, it does not subtract it.

✅ Source-verified in `pypfopt/black_litterman.py` (1.6.0):

- `BlackLittermanModel(cov_matrix, pi=None, absolute_views=None, Q=None, P=None, omega=None,
  view_confidences=None, tau=0.05, risk_aversion=1)`. **`tau` defaults to 0.05 and
  `risk_aversion` to 1** - the latter is not the 2.0-3.0 a reverse optimization usually assumes,
  and it scales the whole implied prior.
- `default_omega(cov, P, tau)` returns `np.diag(np.diag(tau * P @ cov @ P.T))`, the He and
  Litterman (1999) form. Same as this module's default.
- 🚨 **`pi=None` is a warning, not an error**: `_set_pi` does
  `warnings.warn("Running Black-Litterman with no prior.")` and sets `self.pi` to **zeros**. A
  zero prior is an opinion (every asset has zero expected excess return), not equilibrium, and
  the warning is easy to miss in a notebook. `black_litterman()` here **raises** instead unless
  you supply `pi` or `w_mkt`.
- `pi="market"` needs a `market_caps` kwarg; `pi="equal"` sets 1/N; `omega="idzorek"` needs
  `view_confidences`.

⚠️ `tau` has no agreed value in the literature and the model is sensitive to it; treat it as a
trial, not a constant.

## 4. Risk parity and minimum CVaR

**Risk parity** solves for `w_i (Sigma w)_i = b_i` by Newton's method on the convex form
`0.5 w'Sigma w - sum b_i log w_i`. ✅ Measured on the same 25-asset sample covariance:

| rule | risk-share min | risk-share max | weight min | weight max | annual vol |
|---|---|---|---|---|---|
| **risk parity** | 0.0400 | 0.0400 | 0.025 | 0.073 | **14.03 %** |
| 1/N | 0.0169 | 0.0589 | 0.040 | 0.040 | 15.81 % |
| inverse volatility | 0.0158 | 0.0549 | 0.030 | 0.062 | 15.17 % |

Equal *weights* are not equal *risk*: 1/N's risk shares span 1.7 % to 5.9 %, a 3.5x range, and
inverse-vol - the usual shortcut - barely narrows it because it ignores correlations. ✅ Risk
parity's shares are equal to **9.6e-15**. It is mu-free, so section 1's sensitivity is zero for
it too.

**Minimum CVaR** is a linear program (Rockafellar and Uryasev), not a quadratic one:

```
min over w, alpha, z:  alpha + 1/((1-level) T) * sum_t z_t
        s.t.  z_t >= -r_t'w - alpha,  z_t >= 0,  1'w = budget,  w >= 0
```

✅ Measured on 10 assets x 500 months at the 95 % level with `scipy.optimize.linprog(method=
"highs")`: the LP objective is **0.071354** and the sample CVaR of the LP's own weights is
**0.071354** (difference 2.8e-17), so the objective really is the CVaR; `alpha` at the optimum is
the VaR, **0.055653**; 121 HiGHS iterations. The long-only minimum-*variance* portfolio on the
same sample has CVaR 0.072853, **2.1 % worse** - a small gap on this well-behaved data, and the
right comparison to make before adopting the heavier objective.

**HRP is deliberately not reimplemented here.**
`../../../fin-libraries/skills/lib-pyportfolioopt/SKILL.md` owns it, and the
`weight_traps.py` script in that skill measures the prices-versus-returns trap that silently
changes HRP's answer.

## 5. Traps

- 🚨 **Weights without a stability number.** Report turnover per bp of mu (section 1).
- 🚨 **Weights without the 1/N comparison.** Section 2.
- 🚨 **Shrinking Sigma and leaving mu alone.** Worth 0.11 pp of turnover here and *nothing* out
  of sample (-0.028 to -0.068).
- 🚨 **`pi=None` in PyPortfolioOpt's Black-Litterman.** A warning and a zero prior.
- 🚨 **`risk_aversion=1` as a default.** It scales the entire implied prior.
- 🚨 **Reading an unconstrained tangency weight vector as a portfolio.** Gross leverage 5.87 on
  25 assets. And ✅ when `1'Sigma^-1 mu` is negative, dividing by the *signed* sum flips every
  weight and silently turns the portfolio inside out; `tangency()` divides by `abs(...)`
  instead, so the direction is kept and the budget comes back as **-1** where you can see it.
  Check `w.sum()`, not just the weights.
- ⚠️ **SLSQP's resolution is set by the objective's scale.** ✅ Measured: with monthly returns
  the objective is of order 1e-4, and at `ftol=1e-12` the constrained solver lands within
  **2e-4** of the closed-form solution and no closer. That is fine for weights and not fine if
  you difference two solutions - which is exactly what a turnover measurement does, so measure
  turnover with a perturbation well above the solver's own noise.
- 🚨 **An optimizer fed a singular covariance.** It will answer.
  `../covariance-and-risk-models/SKILL.md` section 1 has what that answer looks like.
- ⚠️ **The DGP matters for section 2's size, not its sign.** With no cross-sectional alpha the
  true tangency Sharpe here is within 1.3 % of 1/N's and the comparison is vacuous; the demo
  uses a universe where the true edge is 1.88x and prints both population Sharpes so the setup
  is checked rather than asserted.
- ⚠️ Risk parity, minimum CVaR, the CVaR level, `tau`, the risk aversion and the constraint set
  are all trials -
  `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py` is where they go.
- ⚠️ Riskfolio-Lib and skfolio - the two libraries with the largest menus of objectives - are
  **not installed in this environment** and nothing about them was verified here. See
  `../../../fin-libraries/skills/lib-riskfolio/SKILL.md` and
  `../../../fin-libraries/skills/lib-skfolio/SKILL.md`.

## 6. Scripts

- `scripts/optimizers.py` - all five sections. `mean_variance` (SLSQP with a budget and
  long-only or box bounds), `mean_variance_closed_form`, `tangency`, `min_variance`,
  `implied_returns`, `black_litterman` with `pypfopt_bl_check`, `risk_parity` and
  `risk_contributions`, `min_cvar_lp` with `sample_cvar` as its independent check,
  `perturbation_turnover` and `assert_stable_weights` (the guard), `simulate_universe`,
  `population_sharpe`, `max_population_sharpe` and `oos_backtest`. Without PyPortfolioOpt the
  cross-check prints that it was skipped and the script still exits 0. About 9 s.

## 7. Where this sits

`../covariance-and-risk-models/SKILL.md` produces the covariance and owns the N > T failure;
`../factor-models/SKILL.md` produces the expected returns and owns their look-ahead;
`../risk-measures-var-cvar/SKILL.md` owns the CVaR definition minimised in section 4 and the
reason ES rather than VaR is the objective;
`../../../fin-core/skills/portfolio-and-risk/SKILL.md` owns the Sharpe and turnover conventions
and routes between optimizer libraries;
`../../../fin-core/skills/backtest-validation/SKILL.md` counts the specifications; and
`../../../fin-libraries/skills/lib-pyportfolioopt/SKILL.md`,
`../../../fin-libraries/skills/lib-riskfolio/SKILL.md` and
`../../../fin-libraries/skills/lib-skfolio/SKILL.md` are the per-library deep dives, including
hierarchical risk parity.
