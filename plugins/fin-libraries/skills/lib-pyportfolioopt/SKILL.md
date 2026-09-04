---
name: lib-pyportfolioopt
description: >-
  Textbook mean-variance and Black-Litterman optimizer whose HRPOpt silently accepts a price
  matrix where it requires returns and returns plausible garbage. TRIGGER - pypfopt,
  PyPortfolioOpt, EfficientFrontier, HRPOpt, CovarianceShrinkage, DiscreteAllocation,
  BlackLittermanModel, EfficientCVaR, EfficientSemivariance, CLA, mean_historical_return,
  capm_return, clean_weights, max_sharpe, min_volatility, portfolio_performance,
  risk_models.risk_matrix, "efficient frontier", "whole-share allocation". Memory is stale - the
  repo moved to the PyPortfolio org and 1.6.0 shipped 2026-02-26 after three dormant years under a
  new maintainer. SKIP for Marcenko-Pastur denoising, HERC or NCO (lib-riskfolio) and for
  GridSearchCV over portfolio models (lib-skfolio). SKIP for choosing between libraries - that is
  the domain skill's job.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# PyPortfolioOpt

Textbook mean-variance, Black-Litterman and efficient frontiers with the best prose docs in the category — the right
teaching and prototyping choice, and the wrong one the moment you need denoising, HERC or NCO.

| | |
|---|---|
| pip / import | `PyPortfolioOpt` (resolves as `pyportfolioopt`) / **`pypfopt`** |
| Version | **1.6.0** (2026-02-26) |
| Licence | MIT. `requires_python` is **unset on PyPI** — the package declares no floor at all |
| Status | ⚠️ **revived, cautiously healthy** — repo now `github.com/PyPortfolio/PyPortfolioOpt`, 6,007★ / **113 open issues**, pushed 2026-07-07 |

The old `robertmartin8/PyPortfolioOpt` URL **redirects** to a `PyPortfolio` GitHub org. The project went nearly
dormant — 1.5.5 (2023-05), 1.5.6 (2024-12), no feature release for ~3 years — then shipped 1.6.0 under new
stewardship. Robert Martin remains the PyPI author of record; the project's own `docs/Roadmap.rst` names
**Tuan Tran as primary maintainer** (issue #587). 113 open issues is a real backlog — treat responsiveness as
moderate, not high.

## The trap that costs you money

🚨 **`HRPOpt(returns=...)` takes a RETURNS matrix.** Pass prices and it runs — no exception, no warning — producing a
garbage correlation tree and plausible-looking weights you cannot tell from correct ones. This is the single most
reported PyPortfolioOpt error. Assert your input contains negative values before you hand it over.

The type check is real but **shallow**: `HRPOpt(returns=<numpy array>)` raises `TypeError: returns are not a
dataframe`, so it validates the *container* and never the *content*. A prices DataFrame is a DataFrame, so it sails
through. ✅ Measured with `scripts/weight_traps.py` (seed 0, 3-year 6-asset panel): feeding prices moves **85.18%** of
the book into the single **highest-volatility** name, an **L1 weight distance of 1.5648** out of a possible 2.0, and
**+141.4%** realised annualised vol (0.3486 vs 0.1444) — worse even than 1/N at 0.1976. The mechanism is units:
`cov(prices)` is in dollars-squared, so HRP's inverse-variance split collapses into **inverse-share-price weighting**
(the $8 ticker shows 3,654x less "variance" than the $620 one). The output is still long-only and still sums to 1.0000.

Its `linkage_method` default is **`'single'`** (the AFML original) vs skfolio's **Ward** — the two libraries give
different HRP weights *by design*. State the linkage explicitly when reproducing.

## `CovarianceShrinkage` is Ledoit-Wolf — it is NOT denoising

🚨 The two get conflated constantly and they fix different parts of the estimation-error problem: shrinkage pulls the
whole matrix toward a structured target; Marcenko-Pastur denoising replaces only the eigenvalues inside the
random-matrix bulk. A source grep confirms **no `denoise`, `detone` or Marcenko-Pastur anywhere in `risk_models`**. If
the task says "denoise the covariance", this is the wrong library — go to `lib-riskfolio` (`denoiseCov`) or
`lib-skfolio` (`DenoiseCovariance`).

`risk_models.risk_matrix(method=...)` accepts exactly: `sample_cov`, `semicovariance`, `semivariance`, `exp_cov`,
`ledoit_wolf`, `ledoit_wolf_constant_variance`, `ledoit_wolf_single_factor`, `ledoit_wolf_constant_correlation`,
`oracle_approximating`.

## `EfficientFrontier` objects are single-use

After `ef.max_sharpe()` you must construct a **new** `EfficientFrontier` before calling `ef.min_volatility()` —
reusing one raises or returns stale state. This bites everyone sweeping objectives in a loop. ⚠️ `max_sharpe()` also
does not compose naively with `weight_bounds` plus added objectives: the max-Sharpe transformation homogenizes the
problem, so bound semantics and L1/L2 penalties behave unintuitively. For bounded max-Sharpe with regularization use
`max_quadratic_utility` or `efficient_risk`, or move to Riskfolio/skfolio.

## Units: `frequency` and an ANNUAL risk-free rate

- `expected_returns.mean_historical_return` defaults to `compounding=True`, **`frequency=252`**.
  Weekly, monthly or crypto (365) data without a matching `frequency` silently mis-annualizes, and
  the error lands straight in the weights.
- `risk_free_rate` must be **ANNUAL**, on the same scale as `mu` and `S` (which are already
  annualized). empyrical's identically-purposed `risk_free` is **per-period** — mixing the two
  conventions across one pipeline is the most common real-world Sharpe bug.

## What it uniquely has

The **only** library here offering Ledoit-Wolf against the *constant-variance*,
*constant-correlation* and *single-factor* targets — the others stop at scaled identity. And
**`DiscreteAllocation`**, which converts a weight vector into whole share counts under a cash budget
(greedy or LP): genuinely useful, rarely reimplemented. ⚠️ No HERC, no NCO, no risk budgeting, no Schur; `HRPOpt` is
plain HRP only.

## Minimal correct call

```python
from pypfopt import EfficientFrontier, risk_models, expected_returns, HRPOpt, DiscreteAllocation

# prices: DataFrame, DatetimeIndex. frequency MUST match your bar spacing.
mu = expected_returns.mean_historical_return(prices, frequency=252, compounding=True)
S  = risk_models.CovarianceShrinkage(prices, frequency=252).ledoit_wolf()  # Ledoit-Wolf, NOT denoising

ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.10))
ef.add_objective(lambda w: 0.001 * (w @ w))       # L2 diversification penalty
ef.max_sharpe(risk_free_rate=0.02)                # 🚨 ANNUAL, same scale as mu/S
w = ef.clean_weights()
ef.portfolio_performance(risk_free_rate=0.02, verbose=True)

ef2 = EfficientFrontier(mu, S, weight_bounds=(0, 0.10))   # 🚨 NEW object — ef is spent
w_minvol = ef2.min_volatility()

assert (returns < 0).any().any(), "HRPOpt needs RETURNS, not prices"
w_hrp = HRPOpt(returns).optimize(linkage_method="single")  # state it; skfolio defaults to Ward

alloc, cash = DiscreteAllocation(w, prices.iloc[-1],
                                 total_portfolio_value=100_000).lp_portfolio()
```

⚠️ Before optimizing on `mu` at all: mean-variance is an **error maximizer**, loading onto whatever asset has the most
overstated return and most understated variance. If you cannot defend your expected-return estimates, use
`min_volatility()` or HRP — neither needs a forecast — and always benchmark against **1/N after costs**.

## Scripts

- `scripts/weight_traps.py` — the HRPOpt prices-vs-returns trap, end to end. Reproduces HRP from the AFML definition
  in numpy/pandas/scipy, then verifies that reference against the installed `HRPOpt` (**exact match, max
  |reference − HRPOpt| = 0.00e+00** on both the correct and the trapped input). Prints both weight vectors side by
  side, the dollar-variance table that explains the collapse, the two different quasi-diagonal cluster orders, the
  realised-vol penalty, and that `assert (returns < 0).any().any()` is the only thing between you and the bad number.
  Runs without PyPortfolioOpt installed.

Verified on 1.6.0: `EfficientFrontier` reuse raises `InstantiationError: Adding constraints to an already solved
problem might have unintended consequences.` — it **raises cleanly, it does not return stale state**.

## See also

- `../../../fin-core/skills/portfolio-and-risk/SKILL.md` — optimizer choice, and the metric traps downstream
- `../../../fin-core/skills/portfolio-and-risk/references/pyportfolioopt.md` — the source card
- `../../../fin-core/skills/portfolio-and-risk/references/optimizers.md` — head-to-head table and the estimation-error problem
- `../../../fin-core/skills/portfolio-and-risk/references/_solver-layer.md` — cvxpy, solver classes and licences

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`portfolio-and-risk`** (`../../../fin-core/skills/portfolio-and-risk/SKILL.md`).
