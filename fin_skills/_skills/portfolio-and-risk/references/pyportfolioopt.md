# PyPortfolioOpt

The friendliest mean-variance / Black-Litterman library and the best prose docs in the space — the
right teaching and prototyping choice, and the wrong choice the moment you need denoising, HERC or NCO.

| Field | Value |
|---|---|
| pip / import | `PyPortfolioOpt` (resolves as `pyportfolioopt`) / **`pypfopt`** |
| version | **1.6.0** (2026-02-26) ✅ |
| repo / docs | **`github.com/PyPortfolio/PyPortfolioOpt`** ✅ · <https://pyportfolioopt.readthedocs.io/> |
| stars / open issues | **6,007 / 113** ✅ |
| licence | **MIT** ✅ |
| Python | ⚠️ **`requires_python` is unset on PyPI** ✅ — the package declares no floor at all |
| verdict | ⚠️ **revived, cautiously healthy** — pushed 2026-07-07; 113 open issues is a real backlog |

Verified 2026-09-04 via the PyPI JSON API and the GitHub REST API.

🔑 **The repo moved.** `robertmartin8/PyPortfolioOpt` now **redirects** to the `PyPortfolio` GitHub
*org*. The project went nearly dormant — 1.5.5 (2023-05), 1.5.6 (2024-12), no feature release for ~3
years — then shipped **1.6.0 on 2026-02-26** under new stewardship. Robert Martin remains the PyPI
author of record; per the project's own `docs/Roadmap.rst`, **Tuan Tran is now the primary maintainer**
(handover context: GitHub issue #587) ⚠️. Treat responsiveness as moderate, not high.

## 🚨 Traps

🚨 **`HRPOpt(returns=...)` takes a RETURNS matrix.** Passing prices runs silently and produces garbage
weights — no exception, no warning, plausible-looking output. This is the single most reported
PyPortfolioOpt error. Its `linkage_method` default is **`'single'`** (the AFML original), which differs
from skfolio's **Ward** default: the two libraries give different HRP weights by design.

🚨 **`risk_models.CovarianceShrinkage` is Ledoit-Wolf shrinkage — it is NOT Marcenko-Pastur
denoising.** The two are routinely conflated and they fix different parts of the estimation-error
problem: shrinkage pulls the whole matrix toward a structured target; denoising replaces only the
eigenvalues inside the random-matrix bulk. ✅ `grep` confirms no `denoise`, `detone` or Marcenko-Pastur
anywhere in `risk_models`. **If a task says "denoise the covariance", route to `riskfolio-lib.md` or
`skfolio.md`, not here.**

🚨 **`EfficientFrontier` objects are single-use.** After `ef.max_sharpe()` you must construct a *new*
`EfficientFrontier` to call `ef.min_volatility()` — reusing one raises or returns stale state. Very
common user error when sweeping objectives in a loop.

🚨 **`expected_returns.mean_historical_return` defaults to `compounding=True` and `frequency=252`.**
Feeding it weekly, monthly or crypto (365) data without changing `frequency` silently mis-annualizes,
and the error propagates straight into the weights.

🚨 **`risk_free_rate` must be ANNUAL, on the same scale as `mu` and `S`.** PyPortfolioOpt's μ and Σ are
already annualized, so an annual rate is correct here — but empyrical's identically-purposed
`risk_free` is **per-period**. Mixing conventions across a pipeline is the #1 real-world Sharpe bug;
see `analytics-libraries.md`.

⚠️ **`max_sharpe()` does not compose naively with `weight_bounds` and added objectives.** The max-Sharpe
transformation homogenizes the problem, so bound semantics and L2/L1 penalties behave unintuitively.
The docs warn about it; users hit it constantly. For bounded max-Sharpe with regularization, prefer
`max_quadratic_utility` or `efficient_risk`, or move to Riskfolio/skfolio.

⚠️ **No HERC, no NCO, no risk budgeting, no Schur.** `HRPOpt` is plain HRP only.

## Verified API surface (1.6.0, by introspection) ✅

```
BlackLittermanModel, CLA, CovarianceShrinkage, DiscreteAllocation,
EfficientCDaR, EfficientCVaR, EfficientFrontier, EfficientSemivariance, HRPOpt,
market_implied_prior_returns, market_implied_risk_aversion, get_latest_prices,
objective_functions, risk_models, expected_returns, black_litterman
```

- `risk_models.risk_matrix(method=...)` accepts exactly ✅: `sample_cov`, `semicovariance`,
  `semivariance`, `exp_cov`, `ledoit_wolf`, `ledoit_wolf_constant_variance`,
  `ledoit_wolf_single_factor`, `ledoit_wolf_constant_correlation`, `oracle_approximating`.
- `CovarianceShrinkage` methods ✅: `ledoit_wolf`, `oracle_approximating`, `shrunk_covariance`.
- `expected_returns` ✅: `mean_historical_return`, `ema_historical_return`, `capm_return`.

🔑 **Two things it uniquely does well.** It is the **only** library here offering Ledoit-Wolf against
the *constant-variance*, *constant-correlation* and *single-factor* targets ✅ — the other libraries stop
at the scaled-identity target. And **`DiscreteAllocation`** turns a weight vector into whole share
counts under a cash budget (greedy or LP), which is genuinely useful and rarely reimplemented.

## Minimal correct snippet

This exact shape was executed against 1.6.0 ✅.

```python
from pypfopt import EfficientFrontier, risk_models, expected_returns, HRPOpt, DiscreteAllocation

# prices: DataFrame with a DatetimeIndex. frequency MUST match your bar spacing.
mu = expected_returns.mean_historical_return(prices, frequency=252, compounding=True)
S  = risk_models.CovarianceShrinkage(prices, frequency=252).ledoit_wolf()   # Ledoit-Wolf, NOT denoising

ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.10))
ef.add_objective(lambda w: 0.001 * (w @ w))            # L2 diversification penalty
ef.max_sharpe(risk_free_rate=0.02)                     # 🚨 ANNUAL rate, same scale as mu/S
w = ef.clean_weights()
ef.portfolio_performance(risk_free_rate=0.02, verbose=True)

ef2 = EfficientFrontier(mu, S, weight_bounds=(0, 0.10))   # 🚨 NEW object — ef is now spent
w_minvol = ef2.min_volatility()

hrp = HRPOpt(returns)                                   # 🚨 RETURNS, not prices
w_hrp = hrp.optimize(linkage_method="single")           # state it; skfolio's default is Ward

alloc, leftover = DiscreteAllocation(w, prices.iloc[-1], total_portfolio_value=100_000).lp_portfolio()
```

## When to use something else

| Need | Go to |
|---|---|
| Marcenko-Pastur **denoising** or detoning | `riskfolio-lib.md` (`denoiseCov`) or `skfolio.md` (`DenoiseCovariance`) |
| **HERC, NCO, Schur**, risk budgeting, 26 risk measures, integer constraints | `riskfolio-lib.md` |
| Cross-validating the allocation model with `GridSearchCV` | `skfolio.md` |
| Multi-period, transaction-cost-aware policy simulation | cvxportfolio — 🚨 **GPL-3.0** |
| Whole-share allocation from weights | **stay here** — `DiscreteAllocation` is the best option |

⚠️ **Before optimizing on `mu` at all:** mean-variance is an *error maximizer*, loading onto whatever
asset has the most overstated return and most understated variance. If you cannot defend your
expected-return estimates, use `min_volatility()` or HRP, which need no forecast — and benchmark
everything against **1/N after costs**. See `optimizers.md`.
