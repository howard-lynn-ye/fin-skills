# cvxportfolio

The only mainstream Python library that does genuine **multi-period, transaction-cost-aware**
optimization. Everything else here answers "what weights should I hold?"; this answers "what
**trades** should I place today, given that I will re-optimize tomorrow and that trading costs money
and moves the market?" — and it is the only copyleft library in the category.

| Field | Value |
|---|---|
| pip / import | `cvxportfolio` / `import cvxportfolio as cvx` |
| version | **1.5.1 (2025-07-06)** · 51 releases ✅ |
| repo / docs | `github.com/cvxgrp/cvxportfolio` · <https://www.cvxportfolio.com/> ✅ |
| stars / open issues | **1,261 / 30** ✅ |
| licence | 🚨 **GPL-3.0** ✅ (PyPI `license: "GPLv3"`, GitHub `spdx_id: GPL-3.0`) |
| Python | `requires_python` is **null** ⚠️ — declares nothing; the deps require a modern Python |
| verdict | ✅ **alive, low-velocity** — no release in 14 months, but the repo was pushed **2026-04-27** |

Verified 2026-09-04 via the PyPI JSON API and the GitHub REST API. Maintained by Stephen Boyd's
group (`cvxgrp`), principal author Enzo Busseti. It implements Boyd, Busseti, Diamond, Kahn, Koh,
Nystrup & Speth, *Multi-Period Trading via Convex Optimization* (2017).

## 🚨 Traps

🚨 **GPL-3.0 — the only copyleft library in this skill.** PyPortfolioOpt, riskfolio-lib and skfolio
are all permissive; cvxportfolio is not. Importing it into an application you distribute makes that
application GPL. Internal research use is unaffected, and so is a hosted service (GPL is not AGPL) —
but a shipped desktop tool, a redistributed library, or a customer-installed package is not.
**If this repo's output is ever redistributed, cvxportfolio must be an optional, user-installed
backend, never a dependency.** Compare `skfolio.md` (BSD-3), `pyportfolioopt.md` (MIT).

🚨 **The default data layer silently downloads from Yahoo Finance and caches to disk.**
`cvx.StockMarketSimulator(["AAPL", ...])` looks like it takes a universe; it takes a *download
instruction*. That means (a) survivorship bias — no delisted names, see
`../../market-data-sourcing/references/yfinance.md`; (b) irreproducibility — the same script run
next month resolves different data; and (c) the docs themselves caution that tests swallow
`DownloadError`, so a partly-failed download can degrade into a smaller universe rather than an
exception. **For anything you will defend, pass your own returns via `cvx.UserProvidedMarketData`.**

🚨 **A no-release year on a convex-optimization stack is a solver-drift risk, not just staleness.**
1.5.1 predates a lot of cvxpy/ECOS/Clarabel movement. Pin the solver stack alongside it and re-run
the reference backtest after any solver upgrade — see `_solver-layer.md`.

⚠️ **Multi-period problems blow up.** The problem size scales with `planning_horizon` × assets.
Beyond a few hundred names a low-rank covariance forecaster
(`HistoricalLowRankCovarianceSVD`) is effectively mandatory, not an optimization.

⚠️ **The vocabulary does not map onto the rest of the ecosystem.** Policies, costs, forecasters,
simulators — there is no `fit`/`predict`, no `clean_weights()`, and nothing plugs into sklearn.
Budget real ramp-up time; do not expect to port a PyPortfolioOpt script.

## What it has that nothing else does

- **Cost models as first-class terms in the objective**, not a post-hoc haircut:
  `cvx.TransactionCost` (with the three-halves market-impact term `a·|z| + b·σ·|z|^{3/2}/√V`),
  `cvx.HoldingCost` (borrow and short fees), `cvx.StocksTransactionCost`.
- **`MultiPeriodOptimization(planning_horizon=h)`** — optimizes today's trade against a forecast of
  the next `h` rebalances, so it will *decline* a trade whose cost exceeds its decayed alpha.
  `SinglePeriodOptimization` is the myopic special case.
- **`MarketSimulator` applies the same cost models out-of-sample** that the optimizer saw in-sample.
  Consistency by construction — the property you lose the moment you bolt a generic backtester onto
  a generic optimizer.
- **Forecasters that are causal by construction** (`HistoricalMeanReturn`,
  `HistoricalFactorizedCovariance`, `HistoricalLowRankCovarianceSVD`): each only sees data up to its
  rebalance date, which structurally prevents look-ahead in the risk model. This is a genuinely rare
  design choice and the best reason to use the library even if you ignore the multi-period part.

## Minimal correct snippet

```python
import cvxportfolio as cvx

gamma_risk, gamma_trade, gamma_hold = 5.0, 1.0, 1.0
objective = (
    cvx.ReturnsForecast()
    - gamma_risk  * cvx.FullCovariance()          # low-rank alternative beyond ~200 names
    - gamma_trade * cvx.StocksTransactionCost()   # 🚨 the whole point: keep the cost term in
    - gamma_hold  * cvx.HoldingCost()
)
constraints = [cvx.LeverageLimit(3), cvx.LongOnly(applies_to_cash=False)]

policy = cvx.MultiPeriodOptimization(objective, constraints, planning_horizon=3)

# 🚨 default StockMarketSimulator downloads from Yahoo (survivorship-biased, irreproducible).
# Supply your own returns instead:
market = cvx.UserProvidedMarketData(returns=returns_df, cash_key="cash")
result = cvx.MarketSimulator(market_data=market).backtest(policy, start_time="2020-01-01")
print(result)          # sharpe, turnover, realised costs, drawdown
```

## Choosing between it and the neighbours

| Need | Go to |
|---|---|
| **Trade sizing under real costs, over a horizon** | **cvxportfolio** — 🚨 GPL-3.0 |
| Cross-validate or grid-search the allocation model itself | `skfolio.md` (BSD-3) |
| Widest convex risk-measure menu, integer constraints | `riskfolio-lib.md` |
| Textbook mean-variance / Black-Litterman, discrete share allocation | `pyportfolioopt.md` |
| The comparison table across all four | `optimizers.md` |

`_solver-layer.md` for cvxpy and solver licences · `_attribution.md` for decomposing the realised
result · `../../backtest-validation/SKILL.md` before treating any of these Sharpes as evidence.
