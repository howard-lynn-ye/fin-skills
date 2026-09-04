# Portfolio optimizers — verified metadata and how to choose

Verified 2026-09-03. `[V]` = verified at a primary source · `[D]` = documented by the project ·
`[U]` = unverified.

## At a glance

| Library | pip / import | Version | Licence | ★ / open issues | Verdict |
|---|---|---|---|---|---|
| **PyPortfolioOpt** | `PyPortfolioOpt` / `pypfopt` | **1.6.0** (2026-02-26) | MIT | 6,005 / 114 | ⚠️ **revived**, cautiously healthy |
| **Riskfolio-Lib** | `Riskfolio-Lib` / `riskfolio` | **7.3.0** (2026-05-31) | BSD-3 | 4,479 / **18** | ✅ actively developed, unusually well-tended |
| **skfolio** | `skfolio` | **1.0.3** (2026-08-31) | BSD-3 | 2,341 / 41 | ✅ **fastest-moving; 1.0.0 landed 2026-08-23** |
| **cvxportfolio** | `cvxportfolio` / `cvx` | 1.5.1 (2025-07-06) | 🚨 **GPL-3.0** | 1,260 / 30 | ⚠️ slow but alive |
| deepdow | `deepdow` | 0.2.3 (2024-01) | Apache-2.0 | — | ⚠️ dormant |
| 🔴 mlfinlab | — | — | proprietary | — | **off PyPI, source stubbed** |

## PyPortfolioOpt — a status change worth knowing

🔑 **The repo moved.** `robertmartin8/PyPortfolioOpt` now **redirects** — the project lives at
**`github.com/PyPortfolio/PyPortfolioOpt`**, a GitHub *org*.

It went nearly dormant: 1.5.5 (2023-05), 1.5.6 (2024-12), no feature release for ~3 years. In 2026 it
moved to the org and shipped **1.6.0 on 2026-02-26**, with pushes continuing to 2026-07-07. Robert
Martin remains the PyPI author of record; per the project's own `docs/Roadmap.rst` **Tuan Tran is now
the primary maintainer** (handover context: GitHub issue #587). **114 open issues** is a meaningful
backlog — treat responsiveness as moderate.

**Good for:** textbook mean-variance, Black-Litterman, efficient frontier, teaching, quick prototypes.
**Does not have:** HERC, NCO, or Marcenko-Pastur denoising.

🚨 **`HRPOpt(returns=...)` takes a RETURNS matrix.** Passing prices runs silently and produces
garbage. Its `linkage_method` default is **`'single'`** (the AFML original).
🚨 **`risk_models.CovarianceShrinkage` is Ledoit-Wolf — NOT Marcenko-Pastur denoising.** The two are
routinely conflated; they solve different parts of the estimation-error problem.

## Riskfolio-Lib — the risk-measure moat

**Only 18 open issues against 4,479 stars** is an outlier in this ecosystem — the maintainer closes
things. Cadence: 7.0.1 (2025-05) → 7.1.0 (2025-11) → 7.2.0 (2026-01) → 7.2.1 (2026-02) → 7.3.0
(2026-05).

**The most feature-complete open-source portfolio optimizer in Python.** Its moat is **26 convex risk
measures** across three families — dispersion (std dev, MAD, Gini mean difference, range, VaR range,
tail-Gini range, EVaR range, RLVaR range, kurtosis variants), downside, and drawdown — plus HRP and
HERC across all of them, NCO with four objectives, OWA, robust worst-case optimization, and integer
constraints.

Use it when you need **HRP under CVaR / CDaR / EVaR / Ulcer** rather than variance. Non-sklearn API
(`riskfolio.HCPortfolio` with a `model=` selector).

## skfolio — now a first-class recommendation

Shipped **0.20.2 → 1.0.0 → 1.0.1 → 1.0.2 → 1.0.3 between 2026-08-13 and 2026-08-31.** The 1.0
milestone means the API is stability-committed — before mid-2026 it was "promising but churning";
**as of 2026-09 it is a first-class choice.**

⚠️ **Positioning:** it is **scikit-learn-compatible**, part of the sklearn *ecosystem* — it is **not**
an official scikit-learn sub-project and not `scikit-learn-contrib`-governed. Describe it accurately.

**Unique value: it is the only one of these that is a real estimator library.** Everything is a
sklearn estimator with `fit`/`predict` and `get_params(deep=True)`, so portfolio construction composes
with `Pipeline` and `GridSearchCV`. That means you can **cross-validate the portfolio construction
itself**, not just the return forecast — which nothing else here makes easy.

```python
from skfolio.optimization import (MeanRisk, RiskBudgeting, HierarchicalRiskParity,
                                  HierarchicalEqualRiskContribution, NestedClustersOptimization)
from skfolio.moments import DenoiseCovariance, DetoneCovariance, GerberCovariance
from skfolio.model_selection import CombinatorialPurgedCV, WalkForward
```

🚨 **`CombinatorialPurgedCV.split()` yields `(train_index, [test_0, test_1, ...])`** — multiple test
folds per split, **not** sklearn's 2-tuple contract. Code written against the standard protocol will
mis-handle it.
🚨 **`purged_size`/`embargo_size` are in observations, not time.** With dollar or volume bars a fixed
count is a wildly varying time span.
⚠️ Its `HierarchicalRiskParity` defaults to **Ward** linkage, not López de Prado's **single** —
results differ from the book by design, not by bug.

## cvxportfolio — the multi-period one

🚨 **GPL-3.0.** Every other library here is MIT/BSD/Apache. **If you intend to distribute anything
without releasing source, this is the one to flag.** Internal research use is unaffected.

Last *release* 1.5.1 (2025-07-06) — over a year old — but the repo was pushed 2026-04-27. Stable and
low-velocity, maintained by Stephen Boyd's group (`cvxgrp`), principal author Enzo Busseti.

**Unique value:** the only mainstream Python library doing genuine **multi-period, transaction-cost-
aware** optimization — it simulates the *trading policy*, not a single-period weight vector. If your
question is "what does this strategy cost to run", this is the tool.

## The estimation-error problem

Mean-variance is a **maximizer of estimation error** — it loads on whatever asset has the most
overstated expected return and the most understated variance. The optimizer is doing its job; the
inputs are the problem.

| Technique | Where |
|---|---|
| Ledoit-Wolf / OAS shrinkage | `sklearn.covariance`, `pypfopt.risk_models.CovarianceShrinkage` |
| **Marcenko-Pastur denoising / detoning** | `riskfolio.denoiseCov`, `skfolio.moments.DenoiseCovariance`/`DetoneCovariance` — 🚨 **not in PyPortfolioOpt** |
| Factor-model covariance | build it yourself, or Riskfolio-Lib |
| HRP / HERC / NCO | sidestep matrix inversion entirely |
| Black-Litterman | shrinks expected returns toward equilibrium |

**A practical default:** if you cannot justify your expected-return estimates, **do not optimize on
them**. Minimum-variance, risk parity and HRP need no return forecast and are far more stable
out-of-sample. And benchmark everything against **1/N after costs** — it is a famously hard baseline,
and if your optimizer does not beat it, that is the finding.

See `_solver-layer.md` for cvxpy, solver choice, and what makes a problem infeasible.
