---
name: lib-skfolio
description: >-
  The sklearn-compatible portfolio estimator library whose CombinatorialPurgedCV breaks sklearn's
  own split() contract - it yields (train, [test_0, ...]), and normal two-variable unpacking
  mis-partitions your data without raising. TRIGGER - skfolio, skfolio.optimization, MeanRisk,
  RiskBudgeting, HierarchicalRiskParity, HierarchicalEqualRiskContribution,
  NestedClustersOptimization, skfolio.moments, DenoiseCovariance, GerberCovariance,
  ImpliedCovariance, EmpiricalPrior, EntropyPooling, VineCopula, CombinatorialPurgedCV,
  WalkForward, purged_size, embargo_size, RiskMeasure. Memory is stale and will break code - 1.0.0
  landed 2026-08-23, so every recalled snippet predates the API stability commitment. SKIP for the
  widest risk-measure menu (lib-riskfolio) and for a strictly sklearn-compliant purged splitter
  (lib-purgedcv). SKIP for choosing between libraries - that is the domain skill's job.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# skfolio

The only portfolio library that is a real scikit-learn estimator collection — so portfolio
*construction itself* can be cross-validated, not just the return forecast feeding it.

| | |
|---|---|
| pip / import | `skfolio` / `skfolio` |
| Version | **1.0.3** (2026-08-31) — **1.0.0 landed 2026-08-23** · Python `>=3.10` |
| Licence | **BSD-3-Clause** |
| Status | ✅ **fastest-moving library in the category** — 2,345★ / 41 issues, pushed 2026-09-03; 0.20.2 → 1.0.3 in 18 days |

⚠️ **Positioning, precisely:** skfolio is scikit-learn-***compatible***, part of the sklearn
*ecosystem* — **not** an official sub-project, not `scikit-learn-contrib`-governed.
⚠️ **Pre-1.0 knowledge will break.** 1.0.0 shipped 2026-08-23, so essentially every tutorial, blog post and
model-recalled snippet predates the stability commitment. Check any 0.x example first.

## The trap that costs you money

🚨 **`CombinatorialPurgedCV.split()` yields `(train_index, [test_0, test_1, …])`** — a *list* of test folds per split,
**not** sklearn's 2-tuple contract. Any loop written the normal way — `for train, test in cv.split(X)` — binds `test`
to a list of arrays and then indexes with it. It does not raise. It produces a wrong partition and a wrong score,
which is exactly the failure mode purged CV exists to prevent. The correct shape is in the snippet below.

## `purged_size` and `embargo_size` are OBSERVATIONS, not time

🚨 Under time bars that is merely opaque. Under **dollar or volume bars a fixed count is a wildly varying time span** —
the same `purged_size=10` purges an hour in one regime and three days in another. Convert from your longest label
horizon *in bars*, and redo it whenever the sampling scheme changes. ⚠️ CPCV is also combinatorially expensive:
`C(n_folds, n_test_folds)` fits, so `C(10,3)` is 120 model fits per grid point, multiplied by the grid.

## HRP linkage defaults to Ward, not López de Prado's `single`

🚨 `HierarchicalRiskParity` results therefore differ from the AFML book and from PyPortfolioOpt's `HRPOpt` **by design,
not by bug**. Reproducing a paper? Set `linkage` explicitly. 🚨 **`fit(X)` takes RETURNS.** Prices run without
complaint and produce a nonsense covariance. Guarantee the input contains negative values before you hand it over. ⚠️
`predict` returns a `Portfolio`/`Population`, not a weights dict. Arrivals from PyPortfolioOpt look for
`clean_weights()`; the equivalents are `.weights`, `.summary()`, `.plot_cumulative_returns()`. ⚠️ Cardinality
constraints require a MIP-capable solver (HiGHS, SCIP, Gurobi, MOSEK); failures surface as opaque solver errors.

## What it has that nothing else does

- **The estimator protocol.** Every model exposes `fit`/`predict` and `get_params(deep=True)`, so
  `Pipeline`, `GridSearchCV`, `RandomizedSearchCV` and `cross_val_predict` all work. "Tune the
  shrinkage coefficient and the risk measure jointly under purged CV" is three lines here.
- **Optimizers:** Mean-Risk, Risk Budgeting, Maximum Diversification, Distributionally Robust CVaR,
  Benchmark Tracker; HRP, HERC, NCO, **Schur Complementary Allocation**; Stacking; naive baselines.
- **19 risk measures** — variance, semi-variance, CVaR, EVaR, CDaR, EDaR, max drawdown, VaR, Gini
  mean difference, skew, kurtosis.
- **Covariance estimators:** Empirical, Gerber, **Denoise** (Marcenko-Pastur), **Detone**, EWMA,
  Ledoit-Wolf, OAS, Graphical-Lasso-CV, and **Implied Covariance** (option-implied vols on the
  diagonal) — the last genuinely unique to skfolio.
- Entropy Pooling, Opinion Pooling, Vine Copula synthetic returns; Black-Litterman; factor models
  with 46 descriptors; factor stress-testing; bootstrapped uncertainty sets on both μ and Σ.
- **Model selection:** `WalkForward`, `CombinatorialPurgedCV`, Multiple Randomized CV.

## Minimal correct call

```python
from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.moments import DenoiseCovariance
from skfolio.prior import EmpiricalPrior
from skfolio.model_selection import WalkForward, CombinatorialPurgedCV
from sklearn.model_selection import GridSearchCV

model = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=EmpiricalPrior(covariance_estimator=DenoiseCovariance()),
    min_weights=0.0, max_weights=0.10,        # set bounds explicitly, never rely on defaults
)
model.fit(X_train)                            # X_train: DataFrame of RETURNS, not prices
ptf = model.predict(X_test)                   # a Portfolio object; weights are ptf.weights
print(ptf.summary())

grid = GridSearchCV(model, {"risk_measure": [RiskMeasure.VARIANCE, RiskMeasure.CVAR]},
                    cv=WalkForward(train_size=252, test_size=63))   # sizes in OBSERVATIONS
grid.fit(X)

cv = CombinatorialPurgedCV(n_folds=10, n_test_folds=2,
                           purged_size=5, embargo_size=5)           # OBSERVATIONS, not time
for train_idx, test_idx_list in cv.split(X):                        # 🚨 non-sklearn contract
    for test_idx in test_idx_list:
        ...
```

## See also

- `../../../fin-core/skills/portfolio-and-risk/SKILL.md` — optimizer choice, and the metric traps downstream
- `../../../fin-core/skills/portfolio-and-risk/references/skfolio.md` — the source card
- `../../../fin-core/skills/portfolio-and-risk/references/optimizers.md` — head-to-head table
- `../../../fin-core/skills/backtest-validation/references/purgedcv.md` — CPCV alternatives compared
