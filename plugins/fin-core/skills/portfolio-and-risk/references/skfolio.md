# skfolio

The only portfolio library that is a real scikit-learn estimator collection — so portfolio
*construction itself* can be cross-validated, not just the return forecast feeding it.

| Field | Value |
|---|---|
| pip / import | `skfolio` / `skfolio` |
| version | **1.0.3** (2026-08-31) ✅ — **1.0.0 landed 2026-08-23** |
| repo / docs | `github.com/skfolio/skfolio` · <https://skfolio.org/> ✅ |
| stars / open issues | **2,345 / 41** ✅ |
| licence | **BSD-3-Clause** ✅ |
| Python | `>=3.10` ✅ |
| verdict | ✅ **fastest-moving library in the category** — pushed 2026-09-03; 0.20.2 → 1.0.3 in 18 days |

Verified 2026-09-04 via the PyPI JSON API and the GitHub REST API.

⚠️ **Positioning, stated precisely:** skfolio is **scikit-learn-*compatible***, part of the sklearn
*ecosystem*. It is **not** an official scikit-learn sub-project and not `scikit-learn-contrib`-governed.
Say "sklearn-compatible", never "from scikit-learn".

## 🚨 Traps

🚨 **`CombinatorialPurgedCV.split()` yields `(train_index, [test_0, test_1, …])`** — a list of test
folds per split, **not** sklearn's 2-tuple contract. Any loop written as `for train, test in cv.split(X)`
will bind `test` to a *list of arrays* and then index with it. It does not raise; it produces a wrong
partition and a wrong score.

🚨 **`purged_size` and `embargo_size` are counted in OBSERVATIONS, not time.** Under time bars that is
merely opaque; under **dollar or volume bars a fixed count is a wildly varying time span**, so the same
`purged_size=10` purges an hour in one regime and three days in another. Convert from your longest label
horizon *in bars* each time the sampling scheme changes.

🚨 **`HierarchicalRiskParity` defaults to Ward linkage, not López de Prado's `single`.** Results differ
from the AFML book and from PyPortfolioOpt's `HRPOpt` **by design, not by bug**. If you are reproducing
a paper, set the linkage explicitly rather than assuming a shared default.

🚨 **`fit(X)` takes RETURNS.** Prices run without complaint and produce a nonsense covariance. Guarantee
the input contains negative values before you hand it over.

🚨 **Pre-1.0 code and pre-1.0 model knowledge will break.** 1.0.0 shipped 2026-08-23, so essentially
every tutorial, blog post and LLM-recalled snippet predates the stability commitment. Check any 0.x
example against the current API before running it.

⚠️ **`predict` returns a `Portfolio`/`Population` object, not a weights dict.** People arriving from
PyPortfolioOpt look for `clean_weights()`; the equivalents are `.weights`, `.summary()`,
`.plot_cumulative_returns()`.

⚠️ **Cardinality constraints require a MIP-capable solver** (HiGHS, SCIP, Gurobi, MOSEK). Failures
surface as solver errors, not as a clear "you need a MIP solver" message. See `_solver-layer.md`.

⚠️ **CPCV is combinatorially expensive** — `C(n_folds, n_test_folds)` fits. `C(10,3)` is 120 model fits
per grid point, multiplied by the grid.

## What it has that the alternatives do not

- **The estimator protocol.** Every model exposes `fit`/`predict` and `get_params(deep=True)`, so
  `Pipeline`, `GridSearchCV`, `RandomizedSearchCV` and `cross_val_predict` all work. "Tune the shrinkage
  coefficient and the risk measure jointly under purged CV" is three lines here and a project elsewhere.
- **Optimizers:** Mean-Risk, Risk Budgeting, Maximum Diversification, **Distributionally Robust CVaR**,
  Benchmark Tracker; **HRP, HERC, NCO, Schur Complementary Allocation**; Stacking; naive baselines.
- **19 risk measures** — variance, semi-variance, CVaR, **EVaR**, CDaR, **EDaR**, max drawdown, VaR,
  Gini mean difference, skew, kurtosis.
- **Covariance estimators:** Empirical, **Gerber**, **Denoise** (Marcenko-Pastur), **Detone**, EWMA,
  Ledoit-Wolf, OAS, Graphical-Lasso-CV, and **Implied Covariance** (option-implied vols on the
  diagonal) — the last is genuinely unique to skfolio.
- **Entropy Pooling, Opinion Pooling, Vine Copula** synthetic-return generation; Black-Litterman;
  factor models with 46 characteristics-based descriptors; factor stress-testing.
- **Model selection:** `WalkForward`, `CombinatorialPurgedCV`, Multiple Randomized CV.
- **Uncertainty sets** bootstrapped on both μ and Σ, for robust optimization.

## Minimal correct snippet

```python
from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.moments import DenoiseCovariance
from skfolio.prior import EmpiricalPrior
from skfolio.model_selection import WalkForward
from sklearn.model_selection import GridSearchCV

model = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=EmpiricalPrior(covariance_estimator=DenoiseCovariance()),
    min_weights=0.0, max_weights=0.10,          # set bounds explicitly, never rely on defaults
)
model.fit(X_train)                               # X_train: DataFrame of RETURNS, not prices
ptf = model.predict(X_test)                      # a Portfolio object; weights are ptf.weights
print(ptf.summary())

grid = GridSearchCV(model, {"risk_measure": [RiskMeasure.VARIANCE, RiskMeasure.CVAR]},
                    cv=WalkForward(train_size=252, test_size=63))   # explicit sizes, in OBSERVATIONS
grid.fit(X)
```

Handling the non-standard CPCV contract:

```python
from skfolio.model_selection import CombinatorialPurgedCV
cv = CombinatorialPurgedCV(n_folds=10, n_test_folds=2, purged_size=5, embargo_size=5)  # OBSERVATIONS
for train_idx, test_idx_list in cv.split(X):     # 🚨 second element is a LIST of folds
    for test_idx in test_idx_list:
        ...
```

## Choosing between skfolio and the neighbours

| Need | Go to |
|---|---|
| Cross-validate or grid-search the allocation model itself | **skfolio** — nothing else makes this easy |
| The widest risk-measure menu (26 convex, RLVaR/RLDaR, OWA, MVSK, integer constraints) | `riskfolio-lib.md` |
| Textbook mean-variance / Black-Litterman, friendliest API, discrete share allocation | `pyportfolioopt.md` |
| Multi-period, transaction-cost-aware simulation of the trading *policy* | cvxportfolio — 🚨 **GPL-3.0** |
| Standalone purged CV that is strictly sklearn-protocol compliant | `../../backtest-validation/references/purgedcv.md` |

See also `optimizers.md` (the comparison table), `_solver-layer.md` (cvxpy and solver licences), and
`../../backtest-validation/SKILL.md` before treating any cross-validated Sharpe as evidence.
