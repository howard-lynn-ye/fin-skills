# Riskfolio-Lib

The most feature-complete open-source portfolio optimizer in Python — its moat is **26 convex risk
measures**, so you can run HRP or risk parity under CVaR, CDaR, EVaR or Ulcer instead of variance.

| Field | Value |
|---|---|
| pip / import | `Riskfolio-Lib` / **`riskfolio`** (conventionally `import riskfolio as rp`) |
| version | **7.3.0** (2026-05-31) ✅ |
| repo / docs | `github.com/dcajasn/Riskfolio-Lib` (Dany Cajas) · <https://riskfolio-lib.readthedocs.io/> ✅ |
| stars / open issues | **4,480 / 18** ✅ |
| licence | **BSD-3-Clause** ✅ |
| Python | `>=3.10` ✅ |
| verdict | ✅ **actively developed and unusually well-tended** — pushed 2026-08-18 |

Verified 2026-09-04 via the PyPI JSON API and the GitHub REST API.

🔑 **10 open issues against 4,480 stars is an outlier in this ecosystem** — the maintainer actually
closes things. Cadence: 7.0.1 (2025-05) → 7.1.0 (2025-11) → 7.2.0 (2026-01) → 7.2.1 (2026-02) →
7.3.0 (2026-05).

## 🚨 Traps

🚨 **`rp.Portfolio(returns=...)` and `rp.HCPortfolio(returns=...)` take RETURNS.** Passing a price
DataFrame runs silently and yields a garbage covariance and garbage weights. Same failure mode as
PyPortfolioOpt's `HRPOpt` — check for negative values before you pass.

🚨 **The API is imperative and stateful, not sklearn-like.** You must call `port.assets_stats(...)`
*before* `port.optimization(...)`, and re-call it after changing the data. Skipping it, or mutating
`port.returns` afterwards, optimizes against stale or missing μ/Σ rather than raising. It does **not**
compose with `sklearn.Pipeline` or `GridSearchCV` — if you need cross-validated hyperparameter search
over portfolio models, use `skfolio.md`.

🚨 **EVaR, RLVaR, EDaR and RLDaR need exponential-cone / power-cone support** → Clarabel, SCS or MOSEK.
OSQP cannot express them. The failure looks like *"solver did not converge"*, not like a clear "wrong
solver class" message. See `_solver-layer.md`.

🚨 **Cardinality, buy-in threshold and mutually-exclusive constraints need a MIP-capable solver** —
HiGHS or SCIP for free, Gurobi/MOSEK for anything large. The docs explicitly recommend the commercial
solvers at scale.

⚠️ **pip name ≠ import name.** `pip install Riskfolio-Lib`, then `import riskfolio`. The historical
`riskfolio.Portfolio` vs `riskfolio.src.*` split means older snippets may import from paths that have
moved.

⚠️ **Denoising lives in `riskfolio.src.AuxFunctions`, not the top level** — it is a covariance utility,
not an optimizer method.

⚠️ **Cython/C++ extension** — wheels matter; on an exotic platform you get a source build. ❓ Not
exercised in this pass.

⚠️ Documentation is example-heavy (an excellent Jupyter gallery) but **API-reference-thin**. Expect to
read the source for exact argument semantics.

❓ **`rf=` units are not documented consistently.** Riskfolio works on per-period returns throughout, so
`rf` is almost certainly per-period — but the same-named parameter is *annual* in quantstats,
PyPortfolioOpt and ffn, and *per-period* in empyrical. **Hand-compute one ratio to pin it down before
publishing.** See `analytics-libraries.md` for the measured damage this convention split causes.

## The risk-measure moat

**26 convex risk measures** across three families ⚠️ (from the project docs):

- **Dispersion** — standard deviation, MAD, Gini mean difference, range, VaR range, tail-Gini range,
  EVaR range, RLVaR range, kurtosis variants
- **Downside** — semi-standard deviation, first/second lower partial moments, **CVaR**, tail Gini,
  **EVaR** (entropic), **RLVaR** (relativistic), minimax
- **Drawdown** — average drawdown, Ulcer index, **CDaR**, **EDaR**, **RLDaR**, max drawdown

**Models:** mean-risk (min-risk / max-return / max-utility / max-ratio), **logarithmic mean-risk
(Kelly)**, risk parity across 22 measures, risk budgeting, **HRP and HERC across 37 risk measures**,
**NCO** with four objectives, relaxed risk parity, **worst-case robust mean-variance**, **OWA**,
**MVSK** (mean-variance-skewness-kurtosis via SDP relaxation), **Black-Litterman + Augmented BL +
Bayesian BL**, **entropy pooling**, and risk-factor models.

**Constraints:** cardinality, effective number of assets, buy-in thresholds, turnover, tracking error,
asset-class and factor-exposure groups, graph-information constraints, risk-contribution inequalities,
and integer constraints.

## Estimation-error tooling — verified by source read ✅

`riskfolio/src/ParamsEstimation.py`: `mean_vector`, `covar_matrix`, `cokurt_matrix`,
`forward_regression`, `backward_regression`, `PCR`, `loadings_matrix`, `risk_factors`,
`black_litterman`, `augmented_black_litterman`, `black_litterman_bayesian`, `entropy_pooling`,
`bootstrapping`, `normal_simulation`.

`covar_matrix(X, method=...)` accepts `hist`, `semi`, `ewma1`, `ewma2`, **`ledoit`**, **`oas`**,
`shrunk`, `gl` (graphical lasso), `jlogo`, **`fixed` / `spectral` / `shrink` (Marcenko-Pastur
denoising)**, `gerber1`, `gerber2` — plus `detone=True` and `mkt_comp=` to strip the market mode.

`riskfolio/src/AuxFunctions.py` carries the full López de Prado random-matrix stack: `mpPDF`, `fitKDE`,
`errPDFs`, `findMaxEval`, `getPCA`, `denoisedCorr`, `shrinkCorr`, and
`denoiseCov(cov, q, kind="fixed"|"spectral"|"shrink", bWidth, detone, mkt_comp, alpha)` — plus
`dcorr_matrix`, `mutual_info_matrix`, `var_info_matrix`, `ltdi_matrix` (lower tail dependence),
`two_diff_gap_stat`, `std_silhouette_score`.

🔑 **This is the reason to reach for Riskfolio over PyPortfolioOpt when `T/N` is small.**
PyPortfolioOpt's `CovarianceShrinkage` is Ledoit-Wolf only — it has **no Marcenko-Pastur denoising**.

## Minimal correct snippet

```python
import riskfolio as rp

port = rp.Portfolio(returns=returns)                        # 🚨 RETURNS, not prices
port.assets_stats(method_mu="hist", method_cov="ledoit")    # 🚨 REQUIRED before optimization()
port.lowerret = None                                        # set constraints explicitly, not by default

w = port.optimization(model="Classic", rm="CVaR", obj="Sharpe",
                      rf=0.0, l=0, hist=True)               # ❓ verify rf units against a hand calc

# Hierarchical — needs no expected-return estimate at all
hcp = rp.HCPortfolio(returns=returns)
w_hrp = hcp.optimization(model="HRP", codependence="pearson", rm="CVaR",
                         linkage="single", leaf_order=True)  # 'single' = the AFML original
w_nco = hcp.optimization(model="NCO", obj="MinRisk", rm="CVaR")

# Explicit Marcenko-Pastur denoising (this is NOT what Ledoit-Wolf shrinkage does)
import riskfolio.src.AuxFunctions as af
cov_dn = af.denoiseCov(returns.cov().values,
                       q=returns.shape[0] / returns.shape[1],   # T/N
                       kind="fixed", detone=False)
```

## When to use something else

| Need | Go to |
|---|---|
| sklearn `Pipeline`/`GridSearchCV`, entropy pooling *as an estimator*, vine copulas, implied covariance | `skfolio.md` |
| Friendliest API, Black-Litterman teaching, **discrete whole-share allocation** | `pyportfolioopt.md` |
| Multi-period, transaction-cost-aware policy simulation | cvxportfolio — 🚨 **GPL-3.0** |
| HRP/HERC/NCO with a proprietary-free conscience | either Riskfolio or skfolio — 🚨 **not `mlfinlab`**, which is off PyPI and stubbed |

See `optimizers.md` for the head-to-head table, `_solver-layer.md` for solver classes and licences, and
`risk-measures.md` for what CVaR/CDaR/EVaR actually mean before you optimize one.
