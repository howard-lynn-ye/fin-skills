---
name: lib-riskfolio
description: >-
  The 26-risk-measure portfolio optimizer whose stateful API optimizes against stale or missing mu
  and Sigma - with no error - if you forget assets_stats(). TRIGGER - riskfolio, Riskfolio-Lib,
  "import riskfolio as rp", rp.Portfolio, rp.HCPortfolio, assets_stats, port.optimization,
  hcp.optimization, model="HRP"/"HERC"/"NCO", rm="CVaR"/"CDaR"/"EVaR"/"RLVaR"/"EDaR"/"RLDaR",
  denoiseCov, riskfolio.src.AuxFunctions, ParamsEstimation, entropy_pooling, OWA, MVSK, "solver
  did not converge". Memory is stale - it is at 7.3.0 (2026-05-31) with an unusual 18 open issues
  against 4,480 stars. SKIP for GridSearchCV over portfolio models (lib-skfolio) and for
  whole-share allocation (lib-pyportfolioopt). SKIP when the question is WHICH library to choose
  rather than how to use this one - that belongs to the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# Riskfolio-Lib

The most feature-complete open-source portfolio optimizer in Python. Its moat is **26 convex risk measures**, so you
can run HRP, HERC, NCO or risk parity under CVaR, CDaR, EVaR or Ulcer instead of variance.

| | |
|---|---|
| pip / import | `Riskfolio-Lib` / **`riskfolio`** (conventionally `import riskfolio as rp`) |
| Version | **7.3.0** (2026-05-31) · Python `>=3.10` |
| Licence | **BSD-3-Clause** |
| Status | ✅ **actively developed and unusually well-tended** — `dcajasn/Riskfolio-Lib`, 4,480★ / **18 open issues** (an outlier in this ecosystem), pushed 2026-08-18 |

## The trap that costs you money

🚨 **The API is imperative and stateful, not sklearn-like.** You must call `port.assets_stats(...)` **before**
`port.optimization(...)`, and **re-call it after changing the data**. Skipping it, or mutating `port.returns`
afterwards, optimizes against stale or missing μ and Σ rather than raising. The weights come back looking normal.

It also does **not** compose with `sklearn.Pipeline` or `GridSearchCV`. If you need cross-validated hyperparameter
search over portfolio models, that is `lib-skfolio`.

## Returns, not prices

🚨 `rp.Portfolio(returns=...)` and `rp.HCPortfolio(returns=...)` take **RETURNS**. A price DataFrame runs silently and
yields a garbage covariance and garbage weights — the same failure mode as PyPortfolioOpt's `HRPOpt`. Check for
negative values before you pass.

⚠️ **pip name ≠ import name:** `pip install Riskfolio-Lib`, then `import riskfolio`. The historical
`riskfolio.Portfolio` vs `riskfolio.src.*` split means older snippets import from paths that moved. ❓ **`rf=` units
are not documented consistently.** Riskfolio works on per-period returns throughout, so `rf` is almost certainly
per-period — but the same-named parameter is **annual** in quantstats, PyPortfolioOpt and ffn, and
**per-period** in empyrical. Hand-compute one ratio to pin it down.

## The solver layer decides which risk measures you can actually use

🚨 **EVaR, RLVaR, EDaR and RLDaR need exponential- or power-cone support** → Clarabel, SCS or MOSEK. OSQP cannot
express them, and the failure reads *"solver did not converge"*, not "wrong solver class". 🚨
**Cardinality, buy-in threshold and mutually-exclusive constraints need a MIP-capable solver** — HiGHS or SCIP
for free, Gurobi/MOSEK at scale (the docs say so explicitly).

## The moat: 26 risk measures, and real denoising

Three families: **dispersion** (std dev, MAD, Gini mean difference, range, VaR range, tail-Gini range, EVaR range,
RLVaR range, kurtosis variants), **downside** (semi-std, first/second lower partial moments, CVaR, tail Gini, EVaR,
RLVaR, minimax), **drawdown** (average drawdown, Ulcer, CDaR, EDaR, RLDaR, max drawdown).

Models: mean-risk, **logarithmic mean-risk (Kelly)**, risk parity across 22 measures, risk budgeting,
**HRP and HERC across 37 risk measures**, **NCO** with four objectives, relaxed risk parity,
worst-case robust mean-variance, **OWA**, **MVSK** (SDP relaxation), Black-Litterman + Augmented BL + Bayesian BL,
**entropy pooling**, risk-factor models.

Estimation tooling (source-verified, `riskfolio/src/ParamsEstimation.py`): `mean_vector`, `covar_matrix`,
`cokurt_matrix`, `forward_regression`, `backward_regression`, `PCR`, `loadings_matrix`, `risk_factors`,
`black_litterman`, `augmented_black_litterman`, `black_litterman_bayesian`, `entropy_pooling`, `bootstrapping`,
`normal_simulation`. `covar_matrix(X, method=...)` accepts `hist`, `semi`, `ewma1`, `ewma2`, `ledoit`, `oas`,
`shrunk`, `gl`, `jlogo`, **`fixed`/`spectral`/`shrink` (Marcenko-Pastur denoising)**, `gerber1`, `gerber2` — plus
`detone=True` and `mkt_comp=`. ⚠️ Docs are example-heavy (an excellent Jupyter gallery) but API-reference-thin —
expect to read the source for exact argument semantics. ⚠️ Cython/C++ extension: on an exotic platform you get a
source build.

⚠️ **Denoising lives in `riskfolio.src.AuxFunctions`, not at the top level** — it is a covariance utility, not an
optimizer method. That module carries the full López de Prado random-matrix stack: `mpPDF`, `fitKDE`, `errPDFs`,
`findMaxEval`, `getPCA`, `denoisedCorr`, `shrinkCorr`, `denoiseCov`, plus `dcorr_matrix`, `mutual_info_matrix`,
`var_info_matrix`, `ltdi_matrix`, `two_diff_gap_stat`, `std_silhouette_score`. **This is the reason to reach here over
PyPortfolioOpt when `T/N` is small.**

## Minimal correct call

```python
import riskfolio as rp
import riskfolio.src.AuxFunctions as af

assert (returns < 0).any().any(), "RETURNS, not prices"

port = rp.Portfolio(returns=returns)
port.assets_stats(method_mu="hist", method_cov="ledoit")   # 🚨 REQUIRED, and re-run after data changes
w = port.optimization(model="Classic", rm="CVaR", obj="Sharpe",
                      rf=0.0, l=0, hist=True)              # ❓ verify rf units against a hand calc

# Hierarchical — needs no expected-return estimate at all
hcp   = rp.HCPortfolio(returns=returns)
w_hrp = hcp.optimization(model="HRP", codependence="pearson", rm="CVaR",
                         linkage="single", leaf_order=True)   # 'single' = the AFML original
w_nco = hcp.optimization(model="NCO", obj="MinRisk", rm="CVaR")

# Explicit Marcenko-Pastur denoising — NOT what Ledoit-Wolf shrinkage does
cov_dn = af.denoiseCov(returns.cov().values,
                       q=returns.shape[0] / returns.shape[1],   # T/N
                       kind="fixed", detone=False)
```

## See also

- `../../../fin-core/skills/portfolio-and-risk/SKILL.md` — optimizer choice, and the metric traps downstream
- `../../../fin-core/skills/portfolio-and-risk/references/riskfolio-lib.md` — the source card
- `../../../fin-core/skills/portfolio-and-risk/references/risk-measures.md` — what CVaR/CDaR/EVaR mean before you optimize one
- `../../../fin-core/skills/portfolio-and-risk/references/_solver-layer.md` — which solver each conic risk measure needs

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`portfolio-and-risk`** (`../../../fin-core/skills/portfolio-and-risk/SKILL.md`).
