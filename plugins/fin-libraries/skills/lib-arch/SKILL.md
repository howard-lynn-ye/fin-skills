---
name: lib-arch
description: >-
  The reference GARCH implementation in Python, and the home of SPA/StepM/MCS - which all take
  LOSSES, so passing returns silently inverts the test and names your worst strategy as the best.
  TRIGGER - arch, arch_model, arch.bootstrap, arch.univariate, SPA, RealityCheck, StepM, MCS,
  optimal_block_length, StationaryBootstrap, superior_models, spa.pvalues, mcs.included,
  arch.unitroot, GARCH, EGARCH, GJR-GARCH, TARCH, APARCH, FIGARCH, HARCH, HAR-RV, skewt,
  conditional_volatility, or a GARCH fit emitting convergence warnings. Memory is stale on licence
  and version - it is 8.0.0 (2025-10-21) under NCSA, not one of the three usual permissive
  licences. SKIP for PSR/DSR/PBO (lib-purgedcv) and for reporting Sharpe (lib-quantstats). SKIP
  for choosing between libraries - that is the domain skill's job.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# arch

Two distinct libraries share one package. The volatility half is well known; **`arch.bootstrap` is the most under-used
correct tool in quantitative finance.** 8.0.0 is a compatibility release (Python 3.14 wheels, NumPy 2.4 / pandas 3) —
**not an API break** from 7.x. statsmodels has no equivalent.

| | |
|---|---|
| pip / import | `arch` / `arch` |
| Version | **8.0.0** (2025-10-21) · Python `>=3.10` |
| Licence | **NCSA** — permissive and BSD-like, but *not* one of the usual three; GitHub reports `NOASSERTION`. **Flag it in a licence audit.** |
| Status | ✅ **actively maintained; the reference GARCH implementation in Python** — `bashtage/arch` (Kevin Sheppard, Oxford), 1,558★ / 51 issues, pushed 2026-08-10 |

## The trap that costs you money

🚨 **`SPA`, `RealityCheck`, `StepM` and `MCS` all take LOSSES — lower is better.** Verified against
`arch/bootstrap/multiple_comparison.py`: `SPA`/`StepM` document `benchmark` as *"T element array of benchmark model
**losses**"* and `models` as *"T by k element array of alternative model
**losses**"*; `MCS`'s parameter is literally named `losses`.

Passing returns does not raise and does not warn. **It inverts the test** — the worst strategy is identified as the
best, and the p-value you report is for the opposite hypothesis. For a return series the whole conversion is `losses =
-returns`. Anything already loss-shaped (squared forecast error, negative log-likelihood, absolute error, drawdown)
goes in as-is. A *ratio* — a Sharpe, an information ratio — is not accepted at all; these tests need the per-period
loss column.

## Which procedure answers which question

| Question | Class | Input |
|---|---|---|
| Is the **best of N** better than a benchmark, correcting for the search? | **`SPA`** (Hansen 2005) | benchmark losses + (T×k) model losses |
| Same question, older and less powerful | `RealityCheck` (White 2000) | identical |
| ***Which*** strategies beat the benchmark, under FWER control? | **`StepM`** (Romano-Wolf) | benchmark + (T×k) losses |
| Which models are indistinguishable from the best? | **`MCS`** (Hansen-Lunde-Nason) | (T×k) losses, **no benchmark** |

Verified signatures: `SPA(benchmark, models, block_size=None, reps=1000, bootstrap="stationary", studentize=True,
nested=False, *, seed=None)`; `StepM(benchmark, models, size=0.05, ...)`; `MCS(losses, size, reps=1000,
block_size=None, method="R"|"max", ...)`.

- ✅ **`RealityCheck` is literally `class RealityCheck(SPA): pass`** — a zero-code subclass; White's
  Reality Check is SPA without studentization or recentring. **Use `SPA`**, and mention
  `RealityCheck` only when reproducing a pre-2005 paper.
- 🚨 **`SPA.pvalues` returns three values — `lower`, `consistent`, `upper`. Report `consistent`.**
  They bracket the influence of dominated models on the null; `lower` is liberal, `upper`
  conservative. Also `.critical_values(pvalue=0.05)` and `.better_models(pvalue=0.05)`.
- `StepM.superior_models` names the survivors under FWER control; internally it builds an `SPA`.
  `MCS` takes **no benchmark**, raises with fewer than 2 columns, and gives `.included`/`.excluded`.
- ⚠️ `block_size` defaults to `int(sqrt(T))`. Set it from `optimal_block_length` (Politis-White) —
  preserving serial dependence is the entire reason for a block bootstrap. Cost is
  **O(reps × T × k)**: 1000 reps over hundreds of candidates is minutes, not seconds.
- ⚠️ These control FWER over **the models you pass**, and know nothing about the variants you tried
  and discarded — that is what a trial ledger and the Deflated Sharpe Ratio are for. Use both.

## Volatility models: fit on returns × 100

🚨 **`arch` expects returns in PERCENT.** Decimals give tiny variances, poor optimizer conditioning and convergence
warnings — the single most common `arch` support question, and a usage issue rather than a bug.
**Rescale the forecast back out of percent before you use it.**

- `forecast()` returns **variance**, not volatility — square-root it, then undo the ×100.
  `reindex=False` is what you almost always want; the default reindexes to the full sample.
- ⚠️ **`o=1` is what gives you the leverage effect.** Plain `GARCH(1,1)` treats a −5% day and a +5%
  day identically, empirically false for equities. GJR (`o=1`) or EGARCH is the honest default.

Covers GARCH, EGARCH, GJR-GARCH/TARCH, APARCH, HARCH, HAR-RV, FIGARCH, RiskMetrics, ARCH; Normal / Student-t / skew-t
/ GED innovations; AR and HAR mean models; plus `arch.unitroot` (ADF, DF-GLS, Phillips-Perron, KPSS, Zivot-Andrews,
Variance Ratio), cointegration and long-run covariance estimators. **HAR on realized volatility remains a very strong
baseline.**

## Minimal correct call

```python
from arch import arch_model
from arch.bootstrap import SPA, StepM, MCS, optimal_block_length
am  = arch_model(returns * 100,                 # 🚨 PERCENT, not decimals
                 vol="GARCH", p=1, o=1, q=1,    # o=1 -> GJR asymmetry (leverage effect)
                 dist="skewt", mean="Constant")
res = am.fit(disp="off")
f   = res.forecast(horizon=10, reindex=False)   # variance, not vol
sigma_next = (f.variance.iloc[-1] ** 0.5) / 100 # 🚨 rescale back out of percent

z = (res.resid / res.conditional_volatility).dropna()   # GARCH-filtered VaR: a CONDITIONAL quantile
var_1d = (res.params["mu"] + sigma_next.iloc[0] * 100 * z.quantile(0.05)) / 100

losses_bm  = -bench_returns                     # 🚨 NEGATE. (T,)
losses_mdl = -model_returns                     # 🚨 NEGATE. (T, k)
bs = int(optimal_block_length(losses_bm).iloc[0, 0])

spa = SPA(losses_bm, losses_mdl, reps=1000, block_size=bs, seed=7); spa.compute()
print(spa.pvalues["consistent"])                # 🚨 the one to report

stepm = StepM(losses_bm, losses_mdl, size=0.05, reps=1000, block_size=bs); stepm.compute()
print(stepm.superior_models)                    # WHICH ones survive FWER control

mcs = MCS(losses_mdl, size=0.10, reps=1000, block_size=bs); mcs.compute()
print(mcs.included, mcs.excluded)               # no benchmark argument
```

## See also

- `../../../fin-core/skills/backtest-validation/SKILL.md` — the domain skill for the `arch.bootstrap` half
- `../../../fin-core/skills/factor-and-timeseries-research/SKILL.md` — the domain skill for the GARCH half
- `../../../fin-core/skills/factor-and-timeseries-research/references/arch.md` — the source card
- `../../../fin-core/skills/backtest-validation/references/significance-tests.md` — which test answers which question
