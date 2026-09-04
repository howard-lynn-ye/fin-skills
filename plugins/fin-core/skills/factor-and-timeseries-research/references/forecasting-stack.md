# Forecasting libraries — Nixtla, sktime, darts, and the rest

Verified 2026-09-03. **All five Nixtla packages had commits within three weeks of that date** —
this is the healthiest forecasting ecosystem in Python.

## The routing answer first

| Need | Use |
|---|---|
| Fast classical baselines at scale | **`statsforecast`** |
| Global ML on panels (LightGBM etc.) | **`mlforecast`** |
| Deep forecasting | **`neuralforecast`** |
| Coherent hierarchical reconciliation | **`hierarchicalforecast`** — nothing else in Python does this properly |
| Unified API across all TS *tasks*, incl. classification | `sktime` |
| Unified API, deep-learning leaning | `darts` |
| Volatility (GARCH/HAR) | **`arch`** — see `arch.md` |

🚨 **Finance caveat for all of them:** this stack is built for demand/retail/energy forecasting,
where series are seasonal and predictable. **Asset returns are neither.** It is excellent for
**volatility, volume, spreads, macro series and fundamentals**; it is not a return-prediction
engine. `AutoARIMA` on daily returns will — correctly — collapse to a near-zero forecast.

## Nixtla stack — all Apache-2.0, Python ≥3.10

| Package | Version | What it is |
|---|---|---|
| **`statsforecast`** | 2.1.1 | Numba-JIT classical models: AutoARIMA, AutoETS, AutoTheta, AutoCES, MSTL, TBATS, Croston, plus `Naive`, `SeasonalNaive`, `WindowAverage`. **Unique value is speed** — thousands of series in parallel, orders of magnitude faster than `pmdarima`/statsmodels loops |
| **`mlforecast`** | 1.1.0 | Global ML: wraps any sklearn-compatible regressor with automatic **lag / rolling / date feature** generation and recursive or direct multi-step. **Unique value: it builds the lag features correctly and handles recursive-prediction bookkeeping** — the exact place hand-rolled pipelines leak |
| **`neuralforecast`** | 3.2.1 | NHITS, NBEATSx, TFT, PatchTST, TimesNet, iTransformer, TSMixer, plus TSFM wrappers. **Only 9 open issues against 4,264★** — unusually well triaged |
| **`hierarchicalforecast`** | 1.5.1 | BottomUp, TopDown, MiddleOut, MinTrace (ols/wls/shrink), ERM, plus probabilistic (Bootstrap, Normality, PERMBU). Finance use: reconciling sector → industry → stock, or desk → book → firm P&L |
| `utilsforecast` | 0.2.16 | Shared plumbing; usually a transitive dep |

🔑 **`SeasonalNaive` and `AutoETS` are the correct baselines any fancy model must beat — and
frequently doesn't.** Run them first; if your deep model does not clear them, you have your answer.

**The shared data contract:** every Nixtla library wants a **long DataFrame with columns
`unique_id`, `ds`, `y`**. Consistent across the stack — a genuine design win, but a conversion cost
if you live in wide-format panels.

## sktime

| | |
|---|---|
| Version | **1.1.0** (2026-07-28); **1.0.0 landed 2026-06-11** after ~7 years and 103 releases |
| Licence | BSD-3-Clause · Python `>=3.10,<3.15` |
| GitHub | 9,988★, 2,323 forks, 🚨 **1,422 open issues**, pushed 2026-09-03 |

✅ **What 1.0 means:** the release notes say only *"Major release with some breaking changes"* and
defer to the changelog — ⚠️ I could not verify an itemized breaking-change list. What it signals
concretely is a commitment to **API stability guarantees the 0.x line refused** (0.x broke
interfaces routinely). **Treat the version number as a stability promise, not a feature
announcement.**

**1,422 open issues against 9,989 stars — 14.2% — is the worst ratio in this catalogue by an
order of magnitude** (next worst: vectorbt 1.35%, qlib 0.62%, ccxt 0.53%).

⚠️ **The 2,445 figure that circulates for sktime is wrong, and this repo repeated it.** It is
GitHub's `open_issues_count`, which **counts pull requests as issues** — sktime had 1,029 open
PRs. The correct source is the search API (`is:issue+is:open`). The conclusion survives the
correction; the number did not. Run `scripts/check_repo_stats.py` to re-verify every such count. sktime's scope
(forecasting + classification + regression + clustering + annotation + transformations) is enormous
and correspondingly hard to keep coherent.

**Unique value:** the only library with a genuinely unified sklearn-like interface across *all*
time-series learning tasks, rich composition primitives (`TransformedTargetForecaster`,
`ForecastingPipeline`, `make_reduction`), and adapters to statsmodels/prophet/darts/gluonts.
**For time-series classification it is effectively unrivalled.**
**Cost:** heavy abstraction, slower than Nixtla, steep learning curve around `ForecastingHorizon`.

⚠️ `pytorch-forecasting` is now maintained under the **`sktime` GitHub org** — merged governance.

### 🚨 VERIFIED leakage trap — no purge parameter

From `sktime/split/expandingwindow.py` and `slidingwindow.py` on `main`:

```python
class ExpandingWindowSplitter(...):
    def __init__(self, fh=1, initial_window=10, step_length=1):    # <- no `gap`

class SlidingWindowSplitter(...):
    # params: fh, window_length, step_length, initial_window        # <- no `gap`
```

**There is no embargo or purge mechanism.** Training ends at the cutoff and test starts at
`cutoff + fh`. **If your target is itself a multi-period forward return** (say 10-day), the last 10
training labels were built from prices *inside the test window* — direct label leakage.

Other splitters exist (`expandingcutoff`, `expandinggreedy`, `slidinggreedy`, `singlewindow`,
`cutoff`, `testplustrain`) but **the two canonical ones lack a gap argument.**

**For financial labels with any horizon, use `purgedcv` or `skfolio`'s `CombinatorialPurgedCV`
instead** — see `../../backtest-validation/SKILL.md` §2.

## darts

🚨 **Two verified leakage traps:**
1. `historical_forecasts` **has no purge parameter either.**
2. 🔴 **`retrain=False` makes every backtest point in-sample** — a one-word change turns an honest
   walk-forward into a fitted-on-everything curve. This is the more dangerous of the two, because it
   looks like a performance optimization.

## tsfresh — and why its defaults are already right

✅ **`HYPOTHESES_INDEPENDENT = False` by default**, so `select_features` uses **Benjamini-Yekutieli**
(`fdr_by`), valid under **arbitrary dependence** — not plain Benjamini-Hochberg. Financial features
are heavily dependent, so this is the correct choice.

🚨 **Flipping it to let more features through is exactly the mistake the parameter exists to
prevent.** Separately: **extract per fold** — running extraction across the train/test boundary leaks.

`tsfel` is the lighter alternative (BSD, 0.2.0 2025-08-20).

## prophet — listed, not recommended

Maintained (1.4.0, 2026-08-15) but **among the worst performers in the M5 competition**, and
actively harmful on financial series: it imposes a trend-plus-seasonality structure that returns do
not have, and its changepoint machinery will happily fit noise. Listed here so you recognize it in
someone else's code.

## Time-series foundation models — licences verified via the HuggingFace API

| Model | Licence |
|---|---|
| `Salesforce/moirai-1.0-R-*`, `1.1-R-*`, `moirai-moe-*` | 🔴 **`cc-by-nc-4.0` — NON-COMMERCIAL, all variants** |
| `NX-AI/TiRex` | 🔴 `other` = **`nx-ai-community-license`** |
| `google/timesfm-1.0-200m`, `2.0-500m-pytorch`, `2.5-200m-pytorch` | 🟢 `apache-2.0` |
| ⚠️ `google/timesfm-3.0-*` | ❓ **gated — HTTP 401, licence not readable anonymously.** Do not assume it inherits 2.x's Apache grant |
| `amazon/chronos-t5-*`, `chronos-bolt-*` | 🟢 `apache-2.0` |
| `AutonLab/MOMENT-1-large` | 🟢 `mit` |
| `ibm-granite/granite-timeseries-ttm-r2` (Tiny Time Mixers) | 🟢 `apache-2.0` |

🚨 **`pip install tirex` installs an unrelated 2022 dimensionality-reduction package** — the real one
is **`tirex-ts`**.

**Honest evidence:** on GIFT-Eval only **Chronos-2 and TimesFM-2.5** beat classical AutoTheta at high
frequency, and benchmark contamination is documented. The directly relevant finance study
(arXiv **2607.05291**: 9 TSFMs vs 8 econometric specs, 50 assets, realized volatility) found **only
Tiny Time Mixers beat Log-HAR, narrowly** — and Mincer-Zarnowitz recalibration showed most of that
edge was **better scaling, not better dynamics**.

**Treat TSFMs as a baseline to beat, not a source of alpha.** HAR on realized volatility remains hard
to beat.
