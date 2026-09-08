# sktime · darts

The two "one API for every forecaster" libraries. Both are excellent engineering. **Neither one's
backtesting utility can purge overlapping labels**, so both will hand you a clean-looking
walk-forward number that is leaked — and darts has a one-word setting that makes the entire
backtest in-sample without warning.

| | `sktime` | `darts` |
|---|---|---|
| version | **1.1.0 (2026-07-28)** · **1.0.0 landed 2026-06-11** ✅ | **0.47.0 (2026-09-04)** ✅ |
| releases | 103 | 57 |
| GitHub | `sktime/sktime` — **9,989★**, 2,323 forks, pushed 2026-09-04 | `unit8co/darts` — **9,508★**, 1,035 forks, pushed 2026-09-04 |
| open issues | **1,422 genuine issues** (+ ~1,029 open PRs; GitHub's `open_issues_count` reads **2,451** and counts both) ✅ | **194 genuine issues** (`open_issues_count` 229) ✅ |
| Licence | **BSD-3-Clause** ✅ | **Apache-2.0** ✅ |
| Python | `>=3.10,<3.15` ✅ | `>=3.10` ✅ |
| Verdict | ✅ active, enormous scope, worst issue-to-star ratio in the catalogue | ✅ very active — released the day of this audit |

Verified 2026-09-04 via PyPI JSON, the GitHub REST API and the GitHub search API; splitter and
`historical_forecasts` behaviour read from source.

⚠️ **The 2,445-open-issues figure that circulates for sktime is GitHub's `open_issues_count`, which
includes pull requests.** The honest split is ~1,422 issues and ~1,029 PRs. Still the heaviest
backlog in this category, but say which number you mean.

🚨 **`u8darts` on PyPI is dead — install `darts`.** `u8darts` 0.41.0 carries the classifier
`Development Status :: 7 - Inactive` ✅ and its only dependency is `darts==0.41.0` ✅ — it is a
version-pinning shim that will silently hold you six releases behind. Its own summary reads
*"⚠️ DEPRECATED - Use 'darts' package instead."*

## 🚨 Traps

🚨 **sktime's `ExpandingWindowSplitter` and `SlidingWindowSplitter` have NO gap / purge / embargo
parameter.** Read from `sktime/split/expandingwindow.py` and `slidingwindow.py` on `main` ✅:

```python
class ExpandingWindowSplitter(...):
    def __init__(self, fh=1, initial_window=10, step_length=1):        # no `gap`
class SlidingWindowSplitter(...):
    # params: fh, window_length, step_length, initial_window           # no `gap`
```

Training ends **at** the cutoff; test starts at `cutoff + fh`. If your target is a multi-period
forward return — a 10-day forward return, say — the last 10 training labels were computed from
prices that lie *inside* the test window. That is direct label leakage, it is the López de Prado
purging problem, and sktime does not solve it.

🚨 **darts' `historical_forecasts` has no gap parameter either** ✅. Signature from
`darts/models/forecasting/forecasting_model.py`:

```python
def historical_forecasts(self, series, ..., forecast_horizon=1, stride=1,
                         retrain=True, last_points_only=True, ...)
```

Same overlapping-label exposure, same absence of a fix.

🚨 **`retrain=False` makes every backtest point in-sample.** With `retrain=True` (the default) darts
does an honest expanding-window walk-forward. `retrain=False` is "Pre-trained Mode": no refit. So the
common sequence — `model.fit(full_series)` then `model.historical_forecasts(full_series,
retrain=False)` — evaluates the model on data it was trained on, at every single point. **One word,
no warning, spectacular results.** This is the single most common darts leak. Only ever use
`retrain=False` on a model fitted strictly on data preceding `start`.

🚨 **`last_points_only=True` is the darts default** and it discards all but the final point of each
forecast path. Fine at horizon 1. At horizon > 1 your error metric silently becomes "accuracy at
exactly h steps ahead", not "accuracy over the path" — a different quantity from the one most
write-ups claim to report.

🚨 **darts' `Scaler` is the standard preprocessing leak.** `scaler.fit_transform(full_series)` before
splitting fits the scaling statistics on the test period. Fit on train, transform both.

⚠️ **sktime 1.0 is younger than almost every tutorial about it.** 1.0.0 shipped 2026-06-11 after
seven years of a 0.x line that broke interfaces routinely, so essentially every blog post, Stack
Overflow answer and model-recalled snippet predates the stability commitment. Check any example
against the installed API before running it.

⚠️ **Neither is a return-prediction engine.** Both are built for seasonal, predictable series. They
are excellent for **volatility, volume, spreads, macro series and fundamentals**; `AutoARIMA` on
daily returns will — correctly — collapse to a near-zero forecast. See `forecasting-stack.md`.

## The fix: purge outside the library

Neither library will do this for you, so do it upstream. Build the folds with a splitter that
actually purges and pass explicit index sets:

```python
from skfolio.model_selection import CombinatorialPurgedCV   # or the AFML splitters
# purge/embargo counted in OBSERVATIONS, sized from your LABEL HORIZON in bars
cv = CombinatorialPurgedCV(n_folds=10, n_test_folds=2, purged_size=10, embargo_size=5)
```

Then, in darts, retrain honestly and fit the scaler per fold:

```python
from darts.models import LinearRegressionModel
from darts.dataprocessing.transformers import Scaler

scaler = Scaler().fit(train_series)                       # 🚨 fit on TRAIN only
model  = LinearRegressionModel(lags=20, output_chunk_length=1)

bt = model.historical_forecasts(
    scaler.transform(series),
    start=0.6,
    forecast_horizon=10,
    stride=10,                 # >= forecast_horizon: non-overlapping labels, a poor-man's purge
    retrain=True,              # 🚨 NEVER False unless the fit predates `start`
    last_points_only=False,    # keep the whole path if your metric is path-based
)
```

`stride >= forecast_horizon` is the cheapest honest approximation available inside darts: it makes
consecutive evaluation labels non-overlapping, which removes the leakage the missing `gap` would
have caused. It costs you evaluation points.

## Which one

| Need | Pick |
|---|---|
| Time-series **classification** (regime labels, pattern matching) | **sktime** — effectively unrivalled |
| Deep + probabilistic forecasting, one `TimeSeries` object, naive baselines built in | **darts** |
| Pipelines/composition (`TransformedTargetForecaster`, `make_reduction`) and adapters to everything | **sktime** |
| Raw speed on thousands of series | neither — `statsforecast`, see `forecasting-stack.md` |
| sklearn regressor → forecaster with a clearer backtest utility | `skforecast`, see `forecasting-stack.md` |

darts ships naive/drift/seasonal-naive as first-class models in the same API, which removes every
excuse for omitting the baseline. **Run `NaiveSeasonal` and a linear model first.**

## Cross-references

`forecasting-stack.md` (the Nixtla comparison and the finance caveat) ·
`../../backtest-validation/references/purgedcv.md` (splitters that actually purge) ·
`../../backtest-validation/references/afml-stack.md` (why overlapping labels leak) ·
`../../portfolio-and-risk/references/skfolio.md` (purged CV with an sklearn-compatible contract).
