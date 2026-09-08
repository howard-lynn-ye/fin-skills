# Leakage checklist

Run top to bottom. Each item is phrased so the answer is verifiable from the code, not from memory.
Anything you cannot answer is a **fail**, not an unknown.

## A. Splitting

- [ ] No `train_test_split`, no `KFold(shuffle=True)`, no default `cross_val_score` on a time-indexed frame.
- [ ] The splitter receives **both** the time the feature was known and the time the label resolved
      (`prediction_times` / `evaluation_times`). `purgedcv` enforces this; almost nothing else does.
- [ ] Purge horizon ≥ the **maximum** label horizon, not the mean.
- [ ] An embargo is set **in addition to** purging. Purging removes overlapping labels; the embargo
      removes serial correlation immediately after the test window.
- [ ] If the splitter counts in **observations** (skfolio does), you have checked what that means in
      *time* for your bar type. With dollar or volume bars, a fixed count is a wildly varying span.
- [ ] `sktime`'s `ExpandingWindowSplitter` / `SlidingWindowSplitter` and `darts.historical_forecasts`
      are **not** used for a multi-step financial label — neither has a purge parameter.
- [ ] `darts` `retrain=False` is not silently making every backtest point in-sample.

## B. Fitting

- [ ] Every scaler, encoder, imputer and dimensionality reducer is inside a `Pipeline` governed by
      the splitter — not fitted once on the full frame.
- [ ] Any parameter chosen by a statistical search over the data (fractional `d` by ADF, an optimal
      lag, a changepoint set, a clustering) is fitted on **train only** and then applied.
- [ ] Qlib users: `ZScoreNorm`'s `fit_start_time`/`fit_end_time` cover the training period only.
- [ ] Target encoding / group statistics are computed within-fold.

## C. Labels

- [ ] Label construction uses only data at or before the decision time, except for the outcome
      itself.
- [ ] Volatility targets, barrier widths and event thresholds (e.g. a CUSUM threshold from
      `get_daily_vol()`) are computed on an **expanding or rolling** window, not the full series.
- [ ] Meta-labels are built from **out-of-sample** primary-model sides. If the primary model was
      fitted on the same rows, its `side` is an in-sample fitted value and the meta-labeler learns
      to trust an artificially good signal.
- [ ] Overlapping labels are accounted for in sample weights (average uniqueness) or the effective
      sample size is stated as lower than the row count.
- [ ] Any repainting series (ZigZag, swing highs, fractals) is used **only as a label**, never as a
      feature — and then purged like any other forward-looking label.

## D. Features

- [ ] No centered or two-sided window anywhere: `rolling(center=True)`, `savgol_filter`,
      `scipy.signal.filtfilt`, `statsmodels hpfilter`, `bkfilter`, `seasonal_decompose`.
- [ ] Every `.shift()` has been checked for sign. Negative shift = future.
- [ ] Resample calls have explicit `label=` and `closed=`; the default labels at the bin **start**.
- [ ] Aggregation bars (Renko, range, volume/dollar/tick/imbalance) are indexed by **bar close
      time**, not bar start time.
- [ ] No normalization uses a whole-sample statistic — no `(x - x.mean())/x.std()`, no
      whole-history percentile rank, no min-max over everything.
- [ ] `tsfresh` extraction runs per fold, not across the train/test boundary.
- [ ] Recursive indicators (EMA, RSI, ATR, ADX, MACD) have ≥10–15 × period of burn-in discarded, and
      the same burn-in is used in backtest and live.
- [ ] `assert_causal()` (see `../../signal-construction/scripts/assert_causal.py`) passes on every
      engineered feature.

## E. Data

- [ ] The universe comes from a dated membership snapshot, not a current list.
- [ ] Delisted securities are present, or their absence is stated in the result.
- [ ] Joins are on `(identifier, date)` against a listing table, never on identifier alone.
- [ ] Fundamentals are joined on the **announcement/acceptance** timestamp, never the period end.
- [ ] Restatements are handled: the value used is the latest vintage **known as of** the decision
      date, not the latest vintage overall (`keep='last'` on the full frame is the bug).
- [ ] Macro series are timestamped by `realtime_start` (release date), not observation date.
- [ ] Price adjustment is backward-adjusted or raw+factors, never forward-adjusted.
- [ ] The same adjustment convention is used for the signal and for the execution price.

## F. Evaluation

- [ ] The reported number is cost-after and benchmark-relative.
- [ ] Beta and style exposure are regressed out before "alpha" is claimed.
- [ ] The trial count is recorded and reported.
- [ ] A confidence interval accompanies the point estimate.
- [ ] Doubling the cost assumption does not eliminate the result.

## The generic detector

When in doubt, perturb the future and check the past:

```python
a = feature(df)
df2 = df.copy(); df2.iloc[k:] *= 2.0
assert (a.iloc[:k] == feature(df2).iloc[:k]).all()   # any inequality is look-ahead
```
