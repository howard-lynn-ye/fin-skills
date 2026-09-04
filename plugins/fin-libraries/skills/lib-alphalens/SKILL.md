---
name: lib-alphalens
description: >-
  alphalens-reloaded scores cross-sectional factors, and its forward return starts at date t's OWN
  price - it never lags your factor. TRIGGER - alphalens, alphalens-reloaded, "import alphalens as
  al", get_clean_factor_and_forward_returns, compute_forward_returns,
  factor_information_coefficient, mean_return_by_quantile, factor_returns, quantile_turnover,
  factor_rank_autocorrelation, create_full_tear_sheet, MaxLossExceededError, max_loss=0.35,
  cumulative_returns, information coefficient, IC decay, quantile spread, "pip install alphalens".
  The original quantopian package is dead at 0.4.0 (2020-04-27) and most snippets you recall
  target it or its removed pandas internals. SKIP for lib-qlib, which is the skill for the feature
  pipeline and model. SKIP when the question is WHICH library to choose rather than how to use
  this one - that belongs to the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# alphalens-reloaded

The de-facto factor-evaluation tool. **Its forward-return convention is the single most common source
of overstated factor results**, and it is a default, not a bug.

| | |
|---|---|
| pip / import | `alphalens-reloaded` — **imported as `alphalens`** |
| Version | 0.4.6 (2025-06-02); prior 0.4.5 (2024-09-26); 8 releases total. Repo pushed 2025-12-15 |
| Licence | Apache-2.0 |
| Status | Alive but low-velocity maintenance, not active development. `main` runs ahead of PyPI. 642★ |

**The original is dead.** `quantopian/alphalens` — 4,435★, last push 2024-02-12, last PyPI release
**0.4.0 (2020-04-27)**, classifiers capped at Python 3.5. **Do not `pip install alphalens`.** Note
also the `pandas<3.0,>=1.5.0` pin: on a pandas 3.x environment this holds you back or conflicts.

## The trap that costs you money

**`compute_forward_returns` does `pct_change(period).shift(-period)`.** From `src/alphalens/utils.py`:

```python
for period in sorted(periods):
    if cumulative_returns:
        returns = prices.pct_change(period)
    else:
        returns = prices.pct_change()
    forward_returns = returns.shift(-period).reindex(factor_dateindex)
```

For a factor observed at date `t` with `period=p`, the forward return is **`P[t+p] / P[t] − 1` — it
starts accruing from the price at date `t` itself.** **alphalens does not lag your factor for you.**
If your factor at `t` comes from `t`'s close — overwhelmingly the common case for any close-based
indicator — the `1D` forward return alphalens shows is the `t`-close → `t+1`-close return, earnable
only by transacting at the very close that produced the signal.

Fix it yourself, either by lagging the factor:

```python
factor = factor.groupby(level=1).shift(1)
```

or by passing `prices` that are opens, offset appropriately. The realistic chain is **signal from `t`
close → trade at `t+1` open → hold to `t+1+p`.**

## `cumulative_returns=True` is the default and it overlaps

With the default, consecutive rows share `p−1` days of price path. **Overlapping returns inflate
apparent t-stats** on the mean quantile spread. `cumulative_returns=False` gives 1-period returns
shifted by `p` — a different, and for statistics usually more honest, object.

## The orientation asymmetry — factor long, prices wide

This is the single most common `KeyError` in the library:

1. `factor` **must** be a `pd.Series` with a 2-level MultiIndex `(date, asset)` — not a DataFrame,
   not wide. Level 0 must be a `DatetimeIndex`, and **tz-awareness must match** `prices` or you get a
   silent empty join.
2. `prices` must be **wide**: `DatetimeIndex` rows × asset columns — **the opposite orientation from
   `factor`**.
3. `prices` must extend **at least `max(periods)` bars beyond** the last factor date, or the final
   rows silently become NaN and are dropped.
4. `max_loss=0.35` raising `MaxLossExceededError` is usually **telling you the truth** about bad
   alignment. Do not just raise the threshold.

Two mechanical notes: alphalens **infers a trading calendar** (`infer_trading_calendar`) then sets
`df.index.levels[0].freq`, which is fragile and is the source of most "alphalens breaks on my data"
reports — it needs a clean, gap-consistent calendar. And `backshift_returns_series` (utils.py ~L364)
is latent dead code still using `ix.labels`, removed from pandas in 0.24; never call it.

## Interpretation traps

- **Quantile buckets are equal-weighted by count, not dollars.** In a cap-skewed universe the top
  quantile is microcap-driven and not investable.
- **No transaction costs anywhere.** Turnover is reported; cost is never applied. A factor with
  IC 0.03 and 90% monthly turnover is almost certainly negative after costs.
- **Sparse cross-sections.** With fewer than ~100 names/day, quintiles hold 20 names and the spread
  is noise. Alphalens will plot it happily.
- **IC is Spearman by default** — robust to outliers, but it **discards magnitude**. A factor can have
  excellent IC and terrible dollar returns.
- **Survivorship is yours.** Alphalens analyses whatever universe you hand it.

## Minimal correct call

```python
import alphalens as al

factor = my_factor.groupby(level=1).shift(1)          # LAG IT — alphalens will not
fd = al.utils.get_clean_factor_and_forward_returns(
    factor,                                            # Series, MultiIndex (date, asset)
    prices,                                            # DataFrame, WIDE: dates x assets
    periods=(1, 5, 21), quantiles=5,
    cumulative_returns=False,                          # non-overlapping for statistics
)
ic = al.performance.factor_information_coefficient(fd)
print(ic.mean(), ic.mean() / ic.std())                 # IC and IC IR
al.tears.create_full_tear_sheet(fd)
```

## See also

- `../../../fin-core/skills/factor-and-timeseries-research/SKILL.md` §1.1 — factor evaluation
- `../../../fin-core/skills/factor-and-timeseries-research/references/alphalens-reloaded.md` — card
- `../lib-qlib/SKILL.md` — Alpha158/Alpha360 features and the leakage-safe default label

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`factor-and-timeseries-research`** (`../../../fin-core/skills/factor-and-timeseries-research/SKILL.md`).

