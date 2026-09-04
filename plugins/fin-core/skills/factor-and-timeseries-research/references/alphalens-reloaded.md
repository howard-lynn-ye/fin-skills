# alphalens-reloaded

The de-facto factor-evaluation tool. **Its forward-return convention is the single most common
source of overstated factor results**, and it is a default, not a bug.

| | |
|---|---|
| pip | `alphalens-reloaded` — **import as `alphalens`** |
| Version | **0.4.6** (2025-06-02); prior 0.4.5 (2024-09-26); 8 releases total |
| Licence | Apache-2.0 |
| Python | `>=3.10`, classifiers 3.10–3.13 |
| GitHub | `stefan-jansen/alphalens-reloaded` — 642★, 144 forks, 14 open issues, pushed **2025-12-15** |
| Deps | `pandas<3.0,>=1.5.0`, numpy, statsmodels, `empyrical-reloaded>=0.5.7`, seaborn, IPython |
| Status | ⚠️ **alive but low-velocity maintenance**, not active development. `main` runs ahead of PyPI |

🔴 **The original is dead.** `quantopian/alphalens` — 4,435★ but last push **2024-02-12**, last PyPI
release **0.4.0 (2020-04-27)**, classifiers cap at Python 3.5. **Do not `pip install alphalens`.**

⚠️ **Note the `pandas<3.0` pin.** On a pandas 3.x environment this will hold you back or conflict.

## What it computes

Everything flows from one call:

```python
from alphalens.utils import get_clean_factor_and_forward_returns
factor_data = get_clean_factor_and_forward_returns(
    factor,          # pd.Series, MultiIndex (date, asset)
    prices,          # pd.DataFrame, index=date, columns=asset   <- OPPOSITE orientation
    periods=(1, 5, 10),
    quantiles=5,
    groupby=sectors, binning_by_group=False,
    max_loss=0.35,
)
```

- **IC** — `factor_information_coefficient` = per-date **Spearman rank correlation** with the forward
  return; plus IC decay by horizon and "IC IR" (mean IC / std IC).
- **Quantile returns** — `mean_return_by_quantile`, cumulative returns by quantile, and the
  top-minus-bottom factor-weighted long/short (`factor_returns`, `cumulative_returns`).
- **Turnover** — `quantile_turnover` and `factor_rank_autocorrelation`, the proxy for signal decay.
- Tear sheets: `create_full_tear_sheet` and the returns / information / turnover / event-study variants.

## 🚨 The forward-return trap — verified in source

From `src/alphalens/utils.py::compute_forward_returns` on current `main`:

```python
for period in sorted(periods):
    if cumulative_returns:
        returns = prices.pct_change(period)
    else:
        returns = prices.pct_change()
    forward_returns = returns.shift(-period).reindex(factor_dateindex)
```

For a factor observed at date `t` with `period=p`, the forward return is **`P[t+p] / P[t] − 1` — it
starts accruing from the price at date `t` itself.**

**alphalens does not lag your factor for you.** If your factor at `t` comes from `t`'s close — which
is overwhelmingly the common case for any close-based indicator — the `1D` forward return alphalens
shows is the `t`-close → `t+1`-close return. **You can only earn that by transacting at the very
close that produced the signal.**

**Fix**, either:
```python
factor = factor.groupby(level=1).shift(1)     # Jansen's own convention in MLFT
```
or pass `prices` that are opens, offset appropriately. The realistic chain is
**signal from `t` close → trade at `t+1` open → hold to `t+1+p`.**

## Other verified mechanics

- **`cumulative_returns=True` (default) produces overlapping returns.** Consecutive rows share `p−1`
  days of price path, which **inflates apparent t-stats** on the mean quantile spread.
  `cumulative_returns=False` gives 1-period returns shifted by `p` — a different, and usually more
  honest, object for statistics.
- It **infers a trading calendar** (`infer_trading_calendar`) then sets `df.index.levels[0].freq`.
  Setting `.freq` on a MultiIndex level is fragile and is the source of most "alphalens breaks on my
  data" reports — it needs a clean, gap-consistent trading calendar.
- 🔴 **Latent dead code:** `backshift_returns_series` (utils.py ~L364) still uses `ix.labels`, which
  pandas **removed in 0.24** (renamed `.codes`). It `AttributeError`s on any modern pandas. Not on the
  main path — just never call it.

## Hard requirements people get wrong

1. `factor` **must** be a `pd.Series` with a 2-level MultiIndex `(date, asset)` — not a DataFrame,
   not wide. Level 0 must be a `DatetimeIndex`, and **tz-awareness must match** `prices` or you get a
   silent empty join.
2. `prices` must be **wide**: `DatetimeIndex` rows × asset columns — **the opposite orientation from
   `factor`**. This asymmetry is the single most common `KeyError`.
3. `prices` must extend **at least `max(periods)` bars beyond** the last factor date, or the final
   rows silently become NaN and are dropped.
4. `max_loss=0.35` raising `MaxLossExceededError` is usually **telling you the truth** about bad
   alignment. Do not just raise the threshold.

## Interpretation traps

- **Quantile buckets are equal-weighted by count, not dollars.** In a cap-skewed universe the top
  quantile is microcap-driven and not investable.
- **No transaction costs anywhere.** Turnover is reported; cost is never applied. A factor with
  IC 0.03 and 90% monthly turnover is almost certainly negative after costs.
- **Sparse cross-sections.** With <~100 names/day, quintiles hold 20 names and the spread is noise.
  Alphalens will plot it happily.
- **IC is Spearman by default** — robust to outliers, but it **discards magnitude**. A factor can
  have excellent IC and terrible dollar returns.
- **Survivorship is yours.** Alphalens analyses whatever universe you hand it.

## Minimal correct usage

```python
import alphalens as al

factor = my_factor.groupby(level=1).shift(1)          # <- LAG IT
fd = al.utils.get_clean_factor_and_forward_returns(
    factor, prices, periods=(1, 5, 21), quantiles=5,
    cumulative_returns=False,                          # <- non-overlapping for stats
)
ic = al.performance.factor_information_coefficient(fd)
print(ic.mean(), ic.mean() / ic.std())                 # IC and IC IR
al.tears.create_full_tear_sheet(fd)
```
