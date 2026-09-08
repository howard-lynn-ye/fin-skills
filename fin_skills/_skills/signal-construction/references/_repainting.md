# Repainting and non-causal indicators

An indicator "repaints" when its value for a past bar changes as new bars arrive. On a chart it
looks like a perfect signal; in a backtest it *is* a perfect signal, because the backtest reads the
finalized series. In live trading it does not exist.

## The blacklist

### 1. Centered / two-sided windows
`rolling(center=True)` · `scipy.signal.savgol_filter` · **`scipy.signal.filtfilt`** (zero-phase =
forward *and* backward pass) · **`statsmodels.tsa.filters.hpfilter`** · `bkfilter` (Baxter-King is
symmetric by construction) · `seasonal_decompose` (its trend is a centered MA).

🚨 **`filtfilt` and `hpfilter` are guaranteed look-ahead.** Never build a feature from them.
Verified: `rolling(5, center=True)` at index 10 is the mean of indices 8–12.

### 2. Repainting pivot / swing detectors
ZigZag · Bill Williams Fractals · "swing high/low" · Elliott-wave labellers · harmonic-pattern finders.

A pivot is only *definable* once you know price did not go further, so the most recent pivot moves as
bars arrive. **If you use ZigZag it must be a training LABEL, never a feature** — and then it needs
triple-barrier-style purging like any other forward-looking label.

### 3. Forward-plotted components
Ichimoku Senkou Span A/B are plotted 26 bars ahead; pivot-point projections for the *current* period
are computed from that period's own data. Safe **only** if the shift is genuinely backward in your
frame. Verify empirically rather than reasoning about it.

### 4. Aggregation-bar timestamp confusion
Renko · Kagi · Point & Figure · range bars · volume / dollar / tick / imbalance bars.

The bar is **stamped at its start** but only **complete at its end**. Indexing by start time and
using the bar "as of" that timestamp leaks the entire bar. **Always index by bar close time.**

### 5. "Zero-lag" smoothers
Zero-lag EMA variants that subtract a forward-shifted term; some Ehlers filters. Check the sign of
every `shift()` — a negative shift is the future.

### 6. Full-sample normalization disguised as an indicator
z-scored RSI · min-max scaled anything · `(x - x.mean()) / x.std()` over the whole series ·
percentile rank over the whole history. The mean/std/percentile is a future-dependent statistic.
Use `expanding()`, or fit the scaler on train only inside a `Pipeline`.

## Verified library-specific leaks

**`ta` 0.11.0** (isolated by perturbing only future bars):

| Call | Leak |
|---|---|
| `IchimokuIndicator(..., visual=True).ichimoku_a()` | 🚨 `spana.shift(w2, fill_value=spana.mean())` uses the **whole-sample mean** for the first 26 rows. **26 early bars changed, max \|Δ\| 35.34.** Happens **even with `fillna=False`** because `fill_value` is inside `.shift()` |
| `DPOIndicator(..., fillna=True).dpo()` | 🚨 same pattern; **11 early bars changed, max \|Δ\| 34.43.** Clean with `fillna=False` |
| `ichimoku_b()` | Hardcodes `min_periods=0` ignoring `fillna` — the first 51 Span B values are computed on partial windows and are **silently wrong** |
| `_check_fillna()` | Does `ffill().bfill()` — **`bfill()` propagates future values backwards.** Affects PSAR up/down (NaN by construction on the opposite trend, so their leading NaNs *are* backfilled from the future), Ichimoku, Keltner, Donchian, cumulative return |
| `ta.utils.dropna()` | Silently deletes every row where any numeric column equals `0.0` — destroys zero-volume and zero-return rows |

## Warm-up is not leakage, but it is just as wrong

Recursive indicators (EMA, RSI, ATR, ADX, MACD) emit values before they have converged. Measured
against TA-Lib on identical input, `ta`'s RSI(14) differs by >1.0 through bar 36, >0.1 through bar
67, >0.01 through bar 97. `ta`, `pandas-ta-classic` and `talipp` all emit **before** TA-Lib does,
computed on partial windows.

**Discard the first ~10–15 × period bars, and use the same warm-up in backtest and live.** Different
warm-up lengths between the two is why an identical strategy produces different signals in
production — see `../../backtesting-engines/SKILL.md` §2.3.

## Detection

```python
def assert_causal(fn, df, k, tol=1e-9):
    a = fn(df)
    df2 = df.copy(); df2.iloc[k:] *= 2.0
    b = fn(df2)
    bad = (a.iloc[:k].fillna(-9e99) - b.iloc[:k].fillna(-9e99)).abs() > tol
    assert not bad.any().any(), "LOOK-AHEAD"
```

Full version with a warm-up estimator and a batch scanner: `../scripts/assert_causal.py`.

Off-the-shelf: `freqtrade lookahead-analysis` (truncated-data rerun; only checks *triggered* signals)
and `freqtrade recursive-analysis` (varies `startup_candle_count`; **the only off-the-shelf tool that
measures the warm-up problem directly**).
