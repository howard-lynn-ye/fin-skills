---
name: signal-construction
description: >-
  Compute technical indicators and engineered features without leaking the future. TRIGGER - RSI,
  MACD, moving average, Bollinger, ATR, ADX, Ichimoku, PSAR, stochastic or any named technical
  indicator; TA-Lib, pandas-ta, pandas-ta-classic, ta, talipp, finta; choosing an indicator
  library or reconciling two that disagree; "does this indicator repaint"; warm-up, unstable
  period, or an indicator differing between backtest and live; zigzag, fractals, swing highs. SKIP
  for judging whether a finished signal predicts returns (factor-and-timeseries-research) and for
  the backtest that consumes it (backtesting-engines).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# Signal construction

Two failures dominate: **using a library whose warm-up values are silently wrong**, and **using an
indicator that cannot be computed causally.** §2 and §4 cover them. Both were measured, not assumed.

## 1. What changed since ~2024

| Change | Status |
|---|---|
| **TA-Lib's C dependency pain is SOLVED** | `ta-lib-python` ships prebuilt wheels **bundling the C library** since v0.6.5 (2025-08-07). Latest 0.7.1, 54 wheels. `pip install TA-Lib` just works. |
| 🚨 **`pandas-ta` is effectively gone** | `github.com/twopirllc/pandas-ta` returns **404** (removed ~June 2025). PyPI history **wiped to 2 beta releases**. `pandas-ta.dev` **DNS does not resolve**. **No licence field, no classifier.** |
| **`pandas-ta-classic` is the live successor** | MIT, 0.6.52 (2026-06-24), actively released |
| vectorbt got a 1.x Rust engine | v1.1.0 (2026-07-05) — licence is **Apache-2.0 + Commons Clause**, not OSI open source |

🚨 **Treat `pandas-ta` as do-not-install.** A package with no licence, wiped release history, a dead
upstream repo and a dead homepage is a supply-chain risk regardless of intent.

**Also dead:** `finta` (archived 2022, LGPL-3.0), `tulipy` (2019, LGPL), `bta-lib`.

## 2. ⭐ Measured agreement with TA-Lib

> Executed on TA-Lib python 0.7.1 / C core 0.7.1, pandas 3.0.5, numpy 2.4.6, 500-bar synthetic GBM,
> seed 7. `maxabs` = max abs diff over the overlapping non-NaN region.

**Ranking for "numerically identical to TA-Lib":**
1. **TA-Lib** (reference)
2. **`talipp`** — closest, and the only one that also matches TA-Lib's **warm-up start index** on RSI
3. **`pandas-ta-classic`** (native, `talib=False`) — matches on the core set; **ADX warm-up is the exception**
4. **`ta` (bukosabino)** — matches on non-recursive indicators only

| Library | SMA/BB/CCI/OBV | RSI(14) | EMA(20) | ATR(14) | ADX(14) |
|---|---|---|---|---|---|
| `talipp` 2.7.0 | exact | **1.4e-14, exact incl. warm-up** | 2.8e-14 | warm-up only | — |
| `pandas-ta-classic` 0.6.52 | exact | **3.6e-14** | 2.8e-14 | 8.9e-16 | 🚨 **maxabs 4.25**, emits from bar 13 vs TA-Lib's 27 |
| `ta` 0.11.0 | identical | 🚨 **maxabs 3.84** | 🚨 0.363 | 🚨 0.0477, emits **from bar 0** | 🚨 **0.642**, emits **from bar 0** |

**How long `ta`'s RSI(14) takes to agree with TA-Lib on the same input:**
`|diff| > 1.0` through **bar 36** · `> 0.1` through **bar 67** · `> 0.01` through **bar 97** ·
`> 1e-6` through **bar 217**.

🚨 **`ta`'s Wilder-smoothed indicators need ~15× the period of burn-in before they agree with
TA-Lib.** On short datasets, or when you restart the calculation per symbol on a short slice, `ta`
and TA-Lib produce **materially different signals**.

> **Universal rule: discard the first ~10–15 × period bars of any recursive indicator (EMA, RSI,
> ATR, ADX, MACD), whatever the library.** `ta`, `pandas-ta-classic` and `talipp` all emit values
> *before* TA-Lib does, computed on partial windows — **silently wrong, not NaN.**

✅ **`talipp`'s incremental `.add()` is bit-identical to batch construction** — so backtest and live
produce the same numbers. That is the whole reason the library exists, and it is the correct choice
when a strategy must run streaming.

## 3. 🚨 Verified look-ahead bugs in `ta` 0.11.0

Isolated by perturbing **only future bars** and checking whether early bars changed.

1. **`IchimokuIndicator(..., visual=True).ichimoku_a()` — CONFIRMED LOOK-AHEAD.**
   Source: `spana.shift(self._window2, fill_value=spana.mean())` — the **whole-sample mean** fills
   the first 26 rows. Editing only bars 200+ changed **26 early bars, max |Δ| = 35.34**. Happens
   **even with `fillna=False`**, because `fill_value` sits inside `.shift()`, outside the guard.
   `visual=False` is clean.
2. **`DPOIndicator(..., fillna=True).dpo()` — CONFIRMED LOOK-AHEAD.**
   `self._close.shift(int(0.5*window)+1, fill_value=self._close.mean())`. With `fillna=True` the
   rolling `min_periods` drops to 0 so those whole-sample-mean rows survive. Editing bars 200+
   changed **11 early bars, max |Δ| = 34.43**. Clean with `fillna=False`.
3. **`ichimoku_b()` hardcodes `min_periods=0`**, ignoring `fillna`. Span B's `first_valid_index()`
   is 0 while Span A's is 25 — **the first 51 Senkou Span B values are computed on partial windows
   and are silently wrong.**
4. **`_check_fillna()` does `ffill().bfill()`** — the **`bfill()` propagates future values backwards**
   into leading NaNs. Verified affected: Ichimoku conv/base/spanA/spanB, **PSAR up/down** (which are
   NaN by construction on the opposite trend, so their leading NaNs *are* backfilled from the
   future), Keltner, Donchian, cumulative return.
5. **`ta.utils.dropna()` silently deletes every row where any numeric column equals `0.0`** — it will
   quietly destroy zero-volume or zero-return rows.

**Verdict:** `ta` is fine for a quick pure-Python feature set **with `fillna=False`**; never use
`visual=True` on Ichimoku for modelling; discard warm-up generously.

## 4. 🚨 Indicator classes that inherently leak

| Class | Examples | Why |
|---|---|---|
| **Centered / two-sided windows** | `rolling(center=True)`, `savgol_filter`, **`scipy.signal.filtfilt`** (zero-phase = forward+backward), **`statsmodels hpfilter`**, `bkfilter`, `seasonal_decompose` | The value at *t* depends on data after *t*. **`filtfilt` and `hpfilter` are guaranteed look-ahead — never build a feature from them.** |
| **Repainting pivot/swing detectors** | ZigZag, Williams Fractals, "swing high/low", Elliott-wave and harmonic-pattern finders | A pivot is only *definable* once you know price did not go further; the last pivot moves as bars arrive. **If you use ZigZag it must be a training LABEL, never a feature — and then it needs triple-barrier-style purging.** |
| **Forward-plotted components** | Ichimoku Senkou Span A/B (+26), pivot-point projections for the current period | Safe only if the shift is genuinely backward in your frame. **Verify empirically.** |
| **Aggregation-bar timestamps** | Renko, Kagi, P&F, range bars, volume/dollar/tick/imbalance bars | The bar is *stamped* at its start but only *complete* at its end. **Index by bar close time**, or you leak the whole bar. |
| **"Zero-lag" smoothers** | zero-lag EMA variants subtracting a forward-shifted term, some Ehlers filters | Check the sign of every `shift()`. Negative = future. |
| **Full-sample normalization** | z-scored RSI, min-max anything, `(x-x.mean())/x.std()` over the whole series, whole-history percentile ranks | The statistic is future-dependent. Use `expanding()` or fit-on-train-only. |

## 5. Detect it automatically

The generic test is six lines and catches almost everything — `scripts/assert_causal.py`:

```python
def assert_causal(fn, df, k):
    """Perturb only rows >= k; assert nothing before k moved."""
    a = fn(df)
    df2 = df.copy(); df2.iloc[k:] *= 2.0
    b = fn(df2)
    bad = (a.iloc[:k].fillna(-9e9) - b.iloc[:k].fillna(-9e9)).abs() > 1e-9
    assert not bad.any().any(), f"LOOK-AHEAD: {int(bad.sum().sum())} pre-{k} cells depend on post-{k} data"
```

Off-the-shelf: **`freqtrade lookahead-analysis`** (reruns on truncated data, flags indicators whose
history changes — only checks *triggered* signals) and **`freqtrade recursive-analysis`** (varies
`startup_candle_count` ∈ {199, 499, 999, 1999} and reports % variance — **the only off-the-shelf
tool that measures the unstable-period problem directly**).

## 6. Choosing

| Need | Use |
|---|---|
| Reference correctness, C speed | **TA-Lib** |
| **Streaming / live parity with backtest** | **`talipp`** (bit-identical incremental) |
| Pandas-native breadth, MIT | **`pandas-ta-classic`** (watch ADX warm-up) |
| Pure Python, no build step | `ta` — **with `fillna=False`**, and discard warm-up |
| Large panels, fastest | polars-based indicators, or vectorbt's built-ins |

## 7. Reference files

`references/<library>.md` carries the full per-indicator comparison tables, exact signatures and
gotchas. `references/_repainting.md` is the blacklist above with worked detections.

## Per-library deep dives

The optional `fin-libraries` plugin carries a dedicated skill for each library below. Load one
only after this skill has told you which library you want:

- **`lib-talib`** — talib
