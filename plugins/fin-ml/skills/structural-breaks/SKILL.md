---
name: structural-breaks
description: >-
  Sample events with the symmetric CUSUM filter instead of on a clock, and test for explosive
  behaviour with SADF instead of one full-sample ADF that has no power against a bubble in part
  of the sample. TRIGGER - CUSUM filter, cusum_filter, symmetric CUSUM, event-based sampling,
  t_events, event sampling for labels, SADF, supremum ADF, sup ADF, get_sadf, BSADF, backward
  SADF, Phillips-Shi-Yu, explosiveness test, bubble detection, structural break test, Chow test,
  "how do I pick the bars to label", "my ADF says nothing but the chart is obviously a bubble",
  "when did the regime change", Lopez de Prado chapter 17, AFML structural breaks. SKIP for
  labelling the events once they are sampled (triple-barrier-labeling), for making a feature
  stationary (fractional-differentiation), for the cointegration ADF table
  (stat-arb-cointegration), for hidden-state / volatility regime labels (regime-detection), and
  for unit-root and ARIMA selection generally (time-series-forecasting-models).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Structural breaks: CUSUM sampling and SADF

**Two tools from the same chapter answering different questions.** CUSUM decides *when to sample*;
SADF decides *whether a segment is explosive*. Advances in Financial Machine Learning (Lopez de
Prado 2018), chapter 2 section 2.5.2 and chapter 17. Both are cheap and both have a property
people read past — CUSUM's detection lag, and SADF's refusal to turn off after the bubble ends.

Every number below is printed by `scripts/structural_breaks.py` (numpy, seed 0, **17 s**,
`statsmodels` optional). ⚠️ Snippet and page numbers are as cited in mlfinpy 0.1.2's docstrings;
what is verified here is the code.

## 1. The symmetric CUSUM filter, exactly

✅ Source-verified in mlfinpy 0.1.2 (`mlfinpy/filters/filters.py`, `cusum_filter`, cited to
AFML Snippet 2.4 page 39):

```
s_pos = max(0, s_pos + log_ret[t])
s_neg = min(0, s_neg + log_ret[t])
if   s_neg < -threshold:  s_neg = 0; fire
elif s_pos >  threshold:  s_pos = 0; fire
```

Three details that decide the behaviour and are easy to miss:

- 🚨 **only the side that fired is reset.** The opposite accumulator keeps its state, so the
  filter stays "loaded" in the other direction;
- 🚨 **the branches are `elif`**, so at most one event per bar, and the **negative side is tested
  first** — a bar that would trip both is recorded as a downside event;
- ✅ `threshold` may be a scalar **or a `pd.Series`**, which is how a volatility-scaled threshold
  is passed; anything else raises `ValueError("threshold is neither float nor pd.Series!")`.

## 2. What it costs and what it buys

✅ Measured on 2,000 bars of i.i.d. returns (`sigma = 0.01`) with a **+0.5 sigma per bar drift
planted over bars 1200-1319**:

| threshold | events | bars sampled | mean gap | max gap | clustered (≤3 bars apart) |
|---|---|---|---|---|---|
| 0.5 sigma | 1,317 | 65.8 % | 1.5 | 6 | 99.9 % |
| 1.0 sigma | 866 | 43.3 % | 2.3 | 9 | 97.5 % |
| 2.0 sigma | 405 | 20.2 % | 4.9 | 19 | 60.7 % |
| 3.0 sigma | 238 | 11.9 % | 8.4 | 34 | 24.4 % |
| **5.0 sigma** | **114** | **5.7 %** | 17.5 | **53** | **1.8 %** |

✅ **The non-clustering claim, matched on sampling rate rather than on threshold**: a naive
`|return| > 2 sigma` filter fires **91 times (4.5 % of bars) with 24.2 % of its events within 3
bars of another**; CUSUM at 5 sigma fires **114 times (5.7 %) with 1.8 % clustered**. CUSUM
requires a *run* of one-signed returns, so it does not re-trigger while the series hovers at the
threshold — that is the property the book claims for it, and it holds at a comparable event count.

🚨 **The detection lag is the price, and it grows with the threshold.** ✅ Measured, first event at
or after the planted shift:

| threshold | first event | lag (bars) | events inside the 120-bar shift | uniform sampling would give |
|---|---|---|---|---|
| 0.5 sigma | 1200 | **0** | 80 | 79.0 |
| 1.0 sigma | 1202 | 2 | 53 | 52.0 |
| 2.0 sigma | 1200 | **0** | 29 | 24.3 |
| 3.0 sigma | 1208 | 8 | 19 | 14.3 |
| **5.0 sigma** | 1217 | **17** | 10 | 6.8 |

Read both columns together. A 5-sigma threshold samples 5.7 % of bars, waits **17 bars** to notice
the shift, and still places **1.47x** as many events inside the shift window as uniform sampling
would. A 0.5-sigma threshold is instant and samples two thirds of the bars, which is not sampling.
🚨 **The threshold is a researcher degree of freedom that changes the training set**: record it in
`../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`, and note that changing it
changes the label distribution downstream (`../triple-barrier-labeling/SKILL.md`) and the
concurrency (`../sample-weights-and-uniqueness/SKILL.md`).

## 3. SADF: the supremum ADF

`SADF(t) = sup over backwards-expanding start points of the ADF t-statistic of the window ending
at t` — AFML Snippets 17.1-17.4, pages 258-259. Testing `b` in
`dy_t = a + b y_{t-1} + sum_j phi_j dy_{t-j} + e`, with **large positive** values indicating
explosive behaviour (the opposite direction to a unit-root rejection).

✅ Source-verified in mlfinpy 0.1.2 (`mlfinpy/structural_breaks/sadf.py`), two things worth
knowing before you call it:

- 🚨 `get_sadf`'s signature is `(series, model, lags, min_length, add_const=False, phi=0, ...)`
  and `_get_y_x` builds the trend column **once over the whole sample**
  (`x["trend"] = np.arange(x.shape[0])`), not per window. With `model="linear"` and the default
  `add_const=False` you get **a trend with no intercept**, which is not the specification most
  references mean by "SADF with a constant".
- ✅ For the sub/super-martingale models (`model[:2] == "sm"`) it divides by `y.shape[0] ** phi`
  — the length penalty — and only for those; `phi` is ignored for `linear` and `quadratic`.

The script implements the conventional `nc`/`c`/`ct` specifications and solves every
`(start, end)` window **exactly from cumulative cross-product matrices** instead of refitting.
✅ Checked against a naive least-squares refit at one end point: **max abs diff 1.5e-14**.

## 4. ✅ What SADF finds that a full-sample ADF does not

700 bars: a random walk except for an explosive segment (`rho = 1.02`) over **bars 350-499**.
Lags 1, `min_length` 60, trend `c`. Critical values simulated under the null of a driftless random
walk, 40 paths of 350 bars: **max SADF 90/95/99 % = 1.959 / 2.126 / 2.712**.

| test | statistic | verdict |
|---|---|---|
| `statsmodels.adfuller(p, maxlag=1, autolag=None, regression="c")` on the whole series | **+1.278** vs a 5 % value of -2.866 | **no rejection** |
| the same statistic from this script's design | +1.278 | (identical) |
| **SADF, max over sub-windows** | **+15.582** at bar 499 | far past the 95 % value of 2.126 |

🚨 **The full-sample ADF sees nothing.** The bubble is a fifth of the sample and averaging it
against the rest is exactly what destroys the power. That is the argument for SADF in two rows.

✅ **What it flags**: contiguous runs above the 95 % value are `[(387, 395), (397, 635), (645, 645)]`.

- **112 of the 150 planted bars are caught (recall 75 %)**;
- the first flag is at bar **387, +37 bars after the bubble starts** — SADF needs `min_length`
  observations of the new regime before it can see it, so a real-time bubble alarm is late by
  construction;
- 🚨 the last flag is at bar **645, +146 bars past the bubble's END**. SADF takes the supremum
  over **all** start points, so once a bubble is inside the sample every later end point can
  still find it. **A flagged bar does not mean "explosive now"; it means "an explosive window
  ends at or before now".** Phillips-Shi-Yu's *backward* SADF bounds the window for exactly this
  reason; `get_sadf` as written is the unbounded version.

## 5. 🚨 The critical value is for the supremum, not for a bar

✅ Measured on the null:

- one driftless random walk of 700 bars: max SADF **1.801**, **0.0 %** of its bars above the 95 %
  value, no flagged runs;
- across the 40 null paths the value was calibrated on: **5.0 % of PATHS** exceed it (5 % by
  construction) but only **0.2 % of all BARS** do.

Those are different quantities. A threshold calibrated on the supremum over a sample controls the
chance that *some* bar in that sample flags; it says nothing about the rate at which single bars
flag, and reading a per-bar flag as "a 5 % test" over-states the evidence for every extra bar you
look at.

⚠️ **And the critical value is not a constant.** It depends on the sample length, on
`min_length`, on the lag order and on the trend specification. There is no table to copy; simulate
it for your own configuration, as the script does. The values above are for 350-bar paths with
`min_length = 60` and one lag.

## 6. Traps

- 🚨 **Reading SADF's flagged range as the bubble's extent.** Section 4: +37 bars late at the
  start, +146 late at the end. Use it to say a bubble *happened*, not to date its end.
- 🚨 **Looking up an SADF critical value.** Section 5. Simulate it for your sample.
- 🚨 **Reading a per-bar SADF exceedance as a 5 % event.** Section 5.
- 🚨 **The CUSUM threshold silently changes your dataset.** Section 2: 5.7 % of bars at 5 sigma
  against 65.8 % at 0.5 sigma. Every downstream count — labels, concurrency, uniqueness, trials —
  moves with it.
- 🚨 **Both tools are causal only if you feed them causally.** CUSUM as written uses `log_ret[t]`
  to fire an event *at* `t`, which is fine for sampling but means the event's features must be
  computed from data through `t` and earn `t -> t+1`. Check with
  `../../../fin-core/skills/signal-construction/scripts/assert_causal.py`.
- 🚨 **A volatility-scaled CUSUM threshold needs a causal volatility.** Passing a full-sample
  standard deviation as the threshold leaks the future into which bars were sampled. Use a
  trailing EWMA, the same one the barriers use
  (`../triple-barrier-labeling/SKILL.md`).
- ⚠️ **CUSUM finds runs, not variance changes.** A regime that doubles volatility without changing
  the mean produces more events at a fixed threshold, but the filter is not testing for it. For
  volatility states use `../../../fin-core/skills/regime-detection/SKILL.md`.
- ⚠️ **SADF is O(T^2) regressions if written naively.** The script solves every window from
  cumulative cross-products; mlfinpy parallelises the naive version instead. On 10,000 bars the
  difference is minutes against hours.
- 🔴 **`pip install mlfinlab` cannot work** — ✅ checked 2026-09-09: the PyPI JSON API returns 404
  and the simple index lists zero files. The maintained fork is `mlfinpy` 0.1.2 (MIT), which
  ⚠️ pins `numpy<1.27`. Its `structural_breaks` and `filters` modules were read from the sdist.

## 7. Scripts

- `scripts/structural_breaks.py` — the CUSUM filter and a naive threshold filter for contrast,
  event clustering and detection lag, the ADF design matrix, an exact SADF from cumulative
  cross-products (checked against a naive refit), a simulated null and its critical values, and
  the flagged-run comparison against a planted bubble. numpy only; `statsmodels` is imported
  inside one function for the full-sample ADF cross-check. Seed 0, **17 s**.

## 8. Where this sits

`../triple-barrier-labeling/SKILL.md` consumes the CUSUM events as `t_events` and turns them into
labels. `../sample-weights-and-uniqueness/SKILL.md` owns the concurrency the sampling rate
determines. `../fractional-differentiation/SKILL.md` uses the same ADF machinery for
*stationarity* rather than explosiveness, and its d-scan is the feature-side companion to this.
`../../../fin-models/skills/stat-arb-cointegration/SKILL.md` owns the ADF table for fitted
residuals. `../../../fin-core/skills/regime-detection/SKILL.md` owns hidden-state and volatility
regimes, and the filtered-versus-smoothed timing rule that applies to any break date.
`../../../fin-core/skills/backtest-validation/SKILL.md` owns the trial ledger for the threshold.
