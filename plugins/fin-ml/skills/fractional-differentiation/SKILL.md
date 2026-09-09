---
name: fractional-differentiation
description: >-
  Make a price series stationary without throwing away the memory a model needs - the weight
  recursion, the fixed-width window, and the scan for the smallest d that passes ADF.
  TRIGGER - fractional differentiation, fractional differencing, fracdiff, frac_diff,
  frac_diff_ffd, get_weights_ffd, plotMinFFD, plot_min_ffd, "minimum d", ARFIMA, long memory,
  (1-B)^d, binomial weights, fixed-width window fracdiff; "my model only sees returns", "prices
  are non-stationary so I differenced them", "ADF says my feature is non-stationary", "should I
  feed prices or returns to the model", Lopez de Prado chapter 5, AFML fracdiff. SKIP for the
  cointegration ADF table and the fitted-residual null (stat-arb-cointegration), for unit-root
  and ARIMA model selection generally (time-series-forecasting-models), for making a feature
  causal and its warm-up (signal-construction), and for labels rather than features
  (triple-barrier-labeling).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Fractional differentiation

**Returns are stationary and memoryless; prices remember and fail every stationarity test. `d`
is the dial between them, and almost nobody turns it.** Advances in Financial Machine Learning
(Lopez de Prado 2018), chapter 5. The default pipeline sets `d = 1` by reflex — `df.pct_change()`
— and that is the maximum possible amount of differencing, not the minimum needed.

Every number below is printed by `scripts/frac_diff.py` (numpy + scipy, seed 0, **3.7 s**,
`statsmodels` optional). The test series is a mildly non-stationary log price: an AR(1) with
`phi = 0.999`, T = 3,000, so `E[r_{t+1} | p_t] = (phi - 1) p_t` **exactly** — the level is the
entire edge and a single return contains almost none of it. That is the DGP on which "differencing
destroys memory" is a measurable statement rather than a slogan.

## 1. The weight recursion, and why it is not a moving average

`(1 - B)^d` expanded as a binomial series gives weights on the **levels**, newest first:

```
w[0] = 1        w[k] = -w[k-1] * (d - k + 1) / k
```

✅ Measured: the recursion equals the closed form `w_k = (-1)^k C(d, k)` (`scipy.special.binom`,
an independent route) to **1.1e-16** over the first 500 terms at d = 0.15, 0.40 and 0.75.
✅ `d = 1` returns exactly `[1, -1, 0, 0, ...]` — the first difference — and `d = 0` the identity.

| d | first six weights (newest first) |
|---|---|
| 0.25 | +1.000000 -0.250000 -0.093750 -0.054688 -0.037598 -0.028198 |
| 0.50 | +1.000000 -0.500000 -0.125000 -0.062500 -0.039062 -0.027344 |
| 0.75 | +1.000000 -0.750000 -0.093750 -0.039062 -0.021973 -0.014282 |

The tail decays like `k^{-(1+d)}`, so it never terminates for non-integer `d`. Two fixes exist and
they are different features:

- **expanding window** (book section 5.4.1): weights recomputed to the start of the sample, so
  each output point uses a different number of terms. The book notes this "leads to negative
  drift caused by an expanding window's added weights".
- **fixed-width window / FFD** (section 5.4.2, page 83): truncate at the first `|w_k| < thresh`
  and use the same weight vector everywhere. **This is the one to use** — every output point is
  the same linear filter, so the series is comparable with itself over time.

✅ Source-verified against mlfinpy 0.1.2 (`mlfinpy/util/frac_diff.py`). Its `get_weights` runs
this recursion and returns `np.array(weights[::-1]).reshape(-1, 1)` — **oldest first, as an
(N, 1) column**, the opposite orientation to the array above. Its `get_weights_ffd(diff_amt,
thresh, lim)` `break`s *before* appending the first sub-threshold weight, so the window length
matches. `scripts/frac_diff.py` carries that function transcribed verbatim and checks the two
against each other: **max abs diff 0.0e+00** at d = 0.05, 0.20, 0.35, 0.50, 0.75, 1.00.

🚨 **`sum(w)` is the whole point and it is not zero.** Truncating at a threshold leaves a residual
weight on the level, and *that residual is the memory*:

| d | window width | `sum(w)` | last \|w\| |
|---|---|---|---|
| 0.05 | 4 | +0.899427 | 1.14e-02 |
| 0.20 | 10 | +0.537678 | 1.10e-02 |
| 0.35 | 11 | +0.308786 | 1.01e-02 |
| 0.50 | 9 | +0.185471 | 1.09e-02 |
| 0.75 | 6 | +0.070816 | 1.01e-02 |
| 1.00 | 1 | **+0.000000** | — |

At `d = 1` the weights sum to zero exactly: the output is a pure difference and carries no level
at all. Everything below follows from that one row.

## 2. The scan: the smallest d that passes ADF

The book's procedure (section 5.6, page 85; mlfinpy's `plot_min_ffd`) is: sweep `d`, run
`adfuller(..., maxlag=1, autolag=None, regression="c")` on the FFD series, and plot the ADF
statistic against the correlation with the original level. ✅ The script's own ADF t-statistic
matches `statsmodels` 0.15.0's to **+3.3e-15** on this series (own -1.6860, statsmodels -1.6860,
its 5 % value -2.863).

✅ Measured, seed 0, `thresh = 0.01`, T = 3,000:

| d | width | ADF t | rejects at 5 %? | corr with the level | IC vs next return |
|---|---|---|---|---|---|
| 0.00 | 0 | -1.686 | no | 1.000 | -0.0306 |
| 0.10 | 7 | -2.037 | no | 1.000 | -0.0318 |
| 0.20 | 10 | -2.675 | no | 0.998 | -0.0308 |
| **0.25** | 11 | **-3.183** | **yes** | **0.995** | -0.0298 |
| 0.35 | 11 | -4.455 | yes | 0.985 | -0.0297 |
| 0.50 | 9 | -7.200 | yes | 0.955 | -0.0300 |
| 0.80 | 5 | -19.500 | yes | 0.709 | -0.0243 |
| **1.00** | 1 | **-38.902** | yes | **0.034** | **-0.0046** |

✅ Bisected to tolerance 0.005, the minimum d that passes is **d\* = 0.219** (ADF t -2.885, window
11 bars), and at d\* the correlation with the level is **0.996**. The book's claim reproduces:
**stationarity is bought at d ≈ 0.22 and costs 0.4 % of the correlation with the price level.**

## 3. 🚨 What `d = 1` costs, in the units that matter

One path cannot separate an information coefficient of 0.03 from zero (its standard error at
T = 3,000 is 0.018), so the script repeats the scan over **50 independent seeded paths**:

| d | width | mean ADF t | ADF rejection rate | mean corr(level) | mean IC (s.e.) |
|---|---|---|---|---|---|
| 0.00 | 0 | -1.97 | **0.06** | 1.000 | -0.0359 (0.0015) |
| 0.15 | 9 | -2.68 | 0.26 | 0.999 | -0.0358 (0.0015) |
| 0.25 | 11 | -3.63 | 0.82 | 0.994 | -0.0353 (0.0015) |
| **0.35** | 11 | -4.98 | **1.00** | **0.982** | **-0.0347** (0.0015) |
| 0.50 | 9 | -7.83 | 1.00 | 0.946 | -0.0332 (0.0015) |
| 0.70 | 6 | -14.19 | 1.00 | 0.829 | -0.0288 (0.0017) |
| **1.00** | 1 | -38.57 | 1.00 | **0.035** | **-0.0010** (0.0028) |

- ✅ The level's ADF rejects on **6 %** of paths - the series genuinely is near-unit-root and the
  test says so at its nominal size. Nothing is rigged.
- ✅ At `d = 0.35` the ADF rejects on **100 %** of paths, and the feature keeps **97 % of the
  level's information coefficient** (-0.0347 against -0.0359) and **0.982** correlation with it.
- 🚨 At `d = 1` the feature keeps **3 %** of it: mean IC **-0.0010**, which is **0.4 standard
  errors from zero**. The correlation with the level is 0.035.

**That is the trap in one line: first differencing passed the stationarity test and deleted the
signal, and nothing in the pipeline said so.** A model fitted on returns here has no edge to find;
the same model on the `d = 0.35` feature has essentially the whole edge, from a feature that also
passes ADF on every path.

The decay is monotone in `d` across the whole panel, so `d` is not a hyperparameter to tune on a
validation score — there is no interior optimum to find. Take the **smallest** value that clears
the test.

## 4. 🚨 The weight threshold is a second, undocumented knob

`thresh` sets the window width, and the width sets how much level survives. Two people who both
"used d = 0.35" have different features. ✅ Measured on seed 0:

| `thresh` | width at d = 0.35 | usable obs | corr(level) at d = 0.35 | IC | d\* | width at d\* |
|---|---|---|---|---|---|---|
| 1e-2 | 11 | 2,989 | **0.985** | -0.0297 | **0.219** | 11 |
| 1e-3 | 60 | 2,940 | 0.916 | -0.0321 | 0.113 | 65 |
| 1e-4 | 331 | 2,669 | 0.774 | -0.0222 | 0.113 | 517 |
| 1e-5 | **1,825** | **1,175** | **0.610** | -0.0542 | **0.078** | 1,999 |

🚨 The window moves by **165x** across these four, the correlation with the level falls
**0.985 -> 0.610 at the same d = 0.35**, and the same series reports **four different minimum d**
(0.219, 0.113, 0.113, 0.078). At `thresh = 1e-5` only **1,175 of 3,000** bars survive the warm-up.

⚠️ The two thresholds in circulation come from the same file: mlfinpy's `frac_diff_ffd` defaults
to `thresh=1e-5`, while `plot_min_ffd` — the routine that *finds* d — calls it with
`thresh=0.01`. ✅ Source-verified in `mlfinpy/util/frac_diff.py` 0.1.2. **Search for d and build
the feature at the same threshold, and record it next to d.**

## 5. Traps

- 🚨 **`d = 1` by reflex.** Sections 2-3. `pct_change()` is the maximum-differencing corner of a
  continuum; measure what it costs before accepting it.
- 🚨 **The window is warm-up, and it is longer than you think.** The first `width` values are
  undefined. At `thresh=1e-5, d=0.35` that is 1,825 bars of a 3,000-bar sample. Run the feature
  through `../../../fin-core/skills/signal-construction/scripts/warmup_probe.py`; a run that starts
  at bar 0 with `fillna(0)` is trading a partial filter.
- 🚨 **Fit the weights, not the data.** The weights depend only on `d` and `thresh`, never on the
  sample, so the filter itself cannot leak. But **choosing `d` on the full sample does**: the ADF
  statistic that picked d\* used the test period. Pick `d` on a prior window, or record the scan
  as trials in
  `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py` — a 21-point `d` grid
  crossed with four thresholds is 84 trials.
- 🚨 **Difference the log price, not the price.** The recursion is linear, so applying it to raw
  prices makes the feature's scale proportional to the price level; a stock that doubles doubles
  its feature. Take logs first, as the book's own routine does
  (`np.log(series[["close"]])` in `plot_min_ffd`).
- 🚨 **ADF is not the only stationarity failure.** The FFD series passes a unit-root test; it can
  still have a variance that changes by an order of magnitude across regimes, which breaks a
  model just as thoroughly. Pair the ADF with a rolling-variance plot, and see
  `../../../fin-core/skills/regime-detection/SKILL.md`.
- ⚠️ **This is one series' answer, not a constant.** d\* = 0.219 here, 0.35 for the book's E-mini
  example. Re-scan per instrument and per sample; a `d` copied from a paper is a guess.
- 🚨 **The ADF table is the single-series one, and it is only valid because nothing was fitted
  first.** If your feature is a regression residual instead, the table is wrong and over-rejects
  — `../../../fin-models/skills/stat-arb-cointegration/SKILL.md` measures that (16.5 % rejection
  at a nominal 5 %).
- 🔴 **`pip install mlfinlab` cannot work.** ✅ Checked 2026-09-09: `https://pypi.org/pypi/mlfinlab/json`
  returns **HTTP 404** and `https://pypi.org/simple/mlfinlab/` returns a page with **zero
  distribution links**. The maintained fork that carries these chapters is **`mlfinpy` 0.1.2**
  (MIT, released 2024-10-09, `baobach/mlfinpy`) — ⚠️ and it pins **`numpy<1.27`**, so it will not
  install beside numpy 2.x. That is why this skill transcribes the reference code instead of
  importing it.

## 6. Scripts

- `scripts/frac_diff.py` — the weight recursion and its closed form, the FFD truncation against
  mlfinpy's transcribed `get_weights_ffd`, the causal FFD filter, an ADF t-statistic checked
  against `statsmodels`, the single-path and 50-path `d` scans, the bisection for d\*, and the
  threshold table. numpy + scipy; `statsmodels` is imported inside one function and the demo
  prints its own MacKinnon value without it. Seed 0, **3.7 s**.

## 7. Where this sits

`../triple-barrier-labeling/SKILL.md` is the label side of the same chapter's pipeline; this
skill is the feature side. `../structural-breaks/SKILL.md` uses the same ADF machinery for
explosiveness rather than stationarity.
`../../../fin-models/skills/time-series-forecasting-models/SKILL.md` owns unit-root testing and
ARIMA order selection in general.
`../../../fin-models/skills/stat-arb-cointegration/SKILL.md` owns the ADF table for *fitted*
residuals. `../../../fin-core/skills/signal-construction/SKILL.md` owns `assert_causal` and
`warmup_probe`, both of which apply to any FFD feature.
`../../../fin-core/skills/backtest-validation/SKILL.md` owns the trial count for the `d` scan.
`../../../fin-libraries/skills/lib-purgedcv/SKILL.md` owns the cross-validation once these
features meet overlapping labels.
