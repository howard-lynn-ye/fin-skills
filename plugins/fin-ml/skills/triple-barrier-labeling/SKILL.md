---
name: triple-barrier-labeling
description: >-
  Label a trade by which of profit-taking, stop loss and the holding-period limit is hit FIRST,
  with barriers scaled to the volatility at the event - instead of by the sign of the return N
  bars later, which describes a path you would have been stopped out of.
  TRIGGER - triple barrier, triple-barrier method, getEvents, get_events, getBins, get_bins,
  barrier_touched, add_vertical_barrier, pt_sl, vertical barrier, profit taking and stop loss
  labels, t1 touch time, getDailyVol, get_daily_vol, meta-label 0/1, fixed-time horizon labeling,
  fixed horizon labels, "label = sign of the 5-day forward return", "my classifier is 90 %
  accurate but loses money", "how do I label financial data for ML", Lopez de Prado chapter 3,
  AFML labeling. SKIP for weighting the overlapping labels this produces
  (sample-weights-and-uniqueness), for cross-validating them (lib-purgedcv), for the secondary
  model that sizes the bet (meta-labeling), for the CUSUM event sampler itself
  (structural-breaks), and for making the feature stationary (fractional-differentiation).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Triple-barrier labeling

**A fixed-horizon label tells the model that a trade which fell through its stop and recovered was
a winner.** Advances in Financial Machine Learning (Lopez de Prado 2018), chapter 3. The label is
the target the model optimises; if the label is about a path you could not have held, the model
learns to predict something you cannot trade, and no amount of cross-validation catches it because
the labels and the backtest agree with each other.

Every number below is printed by `scripts/triple_barrier.py` (numpy only, seed 0, **1.2 s**). The
test series is 6,000 bars of clustered volatility (log-vol AR(1), `phi = 0.99`) with **zero
drift**, so every asymmetry below is geometry, not alpha. Bars are integer indices; the book keys
everything by `DatetimeIndex`, which changes none of the arithmetic.

## 1. The three barriers, and why two of them must be volatility-scaled

Per event `t0`: a profit-taking barrier at `+pt * trgt`, a stop-loss barrier at `-sl * trgt`, and
a vertical barrier `vert` bars later. The label is **which one is touched first**.

✅ Source-verified against mlfinpy 0.1.2 (`mlfinpy/labeling/labeling.py`, `mlfinpy/util/volatility.py`):

- `triple_barriers` (AFML Snippet 3.2, page 45) computes
  `cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, "side"]` — **arithmetic**
  returns, signed by the side — and takes `cum_returns[cum_returns > profit_taking[loc]].index.min()`
  for the touch. Strict inequality, first index, so a bar that lands exactly on the barrier does
  not count.
- `get_daily_vol` (Snippet 3.1, page 44) is `(close / close.shift(1) - 1).ewm(span=lookback).std()`
  with **`lookback = 100`** — arithmetic returns, pandas' *adjusted* EWM, unbiased standard
  deviation. ✅ The script's numpy re-implementation matches pandas' `ewm(span=100).std()` to
  **9.7e-17**.
- `get_events` (Snippet 3.6, page 50) applies `target = target[target > min_ret]` **before**
  anything else: 🚨 `min_ret` silently **deletes** every event whose target volatility is below
  it. That is a sample filter that removes calm periods, not a label parameter.

**Why the target must be the local volatility.** ✅ Measured, 1,136 events, `pt_sl = (4, 4)`,
20-bar vertical barrier, split into volatility terciles by the EWMA target at the event:

| volatility tercile | resolve on the vertical barrier, **vol-scaled** | resolve on the vertical barrier, **fixed-width** |
|---|---|---|
| low | 39.8 % | **71.0 %** |
| mid | 37.6 % | 37.0 % |
| high | 46.2 % | **22.4 %** |
| low-to-high spread | **6.3 %** | **48.5 %** (7.7x) |

🚨 With one fixed barrier width, the *meaning of the label changes with the regime*: in the quiet
third of the sample 71 % of events time out and in the noisy third only 22 % do. The model learns
"class 0 means low volatility", which is a fact about the labeller, not about the future. The
realised per-bar volatility here spans **40x** (0.0018 to 0.0717), which is not extreme for real
data.

## 2. What the labels look like

✅ Same 1,136 events, same 20-bar horizon:

| labelling | -1 | 0 | +1 |
|---|---|---|---|
| triple barrier, `pt_sl = (4, 4)` | 26.1 % | **41.2 %** | 32.7 % |
| fixed horizon, threshold = the same `pt` barrier | 15.9 % | **62.2 %** | 21.8 % |
| fixed horizon, sign only (threshold 0) | 46.5 % | **0.0 %** | 53.5 % |

The sign-only version is the one most people write, and it has **no neutral class at all**: every
event is a directional bet, including the 41 % that the barrier method says never went anywhere.

## 3. 🚨 How often the horizon label is wrong about the path

The disagreement is not a rounding error. ✅ Measured, sign-only fixed-horizon labels against
triple-barrier labels over the identical 20-bar horizon and the identical barriers:

- the two labels **disagree on 535 of 1,136 events (47.1 %)**;
- **33 events (2.9 %)** are called +1 by the horizon although the 4-sigma stop was crossed on the
  way; **35 (3.1 %)** are called -1 although the profit target was crossed;
- **68 events (6.0 %)** describe a path you could not have held.

🚨 **And 6.0 % is the number for a very loose stop.** Tighten it and the horizon labels fall apart:

| stop (sigma of the event's target vol) | 1.0 | 2.0 | 3.0 | 4.0 |
|---|---|---|---|---|
| horizon labels whose path went through a barrier | **43.7 %** | 22.1 % | 11.8 % | 6.0 % |

At a 1-sigma stop — a perfectly ordinary risk limit — **44 % of the training labels describe
trades that no longer existed by the horizon.**

**What that costs in the backtest.** Take the ceiling: a *perfect* model of the fixed-horizon
labels, which is the best any model trained on them can be.

| how it is scored | mean return per event |
|---|---|
| the way it was trained: hold to the horizon, no stop | **+0.04243** |
| the same bets, with the same barriers honoured | **+0.03336** |
| **the gap** | **+0.00906 per event, 21 % of the apparent profit** |

That 21 % is not a modelling error, a cost assumption or a slippage estimate. It is profit the
backtest booked from drawdowns the risk limits would not have let you hold.

## 4. 🚨 The drawdown the horizon label assumes you sat through

✅ Among the 608 events the sign-only label calls +1, the worst drawdown before the horizon:

| statistic | in return | in units of the event's target volatility |
|---|---|---|
| median | -0.0082 | **-0.82 sigma** |
| 10th percentile | -0.0330 | — |
| worst | -0.1392 | **-11.09 sigma** |

**44.4 %** of these "winners" went through a 1-sigma stop first, and **5.4 %** through a 4-sigma
stop. The median winner spent part of its life 0.8 sigma under water. Every one of those is a
positive training example.

## 5. 🚨 `pt_sl` sets the class balance, and the class balance is not evidence

✅ Same events, same zero-drift series, only the barrier multiples change:

| `pt_sl` | -1 | 0 | +1 | mean barrier return |
|---|---|---|---|---|
| (1, 1) | 48.6 % | 1.0 % | 50.4 % | +0.00006 |
| (2, 1) | **58.6 %** | 3.8 % | 37.6 % | +0.00006 |
| (1, 2) | 34.4 % | 4.1 % | **61.4 %** | +0.00040 |
| (2, 2) | 42.0 % | 10.0 % | 48.0 % | +0.00096 |
| (1, 0) — no stop | 0.0 % | 29.7 % | **70.3 %** | -0.00000 |

🚨 A wide stop with a tight target produces **70 % positive labels on a series with no drift**. If
you report "the model predicts up moves 70 % of the time", you are reporting `pt_sl`. Class
balance, base rate and any accuracy figure derived from them are properties of the barrier
geometry. `pt_sl` and `vert` are researcher degrees of freedom: record them in
`../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`.

## 6. 🚨 Snippet 3.2 touches on arithmetic returns, Snippet 3.9 relabels on log returns

✅ Source-verified in mlfinpy 0.1.2. `triple_barriers` decides the touch with the **arithmetic**
return against `pt * trgt`. `barrier_touched` (Snippet 3.9, page 55), called from `get_bins`,
then re-derives the label from `out_df["ret"]`, which `get_bins` builds as a **log** return
(`np.log(prices.loc[t1]) - np.log(prices.loc[t0])`), and tests it against
`np.log(1 + target) * events.loc[date_time, "pt"]`.

Those two thresholds are equal only at `pt = 1`. ✅ Measured on the same events:

| `pt_sl` | horizontal touches | relabelled **0** (vertical) by the Snippet 3.9 test |
|---|---|---|
| (1, 1) | 1,125 | **0 (0.0 %)** |
| (2, 2) | 1,022 | 9 (0.9 %) |
| (3, 3) | 855 | **19 (2.2 %)** |

At target 2 % and `pt = 2`, the touch fires at `r > 0.0400` and the relabelling needs
`r > 0.0404`; a touch inside that band is recorded as a **vertical-barrier** label. Small, silent,
and it grows with the multiple. If you use `pt_sl` above 1, decide which convention you want and
apply it in both places.

## 7. Traps

- 🚨 **Fixed-horizon labels.** Sections 3-4. If you must use them, compute the maximum adverse
  excursion alongside and drop or reweight the ones that breached your stop.
- 🚨 **A fixed barrier width across regimes.** Section 1. The label's meaning drifts with
  volatility, and the model learns the regime.
- 🚨 **`min_ret` is a sample filter.** `get_events` drops events, it does not just skip barriers.
  Whatever it removes is missing from the training set and from every count you report.
- 🚨 **`t1` is the touch time, not the vertical barrier.** Everything downstream depends on it:
  purging, embargoing, uniqueness and weights all need the bar where the label actually resolved.
  Passing the vertical barrier instead over-purges; passing `t0 + 1` under-purges and silently
  reintroduces the leak — `../../../fin-libraries/skills/lib-purgedcv/SKILL.md` owns that, and
  `evaluation_times` is where the touch time goes.
- 🚨 **These labels overlap, so the samples are not i.i.d.** A 20-bar horizon sampled every 5 bars
  means four labels share every bar. Weight them by uniqueness before fitting
  (`../sample-weights-and-uniqueness/SKILL.md`) and cross-validate them with purging and an
  embargo (`../../../fin-libraries/skills/lib-purgedcv/SKILL.md`). Neither is optional.
- 🚨 **The barrier is checked on the close, but you would have been stopped intrabar.** This
  script — and the book's snippet — scan closes. A real stop is a resting order against the low or
  the high, so the true touch is earlier and worse than what closes report.
  `../../../fin-core/skills/intraday-microstructure/SKILL.md` owns the bar-versus-tick question.
- ⚠️ **Accuracy on triple-barrier labels is still not profit.** The classes are unbalanced by
  construction (section 5), and each event's return is a different size. Score with the
  return-weighted metrics and the barrier returns, not with `accuracy_score`.
- 🚨 **The event sampler is part of the label.** Sampling every 5 bars, as here, is a placeholder;
  the book samples on a CUSUM filter so that events cluster where something happened
  (`../structural-breaks/SKILL.md`). Changing the sampler changes the label distribution and the
  effective sample size.
- 🔴 **`pip install mlfinlab` cannot work** — ✅ checked 2026-09-09, the PyPI JSON API returns 404
  and the simple index lists zero files. The maintained fork is `mlfinpy` 0.1.2 (MIT), which
  ⚠️ pins `numpy<1.27`. The snippets above were read from its sdist, not from an installed copy.

## 8. Scripts

- `scripts/triple_barrier.py` — the causal EWMA target volatility checked against pandas, the
  first-touch scan, triple-barrier and meta-labels, fixed-horizon labels, the volatility-tercile
  table, the path-conflict counts and the stop-tightness sweep, maximum adverse excursion, the
  `pt_sl` class-balance table, and the Snippet 3.2 / 3.9 return-convention mismatch. numpy only
  (pandas is used once, inside a function, purely to check the EWMA). Seed 0, **1.2 s**.

## 9. Where this sits

`../sample-weights-and-uniqueness/SKILL.md` takes the `t1` touch times from here and turns them
into concurrency, uniqueness and sample weights. `../meta-labeling/SKILL.md` takes the `side`
argument of `get_events` and adds the secondary model.
`../structural-breaks/SKILL.md` owns the CUSUM filter that should choose the events.
`../fractional-differentiation/SKILL.md` is the feature side of the same pipeline.
`../../../fin-libraries/skills/lib-purgedcv/SKILL.md` owns purged and embargoed cross-validation
for the overlapping labels this produces — never re-implement it.
`../../../fin-core/skills/backtest-validation/SKILL.md` owns the trial ledger for `pt_sl` and
`vert`. `../../../fin-core/skills/signal-construction/SKILL.md` owns the causality check on the
features that go with these labels.
