---
name: sample-weights-and-uniqueness
description: >-
  Overlapping labels are not independent observations - compute concurrency, average uniqueness
  and return-attributed weights, and divide your t-statistics by the overlap factor before
  believing any of them. TRIGGER - sample weights, average uniqueness, concurrency, overlapping
  labels, numCoEvents, num_concurrent_events, getAvgUniqueness, get_av_uniqueness_from_triple_barrier,
  tW, sample_weight in fit(), getWeightsByReturn, get_weights_by_return, time decay weights,
  getTimeDecay, sequential bootstrap, seq_bootstrap, indicator matrix, effective sample size,
  "my labels overlap", "my t-stat is 4 but it does not hold up", "overlapping forward returns",
  Lopez de Prado chapter 4, AFML sample weights. SKIP for purged and embargoed cross-validation
  of the same labels (lib-purgedcv - purged CV lives there, do not re-implement it), for
  producing the labels and their t1 touch times (triple-barrier-labeling), for feature importance
  under overlap (feature-importance-financial), and for the multiple-testing deflation of a
  Sharpe ratio (backtest-validation).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Sample weights and uniqueness

**`len(y)` is not your sample size.** Advances in Financial Machine Learning (Lopez de Prado
2018), chapter 4. Label a 20-bar forward outcome at every bar and 1,980 rows arrive at `fit()`,
each sharing 95 % of its outcome with its neighbours. Every model, every standard error and every
confidence interval computed downstream is scaled by that 1,980. The shapes are right, the code
runs, and the reported significance is fiction.

Every number below is printed by `scripts/sample_weights.py` (numpy only, seed 0, **0.4 s**).
⚠️ Snippet and page numbers throughout are as cited in **mlfinpy 0.1.2's own docstrings**; what
this skill verified is the *code* — this skill did not have the book.

## 1. Concurrency and average uniqueness

For each bar, count the labels whose span covers it; for each label, average `1 / concurrency`
over its own span. ✅ Source-verified in mlfinpy 0.1.2 (`mlfinpy/sampling/concurrent.py`):

- `num_concurrent_events` (Snippet 4.1, page 60): `count.loc[t_in:t_out] += 1` — pandas label
  slicing is **inclusive at both ends**, so a "20-bar" label occupies **21** bars.
- `_get_average_uniqueness` (Snippet 4.2, page 62):
  `wght.loc[t_in] = (1.0 / num_conc_events.loc[t_in:t_out]).mean()`.

✅ Measured, 2,000 bars, labels spanning 20 bars, one event every `step` bars — and it matches the
analytic `min(1, step / (span + 1))` exactly away from the ends:

| `step` | events | mean concurrency | mean uniqueness | `min(1, step/(span+1))` |
|---|---|---|---|---|
| **1** | 1,980 | **21.000** | **0.0476** | 0.0476 |
| 2 | 990 | 10.500 | 0.0952 | 0.0952 |
| 5 | 396 | 4.200 | 0.2381 | 0.2381 |
| 10 | 198 | 2.100 | 0.4762 | 0.4762 |
| 20 | 99 | 1.050 | **0.9524** | 0.9524 |
| 40 | 50 | 0.525 | 1.0000 | 1.0000 |

🚨 **`step = span` is not unique.** Sampling every 20 bars with a 20-bar label still leaves
uniqueness at 0.9524, because the inclusive endpoints make adjacent labels share one bar. Sample
at `span + 1` if you want independence from spacing alone.

## 2. 🚨 The effective sample size, and the metric that does not measure it

✅ Measured at `step = 1`, `span = 20`, n = 1,980:

| quantity | value |
|---|---|
| nominal n | 1,980 |
| **sum of average uniqueness** | **95.2** (4.8 % of n) |
| mean average uniqueness | 0.0481 |
| Kish effective size of **uniform** weights | 1,980.0 |
| Kish effective size of the **uniqueness** weights | **1,953.4** |

🚨 **Kish's `(sum w)^2 / sum(w^2)` is the wrong tool here and it will reassure you.** It measures
the *dispersion* of the weights, not the overlap. Every uniqueness weight in this design is
approximately 0.0476, so the weights are nearly uniform and Kish reports 1,953 of 1,980 — a 1.3 %
haircut on a sample that is 95 % redundant. The number that carries the information is
`sum(uniqueness) = 95.2`.

## 3. 🚨 What unweighted fitting does to your evidence

The claim to test: with overlapping labels, an ordinary t-test is not a 5 % test. ✅ Measured over
**400 seeded paths**, driftless log prices, the label being the next 20 bars and the feature being
the *prior* 5-bar return — **independent of the label by construction**, so the true slope is
exactly zero:

| fit | sd of the reported t-statistic | rejection rate of a nominal 5 % two-sided test |
|---|---|---|
| **unweighted OLS** | **2.19** (a correct one is 1.00) | **39.2 %** |
| uniqueness-weighted OLS | 2.20 | 38.2 % |

🚨 **A nominal 5 % test rejects on 39 % of null paths, and weighting does not fix it.** Divide the
reported t by 2.19: a headline `t = 3.0` is really **1.37**, and the finding evaporates.

Two things worth reading carefully:

- ✅ **Weights are not a variance correction.** They change what each observation counts *for*;
  they do not change the correlation between overlapping residuals. The size barely moved
  (39.2 % → 38.2 %). The fixes are purged, embargoed cross-validation
  (`../../../fin-libraries/skills/lib-purgedcv/SKILL.md`) and an overlap-aware standard error
  (Newey-West / Hansen-Hodrick), not `sample_weight=`.
- ⚠️ **Summed uniqueness is the right direction and the wrong magnitude here.** The measured
  inflation implies an effective size of **412 of 1,980 (20.8 %)**, while summed uniqueness
  predicts **95 (4.8 %)** — **4.3x more pessimistic**. The feature is autocorrelated too, and the
  two dependencies do not simply multiply. Use `sum(uniqueness)` as an alarm, and measure the
  actual inflation by simulation before quoting a corrected t.

## 4. Return-attribution and time-decay weights

✅ Source-verified in mlfinpy 0.1.2 (`mlfinpy/sample_weights/attribution.py`):

- `_apply_weight_by_return` (Snippet 4.10, page 69):
  `weights.loc[t_in] = (ret.loc[t_in:t_out] / num_conc_events.loc[t_in:t_out]).sum()`, then
  `.abs()`, with `ret = np.log(close_series).diff()` — **log** returns, so the attributions are
  additive. `get_weights_by_return` then rescales with `weights *= weights.shape[0] / weights.sum()`,
  so the weights **sum to n**, not to 1.
- `get_weights_by_time_decay` (Snippet 4.11, page 70): the decay is linear in
  `av_uniqueness["tW"].sort_index().cumsum()` — **cumulative uniqueness**, not calendar time. A
  crowded stretch of history ages more slowly than a sparse one. `decay = 1` is no decay,
  `decay = 0` sends the oldest observation to zero, `decay < 0` erases the oldest fraction
  outright.

✅ Measured on the same 1,980 events:

| weighting | result |
|---|---|
| by \|return attribution\| | min 0.0016, median 0.8478, max 4.9821, sum 1,980.0 |
| its concentration | top 10 % of events carry **25.9 %** of the weight; the bottom half carry 21.1 % |
| its Kish effective size | **1,264.7** of 1,980 (63.9 %) |

| `decay` | oldest weight | newest | at exactly zero | Kish |
|---|---|---|---|---|
| +1.00 | 1.0000 | 1.0000 | 0.0 % | 1,980.0 |
| +0.50 | 0.5009 | 1.0000 | 0.0 % | 1,910.7 |
| 0.00 | 0.0018 | 1.0000 | 0.0 % | 1,492.7 |
| **-0.50** | 0.0000 | 1.0000 | **50.0 %** | **742.8** |

🚨 `decay = -0.5` **deletes half your training set**, exactly and silently. That is the documented
behaviour, not a bug — but "erased from memory" in a docstring is easy to read past, and the
resulting Kish size (743) is the only place it shows up.

## 5. Sequential bootstrap

✅ Source-verified: `seq_bootstrap` (mlfinpy `mlfinpy/sampling/bootstrapping.py`, cited to
Snippets 4.5-4.6, page 65) draws each label with probability proportional to its average
uniqueness **given the labels already drawn**, recomputed at every pick. `get_ind_matrix` builds
the bars x labels indicator, and `get_ind_mat_average_uniqueness` reports
`uniqueness[uniqueness > 0].mean()`.

✅ Measured, 80 labels of span 10 at random starts over 389 bars (whole-set uniqueness 0.3898),
5 draws each:

| draw | average uniqueness of the drawn sample |
|---|---|
| standard bootstrap | 0.3323 (sd 0.0089) |
| sequential bootstrap | **0.3584** (sd 0.0143) |
| ratio | **1.08x** |

⚠️ **The gain is real but small on this design**, and it is *zero* when the labels are regularly
spaced with identical spans — there is then no less-overlapping subset to find. Sequential
bootstrap pays when the spans vary (triple-barrier touch times do vary). Measure it on your own
`t1` before paying its cost: it is O(n) uniqueness recomputations per draw.

## 6. 🚨 The extra bar in the reference return attribution

`ret = np.log(close).diff()` puts the return *into* bar `t` at index `t`, and
`ret.loc[t_in:t_out]` therefore starts with a return the label did not earn — the move from
`t0 - 1` to `t0`. ✅ Measured: dropping that bar moves each weight by a **median 21.6 %** of its
own value (16.8 % of the mean weight), **72.6 %** of events move by more than 10 %, and the rank
correlation between the two weight vectors is **0.9090**. The *ordering* of the weights survives;
the individual weights do not. Decide which convention you want and be consistent, because the
two are not interchangeable at the level of a single sample's weight.

## 7. Traps

- 🚨 **Reporting `len(y)` as the sample size.** Section 2. Report `sum(uniqueness)` next to it,
  every time.
- 🚨 **Believing a t-statistic from overlapping labels.** Section 3, 39 % size at a nominal 5 %.
  This is the same failure as overlapping forward returns in a factor regression — see
  `../../../fin-models/skills/factor-models/SKILL.md`.
- 🚨 **Thinking `sample_weight=` fixes dependence.** It does not (section 3). Weights fix
  *attribution*; purging fixes *leakage*; a HAC standard error fixes *inference*. Three different
  problems.
- 🚨 **Kish's effective sample size on near-uniform weights.** Section 2. It reports 1,953 of
  1,980 on a sample that is 95 % redundant.
- 🚨 **`t1` must be the actual touch time.** Concurrency, uniqueness and every weight here are
  computed from `[t0, t1]`. Feed the vertical barrier instead of the barrier that was touched and
  every number on this page is wrong in the safe direction; feed `t0 + 1` and they are wrong in
  the dangerous one. `../triple-barrier-labeling/SKILL.md` produces the touch times.
- 🚨 **Inclusive endpoints.** Section 1: a "20-bar" label spans 21 bars in the reference
  implementation. It matters at the margin where you choose the sampling step.
- ⚠️ **Weights change what a metric means.** A weighted accuracy is not an accuracy; a
  return-attributed weight makes the model care about the large moves, which is usually what you
  want and is never what `accuracy_score` reports. State which weights produced which number.
- 🚨 **Do not re-implement purged cross-validation here.**
  `../../../fin-libraries/skills/lib-purgedcv/SKILL.md` owns it, including the
  `evaluation_times` argument that these same `t1` values feed.
- 🔴 **`pip install mlfinlab` cannot work** — ✅ checked 2026-09-09: the PyPI JSON API returns 404
  and the simple index lists zero files. The maintained fork is `mlfinpy` 0.1.2 (MIT), which
  ⚠️ pins `numpy<1.27`. The snippets above were read from its sdist.

## 8. Scripts

- `scripts/sample_weights.py` — concurrency by prefix sum, average uniqueness, return-attribution
  and time-decay weights, Kish effective size, the indicator matrix, sequential bootstrap, and
  the 400-path Monte Carlo that measures the true size of a nominal 5 % t-test on overlapping
  labels. numpy only. Seed 0, **0.4 s**.

## 9. Where this sits

`../triple-barrier-labeling/SKILL.md` produces the `t0`/`t1` spans everything here consumes.
`../../../fin-libraries/skills/lib-purgedcv/SKILL.md` owns purged and embargoed cross-validation
of exactly these labels — this skill is the weights, that one is the folds.
`../feature-importance-financial/SKILL.md` needs both: MDA scored on unpurged folds with
unweighted samples is the same error twice.
`../../../fin-core/skills/backtest-validation/SKILL.md` owns the multiple-testing deflation that
comes after the standard error is right. `../../../fin-models/skills/factor-models/SKILL.md` owns
the overlapping-return problem in its regression form.
