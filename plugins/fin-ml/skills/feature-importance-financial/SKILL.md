---
name: feature-importance-financial
description: >-
  Rank features without believing MDI - it is in-sample, it favours columns with many distinct
  values, and it splits credit between substitutable features; MDA under-states collinear pairs
  and leaks outright on a shuffled k-fold. TRIGGER - feature_importances_, feature importance,
  MDI, mean decrease impurity, Gini importance, MDA, mean decrease accuracy, permutation
  importance, permutation_importance, single feature importance, SFI, clustered feature
  importance, MDA with clustering, "which features matter", "my random forest says this random
  column is important", "importance changes every run", correlated features importance,
  substitution effect, Lopez de Prado chapter 8, AFML feature importance. SKIP for purged and
  embargoed cross-validation itself (lib-purgedcv), for the overlapping-label weights that go
  with it (sample-weights-and-uniqueness), for feature construction and causality
  (signal-construction), for factor-return attribution rather than model attribution
  (factor-models), and for counting the trials a feature search spends (backtest-validation).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Feature importance for financial data

**Three methods, three different lies, and the worst of them is the default.** Advances in
Financial Machine Learning (Lopez de Prado 2018), chapter 8. `feature_importances_` is one
attribute access away and it is in-sample, biased toward high-cardinality columns, and split
between features that substitute for each other. Permutation importance fixes the first two and
not the third. And every one of them is meaningless if the fold was shuffled.

Every number below is printed by `scripts/feature_importance.py` (numpy, seed 0, **23 s**,
`scikit-learn` optional — the script carries its own CART and bagged forest, so the claims hold
on a bare install and are *checked against* sklearn when it is present).

## 1. The design, so "wrong" is measurable

n = 1,500. `y = 1.00 * x0 + 0.60 * x2 + e`, where `e` is a rolling mean of 25 shocks — each row's
target is fully explained by its own features, and neighbouring rows share most of their
*residual*. That is what an overlapping label horizon does, and it is what makes section 5 happen.

| feature | distinct values | corr with x0 | corr with y | truth |
|---|---|---|---|---|
| **x0** | 1,500 | 1.000 | 0.425 | informative, coefficient 1.00 |
| **x1** | 1,500 | **0.995** | 0.426 | a noisy copy of x0 — **no independent information** |
| **x2** | 1,500 | 0.005 | 0.293 | informative, coefficient 0.60 |
| **x3** | **1,500** | 0.032 | 0.016 | **irrelevant**, continuous |
| **x4** | **2** | 0.007 | 0.011 | **irrelevant**, binary |
| **x5** | **5** | 0.018 | 0.024 | **irrelevant**, five levels |
| **x6** | 1,500 | 0.017 | -0.092 | **irrelevant**, a random walk |

## 2. ✅ What sklearn's `feature_importances_` actually computes

The per-node formula lives in the compiled `_tree.pyx`, which the wheel does not ship — so it is
verified **numerically** here, by rebuilding it from the public `tree_` arrays:

```
imp[feature[i]] += wn[i]*impurity[i] - wn[L]*impurity[L] - wn[R]*impurity[R]   # internal nodes
imp /= wn[root];  imp /= imp.sum()
```

✅ Against sklearn 1.4.2 on the same data: **max abs diff 0.00e+00** for a single
`DecisionTreeRegressor`, and **0.00e+00** for a `RandomForestRegressor` when the per-tree
importances are aggregated as `_forest.py` does — ✅ source-verified there:
`np.mean(all_importances, axis=0)` over trees with `node_count > 1`, then `/ np.sum(...)`.

✅ Source-verified in `sklearn/tree/_classes.py` and `sklearn/ensemble/_forest.py`, the docstring
of `feature_importances_` already says it: "impurity-based feature importances can be misleading
for high cardinality features (many unique values)", and points at
`sklearn.inspection.permutation_importance`. **The warning is in the docstring nobody opens, not
in a runtime warning.**

## 3. 🚨 MDI: three failures at once

✅ Measured, 25 trees, depth 5, `max_features` 3 of 7, `min_samples_leaf` 20:

| feature | MDI | rank | truth |
|---|---|---|---|
| x0 informative | 0.3023 | 2 | informative |
| **x1 collinear copy** | **0.3054** | **1** | **no information** |
| x2 informative | 0.2396 | 3 | informative |
| x3 noise, high card | **0.0261** | 5 | irrelevant |
| x4 noise, binary | **0.0063** | 7 | irrelevant |
| x5 noise, 5 levels | 0.0075 | 6 | irrelevant |
| **x6 noise, random walk** | **0.1127** | **4** | **irrelevant** |

- 🚨 **The copy outranks the original.** x1 is x0 plus a little noise and MDI puts it first. The
  pair takes **60.8 %** of the total between them — the group importance is right, the split
  inside it is arbitrary, and it changes with the seed.
- 🚨 **Cardinality bias, measured: x3 scores 4.1x x4**, and both are pure noise. The only
  difference between them is that one has 1,500 distinct values to try thresholds on and the
  other has 2.
- 🚨 **x6 ranks 4th of 7 with 11.3 % of the total.** It is a random walk. MDI is computed on the
  *in-sample* fit, and a slowly-varying column is a usable proxy for "which part of the sample
  this row is in" whenever the residual is autocorrelated — which, with overlapping labels, it
  always is.

✅ sklearn's `RandomForestRegressor` on the same data with the same hyper-parameters:
`0.3604 0.2718 0.2336 0.0234 0.0022 0.0094 0.0992` — same shape, same conclusions.

## 4. MDA and SFI: what each one is actually asking

MDA = the drop in **out-of-sample** R² when one column is permuted. SFI = the out-of-sample R² of
a model fitted on that column **alone**. ✅ Both on an embargoed forward split (3 folds, 30 rows
embargoed either side, 3 permutations per feature):

| feature | MDA | rank | SFI | rank |
|---|---|---|---|---|
| x0 informative | **0.1165** | 2 | **0.1037** | 1 |
| x1 collinear copy | 0.0585 | 3 | 0.0913 | 2 |
| x2 informative | **0.1269** | 1 | 0.0203 | 3 |
| x3 noise, high card | **-0.0018** | 7 | -0.0806 | 6 |
| x4 noise, binary | -0.0011 | 5 | -0.0543 | 4 |
| x5 noise, 5 levels | 0.0016 | 4 | -0.0545 | 5 |
| x6 noise, random walk | **-0.0015** | 6 | -0.4324 | 7 |

- ✅ **MDA has no cardinality bias**: all four irrelevant columns land within ±0.002 of zero, and
  x3 no longer beats x4.
- 🚨 **MDA under-states a collinear pair.** x0 0.1165 and x1 0.0585 sum to 0.175 — about **half**
  of the group's true importance (section 5: 0.3367). Permuting x0 leaves x1 to carry the signal,
  and vice versa. **Two features that look unimportant can be jointly essential.**
- **MDA and SFI disagree on purpose.** MDA ranks x2 above x0 (0.1269 vs 0.1165); SFI ranks x0
  above x2 (0.1037 vs 0.0203). MDA asks *what does this add given the others*; SFI asks *what is
  it worth alone*. Both answers are correct for their own question, and SFI is the only method
  here that says x1 is worth anything at all.
- ⚠️ SFI's blind spot is the mirror image: a feature that is useless alone and essential in
  combination scores zero. There is no such feature in this design; there are plenty in real ones.

## 5. ✅ Clustered MDA: permute the group

Cluster the features first (single linkage on |correlation| ≥ 0.8 — ✅ it recovers exactly
`[[x0, x1], [x2], [x3], [x4], [x5], [x6]]`), then permute each cluster **jointly**:

| cluster | clustered MDA | rank |
|---|---|---|
| **x0+x1** | **0.3367** | 1 |
| x2 | 0.1223 | 2 |
| x5 | 0.0009 | 3 |
| x3 | -0.0004 | 4 |
| x4 | -0.0010 | 5 |
| x6 | -0.0018 | 6 |

✅ The pair scores **0.3367 against 0.1165 for its best single member — a factor of 2.9** — and
x2, which is in a cluster of its own, is unchanged (0.1223 against 0.1269, inside the noise).
This is the answer to substitution: **stop asking which of two interchangeable features matters
and ask whether the group does.**

## 6. 🚨 The split matters more than the metric

Same MDA, same model, one difference — a shuffled k-fold instead of an embargoed forward split:

| feature | MDA, **shuffled** k-fold | MDA, **embargoed forward** |
|---|---|---|
| x0 informative | 0.0758 | 0.1165 |
| x1 collinear copy | 0.0937 | 0.0585 |
| x2 informative | 0.1228 | 0.1269 |
| x3 noise, high card | 0.0007 | -0.0018 |
| x4 noise, binary | -0.0002 | -0.0011 |
| x5 noise, 5 levels | -0.0023 | 0.0016 |
| **x6 noise, random walk** | **+0.0521** | **-0.0015** |

🚨 **The shuffled fold makes a random walk the third most important feature**, at 41 % of x0's
score. The residual is a 25-row rolling mean, so a shuffled test row's neighbours are in the
training set, and a slowly-varying column identifies which neighbours. ✅ The overall out-of-sample
R² tells the same story: **0.2659 shuffled against 0.1626 embargoed** — the shuffled number is 64 %
too high, and it is the number that would have gone in the report.

**The embargoed forward split in this script is the *minimum* honest version.** The real fix needs
each label's actual resolution time and belongs to
`../../../fin-libraries/skills/lib-purgedcv/SKILL.md`. Do not re-implement it.

## 7. Traps

- 🚨 **Reporting `feature_importances_`.** Section 3. In-sample, cardinality-biased,
  substitution-split, and it ranks a random walk fourth.
- 🚨 **Dropping features by MDA.** Section 4: a collinear pair reports half its joint value.
  Cluster first (section 5), then drop clusters.
- 🚨 **Any importance from a shuffled split.** Section 6. With overlapping labels this is not a
  small bias; it invents importance for anything that varies slowly.
- 🚨 **One-hot encoding changes MDI.** Splitting a 50-level categorical into 50 binary columns
  moves it from the high-cardinality end of the bias to the low-cardinality end, and the total
  importance the group receives changes with the encoding, not with the information. Compare only
  within one encoding.
- 🚨 **Importance is not stability.** Rerun with a different seed and the split inside a collinear
  cluster moves. Report importances with an error bar across seeds and folds, or report clusters.
- 🚨 **A feature search is a trial ledger.** Ranking 200 features, keeping the top 20 and
  reporting the backtest of those 20 is a best-of-200 selection.
  `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`.
- 🚨 **Feature selection must happen inside the fold.** Choosing features on the whole sample and
  then cross-validating the chosen set leaks the test folds into the selection. This is the same
  error as section 6 in a different place.
- ⚠️ **MDA needs enough test rows to be stable.** Each fold here is 500 rows and 3 permutations;
  the irrelevant features still scatter to ±0.002. Report the spread across folds, not just the
  mean.
- ⚠️ **Weights change importance too.** With overlapping labels the samples are not equally
  informative; an unweighted MDA over-counts crowded stretches
  (`../sample-weights-and-uniqueness/SKILL.md`).

## 8. Scripts

- `scripts/feature_importance.py` — a CART regression tree with exact variance-reduction splits
  and MDI accumulation, a bagged forest, MDA (single and clustered), SFI, single-linkage
  correlation clustering, shuffled and embargoed splits, and the numerical reconstruction of
  sklearn's `feature_importances_` from `tree_`. numpy only; `sklearn` is imported inside one
  function. Seed 0, **23 s**.

## 9. Where this sits

`../../../fin-libraries/skills/lib-purgedcv/SKILL.md` owns purged and embargoed cross-validation —
section 6 is the reason it matters here, and the fix lives there.
`../sample-weights-and-uniqueness/SKILL.md` owns the weights that go with overlapping labels.
`../triple-barrier-labeling/SKILL.md` produces the labels whose overlap causes section 6.
`../../../fin-core/skills/signal-construction/SKILL.md` owns building the features causally in the
first place. `../../../fin-models/skills/factor-models/SKILL.md` owns attribution of *returns* to
factors, which is a different question from attribution of a *model* to its inputs.
`../../../fin-core/skills/backtest-validation/SKILL.md` owns the trial count of a feature search.
