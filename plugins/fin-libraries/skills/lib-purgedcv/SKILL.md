---
name: lib-purgedcv
description: >-
  The only genuinely sklearn-protocol-compliant purged and embargoed splitter, and the one that
  refuses to run until you state when each label resolved - understate evaluation_times and it
  silently reintroduces the leak while reporting clean folds. TRIGGER - purgedcv, PurgedKFold,
  CombinatorialPurgedCV, PurgedGroupKFold, WalkForwardSplit, prediction_times, evaluation_times,
  purge_horizon, embargo, audit_splitter, reconstruct_paths, path_metrics, overlapping labels,
  triple-barrier touch time, mlfinlab.cross_validation. Memory is stale or absent - this package
  first shipped 2026-05-16 and is at 0.1.6. SKIP for CPCV over portfolio models (lib-skfolio) and
  for SPA/StepM/MCS (lib-arch). SKIP when the question is WHICH library to choose rather than how
  to use this one - that belongs to the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# purgedcv

Purged, embargoed and combinatorial cross-validation for time series with **overlapping labels** — the one splitter
that refuses to run until you tell it when each label actually resolved.

| | |
|---|---|
| pip / import | `purgedcv` / `purgedcv` |
| Version | **0.1.6** (2026-09-04) · Python `>=3.10` (classifiers 3.10–3.14) |
| Licence | **MIT** |
| Status | ✅ **actively developed** — `eslazarev/purged-cross-validation`, 31★, **0 open issues**, pushed 2026-09-04; 18 releases since 2026-05-16, ~117 commits, has a JOSS paper |

## The trap that costs you money

🚨 **`evaluation_times` is a *required* argument, and understating it silently reintroduces the leak.** That
requirement is the point of the library — you cannot call it without stating your label horizon, which is exactly the
fact people omit. But passing `t0 + 1 bar` when the label really resolves ten bars later purges almost nothing: the
splitter reports clean folds and the score stays inflated. With triple-barrier labels, `t1` must be the **actual touch
time returned by the labeler** — not the vertical barrier, and not a fixed horizon. Overstating it (using the vertical
barrier when a barrier was touched early) is merely wasteful;
**understating it is a correctness bug** that looks exactly like a correct run.

## Why purging exists at all

Financial labels overlap. A triple-barrier label stamped at `t0` is not resolved until `t1`, and `t1` routinely lands
inside the next fold. Standard k-fold then trains on a bar whose outcome is already partly encoded in the test set.
The symptom is a CV score far above anything achieved live.

🚨 **Anything that shuffles is catastrophic**, and shuffling is the default in most of sklearn's ergonomic paths:
`train_test_split`, `KFold(shuffle=True)`, `cross_val_score` on a plain DataFrame. 🚨 **Unshuffled `KFold` is still
wrong** — it does not purge the overlap at the fold boundary.

Two corrections, both required. **Purge** — drop training observations whose label-resolution window intersects the
test window. **Embargo** — additionally drop training observations for a span *after* the test window, to kill leakage
through serially correlated features (a 20-day moving average knows about the test set for 20 days afterwards).

🚨 **Purging without an embargo is the common half-fix.** Purging handles the label horizon; the embargo handles the
*feature* horizon. A lookback of 60 bars needs an embargo of at least 60 bars.

## Two more limits worth budgeting for

🚨 **CPCV cost is combinatorial.** With `n_splits=N` groups and `n_test_groups=k` you fit `C(N, k)` models —
`C(6,2)=15`, `C(10,2)=45`, `C(10,3)=120`. The payoff is `k·C(N,k)/N` distinct backtest
*paths* instead of one, which is what makes PBO estimable; the cost is that a 30-second fit becomes
an hour. Choose `N` with that in mind. 🚨 **CV purging does not fix sample weights.** Overlapping labels also make
observations non-independent *within* the training set. Purging fixes the train/test boundary;
**uniqueness and time-decay weights fix the fit itself.** You need both, and the weights go in separately. ⚠️
A clean CV score is still a single draw from a search — purged CV removes leakage, not the selection bias from having
tried 200 configurations.

## Surface, and the alternatives

Exported alongside the splitters: `WalkForwardSplit`, `PurgedGroupKFold`, `purge()`, `apply_embargo()`,
`probabilistic_sharpe_ratio()`, `deflated_sharpe_ratio` / `deflated_sharpe_ratio_full()`, `reconstruct_paths()`,
`path_metrics()`, and **`audit_splitter()`** (2026-08-29 — per-fold leakage diagnostics).

| Option | Status | The catch |
|---|---|---|
| **`purgedcv`** | ✅ MIT, active, py≥3.10 | Young (0.1.x) — expect API movement |
| `skfolio.model_selection.CombinatorialPurgedCV` | ✅ BSD-3, active | 🚨 `split()` yields `(train, [test_0, …])`, not sklearn's 2-tuple; `purged_size`/`embargo_size` are in **observations**, not time |
| `RiskLabAI` | ✅ BSD-3, v3.1.0 (2026-08-26) | 🚨 `requires_python >=3.12,<3.15` — sets the floor for your whole environment |
| `ml4t-diagnostic` | ✅ MIT, v0.1.2 (2026-08-17) | 🚨 py≥3.12, Polars-first; ❓ CPCV import paths unverified |
| 🔴 `mlfinlab.cross_validation` | **unusable** | Off PyPI; the GitHub source is a stub — `PurgedKFold.split()` has body `pass` |
| 🔴 `timeseriescv` | dead | v0.2 (2018) — the design `purgedcv` later refined |

🔑 **Only `purgedcv` is genuinely sklearn-protocol compliant**, so it drops into `cross_val_score`, `GridSearchCV` and
`Pipeline` with no glue. skfolio's splitter composes with *skfolio's own* cross-validation helpers — a different
contract.

## Minimal correct call

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from purgedcv import PurgedKFold, CombinatorialPurgedCV

# t0 = feature-observable time; t1 = ACTUAL label resolution time (triple-barrier touch time,
#      NOT the vertical barrier and NOT a fixed horizon).
cv = PurgedKFold(n_splits=5, prediction_times=t0, evaluation_times=t1,
                 purge_horizon="5D",     # >= your longest label horizon
                 embargo="5D")           # >= your longest feature lookback
clf = RandomForestClassifier(n_estimators=500, random_state=0)
scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc",
                         params={"sample_weight": w_uniqueness})  # weights are a SEPARATE fix
print(scores.mean(), scores.std())       # a mean without the std is not a result

# Combinatorial: C(6,2)=15 fits, for multiple backtest paths -> PBO becomes estimable
cpcv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2,
                             prediction_times=t0, evaluation_times=t1)
```

## Scripts

✅ **`scripts/purge_effect.py`** measures the leak on data with **no signal at all**: N=2000 observations, label
horizon H=100, five AR(1) features (phi=0.995) drawn from a generator that never touches the returns being
labelled, so the honest out-of-sample AUC is 0.500 by construction. Lag-1 label autocorrelation 0.9020. Mean k-NN
AUC over 24 independent datasets (seeds 0-23, 5 folds each, 119-120 scored folds):

| splitter | mean AUC | std err | apparent skill |
|---|---|---|---|
| `KFold(shuffle=True)` | 0.9801 | 0.0007 | **+0.4801** |
| `KFold(shuffle=False)` | 0.5563 | 0.0131 | +0.0563 |
| `TimeSeriesSplit` | 0.5498 | 0.0150 | +0.0498 |
| purged H=100, no embargo | 0.5016 | 0.0141 | +0.0016 |
| purged H=100 + embargo 100 | 0.4991 | 0.0139 | -0.0009 |

🚨 **Measured gap: +0.0572 AUC of skill that is not there**, and `RandomForestClassifier(n_estimators=100)` on
scikit-learn's own splitters reproduces it — 0.5601 vs 0.4999, gap +0.0601 — so it is not an artefact of the toy
k-NN. 🚨 **`TimeSeriesSplit` leaks too** (+0.0498): it is forward-only, but its last training row still abuts the
first test row. ⚠️ In *this* dataset the purge alone closes the gap and the embargo then moves the score by -0.0025
against a standard error of 0.0139 — not measurable, because the features are independent of the labelled return
series, so only the label-overlap channel is live. The embargo closes the *feature* channel, which needs features
built as rolling functions of that same series; it is not exercised here and this run is not evidence against it.
The script's hand-written `KFold` and `TimeSeriesSplit` reproduce scikit-learn's folds index-for-index and its
`roc_auc_score` to 1.11e-16, so the comparison still runs with scikit-learn absent.

## See also

- `../../../fin-core/skills/backtest-validation/SKILL.md` — the domain skill for leakage and multiple testing
- `../../../fin-core/skills/backtest-validation/references/purgedcv.md` — the source card
- `../../../fin-core/skills/backtest-validation/references/afml-stack.md` — labeling and sample weights
- `../../../fin-core/skills/backtest-validation/references/significance-tests.md` — what to do once the folds are clean

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`backtest-validation`** (`../../../fin-core/skills/backtest-validation/SKILL.md`).
