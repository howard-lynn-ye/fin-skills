# purgedcv

Purged, embargoed and combinatorial cross-validation for time series with **overlapping labels** — the
one splitter that refuses to run until you tell it when each label actually resolved.

| Field | Value |
|---|---|
| pip / import | `purgedcv` / `purgedcv` |
| version | **0.1.6** (2026-09-04) ✅ |
| repo | `github.com/eslazarev/purged-cross-validation` — 31 ★, **0 open issues**, pushed 2026-09-04 ✅ |
| licence | **MIT** ✅ |
| Python | `>=3.10` ✅ (classifiers 3.10–3.14 ⚠️) |
| verdict | ✅ **actively developed** — 18 releases since 2026-05-16, ~117 commits, has a JOSS paper ⚠️ |

Verified 2026-09-04 against the PyPI JSON API and the GitHub REST API.

## Why this exists

Financial labels overlap. A triple-barrier label stamped at `t0` is not resolved until `t1`, and `t1`
routinely lands inside the next fold. Standard k-fold then trains on a bar whose outcome is already
partly encoded in the test set. The measured symptom is a cross-validated score far above anything the
strategy achieves live.

🚨 **Anything that shuffles is catastrophic here**, and shuffling is the default in most of sklearn's
ergonomic paths: `train_test_split`, `KFold(shuffle=True)`, and `cross_val_score` on a plain DataFrame.
🚨 **Unshuffled `KFold` is still wrong** — it does not purge the overlap at the fold boundary.

Two corrections, both required:
- **Purge** — drop training observations whose label resolution window intersects the test window.
- **Embargo** — additionally drop training observations for a short span *after* the test window, to
  kill leakage through serially correlated features (a 20-day moving average knows about the test set
  for 20 days afterwards).

## The API

```python
from purgedcv import PurgedKFold, CombinatorialPurgedCV
cv = PurgedKFold(n_splits=5,
                 prediction_times=t0,      # when the feature was observable
                 evaluation_times=t1,      # when the label RESOLVED  <- forces honesty
                 purge_horizon="1D", embargo="2D")
cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2,
                           prediction_times=t0, evaluation_times=t1)
```

Also exported ⚠️ (from the project's own README, not re-executed here): `WalkForwardSplit`,
`PurgedGroupKFold`, `purge()`, `apply_embargo()`, `probabilistic_sharpe_ratio()`,
`deflated_sharpe_ratio` / `deflated_sharpe_ratio_full()`, `reconstruct_paths()`, `path_metrics()`,
and `audit_splitter()` (added 2026-08-29 — per-fold leakage diagnostics).

🔑 **The design point:** `evaluation_times` is a *required* argument, not a convenience. You cannot
call this library without stating your label horizon, which is exactly the fact people omit.

## Traps

🚨 **Understating `evaluation_times` silently reintroduces the leak.** Passing `t0 + 1 bar` when the
label really resolves ten bars later purges almost nothing — the splitter reports clean folds and the
score is still inflated. With triple-barrier labels, `t1` must be the **actual touch time** returned by
the labeler, not the vertical barrier and not a fixed horizon. Over-stating it (using the vertical
barrier when the barrier was touched early) is merely wasteful; under-stating it is a correctness bug.

🚨 **Purging without an embargo is the common half-fix.** Purging handles the label horizon; the
embargo handles the *feature* horizon. If your longest lookback window is 60 bars, an embargo shorter
than that leaks.

🚨 **CPCV cost is combinatorial.** With `n_splits=N` groups and `n_test_groups=k` you fit `C(N, k)`
models — `C(6,2)=15`, `C(10,2)=45`, `C(10,3)=120`. The payoff is `k·C(N,k)/N` distinct backtest
*paths* instead of one, which is what makes PBO estimable; the cost is that a 30-second fit becomes an
hour. Budget for it before choosing `N`.

🚨 **CV purging does not fix sample weights.** Overlapping labels also make observations
non-independent *within* the training set. Purging fixes the train/test boundary; uniqueness and
time-decay weights fix the fit itself. You need both — see `afml-stack.md`.

⚠️ **A clean CV score is still a single draw from a search.** Purged CV removes leakage; it does not
remove selection bias from having tried 200 configurations. Pair it with `significance-tests.md`.

## Alternatives, and how they differ

| Option | Status | The catch |
|---|---|---|
| **`purgedcv`** | ✅ MIT, active, py≥3.10 | Young (0.1.x) — expect API movement |
| **`skfolio.model_selection.CombinatorialPurgedCV`** | ✅ BSD-3, active | 🚨 **`split()` yields `(train, [test_0, test_1, …])`** — *not* sklearn's 2-tuple. Code written to the standard protocol mis-unpacks it. 🚨 **`purged_size` / `embargo_size` are counted in observations, not time** — under dollar or volume bars a fixed count is a wildly varying time span |
| **`RiskLabAI`** | ✅ BSD-3, v3.1.0 (2026-08-26) | 🚨 `requires_python >=3.12,<3.15` — sets the floor for your whole environment |
| **`ml4t-diagnostic`** | ✅ MIT, v0.1.2 (2026-08-17) | 🚨 py≥3.12, Polars-first; ❓ exact import paths for its CPCV unverified |
| 🔴 **`mlfinlab.cross_validation`** | **unusable** | Off PyPI; the GitHub source is a stub — `PurgedKFold.split()` has body `pass` ✅ |
| 🔴 **`timeseriescv`** | dead | v0.2 (2018) — the design `purgedcv` later refined |

🔑 **Only `purgedcv` is genuinely sklearn-protocol compliant**, so it drops into `cross_val_score`,
`GridSearchCV` and `Pipeline` with no glue. skfolio's splitter composes with *skfolio's own*
cross-validation helpers, which is a different contract.

## Minimal correct use

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from purgedcv import PurgedKFold

# t0 = feature-observable time; t1 = ACTUAL label resolution time (triple-barrier touch time).
cv = PurgedKFold(n_splits=5, prediction_times=t0, evaluation_times=t1,
                 purge_horizon="5D",      # >= your longest label horizon
                 embargo="5D")            # >= your longest feature lookback
clf = RandomForestClassifier(n_estimators=500, random_state=0)

scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc",
                         params={"sample_weight": w_uniqueness})  # weights are a SEPARATE fix
print(scores.mean(), scores.std())        # a mean without the std is not a result
```

## See also

- `afml-stack.md` — labeling, sample weights, and the `get_bins` threshold bug that survives in clones.
- `significance-tests.md` — what to do with the score once the folds are clean.
- `../../portfolio-and-risk/references/skfolio.md` — CPCV composed with `GridSearchCV` over portfolio models.
