---
name: backtest-validation
description: >-
  Decide whether a backtest result survives the number of things you tried. Covers purged and
  combinatorial-purged cross-validation (purgedcv, skfolio, RiskLabAI), the Deflated and
  Probabilistic Sharpe Ratio, Probability of Backtest Overfitting, and the multiple-comparison
  procedures in arch.bootstrap — White's Reality Check, Hansen's SPA, Romano-Wolf StepM and the
  Model Confidence Set — plus the López de Prado / AFML labeling and sample-weight stack.
  TRIGGER — use when a backtest or model result must be judged real or spurious; when the task
  mentions overfitting, p-hacking, data snooping, multiple testing, walk-forward, cross-validation
  on time series, deflated Sharpe, PBO, or "I tried N strategies"; when a parameter sweep,
  hyperopt, grid search or AutoML produced a winner; and whenever a Sharpe ratio is about to be
  reported as evidence for trading.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# Backtest validation

A backtest reports the maximum of a search. The question is never "what was the Sharpe" but
**"what was the Sharpe, given how many things were tried, and how much of the sample was
independent."** This skill answers both.

## 1. The 2026 stack

```bash
pip install purgedcv        # purged/combinatorial CV + PSR/DSR.  MIT.  py>=3.10
pip install skfolio         # CPCV composed with sklearn GridSearchCV.  BSD-3
pip install arch            # SPA / Reality Check / StepM / MCS.  THE significance toolkit
pip install RiskLabAI       # broadest AFML reimplementation.  BSD-3.  py>=3.12
pip install ml4t-diagnostic # DSR, PBO, CPCV, feature importance.  MIT.  py>=3.12
pip install jsharpe         # PSR, MinTRL + FWER/FDR corrections.  MIT.  py>=3.11
```

🚨 **Do not plan around `mlfinlab`.** Not installable from PyPI; the GitHub source is stubbed
(**every function body is `pass`**); licence is proprietary all-rights-reserved; silent since 2023.
🚨 **`fracdiff` is archived** (2023-12) and `requires_python <3.10` — unusable on modern Python.
⚠️ **`pypbo`** is the reference PBO/CSCV implementation but is **not on PyPI** and is **AGPL-3.0**.
⚠️ **`sharpebench`** (v0.17.1, 2026-09-02) covers this whole surface but declares **no repository
URL on PyPI** — provenance unverifiable. New and unvetted; do not default to it.

## 2. Cross-validation that does not leak

**Any CV that shuffles is catastrophic here.** `train_test_split`, `KFold(shuffle=True)`, default
`cross_val_score` on a DataFrame. Even unshuffled `KFold` fails without purging.

```python
from purgedcv import PurgedKFold, CombinatorialPurgedCV
cv = PurgedKFold(n_splits=5,
                 prediction_times=t0,     # when the feature was known
                 evaluation_times=t1,     # when the label resolved  <- forces honesty
                 purge_horizon="1D", embargo="2D")
cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2,
                           prediction_times=t0, evaluation_times=t1)
```

`purgedcv` is the best standalone choice precisely because it **requires** `evaluation_times` —
you cannot use it without confronting your label horizon. It is sklearn-protocol compliant, so it
drops into `cross_val_score` / `GridSearchCV` / `Pipeline` with no glue, and `audit_splitter()`
gives per-fold leakage diagnostics.

**Rules:**
- Purge horizon = your **maximum** holding period, not the mean.
- The **embargo is separate from purging**: purging handles label overlap, the embargo handles
  serial correlation *after* the test window. You need both.
- In `skfolio`, `purged_size`/`embargo_size` are counted in **observations, not time** — with
  dollar/volume bars a fixed count is a wildly varying span. Its `split()` also yields
  `(train, [test_0, test_1, ...])`, not sklearn's 2-tuple.

**Evidence that CPCV is the right default:** Arian, Norouzi & Seco, *Knowledge-Based Systems* 305
(2024), doi `10.1016/j.knosys.2024.112477` — in a synthetic controlled environment (Heston, Merton
jumps, drift-burst, regime-switching), **CPCV markedly outperforms K-Fold, Purged K-Fold and
especially Walk-Forward** on both PBO and DSR, with walk-forward showing **weak false-discovery
control and high temporal variability**. ⚠️ *The lead author also authors `RiskLabAI` — peer-reviewed
but not disinterested.*

## 3. ⭐ `arch.bootstrap` — the right tool for "better than the benchmark after searching"

This is the most under-used correct answer in the field. ✅ verified API, executed on arch 8.0.0:

```python
from arch.bootstrap import SPA, RealityCheck, StepM, MCS, optimal_block_length
```

| Question | Class |
|---|---|
| "Is the best of my N strategies better than the benchmark, accounting for having searched N?" | **`RealityCheck`** (White 2000; alias of `SPA`) |
| Same, studentized + recentered so poor models don't distort it | **`SPA`** (Hansen 2005) |
| ***Which*** strategies beat the benchmark, with FWER control | **`StepM`** (Romano-Wolf) |
| The set of models indistinguishable from the best at 1−α | **`MCS`** (Hansen-Lunde-Nason 2011) |

🚨 **ALL THREE TAKE LOSSES, NOT RETURNS.** ✅ Verified from `arch/bootstrap/multiple_comparison.py`:
`SPA`/`StepM` document `benchmark` as *"T element array of benchmark model **losses**"* and `models`
as *"T by k element array of alternative model **losses**"*; `MCS`'s parameter is literally named
`losses`. **Passing returns silently inverts every one of these tests** — it asks which strategy is
*worst*. Convert first: `losses = -returns`.

```python
bm_loss  = -benchmark_returns          # (T,)    NEGATE
mdl_loss = -model_returns              # (T, k)  NEGATE

spa = SPA(bm_loss, mdl_loss, reps=1000, block_size=10, seed=7); spa.compute()
spa.pvalues            # {'lower':…, 'consistent':…, 'upper':…}  -> report 'consistent'

stepm = StepM(bm_loss, mdl_loss, size=0.05, reps=1000, block_size=10); stepm.compute()
stepm.superior_models  # survives FWER control

mcs = MCS(mdl_loss, size=0.10, reps=1000, block_size=10); mcs.compute()
mcs.included, mcs.excluded

optimal_block_length(bm_loss)   # Politis-White; pick block_size from this, not by eye
```

🚨 **Other gotchas:**
- SPA's three p-values are a range — `lower` liberal, `upper` conservative. **Report `consistent`.**
- `RealityCheck` is a zero-code subclass of `SPA` — same interface, same loss convention.
- Choose `block_size` from `optimal_block_length` — preserving serial dependence is the entire point.
- Cost is O(reps × T × N): 1000 reps over hundreds of strategies is minutes, not seconds.
- arch 8.0.0 is a **compatibility release, not an API break.**

## 4. Deflated and Probabilistic Sharpe Ratio

**No dominant maintained package exists** — this is a genuine ecosystem gap and the formulas are
short enough to inline. Available in `purgedcv`, `ml4t-diagnostic`, `jsharpe`, `RiskLabAI`.
🚨 **`quantstats` has `sharpe`/`sortino` but NO PSR/DSR/MinTRL** — do not reach for it here.

```python
import numpy as np
from scipy.stats import norm, skew, kurtosis

def probabilistic_sharpe_ratio(r, sr_benchmark=0.0):
    """P(true SR > sr_benchmark), correcting for skew, kurtosis, sample length.
    r and sr_benchmark must be on the SAME per-period scale."""
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)      # g4 = NON-excess kurtosis
    num = (sr - sr_benchmark) * np.sqrt(T - 1)
    den = np.sqrt(1 - g3 * sr + ((g4 - 1) / 4) * sr ** 2)
    return norm.cdf(num / den)

def deflated_sharpe_ratio(r, n_trials, sr_variance):
    """PSR evaluated against the EXPECTED MAX Sharpe of n_trials independent backtests."""
    e = np.euler_gamma
    sr0 = np.sqrt(sr_variance) * (
        (1 - e) * norm.ppf(1 - 1 / n_trials) + e * norm.ppf(1 - 1 / (n_trials * np.e))
    )
    return probabilistic_sharpe_ratio(r, sr_benchmark=sr0)
```

⚠️ Formula transcription unverified against the source papers — check before publishing.

🔴 **`n_trials` is the input that decides the answer, and it is almost always underreported.** It is
**every** configuration you tried: every barrier multiple, lookback, threshold, feature set, universe
filter, rebalance frequency — including abandoned ones, and including everything an automated search
(RD-Agent, gplearn, Optuna, hyperopt) evaluated. **No library can recover it for you.** Reporting
`n_trials` = "the 5 models I saved" makes DSR a rubber stamp.

**Keep a trial ledger.** Append every candidate and its parameters *before* looking at out-of-sample
results. `scripts/trial_ledger.py` provides a minimal append-only implementation.

Also check whether a given DSR implementation actually applies the skew/kurtosis adjustment or
silently assumes IID normal — the expected-max term should use the Euler–Mascheroni constant.

## 5. The AFML labeling stack — where each method actually lives

| Method | Correct implementation | Note |
|---|---|---|
| Triple-barrier labeling | `RiskLabAI` (`meta_events`, `triple_barrier`), `ml4t.engineer.labeling`, `mlfinpy` (MIT, stale) | mlfinlab's is a `pass` stub |
| Meta-labeling | `RiskLabAI.meta_labeling`, `mlfinpy.get_bins` (side path) | 🚨 primary model must emit **out-of-sample** sides |
| Fractional differentiation (FFD) | `RiskLabAI` (FFD + ADF min-`d`) | `fracdiff` archived, py<3.10 |
| Purged / combinatorial CV | **`purgedcv`**, `skfolio`, `RiskLabAI` | |
| Sample uniqueness / time-decay weights | `RiskLabAI`, `mlfinpy.sampling`, `ml4t-engineer` | Sequential bootstrap is O(n²); effect is modest (uniqueness ~0.6→0.7) |
| PSR / DSR / MinTRL | `purgedcv`, `jsharpe`, `ml4t-diagnostic` | |
| PBO via CSCV | `pypbo` (AGPL, git-only), `ml4t-diagnostic`, `RiskLabAI` | `pip install pbo` does not exist |
| Tick / volume / dollar / imbalance bars | `ml4t.engineer.bars`, `RiskLabAI`, `mlfinpy` | Imbalance bars' EWMA warm-up is arbitrary and materially changes bar counts |
| Bet sizing | `RiskLabAI` | Thinnest coverage of any AFML chapter |

🚨 **A verified bug worth knowing**, present in mlfinlab's API and its faithful clones: `get_bins`
throws away which barrier `get_events` already determined was touched, and **re-infers** the label
by comparing the realized return to thresholds. Two consequences: (a) **threshold inconsistency** —
`barrier_touched` tests against `pt·log(1+target)` while the barrier was placed at
`log(1+pt·target)` (for `target=0.02, pt=2`: 0.0396 vs 0.0392), so borderline events are mislabeled;
(b) `prices = close.reindex(all_dates, method="bfill")` — **backfill** resolves a missing `t1` to the
*next* available price, a forward-looking value. **Fix: capture which barrier was hit inside
`get_events` and carry it through.**

## 6. Honest assessment of AFML itself

AFML is **not seriously contested in the literature; it is under-tested.** No substantive published
rebuttal of triple-barrier or meta-labeling exists. But:
- Every AFML knob — barrier multiples, `min_ret`, CUSUM threshold, bar type, embargo width,
  fractional `d` — **is a researcher degree of freedom that inflates your trial count.**
- **The strongest critique of AFML is López de Prado's own work**: "Pseudo-Mathematics and Financial
  Charlatanism", "The Probability of Backtest Overfitting", "The 10 Reasons Most Machine Learning
  Funds Fail" (*JPM* 44(6), 120–133; SSRN 3104816) apply with full force to AFML pipelines themselves.
- **Meta-labeling's evidence base is largely vendor-produced** (the Hudson & Thames / JFDS series,
  with Joubert an author on all four papers). Independent replication is thin. Treat "meta-labeling
  improves Sharpe" as plausible but **not independently established**.

## 7. The reporting standard

A result is not reportable without: the metric and its annualization factor; the risk-free
convention; the CV scheme with purge and embargo; the **trial count and where the ledger lives**;
the cost model; and the confidence interval. A 3-year Sharpe of 1.0 has a standard error near 0.58.

**A negative result is a result.** "The baseline did not clear the pre-registered threshold" is the
correct output far more often than a positive one, and pre-registering the threshold is what makes
it credible.
