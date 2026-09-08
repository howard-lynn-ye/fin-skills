# Significance tests for backtests — PSR, DSR, PBO, SPA, StepM, MCS

Six procedures that answer six *different* questions about "is this Sharpe real"; picking the wrong one
is common, and feeding the right one the wrong sign is worse.

## 🚨 The trap that dominates everything else: losses, not returns

`arch.bootstrap`'s **`SPA`, `RealityCheck`, `StepM` and `MCS` all take LOSSES — lower is better.**
✅ Verified against `arch/bootstrap/multiple_comparison.py`.

Passing returns does not raise, does not warn, and **inverts the test**: the worst strategy is
identified as the best, and the p-value you report is for the opposite hypothesis. This is the single
most common silent error in this whole area.

```python
losses = -returns          # for a return series, this is the whole conversion
```

Anything already loss-shaped (squared forecast error, negative log-likelihood, absolute error, drawdown)
goes in as-is. If your metric is a *ratio* rather than a per-period series — a Sharpe ratio, an
information ratio — these tests do not take it at all; they need the per-period loss column.

## Which test answers which question

| Your question | Procedure | Class | Input shape |
|---|---|---|---|
| Is *my one* strategy's Sharpe distinguishable from luck, given skew, kurtosis and sample length? | **PSR** | hand-roll / `purgedcv` / `jsharpe` | one return series |
| …and given that I tried **N** configurations before keeping this one? | **DSR** | hand-roll / `ml4t-diagnostic` | one series + honest `n_trials` |
| What fraction of the time does my in-sample-best configuration under-perform the median out-of-sample? | **PBO** (CSCV) | `ml4t-diagnostic`, `pypbo` ⚠️ | a matrix of trial P&Ls |
| Is the **best of N** strategies better than a benchmark, correcting for the search? | **SPA** (Hansen 2005) | `arch.bootstrap.SPA` | benchmark losses + (T×k) model losses |
| Same question, the older less powerful version | **Reality Check** (White 2000) | `arch.bootstrap.RealityCheck` | identical |
| ***Which*** strategies beat the benchmark, with family-wise error control? | **StepM** (Romano-Wolf) | `arch.bootstrap.StepM` | benchmark + (T×k) losses |
| Which models are statistically **indistinguishable from the best**? | **MCS** (Hansen-Lunde-Nason 2011) | `arch.bootstrap.MCS` | (T×k) losses, **no benchmark** |

🔑 **SPA/StepM/MCS vs PSR/DSR/PBO is not a quality ranking, it is a different data requirement.**
SPA/StepM/MCS need the *per-period series of every candidate you tried*. PSR/DSR need only the winner
plus an honest count. If you kept only the winner's equity curve, DSR is your only option — and its
accuracy is then entirely hostage to a number you have to supply from memory.

## `arch.bootstrap` — verified API

`arch` 8.0.0 (2025-10-21) ✅ · licence **NCSA** ✅ (permissive, BSD-like, but not one of the usual
three — flag it in a licence audit; GitHub reports `NOASSERTION`) · `>=3.10` ✅ · 1,558 ★ / 51 issues,
pushed 2026-08-10 ✅ · Kevin Sheppard (Oxford). 8.0.0 is a NumPy-2.4 / pandas-3 compatibility release,
**not an API break** ✅.

```python
SPA(benchmark, models, block_size=None, reps=1000, bootstrap="stationary",
    studentize=True, nested=False, *, seed=None)          # ✅ verified signature
StepM(benchmark, models, size=0.05, block_size=None, reps=1000, ...)   # ✅
MCS(losses, size, reps=1000, block_size=None, method="R"|"max", ...)   # ✅
```

✅ **`RealityCheck` is literally `class RealityCheck(SPA): pass`** — a zero-code subclass. White's
Reality Check is the special case of SPA without studentization or recentring. **Use `SPA`;** mention
`RealityCheck` only when reproducing a pre-2005 paper.

- `SPA.pvalues` returns **three** values — `lower`, `consistent`, `upper`. ✅ They bracket the influence
  of dominated models on the null distribution. **Report `consistent`.** `lower` is liberal, `upper`
  conservative. Also `.critical_values(pvalue=0.05)` and `.better_models(pvalue=0.05)`.
- `StepM.superior_models` names the survivors under FWER control. Internally it builds an `SPA`. ✅
- `MCS` takes **no benchmark** and ✅ raises with fewer than 2 columns. Gives `.included`, `.excluded`,
  `.pvalues`. `method="R"` (range, default) or `"max"`.
- ⚠️ `block_size` defaults to `int(sqrt(T))`. Set it from `optimal_block_length` instead — preserving
  serial dependence is the entire reason for a block bootstrap.
- ⚠️ Cost is **O(reps × T × k)**. 1000 reps over hundreds of candidates is minutes, not seconds.

```python
import numpy as np
from arch.bootstrap import SPA, StepM, MCS, optimal_block_length

losses_bm  = -bench_returns                     # 🚨 NEGATE. (T,)
losses_mdl = -model_returns                     # 🚨 NEGATE. (T, k)
bs = int(optimal_block_length(losses_bm).iloc[0, 0])   # stationary-bootstrap block length

spa = SPA(losses_bm, losses_mdl, reps=1000, block_size=bs, seed=7); spa.compute()
print(spa.pvalues["consistent"])                # <- the one to report

stepm = StepM(losses_bm, losses_mdl, size=0.05, reps=1000, block_size=bs); stepm.compute()
print(stepm.superior_models)                    # WHICH ones survive FWER control

mcs = MCS(losses_mdl, size=0.10, reps=1000, block_size=bs); mcs.compute()
print(mcs.included, mcs.excluded)               # no benchmark argument
```

## PSR / DSR — no dominant package, so inline them

⚠️ Formula transcription below follows Bailey & López de Prado (2012/2014); verify against the papers
before publishing numbers.

```python
import numpy as np
from scipy.stats import norm, skew, kurtosis

def probabilistic_sharpe_ratio(r, sr_benchmark=0.0):
    """P(true SR > sr_benchmark). r and sr_benchmark must be on the SAME per-period scale."""
    T, sr = len(r), r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)          # g4 = NON-excess kurtosis
    return norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1)
                    / np.sqrt(1 - g3 * sr + ((g4 - 1) / 4) * sr**2))

def deflated_sharpe_ratio(r, n_trials, sr_variance):
    """PSR against the EXPECTED MAX Sharpe of n_trials independent backtests."""
    e = np.euler_gamma
    sr0 = np.sqrt(sr_variance) * ((1 - e) * norm.ppf(1 - 1 / n_trials)
                                  + e * norm.ppf(1 - 1 / (n_trials * np.e)))
    return probabilistic_sharpe_ratio(r, sr_benchmark=sr0)
```

🚨 **`n_trials` is the input that decides the answer, and no library can recover it for you.** It is
every configuration you actually tried — including the ones you abandoned, the parameter grid you ran
before narrowing it, and the universes you swapped. It is almost always under-reported by an order of
magnitude, which is what makes a DSR look reassuring.

⚠️ **A same-scale reminder:** `sr` here is *per-period*. Mixing an annualized `sr_benchmark` with a
daily `sr` produces a confidently wrong probability.

Sanity floor before any of this — the standard error of a Sharpe ratio under i.i.d. normality (Lo 2002):

```python
def sharpe_se(sr_annual, n_years, N=252):
    sr_p = sr_annual / np.sqrt(N)
    return np.sqrt((1 + sr_p**2 / 2) / (n_years * N)) * np.sqrt(N)
# sharpe_se(1.0, 3) -> ~0.58 : a 3-year Sharpe of 1.0 is not distinguishable from 0.
```

## Where the implementations live

| Package | Covers | Status |
|---|---|---|
| **`arch`** | SPA, RealityCheck, StepM, MCS, block bootstraps | ✅ 8.0.0, NCSA, active — **the reference** |
| **`purgedcv`** | `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio(_full)` alongside the CV | ✅ 0.1.6 (2026-09-04), MIT |
| **`jsharpe`** | PSR, MinTRL, Sharpe variance, power, **FWER/FDR (Bonferroni/Holm/Šidák)** | ✅ 0.6.3 (2026-07-09), MIT, py≥3.11, 22 ★ |
| **`ml4t-diagnostic`** | DSR, PBO, CPCV, RAS, HAC-adjusted IC | ✅ 0.1.2 (2026-08-17), MIT — 🚨 py≥3.12, Polars-first |
| **`RiskLabAI`** | PSR/DSR/PBO inside the full AFML stack | ✅ 3.1.0, BSD-3 — 🚨 py≥3.12 |
| **`statsmodels.stats.multitest.multipletests`** | Bonferroni, Holm, **BH-FDR**, Benjamini-Yekutieli | ✅ 0.15.0 (2026-08-27), BSD-3 |
| ⚠️ **`pypbo`** | reference PBO/CSCV, PSR, DSR, MinTRL, MinBTL | 🚨 **not on PyPI** (404 ✅) and 🚨 **AGPL-3.0** ✅ — clone only; AGPL rules it out of most commercial work |
| ⚠️ **`sharpebench`** | claims the whole surface, Rust kernel | v0.17.1 (2026-09-02) ✅, MIT OR Apache-2.0 — 🚨 **declares no repository URL on PyPI at all** ✅. Provenance unverifiable. **Do not default to it.** |
| 🔴 **`quantstats`** | `sharpe`, `sortino` — **no PSR, no DSR, no MinTRL** ✅ | do not look for them there |
| ⚠️ **`skfolio`** | `CombinatorialPurgedCV` — the *machinery* PBO needs, not a packaged PBO statistic | ✅ 1.0.3 |

❓ **Ledoit-Wolf's test for equality of two Sharpe ratios** (2008) has no maintained Python package;
the reference code is R/MATLAB. Approximate it with a studentized stationary bootstrap on the Sharpe
difference via `arch.bootstrap.StationaryBootstrap(...).conf_int(fn, reps=5000, method="studentized")`.

## Order of operations

1. Clean folds first — leakage invalidates every test below (`purgedcv.md`).
2. **Keep the per-period P&L of every candidate**, not just the winner. This is the decision that
   determines whether SPA/StepM/MCS are available to you at all.
3. Screen with SPA (`consistent` p-value) → identify survivors with StepM → narrow to MCS.
4. Report DSR with the honest `n_trials`, and the Sharpe standard error alongside the point estimate.
5. Cross-check the underlying Sharpe itself — see `../../portfolio-and-risk/references/analytics-libraries.md`,
   where the same series yields different Sharpes in different libraries.
