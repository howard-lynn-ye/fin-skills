---
name: factor-and-timeseries-research
description: >-
  Judge whether a cross-sectional factor predicts returns, and forecast financial series. TRIGGER
  - information coefficient, IC, quantile returns, factor decay, turnover, alphalens; Fama-French,
  Fama-MacBeth, PanelOLS, linearmodels, cross-sectional asset pricing; event study, abnormal
  returns, CAR, BHAR; Alpha101, Alpha158, symbolic alpha mining, gplearn; or forecasting with
  ARIMA, GARCH, volatility models, arch, Nixtla, statsforecast, mlforecast, sktime, darts, Prophet
  or a time-series foundation model. SKIP for computing the indicator itself (signal-construction)
  and for portfolio weights or Sharpe (portfolio-and-risk).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# Factor research and time-series forecasting

The libraries here are mostly healthy. The danger is that **their default cross-validation and
forward-return conventions do not purge anything**, so a leaky result looks like a clean API call.

## 1. Factor evaluation

### 1.1 alphalens-reloaded — and its forward-return convention

`alphalens-reloaded` 0.4.6 (2025-06-02, Apache-2.0, 642★) is alive but low-velocity. The original
`quantopian/alphalens` is dead (0.4.0, 2020) — do not use it.

🚨 **Verified in source:** `compute_forward_returns` does `pct_change(period).shift(-period)`, so
**the forward return for date *t* starts at *t*'s own price.** It never lags your factor. If your
factor is computed from date *t*'s close, alphalens is scoring you as if you traded that same close.

**Fix: lag the factor yourself before passing it in** — `factor.groupby(level=1).shift(1)` — or
build forward returns from the next open. The IC alphalens reports on an unlagged close-based factor
is not achievable.

### 1.2 Qlib Alpha158 / Alpha360

`pyqlib` 0.9.7 (MIT, 48,255★). ✅ **The label is `Ref($close,-2)/Ref($close,-1)-1`** — deliberately
leakage-safe: it trades at T+1's close and measures to T+2, so the signal at T is never scored
against a price it could see.

🚨 **The real trap is normalization:** `ZScoreNorm` is fit over `fit_start_time..fit_end_time`. Pass
the full sample and you leak the test distribution into every feature, silently. Set the fit window
to your training period only.

### 1.3 Alpha101 — a licensing problem, not a quality problem

🚨 The canonical Python port (`yli188`, 864★) and **nearly every other Alpha101 repo has no licence
file** = all rights reserved. You cannot legally use or redistribute them.
✅ **`Menooker/KunQuant` (Apache-2.0, active 2026-05) is the one safely usable implementation.**

### 1.4 Symbolic alpha mining

`gplearn` **was revived** — 0.4.3 shipped 2026-01-07 after a 3.7-year gap; now needs Python ≥3.11 and
sklearn ≥1.8. Notes calling it abandoned are stale.

🚨 Every expression the search evaluates is a trial. A genetic program that examines 10⁵ candidates
has a trial count of 10⁵ for Deflated Sharpe purposes. See `backtest-validation`.

### 1.5 Fama-French and asset-pricing regressions

✅ **Live-tested:** Ken French's ZIPs return HTTP 200; a downloaded file parsed to 1,200 monthly rows
through **202606**. `pandas-datareader` **0.11.1 (2026-06-24)** is itself revived and its
`famafrench` reader works (its parser was **silently wrong before 0.11.0** — re-pull anything you
fetched with 0.10.0). 🚨 `getFamaFrenchFactors` is dead (2 releases, both 2019-05-18).

**`linearmodels` 7.0** (2025-10-21, NCSA licence, 1,066★) is healthy and **remains the only real
Fama-MacBeth / asset-pricing option**. **`pyfixest` 0.60.0** is fast and modern but **has no
Fama-MacBeth** — complementary, not a replacement.

🚨 **Fama-MacBeth with overlapping returns understates standard errors.** Use Newey-West / HAC. With
panel data, choose the clustering dimension deliberately — clustering on the wrong dimension is how
a t-stat of 2 becomes a t-stat of 5.

### 1.6 Two negative results worth knowing

- 🚨 **No credible open-source Barra-style risk model exists.** Every replication found is
  unlicensed and abandoned (2017–2023). `cvxgrp/cvxrisk` (MIT, active) is credible but is a **risk
  *interface*, not an estimated model** — you still supply the factor exposures.
- 🚨 **Event studies have no usable library.** The only PyPI package, `eventstudy`, is **0.1a12 — an
  alpha from 2021, GPL-3.0, 69★**; alternatives top out at 12★. **Write it yourself.**
  `references/_event-study-method.md` has the full methodology: market-model estimation window,
  event window, CAR vs BHAR, cross-sectional t-statistics, and **Boehmer-Musumeci-Poulsen** for
  event-induced variance (which is the correction almost everyone omits).

## 2. Forecasting

| Need | Use | Note |
|---|---|---|
| Fast classical baselines at scale | **Nixtla `statsforecast`** | All five Nixtla packages Apache-2.0, all pushed within 3 weeks — healthiest forecasting ecosystem in Python |
| Tree/ML forecasting on panels | **`mlforecast`** | |
| Deep forecasting | **`neuralforecast`** | |
| Unified API, many estimators | `sktime` | ✅ **hit 1.0.0 on 2026-06-11** (now 1.1.0) — an API-stability commitment. ⚠️ **1,422 open issues** |
| Unified API, deep-learning leaning | `darts` | See the leakage warning below |
| **Volatility (GARCH/EGARCH/HAR)** | **`arch`** | The reference implementation. Also the home of SPA/StepM/MCS |
| Automated feature extraction | `tsfresh` | ✅ Its defaults are statistically correct — see §2.2 |

🚨 **Prophet** is maintained (1.4.0, 2026-08-15) but was **among M5's worst performers** and is
actively harmful for financial series. Listed for recognition; not recommended.

### 2.1 🚨 Verified leakage in the forecasting CV splitters

- **`sktime`'s `ExpandingWindowSplitter` and `SlidingWindowSplitter` have NO `gap` / purge
  parameter.** With a multi-step horizon, train and test overlap in the information they encode.
- **`darts.historical_forecasts` has no purge either.**
- 🔴 **Worse: darts' `retrain=False` makes every backtest point in-sample** — a one-word change turns
  an honest walk-forward into a fitted-on-everything curve.

**For financial labels with any horizon, use `purgedcv` or `skfolio`'s `CombinatorialPurgedCV`
instead of the forecasting libraries' own splitters.** See `backtest-validation` §2.

### 2.2 tsfresh's multiple-testing defaults are correct — don't "fix" them

✅ `HYPOTHESES_INDEPENDENT = False` by default, so `select_features` uses **Benjamini-Yekutieli**
(`fdr_by`), which is valid under **arbitrary dependence** — not plain Benjamini-Hochberg. Financial
features are heavily dependent, so this is the right choice. **Flipping it to get more features
through is exactly the mistake the parameter exists to prevent.**

Separately: extracting features across the train/test boundary leaks. Extract per-fold.

### 2.3 Time-series foundation models — honest evidence

**🚨 Licences — verified directly from the HuggingFace API, 2026-09-04. Do not assume Apache:**

| Model | Licence |
|---|---|
| `Salesforce/moirai-1.0-R-*`, `moirai-1.1-R-*`, `moirai-moe-1.0-R-*` | 🔴 **`cc-by-nc-4.0` — NON-COMMERCIAL, all weights** |
| `NX-AI/TiRex` | 🔴 `other` = **`nx-ai-community-license`** (Llama-3-derived, source-available, commercial terms aimed at large enterprises) |
| `google/timesfm-1.0-200m`, `2.0-500m-pytorch`, `2.5-200m-pytorch` | 🟢 `apache-2.0` |
| ⚠️ `google/timesfm-3.0-*` | ❓ **gated — returns HTTP 401, licence not readable anonymously.** Verify terms before use; do not assume it inherits 2.x's Apache grant |
| `amazon/chronos-t5-*`, `chronos-bolt-*` | 🟢 `apache-2.0` |
| `AutonLab/MOMENT-1-large` | 🟢 `mit` |
| `ibm-granite/granite-timeseries-ttm-r2` (Tiny Time Mixers) | 🟢 `apache-2.0` |

🚨 **`pip install tirex` installs an unrelated 2022 dimensionality-reduction package** — the real one
is **`tirex-ts`**.

**Does any of it work on prices?**
- On GIFT-Eval, **only Chronos-2 and TimesFM-2.5 beat classical AutoTheta at high frequency**, and
  benchmark contamination is documented.
- The directly relevant finance study (arXiv **2607.05291**: 9 TSFMs vs 8 econometric specifications,
  50 assets, realized volatility) found **only Tiny Time Mixers beat Log-HAR, narrowly** — and
  Mincer-Zarnowitz recalibration showed most of that edge was **better scaling, not better dynamics.**

**Conclusion: treat TSFMs as a baseline to beat, not a source of alpha.** A HAR model on realized
volatility remains hard to beat, and any TSFM claim should be checked against it.

## 3. The recurring traps

1. **Unlagged factors** in alphalens (§1.1) — the most common factor-research bug.
2. **Splitters with no purge** in sktime/darts (§2.1).
3. **Normalizers fit on the full sample** — Qlib's `ZScoreNorm`, any `StandardScaler` outside a
   `Pipeline`.
4. **Overlapping-return standard errors** in Fama-MacBeth — needs Newey-West.
5. **Survivorship in the factor universe** — a factor tested on today's index members is not a
   factor. See `research-integrity-guards` §1.
6. **`.shift()` sign errors** and **resample label placement** (`label=`, `closed=`).
7. **Search-space size as trial count** — gplearn, Optuna, AutoML. See `backtest-validation`.

## 4. Reference files

`references/<library>.md` for exact versions, licences, signatures and quirks;
`references/_event-study-method.md` for the methodology you have to implement yourself.
