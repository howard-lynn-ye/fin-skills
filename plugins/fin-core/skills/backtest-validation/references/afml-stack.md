# The AFML / López de Prado stack in 2026

**Headline: the canonical library is gone.** `mlfinlab` is not installable as an open dependency, and
its GitHub source is stubbed. The live open-source stack is `RiskLabAI`, `purgedcv`, the new `ml4t/*`
family, and `skfolio`.

## 🔴 Hudson & Thames — status verified

### mlfinlab — dead as an open dependency
- ✅ **`pip install mlfinlab` FAILS.** `pypi.org/pypi/mlfinlab/json` → **HTTP 404**; the project page
  → 404; `/simple/mlfinlab/` renders **"Links for mlfinlab" with zero files** — the name is reserved,
  every distribution deleted. Subscribers get it from a private index.
- ✅ GitHub `hudson-and-thames/mlfinlab` — **not archived**, 4,916★ / 1.3k forks, but **repo last
  updated 2023-10-02, last commit 2021-12-01**, only **11 commits** in history, **no releases or tags**.
- 🔴 **THE SOURCE IS A STUB.** The module tree is complete (`labeling`, `cross_validation`,
  `sample_weights`, `sampling`, `bet_sizing`, `structural_breaks`, `features`,
  `microstructural_features`, `codependence`, `clustering`, `data_structures`, `feature_importance`,
  `backtest_statistics`, `ensemble`, `networks`, `regression`, `data_generation`, `multi_product`,
  `filters`, `util`, `datasets`) — but **every function body is `pass`**:

  ```python
  def ml_get_train_times(samples_info_sets, test_times) -> pd.Series:
      """ Advances in Financial Machine Learning, Snippet 7.1, page 106. ... """
      pass
  ```
  Verified across `get_bins`, `PurgedKFold.split()`, `ml_cross_val_score()`,
  `StackedPurgedKFold.split()`. **You get the API surface and AFML page references, nothing executable.**
- **Licence: proprietary, "all rights reserved."** *"The codebase is NOT open-source. All proprietary
  rights are reserved under the Hudson and Thames Quantitative Research copyright."* Research use
  only; commercial use needs a paid licence; no derivative works or redistribution.
- Pricing still advertised: Business **£100 +VAT / month / user**. ⚠️ The page still says
  `pip install mlfinlab`, which is **stale and misleading**.
- ✅ **Licence history:** the original release was **BSD 3-Clause** — confirmed via the Internet
  Archive capture of **2019-09-04**, whose README states *"This project is licensed under the 3-Clause
  BSD License."*
- **Business status:** the blog's most recent posts are **October 2023**. No 2024/2025/2026 posts
  observed. **Treat H&T as a stalled vendor.**

### arbitragelab — they DID open-source this one
✅ `pip install arbitragelab` works. **v1.0.0 (2024-05-12)**, **BSD 3-Clause** (verified in
`LICENSE.txt`), 692★, last updated 2024-05-19, `requires_python >=3.8,<4.0`.
Scope: statistical arbitrage / pairs trading (Krauss taxonomy), cointegration, optimal mean reversion,
ML pair selection. **Not an AFML replacement** — no triple barrier, no purged CV.
**Status: open-sourced, then abandoned.** Nothing since May 2024.

### portfoliolab — gone
`pip` → **404**. The repo exists (185★) under the same proprietary licence. This is where
`mlfinlab.portfolio_optimization` (HRP/NCO/denoising) went. **Superseded entirely by skfolio and
Riskfolio-Lib.**

## ✅ The live stack

### RiskLabAI — the most complete AFML reimplementation
`RiskLabAI` · `github.com/RiskLabAI/RiskLabAI.py` · **v3.1.0 (2026-08-26)** · **BSD-3** ·
🚨 `requires_python >=3.12,<3.15`
Cadence: 3.0.0 (2026-08-26), 2.0.1 (2026-06-20), 2.0.0 (2026-06-19), 1.0.8 (2026-02-22), 1.0.7
(2025-11-17), 1.0.0 (2025-11-09), 0.0.93 (2025-04-05), 0.0.0 (2022-08).
Author: Hamid Arian, York University. Julia companion `RiskLabAI.jl`.

⚠️ **Only 3 GitHub stars** — the repo appears recently re-created/re-orged, so the star count badly
understates maturity. **Judge it on the release history, not the stars.**

**Coverage (broadest of any open package):** triple-barrier, meta-labeling, trend-scanning; uniqueness
and time-decay weights; FFD fractional differentiation; Marcenko-Pastur denoising + targeted shrinkage;
purged and combinatorial purged CV, walk-forward; MDI/MDA/SFI incl. clustered variants; HRP, NCO,
hedging; **PSR/DSR/PBO**; microstructure, entropy, structural breaks; tick/volume/dollar bars.

**Credibility anchor:** Arian is first author of the peer-reviewed CPCV evaluation paper, so this is
the reference code behind that study.

```python
# RiskLabAI.data.labeling.labeling
meta_events(close, time_events, ptsl, target, return_min, num_threads,
            vertical_barrier_times=None, side=None) -> pd.DataFrame
triple_barrier(close, events, ptsl, molecule) -> pd.DataFrame
meta_labeling(events, close) -> pd.DataFrame
# helpers: symmetric_cusum_filter(), daily_volatility_with_log_returns(), vertical_barrier()
```

🚨 **Gotchas:** Python **≥3.12 only** — will not install on 3.9/3.10/3.11. Naming diverges from
mlfinlab (`meta_events` not `get_events`, `meta_labeling` not `get_bins`), so **mlfinlab tutorials do
not transfer verbatim.** 3.x reorganized the namespace toward causal-factor APIs.

### purgedcv — the best standalone purged/combinatorial CV
`purgedcv` · `github.com/eslazarev/purged-cross-validation` · **v0.1.5 (2026-08-30)** · **MIT** ·
`>=3.10` (3.10–3.14) · 31★, 117 commits · **18 releases since 2026-05-16** — very active. Has a
**JOSS paper**.

✅ **Genuinely sklearn-protocol compliant** — drops into `cross_val_score`, `GridSearchCV` and
`Pipeline` with no glue.

```python
from purgedcv import PurgedKFold, CombinatorialPurgedCV, deflated_sharpe_ratio
cv = PurgedKFold(n_splits=5, prediction_times=t0, evaluation_times=t1,
                 purge_horizon="1D", embargo="2D")
cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2,
                           prediction_times=t0, evaluation_times=t1)
```

Also `WalkForwardSplit`, `PurgedGroupKFold`, `purge()`, `apply_embargo()`,
`probabilistic_sharpe_ratio()`, `deflated_sharpe_ratio_full()`, `reconstruct_paths()`,
`path_metrics()`, and `audit_splitter()` (added 2026-08-29 — per-fold leakage diagnostics).

🔑 **Why it matters:** it takes `prediction_times`/`evaluation_times` **explicitly**, which forces you
to be honest about label horizons — the single most common source of AFML leakage.

### ml4t/* (Stefan Jansen) — the most actively maintained
Ten MIT repos under `github.com/ml4t`, **all updated 2026-09-03**. The production successor to
*Machine Learning for Trading* (3rd ed.).

| Repo | pip | ★ | Purpose |
|---|---|---|---|
| `engineer` | `ml4t-engineer` | 21 | **labeling, bars, leakage-safe datasets** |
| `diagnostic` | `ml4t-diagnostic` | 30 | **DSR, PBO, CPCV, feature importance** |
| `data` | `ml4t-data` | 45 | acquisition/storage |
| `backtest` | — | 25 | event-driven backtester |
| `live` | — | 20 | live trading + brokers |
| `models` | — | 14 | latent-factor / portfolio-learning |
| `skills` | — | 6 | **Apache-2.0 agent workflows — 60 skills, the closest peer to this repo** |
| `itch-parser` | — | 4 | Rust ITCH parser |

`ml4t-engineer` v0.1.3 (2026-08-12), MIT, `>=3.12,<3.15`: Polars-native, Numba JIT, 139 commits,
**60 features validated against TA-Lib at 1e-6**.
```python
from ml4t.engineer.labeling import triple_barrier_labels, atr_triple_barrier_labels
from ml4t.engineer.bars import VolumeBarSampler, DollarBarSampler, TickImbalanceBarSampler
```
`ml4t-diagnostic` v0.1.2 (2026-08-17), MIT: `deflated_sharpe_ratio`, `analyze_signal()`,
`PortfolioAnalysis`, PBO, calendar-aware CPCV, **RAS (Rademacher Anti-Serum)**, HAC-adjusted IC,
MDI/PFI/MDA/SHAP with consensus ranking. ❓ Exact import paths for PBO/CPCV unverified (docs 403).

🚨 **Both require Python ≥3.12 and are Polars-first, not pandas-first.** Young (0.1.x) — expect churn.

### jsharpe — Sharpe inference done properly
`jsharpe` · `tschm/jsharpe` · **v0.6.3 (2026-07-09)** · **MIT** · `>=3.11` · 22★.
PSR, MinTRL, Sharpe-ratio variance, power calculations, and **FWER/FDR corrections
(Bonferroni/Holm/Šidák)** for screening many strategies.
🔑 **The multiple-testing correction layer is the honest complement to DSR** — it makes the
number-of-trials problem explicit rather than a single fudge parameter.

### Read-only references
- **`mlfinpy`** (`baobach/mlfinpy`) v0.1.2 (2024-10-09), **MIT**, `>=3.11,<4.0`, 84★ — 🔑 **the only
  place to read a working, MIT-licensed version of mlfinlab's exact API** (`get_events`,
  `add_vertical_barrier`, `get_bins`, `barrier_touched`, `cusum_filter`, imbalance bars), which
  matters because upstream is stubbed. **~2 years stale** — fine to read and vendor, risky to depend on.
- **`pypbo`** (`esvhd/pypbo`) — reference PBO/CSCV, PSR, DSR, MinTRL, MinBTL. 🚨 **not on PyPI**
  (404 — clone it) and 🚨 **AGPL-3.0**. Latest commit 2026-07-06 after a 4-year gap — a one-off
  revival, not sustained maintenance. Deps pinned to `statsmodels 0.8.0`.
- ⚠️ **`sharpebench`** v0.17.1 (2026-09-02) covers this whole surface (MIT OR Apache-2.0, Rust kernel)
  but **declares no repository URL on PyPI at all** — provenance unverifiable. **New and unvetted; do
  not default to it.**

### 🔴 Dead / nonexistent
- **`fracdiff`** — v0.9.0 (2022-12-01), BSD-3, 338★. 🚨 **ARCHIVED by the owner 2023-12-16** and
  🚨 **`requires_python >=3.7.12,<3.10`** — it will **not install on Python 3.10+**. Use RiskLabAI's
  FFD, or vendor its ~200 lines.
- **`timeseriescv`** — v0.2 (2018-09-07), MIT. The design `purgedcv` later refined. 8 years stale.
- **`pbo`**, **`afml`**, **`finml`** → PyPI **404**. They do not exist. The `afml.structural_breaks`
  imports seen in some blog articles are a **local module in those articles**, not a package.
- **`pyfinlab`** — v0.0.30 (2021-12-31). A *portfolio management* wrapper, **not** AFML. Abandoned.
- **mlfinlab forks** (`integracore2/`, `hhy5277/`, `closedloop/`, `allensmile/`) are pre-licence-change
  snapshots. The code is real but **the legal status is murky** — H&T asserts retroactive
  all-rights-reserved terms. The defensible pre-change artifact is the **BSD-3 Internet Archive
  capture of 2019-09-04**. For new work prefer `mlfinpy` (MIT) or `RiskLabAI` (BSD-3).

## 🚨 A verified bug in `get_bins`, inherited by every faithful clone

`get_events` already knows which barrier was touched — it computed the times — but **`get_bins`
throws that away** and re-infers the label in `barrier_touched` by comparing the realized return to
thresholds. Two consequences:

1. **Threshold inconsistency.** `barrier_touched` tests `ret > np.log(1 + target) * events.loc[dt, "pt"]`
   — i.e. `pt · log(1+target)`. But the barrier was placed in `get_events` at `pt · target` on
   *arithmetic* returns, i.e. `log(1 + pt·target)`. **These are not equal**
   (`target=0.02, pt=2` → 0.0396 vs 0.0392), so borderline events are mislabeled: a touched barrier
   can be recorded as a vertical-barrier `0`, and vice versa.
2. **Forward-fill direction.** `prices = close.reindex(all_dates, method="bfill")` — **backfill**
   resolves a `t1` absent from the close index to the *next* available price, a forward-looking value.

Note `ret` is multiplied by `side` **before** `barrier_touched`, so pt/sl thresholds apply to the
side-adjusted return — correct for shorts only if `pt_sl` is symmetric.

**Fix: capture which barrier was hit inside `get_events` and carry it through, rather than
re-inferring from `ret`.**

Also: **`drop_labels(min_pct=.05)` runs recursively** and can silently delete an entire class.

## Recommended install

```bash
pip install RiskLabAI          # AFML methods, BSD-3, py>=3.12
pip install purgedcv           # purged/combinatorial CV + PSR/DSR, MIT, py>=3.10
pip install skfolio            # HRP/HERC/NCO/denoising + CPCV, BSD-3
pip install ml4t-engineer ml4t-diagnostic   # labeling/bars + DSR/PBO, MIT, py>=3.12
pip install jsharpe            # PSR/MinTRL + FWER/FDR, MIT
pip install Riskfolio-Lib      # HRP/HERC/NCO across 26 risk measures, BSD-3
# read-only: git clone https://github.com/baobach/mlfinpy     (MIT, stale)
# PBO ref:   git clone https://github.com/esvhd/pypbo         (AGPL, not on PyPI)
```

🚨 **RiskLabAI and the ml4t family both require Python ≥3.12**, which sets the floor for the whole
stack. On Python 3.11, `purgedcv` + `skfolio` + `jsharpe` still work.
