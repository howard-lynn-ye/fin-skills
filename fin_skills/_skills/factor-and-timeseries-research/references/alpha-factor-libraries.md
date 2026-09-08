# Alpha factor libraries and factor data

## Qlib Alpha158 / Alpha360

`pyqlib` 0.9.7, MIT, 48,255★. Two ready-made feature sets you can use without adopting Qlib's
backtest stack.

✅ **The label is leakage-safe by design:** `Ref($close,-2)/Ref($close,-1)-1` — it trades at T+1's
close and measures to T+2, so a signal computed at T is never scored against a price it could see.
This is a deliberately conservative choice and it is worth copying.

🚨 **The real trap is normalization.** `ZScoreNorm` is fit over `fit_start_time..fit_end_time`. Pass
the full sample and you leak the test distribution into **every feature**, silently. **Set the fit
window to your training period only.**

Alpha158 is ~158 handcrafted price/volume expressions; Alpha360 is a raw 60-day × 6-field lookback
tensor. Alpha360 has far more parameters and correspondingly more overfitting risk.

## 🚨 WorldQuant Alpha101 — a licensing problem, not a quality problem

The canonical Python port (`yli188`, 864★) and **nearly every other Alpha101 repo has no licence
file** = all rights reserved. You cannot legally use or redistribute them, regardless of how widely
they are copied.

✅ **`Menooker/KunQuant` (Apache-2.0, active 2026-05) is the one safely usable implementation.**

The underlying formulas come from Kakushadze (2015), "101 Formulaic Alphas" (arXiv 1601.00991) —
the *paper* is public; the *code ports* are what carry the licence problem.

⚠️ On the alphas themselves: they were published in 2015 and are now in every retail toolkit.
Treat them as a feature basis and a baseline, not as alpha.

## gplearn — symbolic regression for alpha mining

✅ **Revived**: 0.4.3 shipped **2026-01-07** after a 3.7-year gap; now requires **Python ≥3.11 and
sklearn ≥1.8**. Notes calling it abandoned are stale.

🚨 **The overfitting critique is the important part.** A genetic program that evaluates 10⁵ candidate
expressions has a **trial count of 10⁵** for Deflated Sharpe purposes. The search will find an
expression that fits any sample. Mitigations that actually help:
- Hold out a period the search never sees, and evaluate exactly once.
- Constrain the function set and depth — unconstrained operators produce unfalsifiable expressions.
- Record the search size in your trial ledger (`../../backtest-validation/scripts/trial_ledger.py`).
- Require the surviving expression to be **economically interpretable**. "It worked" is not a reason.

## Fama-French factor data in 2026 — ✅ live-tested

✅ **Ken French's ZIPs return HTTP 200**; a downloaded file parsed to 1,200 monthly rows through
**202606**.

✅ **`pandas_datareader` 0.11.1 (2026-06-24) works** and its `famafrench` reader is the easy path.
🚨 **Its parser was silently wrong before 0.11.0** (French changed the file format) — **re-pull
anything you fetched with 0.10.0.**

```python
import pandas_datareader.data as web
ff = web.DataReader("F-F_Research_Data_Factors", "famafrench")   # dict of frames
ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench")
mom = web.DataReader("F-F_Momentum_Factor", "famafrench")
```

**Parsing gotchas** (they apply whether you use the reader or the raw ZIP):
- Factor values are in **percent**, not decimals. Divide by 100 before combining with returns.
- A single file contains **multiple tables** (monthly, then annual) separated by blank lines — the
  reader returns a dict keyed by table index; the raw file needs manual splitting.
- Dates are `YYYYMM` integers for monthly files, `YYYYMMDD` for daily.
- The last row is often a partial period.

🔴 **`getFamaFrenchFactors` is dead** — 2 releases, both 2019-05-18.
⚠️ **`famafrench`** (Christian Jauregui) is a different, more ambitious package, and stale.

**Recommended order:** `pandas_datareader.famafrench` → the raw ZIPs from French's site →
anything else.

## 🚨 Open-source Barra-style risk models — nothing credible exists

Every replication found is **unlicensed and abandoned** (2017–2023). The one credible project,
`cvxgrp/cvxrisk` (MIT, active), is a **risk *interface*, not an estimated model** — you still supply
the factor exposures.

If you need a factor risk model, the realistic options are: build one (PCA or a fundamental factor
set estimated yourself), buy one (Barra, Axioma, Northfield), or use a shrunk/denoised sample
covariance and be honest that it is not a factor model. See `../../portfolio-and-risk/references/_solver-layer.md`.

## Event studies

🚨 The only PyPI package, `eventstudy`, is **0.1a12 — an alpha from 2021, GPL-3.0, 69★**;
alternatives top out at 12★. **Write it yourself** — see `_event-study-method.md` in this directory
for the full methodology including Boehmer-Musumeci-Poulsen.
