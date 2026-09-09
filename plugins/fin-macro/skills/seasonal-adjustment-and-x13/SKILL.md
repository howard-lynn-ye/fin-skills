---
name: seasonal-adjustment-and-x13
description: >-
  Seasonal adjustment is a second, silent vintage - the published seasonally adjusted history
  keeps changing with no new data. TRIGGER - seasonally adjusted, SA vs NSA, "why did last
  year's number change", concurrent seasonal adjustment, seasonal factors, X-13ARIMA-SEATS,
  X-13, X-12-ARIMA, x13_arima_analysis, X13NotFoundError, "x12a and x13as not found on
  path", X13PATH, statsmodels seasonal_decompose, STL, ratio to moving average, annual
  re-adjustment, benchmark revision, "the seasonal factors were revised", residual
  seasonality, seasonally adjusting a series myself. SKIP for revisions to the underlying
  unadjusted value (real-time-macro-backtesting), for release timing
  (macro-release-calendar-and-embargo), for where the series live
  (fundamental-and-macro-data), and for general time-series decomposition and forecasting
  (time-series-forecasting-models).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Seasonal adjustment and X-13

**A seasonally adjusted number is an estimate, and the agency re-estimates it every month.**
The unadjusted data for a month settles in about two months. The *published adjusted* value for
that same month keeps moving for years — not because anyone collected anything, but because
the seasonal factors were recomputed.

`../real-time-macro-backtesting/SKILL.md` owns revisions to the *value*. This skill owns the
revision that arrives with **no new information at all**, which is invisible to every diagnostic
built for the first kind.

Everything marked ✅ Measured is printed by `scripts/seasonal_adjustment.py` (numpy + pandas,
seed 20260909, **6 s**, no network; `statsmodels` probed inside a function). Everything marked
✅ source-verified was read at the URL given on 2026-09-09.

## 1. ✅ What the agency says it does

✅ source-verified at `bls.gov/web/empsit/cesseasadjtn.htm`, verbatim:

> *"The CES program employs a **concurrent** seasonal adjustment methodology to seasonally
> adjust its national estimates of employment, hours, and earnings."*
>
> *"Under concurrent methodology, **new seasonal factors are calculated each month using all
> relevant data up to and including the current month period**."*
>
> *"Once a year, BLS seasonally adjusts CES series again, **replacing the most recent 5 years
> of historical estimates** with newly seasonally adjusted estimates."*

— using *"the U.S. Census Bureau's X-13ARIMA-SEATS application."*

✅ source-verified at `bls.gov/web/empsit/cesvininfo.htm`: *"CES estimates are subject to
revisions for up to 2 months from their original publication due to ongoing receipt of sample
data."*

🔑 **Put those two together and the trap states itself.** Past the two-month window the
unadjusted number is settled. **Anything the published seasonally adjusted value does after
that is arithmetic**, and it happens twelve times a year, and again over five years of history
at each annual re-adjustment.

⚠️ The annual benchmark compounds it from the other direction: ✅ source-verified at
`bls.gov/web/empsit/cesbmart.htm`, *"Twenty-one months of not seasonally adjusted CES estimates
for all data types are revised based on this new March level, **prior to seasonal
adjustment**"*, and the March-2025 benchmark moved the SA level by **−898,000** against
−862,000 NSA. The extra 36,000 is the seasonal adjustment reacting to the new NSA history.
`../real-time-macro-backtesting/SKILL.md` §1 has that table month by month.

## 2. The mechanism, in one paragraph

X-11's core is ratio-to-moving-average: estimate the trend with a **centred** 2x12 moving
average, form SI ratios (`data / trend`), smooth each calendar month's SI ratios across years,
normalise. 🔑 **A centred filter does not exist for the last six months.** The newest factors
are one-sided extrapolations; when six more months arrive they become centred, and they change.
That is not a flaw in the implementation — it is why concurrent adjustment revises at all, and
no filter choice removes it.

`scripts/seasonal_adjustment.py` implements exactly that core in ~40 lines of numpy, so the
effect is visible rather than delegated to a binary that is not installed. It is **not** a
substitute for X-13: no RegARIMA pre-adjustment, no outlier detection, no trading-day or
holiday regressors, no SEATS.

## 3. ✅ Measured — how far a settled month's published value moves

Synthetic monthly series, 300 months, seasonal amplitude 8.2 % of level at its widest and
slowly evolving, irregular noise 0.35 %. Month-on-month movement in the **published** SA value
for a month `age` months back, in percent of level:

| age (months) | concurrent, mean | p95 | max | projected (annual), mean | new data possible? |
|---|---|---|---|---|---|
| 1 | 0.0213 | 0.0538 | 0.0949 | 0.0289 | yes |
| 2 | 0.0213 | 0.0538 | 0.0949 | 0.0288 | yes |
| 3 | 0.0213 | 0.0538 | 0.0949 | 0.0290 | **NO** |
| **6** | **0.2301** | **0.5638** | **1.0198** | 0.0174 | **NO** |
| 12 | 0.0157 | 0.0389 | 0.0693 | 0.0190 | **NO** |
| 24 | 0.0070 | 0.0173 | 0.0308 | 0.0098 | **NO** |

⚠️ Ages 1–3 move by an identical amount here because this simplified filter gives every month
inside the un-centred tail one shared end-normaliser; X-13's asymmetric end filters differ in
that detail. **The rows that carry the point are 6 and beyond.**

🚨 **The largest single rewrite lands at age 6** — the month the centred trend filter finally
reaches it: **0.23 % on average and 1.02 % at worst, in one month, on a number nobody collected
any new data about.** ✅ Measured over its life, a settled month's published SA value ends
**0.204 %** from where it first settled (max 0.576 %) and **travels 0.668 %** getting there.

On US total nonfarm's ~158 million level, 0.01 % is 15,800 jobs — so the 12-month-old row
(0.0157 %) is about **25,000 jobs a month arriving from nowhere**, on a series whose mean
absolute first-to-third *sample* revision is 51,000.

## 4. 🚨 Measured — the phantom signal

For each of 203 months, the over-the-month change **as it was published at the time** against
**as it appears in the series you download today**:

| | value |
|---|---|
| sd of the change, as published today | 601 |
| sd of the change, in real time | 1,089 |
| sd of the difference (revision) | 649 |
| **revision sd / published-today sd** | **1.08** |
| correlation between the two | 0.860 |

🚨 **26.1 % of the variation in the change you read today was not in the change that existed at
the time, and the sign differs in 20.2 % of months.** None of that is new data — every month
here is at least three months old. A momentum, acceleration or surprise feature computed on
today's SA history is partly reading a factor update that arrived *after* the decision it is
supposed to inform.

🔑 **And the re-adjustment is not wrong — that is what makes it dangerous.** ✅ Measured against
the true adjusted series: the settled concurrent estimate has RMSE **0.187 %**, the real-time
one **0.305 %**. Re-adjustment genuinely improves the history. It just makes it **a different
series from the one anyone traded**.

## 5. ✅ Measured — three regimes, and what each one buys

| regime | RMSE vs truth, settled | RMSE, real time | rewrite of a 12-month-old value |
|---|---|---|---|
| **concurrent** (CES today) | **0.187 %** | 0.305 % | 0.0157 % |
| projected, annual factors | 0.199 % | **0.283 %** | 0.0190 % |
| **frozen at the start** | 0.424 % | 0.424 % | **0.0000 %** |

🔑 **That is the whole trade in one table: accuracy of the history against stability of the
history.** Frozen factors are the least accurate on a series whose seasonality drifts, and they
are the only ones that never rewrite a published value. **A backtest needs the second column.**

**The practical form:** keep the **NSA vintage** series, estimate your own factors on the data
available at each decision date, and never let a later factor estimate touch an earlier
published value. If you must use the agency's SA series, carry its vintages
(`../real-time-macro-backtesting/SKILL.md` §2 lists the sources) and say which you used.

## 6. ✅ X-13ARIMA-SEATS in Python

✅ source-verified at `census.gov/data/software/x13as.html` (page updated 2025-07-10):
*"X-13ARIMA-SEATS is seasonal adjustment software produced, distributed, and maintained by the
Census Bureau"*, offering *"ARIMA model-based seasonal adjustment using a version of the SEATS
software"* as well as *"nonparametric adjustments from the X-11 procedure"*, distributed *"for
Windows® PC and Linux/Unix platforms"*.

🚨 **`pip install statsmodels` does not install the adjustment engine.** ✅ source-verified in
`statsmodels/tsa/x13.py` (statsmodels 0.15.0, read in the installed package):

```python
BINARY_NAMES = ("x13as", "x12a", "x13as_ascii", "x13as_html")
BINARY_NAMES += tuple(f"{name}.exe" for name in BINARY_NAMES)   # windows
```

`_find_x12` searches `X13PATH`, then `X12PATH`, then an explicit `x12path=`. With none of them,
✅ Measured by running it on this machine:

```
statsmodels.tools.sm_exceptions.X13NotFoundError:
x12a and x13as not found on path. Give the path, put them on PATH, or set the
X12PATH or X13PATH environmental variable.
```

The wrapper writes a spec file, shells out to the Census binary and parses its output; with no
binary it raises before doing any work. **Download the binary from Census and set `X13PATH`.**
🔑 That also means **X-13 is not reproducible from a `requirements.txt`** — pin the binary
version in your environment notes the way you pin a package.

⚠️ **What you are likely to reach for instead, and what it is not.**
`statsmodels.tsa.seasonal.seasonal_decompose` is a plain ratio-to-moving-average with a *single
fixed* seasonal profile — no evolving seasonality, no outlier handling, no calendar effects.
`STL` (Cleveland et al.) does allow the seasonal to evolve and is a reasonable in-house choice,
but neither reproduces an agency's published SA series, and using one to "check" an official
number compares two different estimators. Both are in
`../../../fin-models/skills/time-series-forecasting-models/SKILL.md`'s territory as
decomposition tools; the point here is that they are **your** adjustment, so they are stable and
point-in-time — which for a backtest is the property that matters.

## 7. Traps

- 🚨 **Treating the SA series as data rather than as an estimate.** §3–4.
- 🚨 **Computing a change or a momentum feature on today's SA history.** §4: a quarter of its
  variation and a fifth of its signs were not there at the time.
- 🚨 **Diagnosing SA revisions with a vintage tool built for value revisions.** A first-to-third
  revision table will show nothing: the NSA input never changed.
- 🚨 **Mixing SA and NSA in one model.** A ratio, a spread or a regression between an SA and an
  NSA series inherits the full seasonal on one side. Check both series' units and adjustment
  status before every join; FRED series IDs often differ only in a suffix.
- 🚨 **Residual seasonality after aggregation.** Summing or differencing independently adjusted
  components does not give an adjusted aggregate, and quarterly GDP has a documented history of
  residual seasonality. If your "adjusted" series still shows a calendar-month pattern in its
  residuals, it is not adjusted.
- 🚨 **Assuming an agency uses concurrent adjustment.** CES does — ✅ verified. Other programmes
  and other countries publish projected factors, sometimes a year ahead, and some publish the
  factors themselves. ⚠️ Check the specific programme's technical note; do not generalise from
  CES.
- 🚨 **Comparing your own STL/seasonal_decompose output with the official SA series and calling
  the difference an error.** They are different estimators. The comparison that means something
  is your own adjustment against *itself* at two vintages.
- 🚨 **The adjustment method is a trial.** Concurrent vs projected vs frozen, X-11 vs SEATS,
  the seasonal filter length, whether to log. Log them in
  `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`.

## ❓ Not verified

Which non-CES US programmes use concurrent versus projected factors · whether BEA's GDP
seasonal adjustment carries published vintages · the exact asymmetric end filters X-13 uses
(this script's simplified end handling is stated as such in §3) · the current X-13ARIMA-SEATS
version number (the Census page showed an update date of 2025-07-10 but no version in the text
that was retrieved).

## 8. Scripts and where this sits

`scripts/seasonal_adjustment.py` — the X-11 ratio-to-moving-average core with time-varying
factors, the three publication regimes (concurrent / projected / frozen) as
publication-date x reference-month matrices, the rewrite profile and settled-drift measures,
the phantom-signal decomposition, the accuracy comparison, and the X-13 availability probe.
numpy + pandas, seed 20260909, 6 s; `statsmodels` imported inside a function and the demo
degrades gracefully without it or without the Census binary.

- Revisions to the unadjusted value, and the vintage A/B —
  `../real-time-macro-backtesting/SKILL.md`.
- Why the unemployment rate barely revises and payrolls do —
  `../macro-regime-and-recession-indicators/SKILL.md` §4.
- When each print lands — `../macro-release-calendar-and-embargo/SKILL.md`.
- Feeding an adjusted or unadjusted series to a mixed-frequency factor model —
  `../gdp-nowcasting-dynamic-factor/SKILL.md`.
- Decomposition and forecasting tools in general —
  `../../../fin-models/skills/time-series-forecasting-models/SKILL.md`.
- Where the SA and NSA series come from —
  `../../../fin-core/skills/fundamental-and-macro-data/SKILL.md`.
