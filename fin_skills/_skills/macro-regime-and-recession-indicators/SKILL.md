---
name: macro-regime-and-recession-indicators
description: >-
  Recession probabilities, the Sahm rule and yield-curve inversion - and the fact that the
  NBER label they are all scored against was assigned years after the fact. TRIGGER - USREC,
  NBER recession dates, recession indicator, recession probability, recession dummy,
  "recession-aware" strategy, regime flag from FRED, RECPROUSM156N, Sahm rule, SAHMREALTIME,
  SAHMCURRENT, "unemployment rose 0.5 points", yield curve inversion, T10Y3M, T10Y2Y, 2s10s,
  inverted curve recession signal, "how long after inversion", business cycle dating,
  "when did NBER announce", labelling recessions for a classifier. SKIP for fitting HMMs and
  the smoothed-versus-filtered timing of estimated regimes (regime-detection), for the
  vintage A/B on a macro strategy (real-time-macro-backtesting), for nowcasting GDP itself
  (gdp-nowcasting-dynamic-factor), and for where the underlying series live
  (fundamental-and-macro-data).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Macro regime and recession indicators

**The NBER does not tell you that you are in a recession. It tells you, on average seven
months later, that you were.** Every recession indicator on FRED is scored against a label
with that lag baked in, and the label itself is served as a clean monthly series with a value
for every month of its long history, including the months nobody could have labelled.

`../../../fin-core/skills/regime-detection/SKILL.md` owns the estimators — HMMs, the Hamilton
filter, smoothed-versus-filtered-versus-predicted probabilities. This skill owns the **label**
and the three published indicators, and it never fits a regime model.

Everything marked ✅ Measured comes from `scripts/recession_indicators.py` (numpy + pandas +
scipy, seed 20260909, **12 s**, no network). §1 is arithmetic on the NBER's own published
announcement dates; §§2–4 run on a synthetic economy whose recession lengths and announcement
lags are the real US ones.

## 1. ✅ The announcement lag, from the committee's own table

✅ source-verified at
`nber.org/research/business-cycle-dating/business-cycle-dating-procedure-frequently-asked-questions`,
read 2026-09-09. The committee publishes the lag itself: *"There is no fixed timing rule
because the committee waits long enough to avoid any doubt about the existence of a peak or
trough"*, and *"The committee also allows sufficient time for standard data revisions in order
to assign an accurate peak or trough date."*

| turning point | announced | elapsed months | | turning point | announced | elapsed months |
|---|---|---|---|---|---|---|
| **peak Jan 1980** | 1980-06-03 | 5 | | **trough Jul 1980** | 1981-07-08 | 12 |
| **peak Jul 1981** | 1982-01-06 | 6 | | **trough Nov 1982** | 1983-07-08 | 8 |
| **peak Jul 1990** | 1991-04-25 | 9 | | **trough Mar 1991** | 1992-12-22 | **21** |
| **peak Mar 2001** | 2001-11-26 | 8 | | **trough Nov 2001** | 2003-07-17 | **20** |
| **peak Dec 2007** | **2008-12-01** | **12** | | **trough Jun 2009** | 2010-09-20 | 15 |
| **peak Feb 2020** | 2020-06-08 | 4 | | **trough Apr 2020** | **2021-07-19** | **15** |

✅ Measured — the script recomputes every lag from the two dates and **all 12 match the
column NBER prints**. Peaks: mean **7.3** months, range 4–12. Troughs: mean **15.2** months,
range 8–21.

🚨 **The December 2007 peak was announced on 2008-12-01.** Lehman had already failed. And the
April 2020 trough was announced on 2021-07-19 — fifteen months after the recovery began.

### The number that follows

✅ Measured on those dates, using FRED's own USREC convention (*"the recession begins the
first day of the period following a peak and ends on the last day of the period of the
trough"*): a recession month is knowable only once the **peak** has been announced.

| recession | months | peak announced | unlabelled at the time | share | trough lag |
|---|---|---|---|---|---|
| 1980-02 → 1980-07 | 6 | 1980-06 | 4 | 67 % | 12 |
| 1981-08 → 1982-11 | 16 | 1982-01 | 5 | 31 % | 8 |
| 1990-08 → 1991-03 | 8 | 1991-04 | **8** | **100 %** | 21 |
| 2001-04 → 2001-11 | 8 | 2001-11 | 7 | 88 % | 20 |
| 2008-01 → 2009-06 | 18 | 2008-12 | 11 | 61 % | 15 |
| 2020-03 → 2020-04 | 2 | 2020-06 | **2** | **100 %** | 15 |
| **total** | **58** | | **37** | **63.8 %** | |

🚨 **64 % of US recession months since 1980 carried no NBER recession label while they were
happening. USREC has one for every single month.** The 1990–91 recession was over before it
was declared.

## 2. 🚨 The vintage label does not abstain — it disagrees

✅ source-verified at `fred.stlouisfed.org/series/USREC`: *"This time series is an
interpretation of US Business Cycle Expansions and Contractions data provided by The National
Bureau of Economic Research (NBER)."* An interpretation of a dated list — so the vintage
USREC for an undeclared recession is not missing. **It is a published, confident zero**, and
after the recession ends it stays at one until the trough is announced.

```python
def usrec_known_at(state, peak_lag, trough_lag, d):
    out = np.zeros(state.size)                    # NOT NaN - that is the whole point
    for i, (s, e) in enumerate(episodes(state)):
        if s + peak_lag[i] > d:
            continue                              # peak not announced: this reads 0
        stop = e if e + trough_lag[i] <= d else min(d, state.size - 1)
        out[s:stop + 1] = 1.0                     # and it overstays at the other end
    out[d + 1:] = np.nan
    return out
```

The honest reconstruction is ALFRED's `USREC` vintages, or this lag applied to the dated list.
The `../real-time-macro-backtesting/SKILL.md` `as_of()` pattern is the same shape.

## 3. ✅ Measured — how much of "recession-aware" is the label

The synthetic economy uses the **real** US recession lengths since 1980 (6, 16, 8, 8, 18, 2
months) and the **real** NBER announcement lags, so its live-label statistics are comparable
to §1's. 900 months, 9 recessions, 9.4 % of months in recession; everything is scored on the
last 480.

✅ Measured — the same series, read two ways:

| signal / training scheme | AUC | median delay | false alarm |
|---|---|---|---|
| **USREC as a signal, retrospective** | **1.000** | 0 | 0.0 % |
| **USREC as a signal, as of that month** | **0.611** | **6 months** | 13.2 % |
| classifier, full-sample retrospective label | 0.959 | 2 | 0.7 % |
| classifier, walk-forward retrospective label | 0.952 | 2 | 0.9 % |
| classifier, walk-forward **vintage** label | 0.949 | 2 | 0.9 % |

🚨 **+0.389 of AUC** between reading USREC retrospectively and reading it as of the month.
Live, it catches **35 %** of recession months at **23 %** precision — which reproduces §1's
real-data 36 %, because the synthetic cycle inherits the real lengths and lags. **A perfect
recession indicator is what USREC looks like from the future.**

🔑 **But a fitted classifier barely notices the label.** ✅ Measured over 12 seeds:

| training window | AUC, retrospective label | AUC, vintage label | difference | se |
|---|---|---|---|---|
| expanding | 0.953 | 0.951 | **0.002** | 0.001 |
| 240 months | 0.942 | 0.935 | **0.007** | 0.002 |
| 120 months | 0.772 | 0.861 | −0.090 | **0.041** |

The mislabelled rows are 0.17 % of an expanding training window (4.3 % of its last 24 months),
so the fitted coefficients hardly move; the cost is **two orders of magnitude below** the 0.389
that reading USREC as the signal costs. At a 120-month window the comparison dissolves into
noise and changes sign. 🔑 **So the fix is not "retrain on vintage labels". It is: never use
USREC as a live flag, and never evaluate against it as though it were one.**

## 4. ✅ The Sahm rule, and what its own inputs do to it

✅ source-verified, `fred.stlouisfed.org/series/SAHMREALTIME`, verbatim: *"Sahm Recession
Indicator signals the start of a recession when the three-month moving average of the national
unemployment rate (U3) rises by 0.50 percentage points or more relative to the minimum of the
three-month averages from the previous 12 months."*

🔑 **FRED publishes the same rule twice.** `SAHMREALTIME` adds: *"This indicator is based on
'real-time' data, that is, the unemployment rate (and the recent history of unemployment
rates) that were available in a given month."* `SAHMCURRENT`'s note is ✅ verified identical
minus that paragraph — same rule, today's series. **The two series exist because the answers
differ, and only one of the two pages tells you which object you are holding.**

✅ source-verified on the same page, the reason U3 was chosen: *"The BLS revises the
unemployment rate each year at the beginning of January… Revisions to the seasonal factors can
affect estimates in recent years. Otherwise the unemployment rate does not revise."* Compare
payrolls, whose mean absolute first-to-third revision is 51k
(`../real-time-macro-backtesting/SKILL.md` §1). **The rule's robustness is a design choice
about its input, not a property of thresholds.**

✅ Measured, with annual seasonal-factor revisions of 0.075 pp reaching back five years:

| what | value |
|---|---|
| mean absolute real-time-minus-revised Sahm value | **0.0212 pp** (max 0.130) |
| months landing on opposite sides of the 0.50 threshold | 6 months (**0.68 %**) |
| recessions where both versions first fire in the same month | **6 of 7** |
| median months from recession start to the first vintage signal | **4** |

Push the input revisions up and the hard threshold is what breaks — ✅ Measured:

| seasonal-factor sd (pp) | mean abs gap | max abs gap | threshold flips | same trigger month | false alarms |
|---|---|---|---|---|---|
| 0.000 | 0.0000 | 0.000 | 0 | 7/7 | 0 |
| **0.075** | 0.0212 | 0.130 | 6 | **6/7** | 0 |
| 0.150 | 0.0459 | 0.277 | 13 | 5/7 | 4 |
| 0.300 | 0.1098 | 0.628 | 69 | 4/7 | 9 |
| 0.600 | 0.2703 | 1.330 | 223 | 4/7 | **32** |

🚨 **The value moves by hundredths and the SIGNAL moves by months.** A hard threshold converts
a rounding-level input revision into a different trigger date and, past 0.15 pp, into false
alarms out of nothing. Any indicator of the form `x > c` inherits this; report the distance to
the threshold, not just the boolean.

## 5. Yield-curve inversion

✅ Measured on the synthetic economy, where the curve is *built* to invert 4–22 months ahead of
each recession plus six inversions that lead to nothing — so the generative claim is explicit
rather than discovered:

| | value |
|---|---|
| inversion episodes / recessions | 13 / 9 |
| lead, first inversion to recession start | **median 14 months, IQR 8–15, range 5–18** |
| inversions with no recession inside 24 months | **4** |
| recessions preceded by an inversion within 24 months | 9 of 9 |

🔑 **The lead is the problem, not the hit rate.** A signal with an interquartile range of seven
months is not a position size, and it is not a stop. And the only way to score whether a given
inversion was a false alarm is the NBER label — which arrives **7.3 months after a peak and
15.2 after a trough**. You cannot backtest the indicator faster than the referee reports.

⚠️ The lead distribution here is the script's own assumption, made visible. Do not read these
as US historical lead times; read the mechanism.

## 6. Traps

- 🚨 **`USREC` as a regime flag, a feature, or a target.** §1 and §3. It is a retrospective
  interpretation of a dated list. If you need a live one, use ALFRED's `USREC` vintages, or
  apply the announcement lag to the dated list yourself.
- 🚨 **`RECPROUSM156N` and the other "smoothed recession probability" series are model
  output, not observation** — and they are estimated on the current vintage of their inputs.
  ⚠️ Whether each such series is available with ALFRED vintages is per-series; check before
  designing a study around it.
- 🚨 **Reporting a boolean instead of a distance.** §4: `sahm >= 0.50` flips on a 0.02 pp
  wobble when the value sits at 0.49. Log the value, and log how far it is from the threshold.
- 🚨 **`SAHMCURRENT` in a backtest.** It is the rule computed on today's unemployment series.
  `SAHMREALTIME` is the one that existed. Nothing in the value tells you which you have.
- 🚨 **Scoring a leading indicator against the label it leads.** An inversion 14 months before
  a recession looks like a false alarm for 14 months, and then like genius — and both readings
  need a turning point that will not be dated for another year.
- 🚨 **Fitting a regime model and calling the smoothed probability a signal.** That is a
  different look-ahead and `../../../fin-core/skills/regime-detection/SKILL.md` measures it:
  a full-sample fit anticipated 47 % of regime switches before they started. Both bugs can be
  in the same pipeline.
- ⚠️ **The 2020 recession is two months long.** Any indicator with a 3-month moving average
  cannot resolve it, and any train/test split that straddles it is dominated by it. State how
  you handled 2020 or the result is not comparable to anyone else's.
- 🚨 **A recession indicator is a trial.** Threshold, smoothing window, which spread, which
  unemployment measure. Log them in
  `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`.

## ❓ Not verified

Whether ALFRED carries per-observation vintages for `USREC` specifically (the announcement
dates in §1 are the primary source either way) · the historical US inversion-to-recession lead
distribution — §5's numbers are the script's own DGP, not history · `RECPROUSM156N`'s
estimation vintage.

## 7. Scripts and where this sits

`scripts/recession_indicators.py` — the NBER announcement table with its own self-check, the
unlabelled-month computation, `usrec_known_at()` / `live_usrec()`, the classifier comparison
and the 12-seed label-cost sweep, the Sahm rule with its vintage/revised comparison and
threshold-tolerance sweep, and the inversion record. numpy + pandas + scipy, seed 20260909,
12 s, no network.

- HMMs, the Hamilton filter, and smoothed vs filtered vs predicted regime probabilities —
  `../../../fin-core/skills/regime-detection/SKILL.md`. That skill owns the estimator; this
  one owns the label.
- The vintage A/B for a macro strategy, and `as_of()` —
  `../real-time-macro-backtesting/SKILL.md`.
- Why unemployment's seasonal factors move at all —
  `../seasonal-adjustment-and-x13/SKILL.md`.
- Nowcasting the quarter you are in — `../gdp-nowcasting-dynamic-factor/SKILL.md`.
- When the underlying prints land — `../macro-release-calendar-and-embargo/SKILL.md`.
- Where `USREC`, `SAHMREALTIME` and `T10Y3M` come from —
  `../../../fin-market-data/skills/fundamental-and-macro-data/SKILL.md`.
- Whether a regime-conditional result survives its own sample —
  `../../../fin-core/skills/regime-detection/scripts/regime_coverage.py`.
