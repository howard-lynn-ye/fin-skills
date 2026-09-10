---
name: real-time-macro-backtesting
description: >-
  Run a macro strategy twice - once on today's revised series and once on the vintage that
  existed at each decision date - and report both Sharpes. TRIGGER - real-time data, vintage
  data, data vintages, point-in-time macro, ALFRED, realtime_start, realtime_end, vintage_dates,
  get_series_as_of_date, first release vs latest, initial estimate, "my macro backtest uses
  revised data", "does this have look-ahead", payroll revisions, GDP revisions, annual benchmark
  revision, QCEW benchmark, restated macro history, as-of join on a macro series, "which number
  did I actually see on the day". SKIP for where to GET the series and the fredapi bugs
  (fundamental-and-macro-data), for release times and embargo mechanics
  (macro-release-calendar-and-embargo), for seasonal-adjustment revisions specifically
  (seasonal-adjustment-and-x13), for recession labels assigned after the fact
  (macro-regime-and-recession-indicators), and for company fundamentals rather than macro
  (fundamental-and-macro-data).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Real-time macro backtesting

**A macro backtest has two answers and you have to print both: the one on today's series, and
the one on the numbers that were actually on the wire.** They are different objects. The first
is a study of a series that did not exist; the second is a strategy.

`../../../fin-market-data/skills/fundamental-and-macro-data/SKILL.md` owns where the series come from
— FRED/ALFRED, the Philadelphia Fed RTDSM, the three verified `fredapi` 0.5.2 bugs, the SDMX
licence table. This skill starts one step later: you have the vintages, now what does the A/B
measure, and which of the two look-aheads is actually costing you.

Everything marked ✅ Measured is printed by `scripts/vintage_backtest.py` (numpy + pandas, seed
20260909, about **11 s**, no network — the vintage panel is synthetic and built in the script).
Everything marked ✅ source-verified was read at the URL given, on 2026-09-09.

## 1. ✅ The size of the problem, from BLS's own table

BLS publishes its own revision record. ✅ source-verified at
`bls.gov/web/empsit/cesnaicsrev.htm` ("Summary of ABSOLUTE MEAN revisions between nonfarm
payroll employment over-the-month estimates, 1979-present", page last modified 2026-09-04),
**mean absolute** revision to the over-the-month change, thousands of jobs:

| period | SA 2nd−1st | SA 3rd−2nd | **SA 3rd−1st** | NSA 2nd−1st | NSA 3rd−2nd | NSA 3rd−1st |
|---|---|---|---|---|---|---|
| 1979–2003 | 48 | 29 | **61** | 46 | 53 | 83 |
| **2003–present** | 33 | 34 | **51** | 46 | 18 | 53 |
| all periods | 41 | 31 | **57** | 46 | 36 | 68 |

**51,000 jobs, seasonally adjusted, since 2003** — between the print that moved the market and
the third estimate two months later. That is the same order as the month-to-month signal most
payroll strategies trade.

🚨 **And the third estimate is not final.** ✅ source-verified at `bls.gov/web/empsit/cesbmart.htm`
(CES National Benchmark Article): CES is benchmarked once a year to the QCEW count of
UI-covered employment — *"About 97 percent of total nonfarm employment within the scope of the
establishment survey is covered by UI"* — and *"Twenty-one months of not seasonally adjusted CES
estimates for all data types are revised based on this new March level"*. The March-2025
benchmark: *"total nonfarm employment had a revision of −898,000 or −0.6 percent"* seasonally
adjusted, −862,000 NSA. Its own Table 1, the effect on 2025's **over-the-month changes** in
thousands, ✅ Measured off that transcription:

| | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as published | **111** | 102 | 120 | 158 | 19 | −13 | 72 | −26 | 108 | −173 | 56 | 50 |
| as revised | **−48** | 42 | 67 | 108 | 13 | −20 | 64 | −70 | 76 | −140 | 41 | 48 |
| difference | **−159** | −60 | −53 | −50 | −6 | −7 | −8 | −44 | −32 | +33 | −15 | −2 |

✅ Measured: mean absolute difference **39.1k**, largest **159k**, and **January 2025 changes
sign** — +111k on the day, −48k in the series you download now. A payroll-momentum backtest run
today sees January 2025 as a month of job losses. Nobody could have traded that. The count of
months printed as losses goes from 3 to 4.

✅ source-verified on the same page: *"Over the prior 10 years, the annual benchmark revision
at the total nonfarm level has averaged **0.2 percent** (in absolute terms), with a range of
less than 0.05 percent to **0.4 percent**."* 2025's −0.5 % is outside that range. Read the
level of each benchmark, not the long-run average.

## 2. ✅ What ALFRED gives you that FRED does not

✅ source-verified at `alfred.stlouisfed.org`: ALFRED is *"Archival Federal Reserve Economic
Data"* and lets you *"retrieve each economic data release (vintage) that was available on a
specific date in history."* FRED serves one vintage — the current one.

🚨 **The API default is the trap, and it is silent.** ✅ source-verified in the FRED API docs
(`fred.stlouisfed.org/docs/api/fred/series_observations.html`): `realtime_start` and
`realtime_end` both **default to "today's date"**. A bare `series/observations` call is
therefore a *current-vintage* pull that looks exactly like a historical one. The three
parameters that change that:

| parameter | what it does |
|---|---|
| `vintage_dates` | *"used to download data as it existed on these specified dates in history"*; default: none set |
| `realtime_start` / `realtime_end` | the real-time period; **default `today`** for both |
| `output_type=4` | *"Observations, Initial Release Only"* — the first print for every reference period, in one call |

`output_type` 1–4 are *"Observations by Real-Time Period"*, *"by Vintage Date, All
Observations"*, *"by Vintage Date, New and Revised Observations Only"*, and *"Initial Release
Only"*. `output_type=4` is the fastest honest answer to "what did I see on the day", and it is
the one nobody uses because the wrapper does not surface it.

**Vintage sources FRED/ALFRED does not reach**, and where to go instead:

| source | what it holds | ✅ verified |
|---|---|---|
| **BLS CES vintage data**, `bls.gov/web/empsit/cesvininfo.htm` | *"the CES published employment values for a given reference month across time"* — SA Excel per supersector, SA and NSA CSV for detailed industries. *"Total nonfarm has reference months available back to 1939"* | read 2026-09-09 |
| BLS CES benchmark article | the annual benchmark, month by month (§1) | read 2026-09-09 |
| Philadelphia Fed RTDSM | 244-vintage matrices back to 1965Q4 | ⚠️ taken from `../../../fin-market-data/skills/fundamental-and-macro-data/SKILL.md` §5, which owns it — not re-verified here |

✅ source-verified on the CES vintage page: *"CES estimates are subject to revisions for up to 2
months from their original publication due to ongoing receipt of sample data."* That is the
2-month window; the benchmark is a separate, later event.

## 3. The reconstruction, and the mistake inside it

A vintage panel is long: one row per *publication*, keyed by `(reference_period,
release_date, value)`. The as-of read is four lines:

```python
def as_of(panel, when):
    known = panel[panel["release"] <= when]          # nothing published later exists yet
    return (known.sort_values("release", kind="mergesort")
            .drop_duplicates("ref", keep="last")     # newest vintage AMONG THOSE
            .set_index("ref")["value"].sort_index())
```

🚨 **`groupby("ref").last()` over the whole panel is the bug**, and it is one filter away from
the fix: without the `release <= when` line it takes the newest vintage regardless of when it
appeared. ✅ Measured on the script's panel, reference month 200 published four times:

| released in month | value | which estimate |
|---|---|---|
| 201 | 318.7k | 1st print |
| 202 | 290.3k | 2nd estimate |
| 203 | 244.5k | 3rd estimate |
| 207 | 297.1k | annual benchmark |

Truth 244.5k. Four numbers, each correct on its own date. `as_of(panel, 201)` gives 318.7k,
`as_of(panel, 203)` gives 244.5k, `as_of(panel, 479)` gives 297.1k.

🔑 **The trailing window has to be vintage too.** A rule that compares this month's print with
its trailing 12-month mean must take that mean out of the *same* vintage row, not out of
today's series. Lagging only the newest point is the half-fix that everyone ships.

## 4. ✅ Measured — the panel, and the A/B

The script builds the vintage panel from a latent truth series plus **nested-sample** revision
noise: the second estimate contains the first's respondents plus late ones, so
`var(e_k − e_j) = s²(1/n_j − 1/n_k)` and the stage variances add. Two stage variances are set
from BLS's mean absolute first-to-third revision alone; the other two moments fall out:

| revision | BLS 2003–present | this panel |
|---|---|---|
| 2nd − 1st | 33 | **33.9** |
| 3rd − 2nd | 34 | **34.6** |
| 3rd − 1st (targeted) | 51 | **49.7** |
| **final − 1st** (incl. benchmark) | not published | **71.0** |

Only the third row was targeted; the first two land within 1.5k of BLS's own numbers, which is
the check that the nested-sample shape is right. ✅ Measured: **first-to-final is 1.43x the
first-to-third everyone quotes**, and the sign of the over-the-month change flips between the
first print and the final value in **10.8 %** of months (7.7 % by the third estimate alone).

One rule — long the market when the newest payroll change is above its own trailing 12-month
mean, short otherwise — fed three ways. 480 months, seed 20260909:

| treatment | Sharpe | gap | hit rate | disagrees with vintage |
|---|---|---|---|---|
| **vintage** (only what was published) | **0.187** | — | 51.3 % | — |
| **revised** (today's values, right reference month) | 0.471 | +0.284 | 54.6 % | **14.7 %** |
| **ref-date** (today's values, reference month traded in itself) | **0.875** | **+0.687** | 59.1 % | **38.5 %** |

Over 200 seeds x 480 months, because one seed is one draw:

| treatment | mean Sharpe | sd | p05 | p95 | beats vintage |
|---|---|---|---|---|---|
| vintage | 0.485 | 0.174 | 0.181 | 0.752 | — |
| revised | 0.506 | 0.171 | 0.209 | 0.783 | **58.0 %** |
| ref-date | **0.797** | 0.161 | 0.549 | 1.036 | **93.5 %** |

## 5. 🚨 The result that changes what you look for

🚨 **At payrolls' own revision size, using revised VALUES is worth almost nothing — and the
reference-date join is worth a third of a Sharpe.** ✅ Measured, mean Sharpe gap over the
vintage run, 60 seeds x 480 months per cell; rows scale BLS's revision magnitudes, columns are
the momentum window in months:

| revision size | **revised values** w=1 | w=3 | w=12 | | **ref-date join** w=1 | w=3 | w=12 |
|---|---|---|---|---|---|---|---|
| 1x (payrolls) | 0.021 | 0.046 | 0.029 | | −0.031 | 0.114 | **0.348** |
| 2x | 0.076 | 0.100 | 0.056 | | 0.024 | 0.162 | **0.322** |
| 4x | **0.153** | **0.165** | **0.105** | | 0.103 | 0.234 | **0.310** |

🔑 **The two bugs have different signatures, and that is the diagnostic.** The value gap rises
monotonically with the revisions at every window — 4x the revision size, 3.6x to 7x the gap.
The timing gap barely moves with revision size at all, because it is not a data-quality problem:
it is a month of the future. **If your Sharpe collapses under the A/B, look at the join before
you look at the vintages.**

⚠️ The magnitudes are DGP-specific. In this DGP returns load on the true state of the economy
(0.13 on last month's, 0.30 on this month's) and a published estimate is a noisy reading of it;
that is why a *better* thermometer helps only a little at payroll-sized noise. If your signal is
the release *surprise* rather than the level, the market reacts to the print itself and the
mechanism is different — that is a separate skill's problem, not this one's. Read the ordering
and the scaling, not the decimals.

## 6. Traps

- 🚨 **`groupby('ref').last()` on the whole panel.** §3. Filter by `release <= when` first.
- 🚨 **Lagging the latest print and not the history.** The trailing mean, the z-score's mean and
  standard deviation, the rolling rank — each of those reads the past, and the past has been
  rewritten. ✅ Measured: it is worth 14.7 % of position signs on its own.
- 🚨 **Standardising with full-sample moments.** `(x − x.mean()) / x.std()` over the whole
  series is a look-ahead independent of vintages and survives every vintage fix. Use expanding
  windows; `../../../fin-core/skills/signal-construction/scripts/assert_causal.py` catches it.
- 🚨 **Splicing a first-release series onto a revised tail.** Pulling `output_type=4` for the
  history and today's values for the recent months makes the recent months systematically
  better-measured than the old ones — a trend in signal quality that a backtest reads as alpha.
- 🚨 **Treating the third estimate as final.** §1: the benchmark moves the over-the-month change
  by 39k on average and can flip its sign. First-to-final is 1.43x first-to-third here.
- 🚨 **A vintage panel with no gaps is not a vintage panel.** Real ALFRED series have reference
  periods with one vintage and periods with twenty. Absence is data; do not forward-fill across
  a release that never happened. `../../../fin-market-data/skills/fundamental-and-macro-data/SKILL.md`
  §5 has the forward-fill-from-release rule.
- ⚠️ **Not every FRED series has vintages.** ALFRED coverage is per-series. Check before you
  design a study around it; the fallback is the agency's own vintage files (§2).
- 🚨 **The A/B is a trial.** Vintage-vs-revised, first-release-vs-third, benchmark-in-or-out are
  three more researcher degrees of freedom. Log them in
  `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`.

## 7. Scripts and where this sits

`scripts/vintage_backtest.py` — BLS's published revision tables transcribed and summarised, the
synthetic nested-sample vintage panel and its calibration check, `as_of()` / `first_release()` /
`latest()` / `vintage_matrix()`, the three-treatment A/B, the 200-seed sweep and the
revision-size x window grid. numpy + pandas, seed 20260909, about 11 s, no network.

- Where the series come from, the `fredapi` 0.5.2 bugs, the Philly Fed RTDSM, forward-fill from
  the release date — `../../../fin-market-data/skills/fundamental-and-macro-data/SKILL.md`.
- The same discipline for company fundamentals, and the availability rule in general —
  `../../../fin-core/skills/research-integrity-guards/SKILL.md` §2 and its
  `../../../fin-market-data/skills/fundamental-and-macro-data/scripts/pit_fundamentals.py`.
- **When** a number becomes tradeable, to the minute, and what changed in 2018 and 2020 —
  `../macro-release-calendar-and-embargo/SKILL.md`. The A/B needs a release timestamp; that
  skill builds it.
- Seasonal adjustment as a second vintage that rewrites history with no new data —
  `../seasonal-adjustment-and-x13/SKILL.md`.
- Recession labels that were assigned years after the fact —
  `../macro-regime-and-recession-indicators/SKILL.md`.
- Nowcasting the current quarter from a ragged edge — `../gdp-nowcasting-dynamic-factor/SKILL.md`.
- Whether the surviving Sharpe is real at all —
  `../../../fin-core/skills/backtest-validation/SKILL.md`.
