---
name: macro-release-calendar-and-embargo
description: >-
  Build the timestamp at which a macro number becomes tradeable - release date, clock time,
  timezone - and know where the release mechanics changed under your sample. TRIGGER -
  release calendar, economic calendar, release date vs reference date, available_at,
  as-of join on a macro series, "when was this number published", 8:30 ET, embargo,
  press lock-up, media lockup, pre-release access, WASDE noon, EIA Wednesday 10:30, natural
  gas storage Thursday, holiday release schedule, "why is my macro feature one day early",
  forward-fill a monthly series onto daily bars, DST offset on a release timestamp,
  event-study window around a print. SKIP for vintages and revisions to the value itself
  (real-time-macro-backtesting), for where the series live (fundamental-and-macro-data), for
  exchange sessions and holidays (us-market-rules), and for measuring fills you already have
  (execution-cost-analysis).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Macro release calendar and embargo

**A macro observation has two timestamps and most pipelines carry the wrong one.** The
reference period says what the number is about. The release timestamp says when it began to
exist. Joining on the first is the most common look-ahead in macro research; joining on the
release *date* without its clock time is the one that survives review.

And the release mechanics are not constant. **USDA stopped admitting reporters to its lock-up
on 2018-08-01; DOL abolished its lock-up entirely on 2020-06-03.** A backtest that spans either
date is pricing two different microstructures with one fill assumption.

Everything marked ✅ Measured is printed by `scripts/release_calendar.py` (numpy + pandas, seed
20260909, **under 1 s**, no network). Every date, time and schedule below was read at the
primary source on 2026-09-09.

## 1. ✅ Three embargo regimes in one calendar

| when | agency | what changed | ✅ source |
|---|---|---|---|
| **2018-08-01** | USDA NASS | *"NASS used to allow pre-release media Lockup access, but that policy has changed, effective August 1, 2018"* | `nass.usda.gov/About_NASS/ASB_and_Lockup/Lockup_QA.pdf`, dated 2018-07-10, read in full |
| 2020-03-04 | DOL | commits to *"at least 14 calendar days' notice before implementing any policy changes related to pre-release media lock-ups"* | BLS Commissioner's letter, 2020-05-19 |
| 2020-03-20 | DOL | *"suspend all pre-release media lock-ups until further notice"* (COVID-19) | same letter |
| **2020-06-03** | DOL | *"effective June 3, 2020, the Department will permanently discontinue media lock-ups for all releases"* | same letter; Federal Register **85 FR 31810**, 2020-05-27, doc 2020-11297 |

✅ source-verified, the BLS Commissioner's own letter: *"BLS and DOL's Employment and Training
Administration (ETA) will continue to make their data available to the general public
immediately upon their 8:30AM Eastern Time release through the Web and other sources."* The
Federal Register notice describes what ended: the lock-up gave journalists *"a period of time
(typically 30 minutes) to review data prior to the official release time."*

🚨 **The letter also says, quoting DOL Inspector General Report 17-14-001-03-315 (2014-01-02),
exactly what that half-hour was worth:**

> *"Several news organizations that participate in the DOL press lock-up are able to profit
> from their presence in the lock-up by selling, to traders, high speed data feeds of economic
> data formatted for computerized algorithmic trading. Because these news organizations have
> pre-release access, they are able to pre-load the data … allowing their clients to get this
> information faster than the general public, which has to wait to download the data after it
> gets posted to the Department of Labor websites."*

**Pre-loaded, machine-readable, on the tape at 08:30:00.000. That mechanism does not exist
after 2020-06-03.** §5 prices the difference.

🔑 **USDA ended *media* access but kept its own lock-up**, and its description is the clearest
statement of what an embargo is. ✅ source-verified, NASS's own Q&A: *"an officer is stationed
at the single entry point, the doors are locked and alarmed, the windows are covered with
shades that are secured with tamper resistant seals, and all telephone and Internet connections
are switched off … Everyone entering Lockup leaves cell phones and other wireless devices
outside and signs an agreement to that effect. Additionally, the Lockup area is continually
monitored for cell phone and wireless transmissions."* And: *"Not even the Secretary of
Agriculture knows a report's contents until he enters the Lockup area to sign the report just
prior to release."*

| agency | release | clock (agency local) | day rule | ✅ source |
|---|---|---|---|---|
| BLS | Employment Situation, CPI | **08:30 ET** | first Friday / mid-month weekday | BLS letter above |
| USDA | **WASDE** | **12:00 ET** | a day between the 8th and 12th | usda.gov WASDE page |
| EIA | Weekly Petroleum Status Report | **10:30 ET**, full report **13:00 ET** | Wednesday | `eia.gov/petroleum/supply/weekly/schedule.php` |
| EIA | Weekly Natural Gas Storage Report | **10:30 ET** | Thursday | `ir.eia.gov/ngs/schedule.html` |
| BEA | GDP advance / second / third | 08:30 ET | monthly, by quarter | ⚠️ not read at a primary source in this pass |

⚠️ **BEA is Commerce, not Labor.** The 2020 DOL policy does not reach it; do not assume one
department's embargo history applies to another's releases.

⚠️ Whether EIA operates a lock-up is **not verified** in this pass. ⚠️ There was also an
earlier, superseded DOL policy barring **electronic devices** from the lock-up room — the
Commissioner's letter cites it as *"85 Fed. Reg. 7333"* but its effective date was not
verified here, so it is not in the timeline above.

## 2. ✅ `available_at`, and the offset that moves twice a year

```python
def available_at(release_date, clock="08:30", tz="America/New_York"):
    return pd.Timestamp(f"{release_date} {clock}").tz_localize(tz).tz_convert("UTC")
```

🚨 **A fixed UTC offset is wrong for part of every year, silently.** ✅ Measured on 72 monthly
08:30 ET releases (2017–2022): 25 of them are UTC−5 and 47 are UTC−4. Hardcode the modal
offset and **25 of 72 releases (35 %) are stamped one hour off** — always in the same season,
which is why it reads as a seasonal effect rather than a bug. `2017-01-06 08:30 ET` is
`13:30Z`; `2017-07-07 08:30 ET` is `12:30Z`.

## 3. 🚨 "Wednesday at 10:30" is a rule, and the agency publishes the exceptions

✅ Measured on EIA's own published schedules, 2025–2026:

| report | nominal | published exceptions | on a different weekday | latest shift | clock times actually used |
|---|---|---|---|---|---|
| Weekly Petroleum Status | Wed 10:30 ET | **14** | **14** | **+6.5 h** | 10:30, 11:00, 12:00, **17:00** |
| Weekly Natural Gas Storage | Thu 10:30 ET | **9** | **9** | +1.5 h | 10:30, 12:00 |

🚨 The worst single case is **Monday 2025-12-29 at 17:00 ET** — a different weekday and six and
a half hours later, for a report your code believes lands Wednesday morning. Every one of those
is an observation your join attaches to the wrong session. **Read the agency's schedule file;
do not infer the rule from the modal weekday.**

⚠️ The same applies to BLS: the Employment Situation is usually the first Friday, but BLS shifts
it around holidays and the annual benchmark. The script's `payroll_release_dates()` is a
stand-in and says so.

## 4. ✅ Measured — four ways to stamp the same number

A monthly series joined to daily decisions, 2017–2022, 1,541 business days. "Look-ahead" is
the gap between the decision instant and the moment the value used actually existed.

**Decisions at the 09:30 ET open:**

| stamp | days with look-ahead | share | mean | max |
|---|---|---|---|---|
| **reference period start** | **1,540** | **99.9 %** | **18.5 days** | 37.0 days |
| **reference period end** | 230 | 14.9 % | 2.4 days | 7.0 days |
| release date, midnight | 0 | 0.0 % | — | — |
| **release timestamp** | **0** | **0.0 %** | — | — |

**Decisions at 06:00 ET (pre-market, or a previous-close fill):**

| stamp | days with look-ahead | share | mean | max |
|---|---|---|---|---|
| reference period start | 1,541 | 100.0 % | 18.7 days | 37.1 days |
| reference period end | 302 | 19.6 % | 2.0 days | 7.1 days |
| **release date, midnight** | **72** | **4.7 %** | **2.5 h** | 2.5 h |
| **release timestamp** | **0** | **0.0 %** | — | — |

🚨 **Indexing by reference period is not a subtle bug: every single day of the sample trades on
a number that does not exist yet**, by 18 days on average.

🔑 **And stamping at the release *date* survives the 09:30 test by luck.** The print lands at
08:30, before the decision, so midnight-stamping looks clean — until the decision moves
earlier. At 06:00 it hands you the number **2.5 hours early on all 72 release days**. A
convention that is correct for one decision clock and wrong for another is not a convention;
it is an accident. `pd.Series.resample("D").ffill()` on a period-indexed macro series produces
the first row of both tables.

## 5. 🚨 The splice: one fill convention across 2020-06-03

⚠️ **The two half-lives here are the script's own assumption** (40 ms with the lock-up, 1,500 ms
without). What is ✅ source-verified is their *ordering* and its cause — the DOL Inspector
General's finding in §1 that lock-up participants pre-loaded the data into algorithmic feeds
while the public waited to download a file.

✅ Measured — share of the release move already in the price when your order arrives:

| latency | lock-up era | after 2020-06-03 |
|---|---|---|
| 10 ms | 15.9 % | 0.5 % |
| 50 ms | 58.0 % | 2.3 % |
| **250 ms** | **98.7 %** | **10.9 %** |
| 1,000 ms | 100.0 % | 37.0 % |
| 5,000 ms | 100.0 % | 90.1 % |

✅ Measured — a strategy that reads the print and takes liquidity at 250 ms, 260 weekly events
straddling the boundary:

| | events | captured before you arrive | true edge |
|---|---|---|---|
| before 2020-06-03 | 126 | 98.7 % | **0.23 bp** |
| after 2020-06-03 | 134 | 10.9 % | **16.48 bp** |
| one spliced assumption (50 % everywhere) | 260 | 50 % | reports **9.09 bp** throughout |

🚨 **Overstated by 8.69 bp before the change, understated by 7.23 bp after — and off by
+0.48 bp on the full sample.** The two errors nearly cancel in aggregate, which is exactly why
nobody notices: the headline number looks fine and every year of it is wrong. **Split the
sample at 2020-06-03 (and at 2018-08-01 for USDA releases), fit the fill assumption on each
side, or start the sample after the change.**

## 6. Traps

- 🚨 **Joining on the reference period.** §4. 99.9 % of days, 18 days of look-ahead.
- 🚨 **A release date with no clock time.** §4. It is correct for one decision clock by
  coincidence and wrong for every other.
- 🚨 **A hardcoded UTC offset.** §2. Wrong on 35 % of releases, always in the same season.
- 🚨 **Inferring the release day from the modal weekday.** §3. 23 published exceptions across
  two EIA reports in two years, one of them six and a half hours late on a Monday.
- 🚨 **Splicing across a change in release mechanics.** §5. Also applies to any change in the
  publication *content* — a redefinition, a rebasing, a new seasonal-adjustment method
  (`../seasonal-adjustment-and-x13/SKILL.md`).
- 🚨 **`interpolate()` on a macro series.** Interpolation is bidirectional: it pulls a future
  release backwards into days before it existed. Forward-fill from `available_at`, and let the
  gaps be gaps. `../../../fin-core/skills/fundamental-and-macro-data/SKILL.md` §5 states the
  rule; this skill supplies the timestamp it needs.
- 🚨 **Assuming the first print is the number.** It is the number *for that instant*. The value
  changes later without the timestamp changing — that is `../real-time-macro-backtesting/SKILL.md`.
- ⚠️ **A vendor "economic calendar" is a product, not a source.** Its historical release
  timestamps are frequently the date only, are sometimes back-filled, and its consensus column
  is a separate point-in-time object with its own vintages. Verify against the agency schedule
  before you build a study on it.
- 🚨 **Every timing choice is a trial.** Fill at the release, at the next bar, at the next
  open; 250 ms or 5 s. Log them in
  `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`, and run the feature
  through `../../../fin-core/skills/signal-construction/scripts/assert_causal.py`.

## 7. Scripts and where this sits

`scripts/release_calendar.py` — the verified release-time table and embargo timeline,
`available_at()` and the DST audit, the EIA holiday-shift audit over the agency's own published
exceptions, the four-stamp as-of join at two decision clocks, and the latency/capture splice
study. numpy + pandas, seed 20260909, under 1 s, no network.

- The value at that timestamp, and how it changes afterwards —
  `../real-time-macro-backtesting/SKILL.md`.
- Seasonal adjustment as a revision with no new information —
  `../seasonal-adjustment-and-x13/SKILL.md`.
- Recession labels that arrive 7 to 21 months after the fact —
  `../macro-regime-and-recession-indicators/SKILL.md`.
- The general availability rule (`available_at = max(publication, retrieval, processing)`) and
  the filing-date version of this bug —
  `../../../fin-core/skills/research-integrity-guards/SKILL.md` §2.
- Where the series come from, and the forward-fill-from-release rule —
  `../../../fin-core/skills/fundamental-and-macro-data/SKILL.md`.
- Exchange sessions, halts and holidays on the price side —
  `../../../fin-core/skills/us-market-rules/SKILL.md`.
- What a fill actually costs once you have one —
  `../../../fin-core/skills/execution-cost-analysis/SKILL.md`.
