---
name: combining-data-sources
description: >-
  Combine information of DIFFERENT kinds into one research view whose every number can be traced
  back to what was knowable when. TRIGGER - joining prices to fundamentals, earnings or a macro
  series; building a signal from a filing plus a price; merging two vendors into one series;
  "which vendor do I trust"; combine_first / fillna / bfill across sources; assembling a company
  dossier, profile or research view from several places; a field that must carry its provenance;
  "when did this number become knowable"; period_end vs filed vs available; a ticker that changed
  company mid-sample; filling a ResultCard's data block. SKIP for the mechanics of ONE as-of join
  (safe_asof, in market-data-engineering); for reconciling two price series from two vendors
  (reconcile_sources, in market-data-sourcing); for auditing a finished backtest
  (research-integrity-guards); and for resolving a ticker to a permanent id
  (security-master-and-symbology).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-10"
---

# Combining data sources

Fetching is solved. Finding is solved. **Combining is where the number stops being true**, because
each kind of data arrives on its own clock, in its own convention, under its own identifier — and a
join erases all three. Everything below is **measured** by the two scripts in this skill; nothing
here is asserted.

## 1. The findings

| # | Finding | Measured |
|---|---|---|
| 1 | 🚨 **A signal joined on `period_end` instead of `available_at` scores 3.37 Sharpe against 1.62.** Same panel, same signal, one clock apart — **+1.76 of pure look-ahead, 2.1x** | `scripts/availability_clock.py` |
| 2 | 🚨 **Taking the EARLIEST input clock is the error that looks fine.** The joined row has a price in it, an EPS in it, and no NaN anywhere. Nothing about its shape says half of it was published 53 days later | `scripts/availability_clock.py` |
| 3 | 🚨 **Stamping the join date instead fails SILENTLY** — 0 of 1512 sessions carry the fact, so the backtest runs, holds nothing, and reports a flat line rather than an error | `scripts/availability_clock.py` |
| 4 | 🚨 **`combine_first` resolves by NULLITY, not by precedence.** On 2,316 overlapping sessions the two vendors disagree by >10 bps on **171 (7.4%)**, and `b.combine_first(a)` takes the non-authoritative number on **every one of them**, keeping **0 records** | `scripts/source_merge.py` |
| 5 | 🚨 **The corrupted series passes every check you actually run**: lag-1 autocorrelation **−0.055 vs +0.012**, annualised vol **20.6% vs 19.2%**, and **0** returns beyond 10 sigma. It hands a one-day reversal rule **+0.32 Sharpe against −0.38** on the truth | `scripts/source_merge.py` |
| 6 | 🚨 **Splicing two adjustment conventions puts the split on the wrong day** — a **−49.8%** return on a date nothing happened and **−0.5%** on the date the split actually did; annualised vol 19.2% → **24.9% (1.29x)** | `scripts/source_merge.py` |
| 7 | 🚨 **A recycled ticker glues two issuers into one series**: a **−83.9%** one-day return, vol 18.4% → **32.7% (1.77x)**, and a 60-day momentum rule scoring **−0.04 glued against +0.86** on the first issuer alone | `scripts/source_merge.py` |
| 8 | ✅ **The right `combine_first` order returns the SAME numbers as a precedence merge.** The difference is not the values — it is that one of them can tell you it made 171 choices and the other cannot | `scripts/source_merge.py` |

## 2. 🚨 The availability clock

Three timestamps, and only one of them may decide anything:

| Timestamp | What it is | What it is not |
|---|---|---|
| `period_end` | the period the value **describes** | ❌ a knowability date. A 30–90 day look-ahead for a filing |
| `filed_at` | when it was **filed or released** | ❌ tradable. A post-close release is next session's information |
| `available_at` | when it became **publicly knowable** | ✅ the only one that may enter a signal |

**The rule, and it is the whole skill:**

> **A combined fact is knowable only when its LAST input is.**
> `available_at = max(inputs)` — never `min(inputs)`, and never the date the join ran.

**Why `min` is the seductive error.** An earnings yield is a price *divided by* a fundamental. The
price is known at `t`. The fundamental describes a quarter that ended weeks ago. Join them on the
fundamental's period end and **every column of the output row is populated on date `t`** — a price,
an EPS, no NaN, no warning. The row is a fact about `t` in every respect except the one that
matters.

✅ **Measured** (`scripts/availability_clock.py`, seed 20260910): 40 names, 1,512 sessions, 1,040
quarterly facts, filings knowable 30–76 calendar days after the period ends (median **53**), the
market pricing each quarter in over the following 63 sessions — so about **60%** of the drift is
over by the median release.

| Clock | What it means | Sharpe | Ann. return |
|---|---|---|---|
| `period_end` (= `min` inputs) | trade it when the quarter ends | **3.37** | 15.4% |
| `available_at` (= `max` inputs) | trade it when it is published | **1.62** | 7.4% |
| join date | stamp it with the moment the research ran | **0.00** | 0.0% |

🔑 **The cost of the earliest-input clock is +1.76 Sharpe, and it is entirely look-ahead.** The
join-date clock is the opposite failure: it leaves **0 of 1,512 sessions** with a signal and
reports that as a flat equity curve, not as an error.

**The clock is per FACT, not per join.** Prices, filings, macro prints and documents each have
their own, and the combined one is the max over whichever inputs that particular field used. That
is why it has to travel with the value rather than live in the pipeline.

## 3. 🚨 The silent vendor pick

`a.combine_first(b)` takes `a` wherever `a` is **not NaN** and `b` elsewhere. That is not a
precedence rule. **A vendor printing a wrong number is not printing a null**, so a stale repeat of
yesterday's close, an auction print and a correct quote are all equally "not NaN".

Two consequences, both measured on 2,520 sessions with vendor A missing 204 of them and vendor B
complete but carrying 150 stale repeats and 48 bad prints:

- ✅ **Typing order decides the result.** The two vendors both print on 2,316 sessions and disagree
  by more than 10 bps on **171** of them (median **112 bps**, max **622 bps**).
  `b.combine_first(a)` — the order you write when B has the better coverage — takes B on **all
  171**, and lands more than 10 bps from the truth on **187 of 2,520 sessions (7.4%)**.
- 🚨 **Neither order leaves a record.** `a.combine_first(b)` returns the *same numbers* as a
  precedence merge, and is still wrong on the **16** sessions where A had a hole and B filled it
  badly. The difference between them is not the values; it is that one **can say it made 171
  choices**.

🚨 **And the damage is invisible to every check you actually run:**

| Series | Ann. vol | max \|ret\| | >10 sigma | autocorr(1) | 1-day reversal Sharpe |
|---|---|---|---|---|---|
| truth | 19.2% | 534 bps | 0 | +0.012 | **−0.38** |
| precedence merge (= `a.combine_first(b)`) | 19.3% | 534 bps | 0 | +0.006 | −0.29 |
| naive `b.combine_first(a)` | 20.6% | 688 bps | **0** | **−0.055** | **+0.32** |

🔑 **A bad print and its snap-back IS negative autocorrelation** — which is exactly what a
mean-reversion signal is built to find. The artefact adds 1.4 points of annualised vol, produces
**zero** returns beyond 10 sigma, and flips a losing reversal rule into a winning one. There is no
outlier filter and no vol check that fires here.

**What to do instead:** state a ranking before you look at the data, take the highest-ranked source
present at each timestamp, and emit a record for every timestamp where the sources differ beyond a
stated threshold.

### Two merges that cannot be correct — refuse, do not resolve

- 🚨 **Declared `Adjustment` differs.** A raw series and a split-adjusted one are not two
  measurements of one quantity. Measured: splicing them puts a **−49.8%** return on the switch date
  (nothing happened that day) and **−0.5%** on the day the 2-for-1 split actually occurred;
  annualised vol goes 19.2% → **24.9% (1.29x)**. Re-adjust to one convention **first** — which
  needs the corporate-actions table, and needing it is the point.
- 🚨 **Symbology is unresolved.** See §4.

## 4. 🚨 The identifier that changed entity mid-sample

A ticker is a slot in an exchange's namespace, not a company. It is reassigned after a delisting
and it changes on a rename, so **joining two vendors on a ticker joins whatever each of them meant
by it on each date** — a different question from whether their numbers agree.

✅ **Measured** (`scripts/source_merge.py`): ticker `XYZ` is issuer E1 (about $40) for 1,260
sessions, delists, and is reassigned to issuer E2 (about $6) for 1,260 more.

- a **−83.9%** one-day return on the handover date
- annualised vol **18.4% → 32.7% (1.77x)**
- exactly **one** session beyond 10 sigma — one row, easily written off as a real crash
- a 60-day momentum rule scores **−0.04** on the glued series against **+0.86** on E1 and **+0.07**
  on E2: the artefact does not merely add noise, it **destroys a signal that was really there**

🔑 **Resolve to a permanent id (`cik`, `figi`, `permno`, `isin`) before the join, not after.** The
identifier is the only one of the three failure modes that a later check cannot undo, because once
the histories are glued there is nothing in the series that says where the seam was.

## 5. The API

`fin_skills.synthesis` implements all three rules, and refuses rather than guessing.

```python
from fin_skills.synthesis import Dossier, Fact, Timeline, price_fact, MergePolicy, merge_series
from fin_skills.api import check

tl = Timeline([price_fact("close", "cik:0000320193", 189.4, "2024-02-02", source="vendor-a")])
tl.add(Fact(field="revenue", entity="cik:0000320193", value=1.19e11,
            period_end="2023-12-30", filed_at="2024-02-01", available_at="2024-02-14",
            kind="fundamental", source="edgar"))
ey = tl.combine("earnings_yield", tl.as_of("2024-03-01"), 0.062)   # clock = max(inputs)
ey.available_at            # 2024-02-14, not 2024-02-02

d = Dossier(entity="cik:0000320193", as_of="2024-03-01", timeline=tl)
print(d.render())          # compact ASCII: value, clock, lag and source per field
print(d.explain("earnings_yield"))   # every input, TRANSITIVELY, with the vintage of each
d.provenance_block()       # list[DataSource] - the `data=` argument of a ResultCard
print(check(d.to_bundle()).summary())      # the guards run on a SYNTHESISED view
```

| Call | Refuses |
|---|---|
| `timeline.combine(...)` | an `available_at` earlier than the last input (`AvailabilityError`) |
| `merge_series(sources, policy)` | sources whose declared `Adjustment` differs; an unresolved or mismatched entity; a source nobody ranked (`MergeRefused`) |
| `Timeline.check_availability()` | — returns the facts some other code path back-dated |
| `Dossier.explain(field)` | — returns the chain, and says when it stops short of a source |

**The guard.** `synthesis_integrity` fails on exactly three things: a combined fact claiming an
availability earlier than the latest of its inputs; two merged sources disagreeing beyond the
stated threshold with **no** `Disagreement` recorded; and a field whose provenance chain ends at a
root that names no source. It runs from `fin_skills.api.check(dossier.to_bundle())`.

⚠️ It is **excluded from the JSON tool set** on purpose: a live `Timeline` does not survive
`json.dumps`, and a fabricated one would not be the caller's. Call it from Python.

## ❓ Not verified

- **Every number here is from seeded synthetic data**, not from a vendor feed. The error *rates*
  (8% gaps, 6% stale prints, 2% bad prints, a 30–76 day filing lag) are plausible parameters, not
  measurements of any real vendor. What is measured is the **consequence** of those rates, which is
  the part that transfers.
- **The Sharpe figures are one panel, one seed.** Rerunning with another seed keeps the ordering
  (`period_end` > `available_at` — asserted in the tests) but not the magnitudes.
- **No vendor-specific availability lags are claimed here.** For the SEC's `filed` vs
  `acceptanceDateTime` and FRED's `realtime_start`, see `fundamental-and-macro-data`.

## Where this sits

- **`market-data-engineering`** — the mechanics of one as-of join: `allow_exact_matches`,
  `tolerance`, sortedness. That skill is about the join; this one is about what the joined row is
  allowed to claim.
- **`market-data-sourcing`** — `reconcile_sources`, which explains why two price series differ.
  Same kind, two sources; this skill starts where the kinds differ or a series has to be produced.
- **`research-integrity-guards`** — the audit of a finished result. This skill is what makes the
  data half of a `ResultCard` fillable without typing it in twice.
- **`security-master-and-symbology`** — resolving an identifier to a permanent entity. §4 measures
  what happens if you skip it; that skill is how not to.
- **`fundamental-and-macro-data`** — the vendor-specific clocks (`filed` vs `acceptance`, FRED
  vintages) that populate a `Fact`'s three timestamps.
