---
name: congressional-trading-disclosures
description: >-
  Build a congressional-trading signal on the disclosure date instead of the transaction
  date, and price what the amount brackets cost you. TRIGGER - congressional trading,
  congress stock trades, STOCK Act, Periodic Transaction Report, PTR, House Clerk financial
  disclosure, disclosures-clerk.house.gov, efdsearch.senate.gov, Senate eFD, senator trade
  trackers, "copy the trades Congress makes", "backtest congressional trades", "what did
  Congress buy", amount ranges, "$1,001 - $15,000", "how do I weight a trade disclosed as a
  bracket", the 45-day filing deadline, late PTR filings, scraping the House or Senate
  disclosure sites. SKIP for corporate insiders and Form 4 transaction
  codes (insider-form-4), for fund holdings and 13F staleness (institutional-13f), for
  social posts and influencer panels (social-and-influencer-feeds), for EDGAR endpoints and
  XBRL point-in-time fundamentals (fundamental-and-macro-data), and for the general
  availability rule and the five-gate audit (research-integrity-guards).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-10"
---

# Congressional trading disclosures

**The transaction date is on the filing, and it is not the date the information existed.**
A Periodic Transaction Report is due up to 45 days after the trade, and the routine penalty
for missing that is a $200 fee. Keying a backtest on the transaction date is the same bug
`../../../fin-core/skills/fundamental-and-macro-data/SKILL.md` documents for a fiscal period
end, in a domain where it is easier to make because the filing prints both dates side by side.

Everything marked ✅ Measured is printed by `scripts/congress.py` (numpy + pandas, seed
20260910, about 15 s, no network). Its filings are **synthetic and seeded** - the lag
distribution is one this repo chose and then measured back, not a tally of real filings.
Everything marked ✅ source-verified was read at the URL given, on 2026-09-10.

## 1. ✅ The regime, at primary sources

| Fact | Value | Source, read 2026-09-10 |
|---|---|---|
| **PTR deadline** | *"Not later than 30 days after receiving notification of any transaction"*, and *"in no case later than 45 days after such transaction"* | **5 U.S.C. 13105(l)** (`uscode.house.gov`) |
| Reporting trigger | *"the gross amount of a single purchase or sale transaction exceeds $1,000"* - **gross**, so a losing trade still counts | House PTR form CY2025 (`ethics.house.gov`), 5 U.S.C. 13104(a)(5)(B) |
| **Late-filing fee** | *"$200 penalty shall be assessed against anyone who files more than 30 days late"* | 5 U.S.C. 13106(d); House PTR form; Senate PTR instructions |
| Civil penalty | base **$50,000** for a knowing and willful falsification, inflation-adjusted to **$75,540** for 2025 | 5 U.S.C. 13106(a)(1); 90 FR (2025-01-15) |
| Use restriction | unlawful for commercial purposes, credit rating or solicitation; penalty *"not to exceed $10,000"* | **5 U.S.C. 13107(c)** |
| Online posting | STOCK Act sec. 8's *"search, sort, and download"* mandate was suspended by **Pub. L. 113-7 (2013-04-15)** for staff, but the carve-out expressly preserves *"(C) Any Member of Congress. (D) Any candidate for Congress."* | `govinfo.gov` PLAW-113publ7 |

🚨 **The statute is 5 U.S.C. 13105(l), not 13104(l).** 13104 is *Contents of reports* and has
no subsection (l). Both numbers circulate; only one resolves.

🔑 **The deadline is not the only clock.** The 30-day leg starts at *notification*, and the
$200 fee only attaches past **30 days late**, i.e. day 75 after the trade. Nothing in the
filing tells you which leg the filer was on.

### The amount brackets - the amount itself is never disclosed

✅ House PTR form CY2025, columns A-J, plus a column K that has no numeric equivalent:

| bracket | width | | bracket | width |
|---|---|---|---|---|
| $1,001-$15,000 | **15.0x** | | $1,000,001-$5,000,000 | 5.0x |
| $15,001-$50,000 | 3.3x | | $5,000,001-$25,000,000 | 5.0x |
| $50,001-$100,000 | 2.0x | | $25,000,001-$50,000,000 | 2.0x |
| $100,001-$250,000 | 2.5x | | over $50,000,000 | **open** |
| $250,001-$500,000 | 2.0x | | **K: spouse/dependent child over $1,000,000** | **open** |
| $500,001-$1,000,000 | 2.0x | | | |

🚨 **Column K collapses G, H, I and J into one bucket** for an asset owned by a spouse or
dependent child in which the filer has no interest (5 U.S.C. 13104(e)(1)(F)). A $1.1M trade
and a $60M trade are the same disclosure. ⚠️ The *statutory* categories at 13104(d)(1) start
at *"not more than $15,000"*; the $1,001 floor is a form convention that follows from the
$1,000 reporting trigger. ⚠️ The Senate's own bracket list was **not verified** - the Senate
PTR instructions do not print it and the form sits behind the click-through in §2.

## 2. Where the filings live, and what is NOT knowable

✅ **House** - `disclosures-clerk.house.gov/FinancialDisclosure`, read 2026-09-10. Annual
archives at `/public_disc/financial-pdfs/<YEAR>FD.zip`, 2008-2026. The path segment is
literally `financial-pdfs`. A displayed notice quotes the 5 U.S.C. 13107(c) use prohibition;
there is no checkbox and **no documented public API**.

✅ **Senate** - `efdsearch.senate.gov/search/home/`, read 2026-09-10. Covers *"Senators, former
Senators and Senate candidates filed from 2012 to present"*. Access is gated by a
**click-through**: a checkbox reading *"I understand the prohibitions on obtaining and use of
financial disclosure reports."* ✅ Both sites return **HTTP 404 for `robots.txt`**, so the
restriction is the statute and the agreement, not a crawl directive.

⚠️ **What this repo did not verify, and will not assert:**

- **The internal structure of the House ZIP.** Every description found of an XML index inside
  it came from a data vendor, not from the Clerk. Not opened here.
- **What the Senate serves per report** (HTML tables for e-filers, PDFs for paper filers is
  the common claim) and whether a bulk download exists. Both sit behind the agreement, and
  accepting an agreement on a user's behalf is not something this repo does.

🚨 **The consequences for a pipeline, and they are the reason this skill ships no scraper:**

1. The unit of publication is a **document**, not a record. A PTR is a PDF or semi-structured
   HTML listing several transactions. Parsing it is an OCR/layout problem with its own error
   rate, and that error rate is a data-quality term nobody reports.
2. **Several aggregators resell this data.** They add a ticker mapping, an asset-type guess
   and a "disclosed" timestamp - all three are their inference, not the filing's content.
   A vendor's `disclosure_date` may be its *ingestion* date, which is later again.
3. This library ships **no scraper and no credential path** for either site. Read the terms
   yourself, and if you use a vendor, ask which of its fields are transcribed and which are
   derived.

⚠️ **Ticker resolution is a research step, not a lookup.** Filings name assets in prose
("Apple Inc. - Common Stock"), and the same prose covers options, bonds and funds. Resolving
that string to a security is the same `(identifier, date)` problem as
`../../../fin-core/skills/research-integrity-guards/SKILL.md` §1, and it is where a
survivorship bias enters if you map against today's ticker table.

## 3. ✅ Measured - the lag this file chose, and what it produced

`draw_lags()` sets the distribution: a Beta squeezed onto [2, 45] days because a deadline is a
target rather than a mean, plus a **12% tail past the deadline** because the fee is $200.
Measured off 1,500 seeded filings, calendar days from transaction to disclosure:

| mean | p05 | p25 | **median** | p75 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 43.1 | 11.8 | 22.3 | **31.2** | 38.6 | **141.2** | 306.6 | 471.2 |

✅ Measured: **11.7%** past the 45-day deadline, **9.3%** past 90 days, **0.5%** past a year.

⚠️ These are properties of a generator, not a census of Congress. What is *not* a choice is
the shape: the median filing is a month old when it becomes public, and the tail is long.
**Measure your own vendor's lag distribution before you quote any number from anywhere,
including this table.**

## 4. ✅ Measured - the A/B

One rule: long the disclosed name for 63 trading days, short the equal-weight universe,
entered at the **close of the key day** so the first return earned is the next day's. Same
trades, same names, same holding period; only the date you may act on differs.

| key | Sharpe | ann. return | ann. vol | retained |
|---|---|---|---|---|
| **transaction-date** (look-ahead) | **0.880** | 4.33% | 4.92% | 100% |
| **disclosure-date** (honest) | **0.362** | 1.82% | 5.01% | **41.2%** |

Over 40 seeds, because one seed is one draw:

| | mean | sd | p05 | p95 | > 0 |
|---|---|---|---|---|---|
| transaction-date | **1.038** | 0.266 | 0.535 | 1.428 | 100% |
| disclosure-date | **0.449** | 0.275 | 0.055 | 0.839 | 100% |
| **gap** | **0.589** | 0.304 | 0.226 | 1.077 | **100%** |

🔑 **The gap is 0.589 of Sharpe and it is positive in every one of 40 seeds.** The volatility
of the two runs is the same to two decimal places - the look-ahead is not buying you a smoother
ride, it is handing you a return you could not have earned.

## 5. 🚨 Measured - the gap is a ratio, not a number of days

Disclosure-date Sharpe as a **fraction** of the transaction-date Sharpe. Rows scale the whole
lag draw (mean calendar days in brackets); columns are the half-life of the information in
trading days. 12 seeds per cell.

| lag | half-life 5d | half-life 21d | half-life 63d |
|---|---|---|---|
| 0.1x (4 days) | 65.8% | 87.3% | 91.7% |
| 0.25x (11 days) | 48.5% | 79.4% | 86.7% |
| 0.5x (22 days) | 34.8% | 70.5% | 80.7% |
| **1x (44 days)** | **19.9%** | **52.3%** | **65.3%** |
| 2x (89 days) | 3.1% | 19.5% | 29.7% |

🔑 **Read down a column, not across the table.** At a 5-day half-life the statutory lag alone
removes 80% of the signal; at a 63-day half-life the same lag leaves two thirds of it. "45
days" says nothing until you say what it is 45 days *of*. ⚠️ The decay model is a choice
(exponential, stated in the script). The invariant is that the loss depends on
`lag / half-life`, so **the first thing to measure on your own data is the half-life**, by
running the disclosure-keyed rule at several holding periods.

## 6. ✅ Measured - what the brackets do to a size-weighted signal

Three measurements, in increasing order of how much they lean on the generator.

**(a) Ordering - a property of the brackets alone.** Two filings in the same bracket are
indistinguishable no matter which imputation you pick. ✅ Measured on 18,000 pooled seeded
filings: **22.6% of all pairs are exact ties**, and the modal bracket ($15,001-$50,000) holds
32.5% of filings. Column K, the spouse/dependent-child catch-all, hides a **p90/p10 ratio of
12.9x** inside a single disclosed value.

**(b) The dollar aggregate - "Congress bought $X of this name".** Imputed total as a fraction
of the true total, columns being the value you assign the **open-ended** top brackets as a
multiple of their lower edge - a number nobody publishes:

| imputation | top x1 | top x2 | top x5 |
|---|---|---|---|
| bracket **midpoint** | **81.2%** | 92.0% | **124.4%** |
| geometric mean | 68.5% | 79.3% | 111.8% |
| lower bound | 43.0% | 43.0% | 43.0% |

🚨 ✅ Measured: **1.2% of filings sit in an open-ended bracket and carry 42.0% of the true
dollars.** The headline aggregate moves from 81% to 124% of truth on one unpublished
assumption. A dollar aggregate is not a measurement; it is your prior about the tail.

**(c) The Sharpe cost.** Dollar-weight the disclosure-keyed portfolio five ways, 12 seeds.
Only `true` needs a number nobody published:

| weight | Sharpe | sd | cost vs true |
|---|---|---|---|
| **true amount** (unknowable) | **0.839** | 0.292 | - |
| bracket midpoint | 0.718 | 0.342 | **-0.121** |
| geometric mean | 0.743 | 0.343 | -0.095 |
| lower bound | 0.799 | 0.334 | -0.040 |
| equal weight (ignore size) | 0.590 | 0.327 | **-0.249** |

✅ Measured: size information is worth **0.249** of Sharpe here, and **the bucketing throws
away about half of it**. Re-valuing the open-ended brackets at 5x their lower edge moves the
midpoint run from 0.718 to 0.774 - the same free parameter as in (b). ⚠️ The level depends on
the alpha-to-size link the script sets (`dollar_exponent=0.5`, sublinear); the ordering does
not. 🔑 The lower bound beats the midpoint here, which is the point: **the midpoint is an
estimator with a bias, not a neutral default**, and each of these five is a trial to log in
`../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`.

## 7. Traps

- 🚨 **Keying on `transaction_date` because the column exists.** §4: 0.589 of Sharpe, positive
  in 40 of 40 seeds. Use the date the filing became *public*, and prefer the vendor's ingestion
  timestamp over its `disclosure_date` when the two differ.
- 🚨 **Dropping the late filings.** They are 11.7% of this panel and they are not missing at
  random - a filer who is late is different from one who is not. Dropping them is a selection
  on the disclosure process itself.
- 🚨 **Backfilling an amendment onto its original date.** An amended PTR is a *new* publication.
  Use the amendment's own date, the same rule as a restated 10-K.
- 🚨 **Treating the bracket midpoint as the amount.** §6. And the top bracket has no midpoint.
- 🚨 **Aggregating "Congress" as one trader.** 500+ filers with different mandates, staff and
  brokers. An equal-weighted aggregate over all of them is a survey, not a portfolio.
- 🚨 **Selecting the members after seeing the returns.** "Follow the five best members" chosen
  in-sample is the same defect as an influencer panel - see
  `../social-and-influencer-feeds/SKILL.md` §on selection, which measures it.
- ⚠️ **Sales are not the mirror of purchases.** A sale can be tax, a blind-trust rebalance or a
  divestiture required on taking a committee seat. The same asymmetry the insider literature
  reports (`../insider-form-4/SKILL.md` §on codes) applies here with less documentation.
- 🚨 **Filings are documents.** Your parser's error rate belongs in the result. If you cannot
  state it, say the Sharpe is an upper bound.

## 8. Scripts and where this sits

`scripts/congress.py` - the transcribed statutory constants; `draw_lags()` and
`lag_summary()`; `bracket_of()` / `impute_size()` with the five imputations and the
open-ended-bracket parameter; `make_panel()`; `portfolio()` / `run_ab()` / `ab_over_seeds()`;
`lag_grid()`; `ordering_loss()`, `aggregate_bias()`, `weighting_table()`. numpy + pandas,
seed 20260910, about 15 s, no network, no file writes, no scraper.

- `../insider-form-4/SKILL.md` - the same argument with a **two-business-day** deadline instead
  of 45, and the transaction-code filter that decides whether there is a signal at all.
- `../institutional-13f/SKILL.md` - the same argument where the lag is measured from a *quarter
  end*, so the position can be four months old before you see it.
- `../social-and-influencer-feeds/SKILL.md` - the selection and deletion biases that a
  "top members" or "top accounts" panel shares with this one.
- `../../../fin-core/skills/research-integrity-guards/SKILL.md` §2 - the availability rule this
  skill is one instance of, and the `(identifier, date)` join that ticker resolution needs.
- `../../../fin-core/skills/fundamental-and-macro-data/SKILL.md` - the period / filed /
  acceptance distinction, and its `pit_fundamentals.py`, the same as-of read in code.
- `../../../fin-macro/skills/real-time-macro-backtesting/SKILL.md` - the identical A/B on macro
  vintages, and the finding that the *timing* leg usually dominates the *value* leg.
- `../../../fin-core/skills/backtest-validation/SKILL.md` - a Sharpe of 0.449 with sd 0.275 is
  not distinguishable from a good one; the imputation choices in §6 are trials.
