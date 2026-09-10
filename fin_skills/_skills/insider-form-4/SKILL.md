---
name: insider-form-4
description: >-
  Filter Form 4 to open-market purchases by transaction code, then key the signal to the
  first session that can trade the acceptance timestamp. TRIGGER - Form 4, Forms 3/4/5,
  Section 16, insider buying, insider selling, corporate insider trades, "officer bought
  shares", transaction codes P S A M F G X C, code P vs code A, 10b5-1 plan, EDGAR ownership
  XML, ownershipDocument, transactionCode, two business days, "insider trading signal",
  "should I follow insider buys", openinsider-style screens, "why does my insider backtest
  look amazing", stripping option exercises and tax withholding out of insider data. SKIP
  for congressional trades and the STOCK Act (congressional-trading-disclosures), for fund
  holdings and 13F staleness (institutional-13f), for social posts and influencer panels
  (social-and-influencer-feeds), for EDGAR endpoints, edgartools and XBRL fundamentals
  (fundamental-and-macro-data), and for the general availability rule
  (research-integrity-guards).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-10"
---

# Insider Form 4

**Two things decide whether a Form 4 study is real, and the deadline is neither of them.**
The deadline is two business days, so decay barely costs you anything. What costs you is
(1) letting grants, option exercises and tax withholdings into the portfolio alongside
actual purchases, and (2) filling at the close of a day whose filing was accepted at
20:30 ET.

Everything marked ✅ Measured is printed by `scripts/form4.py` (numpy + pandas, seed
20260910, about 5 s, no network). Its filings are **synthetic and seeded** - the code mix
and the lag distribution are set by the script and measured back, not a tally of EDGAR.
Everything marked ✅ source-verified was read at the URL given, on 2026-09-10.

## 1. ✅ The rule, at primary sources

| Fact | Value | Source, read 2026-09-10 |
|---|---|---|
| **Deadline** | *"before the end of the second business day following the day on which the subject transaction has been executed"* | **15 U.S.C. 78p(a)(2)(C)** (`govinfo.gov`) and **17 CFR 240.16a-3(g)(1)** (eCFR) |
| Deferred execution date | for a Rule 10b5-1(c) plan trade where *"the reporting person does not select the date of execution"*, and for a 16b-3(f) discretionary transaction, the deemed execution date is the date the broker or administrator *"notifies the reporting person"* | 17 CFR 240.16a-3(g)(2), (g)(3) |
| 🔑 **The cap nobody quotes** | 16a-3(g)(**4**): if notification is later than the **third** business day after the trade, the execution date is deemed to be that third business day. **Outer limit T+5 business days.** | 17 CFR 240.16a-3(g)(4) |
| Gifts | 16a-3(g)(1) as amended now covers *"dispositions by bona fide gifts"* on Form 4. They used to sit on the Form 5, due up to 45 days after fiscal year end. | 17 CFR 240.16a-3(g)(1) |
| **Filing window** | a filing after **17:30 ET** is *"deemed filed as of the next business day"*, **but** a Form 3, 4 or 5 *"commencing on or before 10 p.m. Eastern"* is *"deemed filed on the same business day"* | **17 CFR 232.13(a)(2)** and **(a)(4)** |
| Same 10 p.m. treatment | ⚠️ (a)(4) is **not** Forms 3/4/5 only - it also names **Schedule 14N, Form 144 and Schedules 13D/13G** | 17 CFR 232.13(a)(4) |
| Format | Ownership XML, submission types *"3, 4, 5, 3/A, 4/A, and 5/A"*; a live `form4.xml` on EDGAR returns `Content-Type: text/xml`, root `<ownershipDocument>` | EDGAR Ownership XML Technical Specification; an Archives fetch |
| Access | *"Current max request rate: 10 requests/second"*, and *"Please declare your user agent in request headers"* | `sec.gov/os/accessing-edgar-data` |

🚨 **`filingDate` is a date; the 22:00 ET window means it is not an availability
timestamp.** Use `acceptanceDateTime` and the availability rule in
`../../../fin-core/skills/research-integrity-guards/SKILL.md` §2a - which already flags
22:00 ET for Forms 3/4/5 specifically. §5 below prices the difference.

## 2. ✅ The 20 transaction codes, in the SEC's own groups

Form 4 General Instruction 8 / `sec.gov/edgar/searchedgar/ownershipformcodes.html`,
read 2026-09-10. There are **twenty**, not the eighteen or nineteen usually listed - the
one most often dropped is **V**.

| group | codes | the ones that decide a signal |
|---|---|---|
| **General** | P, S, **V** | **P** *"Open market or private purchase..."*; **S** *"Open market or private sale..."*; **V** *"Transaction voluntarily reported earlier than required"* |
| **Rule 16b-3** | A, D, F, I, M | **A** *"Grant, award or other acquisition pursuant to Rule 16b-3(d)"*; **F** *"Payment of exercise price or tax liability by delivering or withholding securities..."*; **M** *"Exercise or conversion of derivative security exempted pursuant to Rule 16b-3"* |
| **Derivative** (*"Except for transactions exempted pursuant to Rule 16b-3"*) | C, E, H, O, X | **X** *"Exercise of in-the-money or at-the-money derivative security"*; **O** the out-of-the-money version; **C** *"Conversion of derivative security"* |
| **Other 16(b) Exempt and Small Acquisition** | G, L, W, Z | **G** *"Bona fide gift"*; W *"...by will or the laws of descent and distribution"* |
| **Other** | J, K, U | J *"Other acquisition or disposition (describe transaction)"*; K is an equity swap and is **appended** to another code, *"e.g., 'S/K' or 'P/K.'"* |

🚨 **The code is not the direction.** I, J, G, W, Z, K and V have no direction in their own
SEC description - I is literally *"resulting in acquisition or disposition"*. Direction is a
separate acquired/disposed flag on each transaction line. A signal keyed on the code string
alone is guessing on seven of twenty codes.

⚠️ **"Exempt" is a heading, not a field.** Only G, L, W and Z sit under a heading containing
the word; A, D, F, I and M sit under *"Rule 16b-3 Transaction Codes"*, and Rule 16b-3 is the
exemptive rule. C, E, H, O and X are headed *"Except for transactions exempted pursuant to
Rule 16b-3"*.

⚠️ 🔑 **V is a selection effect, not a curiosity.** A filing marked V arrived *before* it had
to. Any lag distribution you measure is a mixture of on-time filers, early filers and the
deferred branch, and V is the only one of the three the form labels.

### ⚠️ Which codes carry information - academic, not SEC

🔴 **No SEC publication says any code is informative.** Do not attribute it to them. The
finding is academic and it is ⚠️ secondhand here:

- **Lakonishok & Lee**, *"Are Insider Trades Informative?"*, **Review of Financial Studies
  14(1), 2001, 79-111**: *"informativeness of insiders' activities is coming from purchases,
  while insider selling appears to have no predictive ability."* NYSE/AMEX/Nasdaq, 1975-1995.
- **Jeng, Metrick & Zeckhauser**, *Review of Economics and Statistics* **85(2), 2003**,
  453-471. From the NBER working-paper abstract (w6913): the purchase portfolio *"earns
  abnormal returns of about 40 basis points per month"*; *"The sale portfolio does not earn
  abnormal returns."* ⚠️ The widely quoted ">6% per year" figure from the published version
  could **not** be reached at a primary source; use the 40 bp/month number or verify it.
- **Cohen, Malloy & Pomorski**, *"Decoding Inside Information"*, **Journal of Finance 67(3),
  2012**, 1009-1043: *"opportunistic insider trades yields value-weight abnormal returns of
  82 basis points per month, while the abnormal returns associated with routine traders are
  essentially zero."* 🚨 **This is routine-vs-opportunistic *timing*, not the P-vs-S code
  split.** Conflating the two is common and wrong.

## 3. ✅ Measured - the lag and the hour

6,000 seeded filings. `draw_lag_and_hour()` sets both distributions; these are measured back.

| business days, execution to acceptance | mean | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| | **2.50** | **2** | 4 | 5 | 16 | 49 |

✅ Measured: **75.2%** inside the two-business-day deadline, **96.8%** inside the T+5 outer
limit that 16a-3(g)(4) implies for the deferred branch, **3.2%** past it. And the hour:

✅ Measured: **62.1%** of filings accepted at or after the 16:00 close, **50.7%** after 17:30,
latest 22.0 ET.

⚠️ Those two percentages are the generator's, not EDGAR's - **measure your own feed's**. What
is *not* a choice is the shape: the deadline is two business days rather than 45, and the
ownership-form window runs six hours past the closing bell.

## 4. ✅ Measured - the A/B, and where the gap actually comes from

Long the purchased name for 63 trading days, short the market, entered at the **close of the
key day**. Open-market purchases only (code P). Seed 20260910:

| key | Sharpe | ann. return | ann. vol | gap |
|---|---|---|---|---|
| **transaction-date** (look-ahead) | **1.869** | 13.58% | 7.26% | **+0.768** |
| **filing day's close** (look-ahead) | 1.631 | 11.92% | 7.31% | +0.529 |
| **first tradeable session** (honest) | **1.102** | 7.98% | 7.25% | - |

Now decompose it. `announce_share` is how much of each purchase's alpha arrives as a jump on
the first session the market can trade the filing, rather than as drift from the execution
date. 10 seeds per row:

| announce share | transaction key | first tradeable | gap | retained |
|---|---|---|---|---|
| **0%** | 1.628 | **1.436** | **0.191** | **88.2%** |
| 20% | 1.624 | 1.138 | 0.486 | 70.1% |
| **40%** | 1.616 | **0.840** | **0.776** | **52.0%** |
| 60% | 1.605 | 0.542 | 1.064 | 33.7% |

🔑 🚨 **At a two-business-day deadline the lag itself is nearly free - 88.2% retained with
no announcement effect at all.** The gap in real Form 4 data is not the deadline; it is the
market's reaction to the filing, which happens on the session you *enter at the close of*.
This is the opposite diagnosis from `../congressional-trading-disclosures/SKILL.md`, where a
45-day lag destroys the signal by decay. **Same look-ahead, different mechanism, and the fix
is different: for Form 4 you cannot buy the pop back by trading faster on a daily bar.**

⚠️ The announcement share is a parameter, not a measurement of the market. Estimate your own
by comparing the filing-day return of code-P names against a matched sample.

## 5. 🚨 Measured - what filling at the filing day's close is worth

| fill | Sharpe | sd |
|---|---|---|
| **filing day's close**, whatever hour it was accepted | **1.272** | 0.351 |
| **first tradeable session** | **0.840** | 0.363 |

✅ Measured: **0.432 of Sharpe**, bought entirely with the **61.3%** of filings accepted after
the bell. This is the 90-minute post-close leak that
`../../../fin-core/skills/fundamental-and-macro-data/SKILL.md` §2.1 describes for periodic
reports, except the ownership-form window makes it a **six-hour** one.

## 6. 🚨 Measured - the code filter is the whole game

All four signals honestly keyed to the first tradeable session, same filings, same names,
same 63-day hold. 10 seeds. Only the code filter differs:

| signal | Sharpe | sd | % of filings used | % of those informed |
|---|---|---|---|---|
| every Form 4 acquisition (P, A, M, C, X) | **0.264** | 0.349 | 53.1% | 23.0% |
| "net insider buying" (acquisitions minus dispositions) | **0.196** | 0.284 | 96.0% | 12.5% |
| purchases and sales (P long, S short) | 0.453 | 0.339 | 36.2% | 32.5% |
| **open-market purchases only (P)** | **0.840** | 0.363 | **12.1%** | **100%** |

✅ Measured spread: **0.644 of Sharpe from the filter alone.** 🔑 **The most inclusive signal
is the worst one.** "Net insider buying" reads 96% of filings and only 12.5% of what it reads
is a decision - the rest is the issuer's calendar: annual grants (A), the exercises that
follow them (M) and the shares withheld to pay the tax on them (F). Shorting F is shorting a
payroll deduction.

⚠️ In this generator only code P carries alpha, which is the strongest possible version of
§2's academic finding. If sales carried information the P-and-S row would rise. The ordering
that does **not** depend on that choice is *inclusive < selective*, because the denominator
is real: grants and exercises outnumber purchases and they are scheduled, so they also
cluster in calendar time, which concentrates the naive portfolio.

## 7. Traps

- 🚨 **Keying on the execution date.** §4, +0.768 of Sharpe on one seed.
- 🚨 **Filling at the filing day's close.** §5, +0.432. Use `acceptanceDateTime`, convert to
  exchange local time, and if it is at or after 16:00 the earliest bar is the next session.
- 🚨 **No code filter, or a filter built from the code string alone.** §6, and §2: seven of
  twenty codes carry no direction in their own description.
- 🚨 **Treating F as a sale.** It is tax withholding on vesting. So is D (disposition back to
  the issuer). Neither is a view.
- 🚨 **Treating M as a purchase.** An option exercise is a conversion, usually on a schedule,
  and is very often paired with an immediate S on the same form.
- 🚨 **Counting shares instead of reading the A/D flag.** A single Form 4 routinely carries an
  M, an F and an S on three lines with opposite directions. Summing "shares transacted" across
  them produces a number that means nothing.
- ⚠️ **10b5-1 plan trades are pre-committed.** §1's deferred branch exists because the insider
  did not choose the date. A plan trade is close to the "routine" category in Cohen/Malloy/
  Pomorski; treating it as a view is what that paper says not to do.
- 🚨 **Amendments (4/A) backfilled onto the original date.** A 4/A is a new publication with
  its own acceptance timestamp. Same rule as a restated 10-K.
- 🚨 **Survivorship in the issuer universe.** Insider buying concentrates in small, distressed
  names - exactly the ones a current-constituents universe has dropped.
  `../../../fin-core/skills/research-integrity-guards/SKILL.md` §1.
- ⚠️ **Business days are not trading days.** The deadline counts business days; the exchange
  calendar has holidays the federal one does not, and vice versa. This script treats them as
  the same and says so.

## 8. Scripts and where this sits

`scripts/form4.py` - the transcribed deadline, deferred branch, filing window and all 20
codes in `TRANSACTION_CODES`; `draw_lag_and_hour()` and `lag_summary()`; `make_panel()` with
`announce_share` and scheduled-code clustering; `portfolio()`, `run_ab()`, `announce_split()`,
`post_close_cost()`, `signal_sides()` and `code_filter_table()`. numpy + pandas, seed
20260910, about 5 s, no network, no file writes, no scraper.

- `../congressional-trading-disclosures/SKILL.md` - the same A/B with a **45-day** deadline,
  where the gap comes from decay rather than the announcement, and where the amounts are
  disclosed as brackets.
- `../institutional-13f/SKILL.md` - the same A/B where the clock starts at a *quarter end*.
- `../social-and-influencer-feeds/SKILL.md` - the same trap with no filing at all.
- `../../../fin-core/skills/fundamental-and-macro-data/SKILL.md` §1 and §2.1 - the EDGAR
  endpoints, the required User-Agent, the 10 req/s limit, and the period/filed/accepted
  distinction. `edgartools` is the library it recommends for typed Form 4 objects.
- `../../../fin-core/skills/research-integrity-guards/SKILL.md` §2a - the availability rule,
  which already names the 22:00 ET ownership-form cutoff.
- `../../../fin-libraries/skills/lib-edgartools/SKILL.md` - the per-library page.
- `../../../fin-core/skills/backtest-validation/SKILL.md` - four code filters is four trials,
  and a Sharpe of 0.840 with sd 0.363 across seeds needs an interval, not a decimal.
