---
name: institutional-13f
description: >-
  Clone or study 13F holdings without the quarter-end look-ahead, and report the full age
  distribution of the positions instead of the 45-day deadline. TRIGGER - 13F, Form 13F,
  13F-HR, institutional holdings, "what does Berkshire own", hedge fund holdings, whale
  watching, cloning a manager's portfolio, guru portfolios, 13F information table XML,
  infoTable, $100 million threshold, 45 days after quarter end, confidential treatment
  request, "positions omitted from the 13F", "backtest a 13F clone", quarterly holdings
  turnover, WhaleWisdom-style data, reconstructing a fund's book from filings. SKIP for
  corporate insiders and Form 4 codes (insider-form-4), for congressional trades and the
  STOCK Act (congressional-trading-disclosures), for social posts and influencer panels
  (social-and-influencer-feeds), for EDGAR endpoints and XBRL fundamentals
  (fundamental-and-macro-data), and for the general availability rule
  (research-integrity-guards).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-10"
---

# Institutional 13F

**The 45-day deadline is the smaller half of the lag, and the number everyone quotes.** A
13F is a snapshot of a *quarter end*, filed up to 45 days later. A position opened on the
first day of that quarter is already 91 days old when the quarter closes. ✅ Measured below,
the median position is **82 calendar days old** at the moment its table becomes public, and
the p95 is **127**.

Everything marked ✅ Measured is printed by `scripts/thirteen_f.py` (numpy + pandas, seed
20260910, about 15 s, no network). Its holdings are **synthetic and seeded**. Everything
marked ✅ source-verified was read at the URL given, on 2026-09-10.

## 1. ✅ The rule, at primary sources

| Fact | Value | Source, read 2026-09-10 |
|---|---|---|
| **Deadline** | *"within 45 days after the last day of such calendar year"* and *"within 45 days after the last day of each of the first three calendar quarters"* | **17 CFR 240.13f-1(a)(1)** (eCFR); Form 13F General Instruction 1 |
| **Threshold** | accounts holding section 13(f) securities *"having an aggregate fair market value on the last trading day of any month"* of *"at least $100,000,000"* | 17 CFR 240.13f-1(a)(1) |
| 🚨 Threshold never indexed | the rule's own source credit starts **43 FR 26705, June 22, 1978** and the figure survives every amendment. The SEC's 2020 proposal said it would be raising it *"for the first time in 45 years, ... from $100 million to $3.5 billion"* | Rel. 34-89290 (`govinfo.gov`) |
| ⚠️ **The proposal was withdrawn** | **2021-05-11**, RIN 3235-AM65, *"This item is being withdrawn."* The $100M threshold stands. The SEC's own rule page still shows the proposal with no withdrawal note. | OMB/OIRA Unified Agenda |
| Format | *"On May 20, 2013, the text-based ASCII format for 13F filings was discontinued."* XML information table since. | 13F FAQ |

## 2. 🚨 What is in a 13F, and what is not

| instrument | status | source, read 2026-09-10 |
|---|---|---|
| US exchange-traded stock | reported | FAQ Q7: the Official List *"primarily includes U.S. exchange-traded stocks"* |
| closed-end funds, ETFs | reported | FAQ Q7 names both |
| convertible debt, equity options, warrants | reported *if on the Official List* | FAQ Q7: *"Certain convertible debt securities, equity options, and warrants are on the Official List"* |
| 🚨 **short positions** | **NOT reported** | **FAQ Q41 verbatim: *"You should not include short positions on Form 13F."*** |
| 🚨 **long netted against short** | **NOT permitted** | FAQ Q41: do not subtract shorts from longs; *"report only the long position"* |
| open-end mutual fund shares | NOT reported | FAQ Q7: shares of open-end investment companies *"should not be reported"* |
| securities on non-US exchanges | NOT reported | FAQ: shares that trade on non-United States exchanges are not reported |
| ⚠️ bonds, commodities, currencies, cash | NOT reported | **INFERRED** from 13f-1(c) - equity securities on the Official List only. The SEC does not state these exclusions in those words. |

### ✅ Confidential treatment - the positions that are not there

Requests are made *"in accordance with rule 24b-2(i) under the Exchange Act"*; the public
filing must show *"that confidential information has been omitted"*. The FAQ places the
commercial rationale under **FOIA Exemption 4**, which protects *"trade secrets and
commercial or financial information"*; the requested period *"may not exceed one (1) year"*.
On denial or expiry an amendment follows *"within six business days of the denial ... or the
expiration"*, and it *"must not be a restatement"* - it *"adds new holdings entries"*.

🔑 **So the public table tells you that something is missing, never what.** §7 measures what
that is worth.

## 3. ✅ Measured - the age distribution, which is the point

`draw_filing_lags()` sets the filing lag; the mass piles against day 45, as a deadline does.
Measured back, calendar days after quarter end:

| mean | p05 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|
| 42.5 | 34.6 | 39.8 | 42.5 | 44.0 | 45.0 | 173.5 |

Now the number that matters: the **age of the position** when its table becomes public,
across 9,792 seeded disclosed positions.

| min | p05 | p25 | **median** | mean | p75 | **p95** | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 24 | 44 | 60 | **82** | 83 | 106 | **127** | 136 | 224 |

✅ Measured: **40.9% of positions are more than 90 days old** when you learn of them, and
**11.5% are more than 120 days old**. It decomposes into **40.9 days** of
establishment-to-quarter-end plus **42.4 days** of quarter-end-to-filing.

🔑 🚨 **"45 days" is one of the two legs and it is roughly half the total.** Quoting it as
the staleness of 13F data understates it by a factor of about two, and it hides that the
staleness is a *distribution* - the same filing carries positions 24 days old and positions
224 days old, and nothing in the table says which is which.

## 4. 🚨 Measured - what the table never contains

- ✅ **21.5%** of positions in this panel are opened *and* closed inside one quarter, so
  **they never appear in any 13F**. Their mean holding period is **19 trading days**; the
  ones you do see average **154**.
- ✅ **20.7%** of the positions you do see have **already been closed by the filing date**,
  and **52.4%** are closed within a quarter of it.

🔑 **13F is a quarter-end snapshot, not a trade record, so what you can copy is selected on
being slow.** Any statement about "the manager's turnover" or "the manager's trades" built
from consecutive 13Fs is a statement about the slow half of the book. That is a survivorship
bias in the *time* dimension, and it is not in the five gates of
`../../../fin-core/skills/research-integrity-guards/SKILL.md` because it is domain-specific.

## 5. ✅ Measured - the A/B

Weight-clone the disclosed longs for 63 trading days, short the equal-weight universe,
entered at the close of the key day. Seed 20260910:

| key | Sharpe | ann. return | ann. vol |
|---|---|---|---|
| **quarter-end date** (look-ahead) | **1.298** | 3.47% | 2.67% |
| **filing date** (honest) | **0.596** | 1.59% | 2.66% |

Over 24 seeds:

| | mean | sd | p05 | p95 | > 0 |
|---|---|---|---|---|---|
| quarter-end | **1.176** | 0.395 | 0.504 | 1.668 | 100% |
| filing | **0.640** | 0.305 | 0.172 | 1.129 | 100% |
| gap | **0.536** | 0.440 | -0.214 | 1.193 | **83%** |

✅ Measured pooled retention: **54.4%** of the look-ahead Sharpe survives honest keying.

⚠️ 🔑 **Read the sd on that gap.** At this alpha level the A/B on one panel does **not**
settle anything - it is negative in 17% of seeds. That is the honest report, and it is also
the point: a single-run 13F A/B is underpowered, so run it over managers or seeds and quote
the spread. Compare `../insider-form-4/SKILL.md`, where the same A/B is decisive in every
seed because the mechanism there is a jump rather than a slow decay.

## 6. 🚨 Measured - a 13F clone is not the manager's portfolio

True book: 130% long 13F-visible equity, **60% short equity that FAQ Q41 says is not
reported**, and a 20% non-13F sleeve. The reconstruction is the long sleeve alone at 100%,
because that is all the table contains. 40 managers:

| | mean | sd | p05 | p95 |
|---|---|---|---|---|
| correlation with the true book | **0.932** | 0.016 | 0.909 | 0.954 |
| **tracking error, ann.** | **8.5%** | 0.7% | 7.6% | 9.7% |
| vol of the true book | 23.5% | 1.0% | 22.2% | 25.1% |
| vol of the reconstruction | 21.9% | 0.5% | 21.2% | 22.5% |
| **market beta, true** | **0.741** | 0.095 | 0.636 | 0.881 |
| **market beta, reconstructed** | **0.999** | 0.025 | 0.965 | 1.036 |

🚨 **A correlation of 0.93 is what makes this dangerous.** It reads as a noisy copy. It is
not: the market beta is wrong by a third and the tracking error is **8.5% a year**. The
error is a *systematic exposure*, not noise, and it does not average away with more quarters.
🔑 **A "manager's returns" series computed from 13F filings is someone else's book.** Say so,
or do not publish the series.

## 7. ✅ Measured - confidential treatment

Omit a share of positions from the public table, re-run the filing-date clone. One panel per
seed, 20 seeds, so the rows differ only in what was omitted:

| omitted | top-conviction omitted | randomly omitted | s.e. | cost of selection |
|---|---|---|---|---|
| 0% | 0.628 | 0.628 | 0.073 | 0.000 |
| 5% | 0.600 | 0.614 | 0.074 | 0.013 |
| **10%** | **0.540** | 0.610 | 0.070 | **0.070** |
| 20% | 0.525 | 0.599 | 0.077 | 0.074 |

🔑 **The random column barely moves at any level; the top-conviction column falls
monotonically.** It is not *how many* positions are withheld, it is *which* - and the
adversarial assumption is the working one, because a manager requests confidentiality for a
position it is still building. ⚠️ Read the cost against the standard error: below about 10%
it is inside the noise here. The request can run a year, so the amendment that finally
reveals the position arrives after the information in it is spent.

## 8. Traps

- 🚨 **Keying on the quarter-end date.** §5. Use the filing's own acceptance timestamp.
- 🚨 **Quoting "45 days" as the staleness.** §3: the median position is 82 days old and the
  spread runs 24 to 224. Report the distribution.
- 🚨 **Calling a long-only reconstruction the manager's portfolio.** §6.
- 🚨 **Inferring trades by differencing consecutive 13Fs.** The difference is
  `held at Q_end` minus `held at Q-1_end`. Everything opened and closed between them is
  invisible (§4, 21.5% here), and a "new position" may have been opened 91 days before you
  see it and closed before you can act (20.7% here).
- 🚨 **Treating an amendment as a correction to the original.** A confidential-treatment
  amendment *"must not be a restatement"* - it **adds** entries. Merging it back onto the
  original filing date rewrites history with information that was withheld at the time.
- 🚨 **Building the manager universe from managers that file today.** A fund that closed
  stopped filing. The list of 13F filers is a survivorship-biased universe in exactly the way
  `../../../fin-core/skills/research-integrity-guards/SKILL.md` §1 describes.
- ⚠️ **The $100M threshold has not moved since 1978**, so the filer population has grown
  enormously in real terms and is not comparable across decades. A study spanning 1980-2026
  is comparing different populations.
- 🚨 **Share counts are not weights.** The table reports shares and a value as of quarter end.
  Converting to a portfolio weight needs a price *as of that quarter end* and a total that
  excludes everything in §2 - so the weights do not sum to the manager's book.
- ⚠️ **Position size is reported for the whole 13F filer**, which may aggregate several funds
  with different mandates, and joint filings are allowed. "The manager" may be many managers.
- 🚨 **CUSIP-to-ticker mapping is a dated join.** Same `(identifier, date)` rule as everywhere
  else; a current CUSIP table reintroduces survivorship.

## 9. Scripts and where this sits

`scripts/thirteen_f.py` - the transcribed deadline, threshold, coverage table and
confidential-treatment mechanism; `draw_filing_lags()`, `lag_summary()`, **`age_summary()`**
and `invisibility()`; `make_panel()`; `portfolio()` (with `neutral=False` for the raw long
book), `run_ab()`, `ab_over_seeds()`, `reconstruction_error()` and
`confidential_treatment()`. numpy + pandas, seed 20260910, about 15 s, no network, no file
writes, no scraper.

- `../insider-form-4/SKILL.md` - the same A/B with a **two-business-day** deadline, where the
  gap is an announcement rather than decay and is decisive in every seed.
- `../congressional-trading-disclosures/SKILL.md` - the same A/B at 45 days from the
  *transaction*, and the bracketed amounts.
- `../social-and-influencer-feeds/SKILL.md` - selecting the manager or the account after
  seeing the returns, which is the bias this skill's universe trap shares.
- `../../../fin-core/skills/fundamental-and-macro-data/SKILL.md` - the EDGAR endpoints, the
  User-Agent requirement, and the period/filed/accepted distinction 13F inherits.
- `../../../fin-core/skills/research-integrity-guards/SKILL.md` §1 and §2 - the survivorship
  and availability gates this skill is a special case of.
- `../../../fin-core/skills/portfolio-and-risk/SKILL.md` - if you are going to report a
  reconstructed book's beta and tracking error, that is where the attribution belongs.
- `../../../fin-core/skills/backtest-validation/SKILL.md` - a gap of 0.536 with sd 0.440 is
  not a result; it is a reason to run more panels.
