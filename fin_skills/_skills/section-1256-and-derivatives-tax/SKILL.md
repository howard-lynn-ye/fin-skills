---
name: section-1256-and-derivatives-tax
description: >-
  Futures and broad-based index options are marked to market on the last business day of the year
  and split 60/40 long/short regardless of holding period, so two options with the same payoff can
  have different after-tax P&L. TRIGGER - section 1256, 1256 contract, 60/40, sixty forty, mark to
  market at year end, marked to market December 31, regulated futures contract, nonequity option,
  broad-based index option, narrow-based security index, SPX vs SPY tax, XSP, VIX options, futures
  tax treatment, Form 6781, blended rate on futures, "do I owe tax on an open position", net
  section 1256 loss carryback, qualified board or exchange. Modelling assumptions for backtests,
  not tax advice. SKIP for stock lots and cost basis (tax-lot-matching-and-cost-basis), for the
  wash-sale rule that does not reach these contracts (wash-sale-rules), for reporting an after-tax
  Sharpe (after-tax-backtesting), and for option pricing and lifecycle mechanics
  (options-backtesting).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Section 1256 and derivatives tax

**Two options can have the same underlying exposure, the same premium and the same pre-tax P&L,
and be taxed under two different regimes. One of them owes tax on December 31 for a position it
never closed.**

These are **modelling assumptions for a backtest, not tax advice.** Every rule carries the
authority and the date this library read it. Rules change — one of them changed nine days before
this file was written, §7 — so confirm current law with a qualified professional before relying
on any of it.

✅ Measured comes from `scripts/section_1256.py` — numpy + pandas, seed 20260909, one premium
series traded once and taxed twice, **under 1 s**. ✅ source-verified material was read on
**2026-09-09** in **IRC §1256** (uscode.house.gov, prelim), **IRS Publication 550 (2025)**
pp. 56–57 and p. 82, **15 U.S.C. §78c(a)(55)**, and **Rev. Rul. 2026-16**.

## 1. The two rules

✅ source-verified — **IRC §1256(a)(1)**: each section 1256 contract held at the close of the
taxable year *"shall be treated as sold for its fair market value on the last business day of
such taxable year (and any gain or loss shall be taken into account for the taxable year)."*

✅ source-verified — **IRC §1256(a)(3)**: gain or loss is *"short-term capital gain or loss, to
the extent of 40 percent"* and *"long-term capital gain or loss, to the extent of 60 percent"*.
Pub 550 (2025) p. 82 adds the part that matters: *"regardless of how long the contracts were
held."*

🔑 **`last business day`, not December 31.** ✅ Measured — `last_business_day()` returns
**2022-12-30**, **2023-12-29**, **2024-12-31**, **2025-12-31**. ⚠️ It is a weekday rule and does
not know the exchange calendar; if the market closes on the last weekday of the year the
statutory date moves, so check the calendar for the years your sample actually spans.

Pub 550 (2025) p. 57 works the example this library's tests reproduce exactly: a regulated
futures contract bought 2024-06-03 for $50,000 is worth $57,000 on 2024-12-31, so **$7,000 is
recognised on the 2024 return** 60/40; sold 2025-02-03 for $56,000, a **$1,000 loss** is
recognised in 2025, also 60/40. The year-end mark becomes the new reference. **`recognise_1256`
implements exactly that chain.**

## 2. ✅ source-verified — what qualifies, and where the sources stop

**IRC §1256(b)(1)**: (A) regulated futures contract, (B) foreign currency contract, (C)
**nonequity option**, (D) dealer equity option, (E) dealer securities futures contract. §1256(b)(2)
excludes swaps, caps, floors *"or similar agreements"*, and non-dealer securities futures.

The chain that decides an option, all in the statute:

| Step | Authority |
|---|---|
| a **nonequity option** is any listed option that is **not an equity option** | §1256(g)(3) |
| an **equity option** is an option *"to buy or sell stock"*, or one *"the value of which is determined directly or indirectly by reference to any stock or any narrow-based security index"* | §1256(g)(6) |
| and it *"includes such an option on a group of stocks **only if** such group meets the requirements for a narrow-based security index"* | §1256(g)(6), final sentence |
| **narrow-based** = 9 or fewer components, **or** one component over 30% of weighting, **or** the top five over 60%, **or** the lowest-weighted quarter has aggregate ADV under $50m (\$30m for a 15+ component index) | 15 U.S.C. §78c(a)(55)(B) |

✅ **The S&P 500 is not a close call, and the IRS says so in its own words.** Pub 550 (2025)
p. 56, verbatim: *"Nonequity options include debt options, commodity futures options, currency
options, and **broad-based stock index options**. A broad-based stock index is based on the value
of a group of diversified stocks or securities (such as the **Standard and Poor's 500 index**)."*
So an **SPX-style broad-based index option is a §1256 contract** — statute *and* the
Publication's own example. Pub 550 p. 56 also carries the mechanism for the borderline cases:
*"Cash-settled options based on a stock index ... are nonequity options if the SEC determines
that the stock index is broad based."*

⚠️ **The ETF-option half is NOT verified, and this library will not pretend otherwise.** Every
tax practitioner treats an option on SPY, QQQ or IWM as an ordinary equity option — the reasoning
being that an ETF share is stock (or an interest treated as stock), so §1256(g)(6)(A) reaches it
directly and the group-of-stocks sentence never applies. That reasoning is sound. But **neither
§1256(g) nor Pub 550 (2025) mentions exchange-traded funds anywhere**, and a search of published
IRS guidance on 2026-09-09 turned up no ruling squarely on the point. `classify("etf_option")`
therefore returns **`"unclear"`**, with the warning in the `authority` field, rather than a
regime this library cannot source. **Get advice before you build a backtest whose whole edge is
the difference.**

## 3. ✅ Measured — same trades, two regimes

One seeded premium series, 38 positions of 10 contracts on 42-business-day holds, plus **one
long-dated position opened early and never closed**. The economics are identical by construction.

**Pre-tax P&L, identical under both regimes: 41,280.10.** Assumed rates — an input the script
prints, never a default it supplies — short 37%, long 20%, so the 60/40 blend is **26.8%**.

| year | §1256 recognised | §1256 tax | equity-option recognised | equity-option tax |
|---|---|---|---|---|
| 2022 | 12,486.06 | 3,346.27 | 6,135.21 | 2,270.03 |
| 2023 | −10,637.53 | −2,850.86 | −9,238.47 | −3,418.23 |
| 2024 | 39,535.69 | 10,595.56 | 32,259.23 | 11,935.91 |
| 2025 | −52.06 | −13.95 | −504.92 | −186.82 |
| **TOTAL** | **41,332.16** | **11,077.02** | **28,651.05** | **10,600.89** |

✅ Measured, and it is the internal check that says the implementation is right:

| | effective rate on recognised income |
|---|---|
| §1256 | 11,077.02 / 41,332.16 = **26.8%** — the 60/40 blend, exactly |
| equity option | 10,600.89 / 28,651.05 = **37.0%** — the short-term rate, exactly (all 37 closes are short-term) |

🚨 **The position never closed books 12,681.11 under §1256, across three year ends, and 0.00 as
an equity option.** §1256 has already taxed **41,332.16** of a book that has made **41,280.10** —
more than the book is worth — because the last mark is above the current price and nothing has
been sold.

🔑 **Two effects, and they point in opposite directions:**

- **Rate.** 60/40 on 41,332.16 saves **4,215.88** against the short-term rate.
- **Timing.** §1256 recognised **12,681.11 more** income, years earlier, **with no cash from a
  sale to pay the tax with**.
- **Net on this book: §1256 pays 476.13 MORE tax.**

## 4. 🚨 "60/40 is better" is only true for a short-held gain

✅ Measured — one $10,000 gain held **516 days** and closed:

| | tax at 37/20 |
|---|---|
| §1256 (60/40) | **2,680.00** |
| equity option (long-term) | **2,000.00** |

**§1256 costs 34% more on that trade.** 60/40 is a blend, and a blend is worse than the better
half. It beats a short-term rate and loses to a long-term one — so the sign of the "1256
advantage" is a function of your holding period, and for a strategy that would naturally hold
over a year it is a disadvantage.

⚠️ And on a **net loss**, the same arithmetic runs backwards: 60% of the loss is stuck in the
long-term bucket at the lower rate. `blended_rate()` is symmetric; your P&L is not.

⚠️ **One genuine consolation on the loss side**, Pub 550 (2025) p. 57: an individual with a **net
section 1256 contracts loss** may elect to **carry it back 3 years**, capped by the net §1256
gain in each of those years, 60/40 in each carryback year. This module does not implement the
carryback — it reports the per-year schedule that a carryback calculation needs.

## 5. What the mark-to-market rule does to a backtest

🚨 **Tax on an unrealised position is a cash-flow event with no matching cash inflow.** A
backtest that compounds after-tax returns must debit the tax at the year end, not at the eventual
close. If your engine only taxes closes, a §1256 book will show cash it does not have.

🚨 **The mark resets the reference every year, so the P&L is a chain, not a difference.** Getting
this wrong double-counts: the naive `close − open` for a position that spanned two year ends
books the whole move twice if the marks were also booked. `recognise_1256` carries the reference
forward, and the test suite checks that the sum over years equals the total economic move to
floating-point tolerance.

⚠️ **Terminations and transfers count too.** Pub 550 (2025) p. 57: the rule applies where rights
*"are terminated or transferred during the tax year"* — offset, delivery, exercise, assignment or
lapse, valued at the time. An expiry is not a non-event.

⚠️ **Not modelled here, and each one changes the answer:** the hedging-transaction exception
(§1256(e), and the identification must be made **before the end of the day** the transaction is
entered into), straddles and the loss-deferral rules, the mixed-straddle elections, §988 ordinary
treatment for foreign currency, and the limited-partner/limited-entrepreneur carve-out from 60/40
for dealer contracts.

⚠️ **Wash sales do not reach these contracts.** Pub 550 (2025) p. 87: the wash sale rules *"do
not apply to losses from sales or trades of commodity futures contracts and foreign currencies."*
See `../wash-sale-rules/SKILL.md` for the regime that does — and note the straddle rules can
still defer a loss here, by a different mechanism.

## 6. 🚨 Dated — the qualified-exchange list moves

✅ source-verified — **Rev. Rul. 2026-16**, read 2026-09-09 at `irs.gov/pub/irs-drop/rr-26-16.pdf`:
the IRS determines that **ICE Endex**, a regulated exchange of the Netherlands, **is a qualified
board or exchange within §1256(g)(7)(C)** for as long as it holds a valid CFTC foreign-board-of-
trade Order of Registration. **Effective for ICE Endex contracts entered into on or after
2026-09-01** — nine days before this file was written — on a cut-off basis, with consent to the
method change granted and Form 3115 waived.

🔑 **Whether a contract is a §1256 contract at all depends on a list the IRS keeps adding to, by
ruling, with prospective effective dates.** A backtest that spans the effective date of one of
these rulings spans two tax regimes for the same contract. §1256(g)(7) is (A) an SEC-registered
national securities exchange, (B) a CFTC-designated contract market, or (C) whatever the Treasury
has separately determined — and (C) is the moving part. Re-check it for the venues you trade.

## 7. Scripts and where this sits

`scripts/section_1256.py` — `classify` (regime plus the authority for it, `"unclear"` where the
sources stop), `sixty_forty`, `blended_rate`, `last_business_day`, `recognise_1256` (year-end
marks chained into the close), `recognise_equity_option`, `tax_schedule` (rates are arguments),
`pretax_pnl`, `compare_regimes`, and `make_option_book` for the seeded demo. numpy + pandas, seed
20260909, under 1 s.

- Stock lots, cost basis, and the short/long split for everything that is not a §1256 contract —
  `../tax-lot-matching-and-cost-basis/SKILL.md`.
- The loss-disallowance regime that stops at the edge of this one —
  `../wash-sale-rules/SKILL.md`.
- Reporting an after-tax Sharpe with the rate, jurisdiction and lot method stated —
  `../after-tax-backtesting/SKILL.md`.
- Option lifecycle, expiry, assignment and the data a backtest needs before any of this applies —
  `../../../fin-core/skills/options-backtesting/SKILL.md`.
- Futures contract specs, roll rules and the continuous series a futures backtest runs on —
  `../../../fin-futures-fx/skills/futures-continuous-contracts/SKILL.md`.
- The US rulebook this sits inside — `../../../fin-core/skills/us-market-rules/SKILL.md`.
