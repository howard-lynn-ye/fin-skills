---
name: wash-sale-rules
description: >-
  A wash sale defers a loss into the replacement's basis rather than destroying it, and a
  monthly-rebalanced strategy triggers one on almost every trade. TRIGGER - wash sale, wash sales,
  IRC 1091, section 1091, 30 day rule, 61 day window, substantially identical, disallowed loss,
  loss disallowed, basis adjustment, replacement shares, tax loss harvesting, harvesting rules,
  buy back within 30 days, "can I sell and rebuy", Form 8949 code W, 1099-B box 1g, wash sale in
  an IRA, Rev. Rul. 2008-5, holding period carryover, "my backtest books every loss", after-tax
  turnover penalty. Modelling assumptions for backtests, not tax advice. SKIP for choosing which
  lot to sell in the first place (tax-lot-matching-and-cost-basis), for futures and broad-based
  index options where the rule does not apply (section-1256-and-derivatives-tax), for wiring it
  into a backtest and reporting an after-tax Sharpe (after-tax-backtesting), and for the A-share
  tax code (china-ashare-trading-taxes).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Wash sale rules

**The wash sale window is 30 days before AND 30 days after — 61 days in total. A strategy that
rebalances monthly puts every trade inside the previous trade's window and inside the next one's.
So this is not an edge case for a systematic book; it is the normal state.**

These are **modelling assumptions for a backtest, not tax advice.** Every rule below carries the
authority and the date this library read it. Rules change; confirm current law with a qualified
professional before relying on any of it.

Everything marked ✅ Measured comes from `scripts/wash_sales.py` — numpy + pandas, seed 20260909,
one 36-month price path traded two ways, **under 1 s**. ✅ source-verified material was read on
**2026-09-09** in **IRC §§1091, 1223** (uscode.house.gov, prelim), **IRS Publication 550 (2025)**
(pp. 86–87), and **Rev. Rul. 2008-5, I.R.B. 2008-3** (2008-01-22).

## 1. The rule, in the two halves people separate

✅ source-verified — **IRC §1091(a)**, verbatim: a loss is disallowed where, *"within a period
beginning 30 days before the date of such sale or disposition and ending 30 days after such date,
the taxpayer has acquired ... or has entered into a contract or option so to acquire,
substantially identical stock or securities."*

✅ source-verified — Pub 550 (2025) p. 86 lists **four** ways the acquisition can happen, and the
fourth is the one nobody models: *"1. Buy substantially identical stock or securities, 2. Acquire
substantially identical stock or securities in a fully taxable trade, 3. Acquire a contract or
option to buy substantially identical stock or securities, or 4. Acquire substantially identical
stock for your individual retirement arrangement (IRA) or Roth IRA."* The same page adds: *"If
you sell stock and your spouse or a corporation you control buys substantially identical stock,
you also have a wash sale."*

And the half that makes it a deferral, Pub 550 (2025) p. 86 verbatim:

> *"If your loss was disallowed because of the wash sale rules, add the disallowed loss to the
> cost of the new stock or securities (except in (4) above). ... This adjustment postpones the
> loss deduction until the disposition of the new stock or securities. Your holding period for
> the new stock or securities includes the holding period of the stock or securities sold."*

Statutory backing for both adjustments: **§1091(d)** for the basis, **§1223(3)** for the holding
period. 🔑 **A wash sale is a timing rule, not a penalty** — with one exception, §4 below.

## 2. ✅ Measured — a monthly rebalance is a wash-sale machine

Seeded 36-month path, constant $100k notional in one name, rebalanced every **21 business days**,
FIFO lots, marked at 60.1699.

| | |
|---|---|
| consecutive trades are | **29–31 calendar days apart** |
| gaps inside the 30-day window | **29 of 36** |
| loss sales | 21 |
| **of which wash sales** | **12** |
| gross realised loss | −41,164.30 |
| **disallowed (deferred)** | **−22,941.47** |
| **fraction of the loss deferred** | **55.7%** |
| realised as reported | −15,863.59 |
| what a naive backtest books | −38,805.06 |

🚨 **At an assumed 37% rate the naive backtest claims 8,488.34 of tax saving it does not have
this year.** The rate is an assumption the script prints, not a default it supplies.

🔑 **Look at the first two rows before anything else.** 21 business days is **29 to 31 calendar
days** depending on holidays, so a monthly rebalance sits *exactly on the boundary* of the
30-day window: 29 of 36 gaps fall inside it and 7 fall outside. Your wash-sale exposure is a
function of **how many business days happened to fall in that month** — which is not a modelling
choice anyone made, and it will not be stable when you re-run the strategy on a different
calendar.

## 3. ✅ Measured — deferral, not destruction, to the cent

Liquidate the surviving position 400 days later, outside every window:

| | |
|---|---|
| reported realised, cumulative | −42,475.73 |
| economic P&L of the book | −42,475.73 |
| **difference** | **0.00** |

🔑 **That zero is the invariant to test your own implementation against.** If your wash-sale
engine does not reproduce it, you have either dropped a basis adjustment or double-counted a
loss. It holds because every disallowed dollar was pushed into the basis of shares still held,
and the final sale gives it back.

⚠️ **The zero is an identity about totals, not about years.** What the rule moves is *when*, and
if the deferred loss lands in a year with no gains to offset it, the $3,000 capital-loss
limitation and the carry-forward mechanics take over — neither of which this module models.

## 4. 🚨 The one case where the loss is destroyed: the IRA

✅ source-verified — **Rev. Rul. 2008-5**, I.R.B. **2008-3** (January 22, 2008), holding,
verbatim: *"The loss on the Sale of stock is disallowed under § 1091. A's basis in the individual
retirement account or Roth IRA is not increased by virtue of § 1091(d)."* Pub 550 (2025) p. 86
carries the same rule in its four-line parenthesis: **"except in (4) above"**.

✅ Measured — the identical rebalance blotter with every repurchase routed to an IRA:

| | |
|---|---|
| disallowed | −22,941.47 |
| **of which permanent** | **−22,941.47** |

**All of it.** No basis step-up, in the taxable account or in the IRA, ever. `summarise()` reports
`permanent_loss` as a separate line for exactly this reason: the deferral invariant in §3 **fails
by that amount**, and a model that silently treats an IRA replacement like any other replacement
will show a loss coming back that never does.

🔑 **This is the failure mode that survives a code review**, because the taxable account has no
record of the IRA purchase, the 1099-B cannot see it, and the number looks completely normal.

## 5. 🚨 The chain — and the number it inflates

✅ Measured — the **same price path**, traded as a monthly tax-loss harvest with a next-day
repurchase (the canonical "harvest and stay invested"), 1,000 shares:

| | |
|---|---|
| loss sales | 10 |
| **of which wash sales** | **10 — every one** |
| gross realised loss | **−292,200.11** |
| disallowed | −292,200.11 (**100%**) |
| deepest chain: a lot re-based before being sold at a loss again | **9 times** |
| economic P&L of the same 1,000 shares | −44,315.39 |
| **deduction actually available this year** | **0.00** |

🚨 **The reported gross loss is 6.6x the economic loss.** Each disallowed loss goes back into the
basis, so next month's sale is under water by the new decline *plus everything disallowed before
it*, and the gross-loss line ratchets. A harvesting P&L report that quotes gross realised losses
is quoting a number that grows with the number of times you churned, not with how much you lost.

🔑 **A harvesting strategy that always repurchases immediately harvests nothing.** The deduction
is zero until the position is finally sold and stays sold for 31 days. The real designs buy a
*correlated but not substantially identical* replacement — and ⚠️ **"substantially identical" has
no bright line**: Pub 550 (2025) p. 87 says *"you must consider all the facts and circumstances
in your particular case"*, then gives only the convertible-preferred and reorganization examples.
Two S&P 500 ETFs from different issuers are the standard trade and the standard argument; this
library does not tell you the answer, and neither does the Publication.

## 6. ✅ Measured — the lot method decides whether you have wash sales at all

The **same rebalance blotter**, matched LIFO instead of FIFO:

| rule | loss sales | wash sales | reported realised |
|---|---|---|---|
| FIFO | 21 | 12 | −15,863.59 |
| **LIFO** | **0** | **0** | **+8,373.07** |

🚨 **On a declining name the newest lot is the cheapest, so LIFO never sells at a loss** — and it
reports a **+8,373.07 realised gain on a book that lost 42,475.73**. Nothing here is wrong;
`../tax-lot-matching-and-cost-basis/SKILL.md` is what decides which lot the sale hits, and this
skill only prices the consequence. 🔑 **Run lot matching and wash sales as one pass**, not two:
the lot method changes the loss sales, and the wash sale changes the basis of the lots the next
sale will hit.

## 7. Implementation notes that are not optional

🚨 **The window is symmetric, and the forward half is where the classic wash sale lives.** Sell
today, buy back tomorrow — the replacement is *after* the sale. An engine that streams the
blotter and searches only the purchases it has seen so far implements a **−30-day** rule and
silently under-reports the disallowance. `apply_wash_sales` builds the full candidate list up
front for this reason, and queues basis adjustments for lots that do not exist yet.

**Matching when the quantities do not line up.** ✅ source-verified, Pub 550 (2025) p. 87: *"You
do this by matching the shares bought with an equal number of the shares sold. Match the shares
bought in the same order that you bought them, beginning with the first shares bought."* The
module consumes each purchase's replacement capacity once, in acquisition order, and the tests
reproduce **both** of the Publication's worked examples exactly — the 75-of-100 pro rata split
(basis 2,750 → 3,250 and 1,125 → 1,375) and the four-daily-purchase case (500 to the first lot,
500 to the second, nothing to the third and fourth).

⚠️ **One deliberate deviation from the Publication's literal ordering, and it only bites at high
turnover.** Pub 550's rule is written for the normal case in which the replacement shares are
still held. Taking the *earliest* in-window purchase when that purchase has itself already been
sold leaves the disallowed loss with no basis to attach to — which converts a deferral into a
permanent loss no broker's 1099-B would ever report. ✅ Measured: on a daily sell-and-rebuy book,
following the literal order orphans **100%** of the disallowed loss and inflates the tax bill by
an order of magnitude. So `apply_wash_sales` tries **still-held (or not-yet-bought) replacements
first**, in acquisition order, and falls back to disposed ones only when nothing else is
available — counting whatever is left in `orphaned_disallowed` so the deviation is visible. Both
demo blotters here report `orphaned_disallowed == 0.0`, and all three of the Publication's worked
examples are unaffected.

| Not modelled | Why it matters |
|---|---|
| "Substantially identical" across **different tickers** | The module compares one symbol against itself. Every real harvesting decision is about two different symbols, and that judgment is not automatable |
| Purchases in **your spouse's** account, or a **controlled corporation's** | Pub 550 p. 86 says they count. No broker feed contains them |
| A replacement already **sold** before the loss sale | Reported as `orphaned_disallowed` rather than silently dropped |
| The **$3,000** capital-loss limitation and carry-forwards | §3's invariant is about totals, not about a single year's deduction |
| **Short sales** (§1091(e)) and options as replacements | `apply_wash_sales` raises rather than netting a short into a long lot stack |

⚠️ **Where the rule does not reach.** Pub 550 (2025) p. 87: *"The wash sale rules apply to losses
from sales or trades of contracts and options to acquire or sell stock or securities. They do not
apply to losses from sales or trades of commodity futures contracts and foreign currencies."* See
`../section-1256-and-derivatives-tax/SKILL.md`, which is a different regime entirely.

⚠️ **Digital assets, as of 2026-09-09.** §1091 reaches *"stock or securities"*; the IRS treats
convertible virtual currency as **property** (Notice 2014-21), which is the basis of the common
statement that wash sales do not apply to directly held crypto — while **crypto ETFs and
crypto-related equities are securities and the rule applies in full**. Bills to extend §1091 to
digital assets have been introduced repeatedly and none has been enacted as of this date. **This
is a dated, legislative claim: re-check it before relying on it.** Separately, Rev. Proc. 2024-28
ended universal across-wallet basis tracking from 2025-01-01 — see
`../../../fin-crypto/skills/crypto-data-and-execution/SKILL.md` for the data side of a crypto
book, and note that this library does not yet carry a crypto tax-lot skill.

## 8. Scripts and where this sits

`scripts/wash_sales.py` — `apply_wash_sales` (lot matching + disallowance + basis and
holding-period adjustment, with the forward half of the window handled), `in_window`,
`summarise`, `economic_pnl`, `naive_error`, and two seeded demo blotters on one price path:
`make_rebalance_blotter` and `make_harvest_blotter`. numpy + pandas, seed 20260909, under 1 s.

- Which lot the sale hits in the first place — `../tax-lot-matching-and-cost-basis/SKILL.md`.
- Contracts this rule does not reach, and the December 31 mark —
  `../section-1256-and-derivatives-tax/SKILL.md`.
- Reporting an after-tax Sharpe with the assumptions stated —
  `../after-tax-backtesting/SKILL.md`.
- The one-paragraph summary in the US rulebook that this skill expands —
  `../../../fin-core/skills/us-market-rules/SKILL.md` §5.
- Turnover, which is what actually drives the disallowed fraction —
  `../../../fin-core/skills/execution-cost-analysis/SKILL.md`, and
  `../../../fin-core/skills/backtest-validation/scripts/cost_curve.py` for turning a drag into a
  breakeven.
