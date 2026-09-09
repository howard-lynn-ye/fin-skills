---
name: tax-lot-matching-and-cost-basis
description: >-
  The same trades produce four different reported P&Ls depending on which lot you sold, and only
  one of the four methods is a statutory default. TRIGGER - tax lot, tax lots, lot matching, cost
  basis, cost basis method, which shares did I sell, FIFO, LIFO, HIFO, LOFO, tax
  lot optimizer, specific identification, specific share identification, adequate identification,
  average cost basis, average basis, short-term vs long-term capital
  gain split, holding period more than one year, Publication 550, Form 8949, 1099-B basis
  mismatch, "my broker says HIFO", "which lot method should the backtest use", after-tax P&L of a
  blotter. Modelling assumptions for backtests, not tax advice. SKIP for the loss you are not
  allowed to book at all (wash-sale-rules), for futures and index options that never use lots at
  all (section-1256-and-derivatives-tax), for wiring all of it onto a backtest and reporting an
  after-tax Sharpe (after-tax-backtesting), and for A-share stamp duty and dividend tax
  (china-ashare-trading-taxes).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Tax lot matching and cost basis

**A backtest reports the P&L of a position. A tax return reports the P&L of a set of lots, and
which lot you sold is a choice. Until you say which choice, "the strategy made $X" is not a
statement anyone can tie to a tax return.**

These are **modelling assumptions for a backtest, not tax advice.** Every rule below carries the
publication and section it came from and the date this library read it. Tax rules change; a
dated claim that is wrong next year is fixable, an undated one is not. Confirm current law with
a qualified professional before relying on any of it.

Everything marked ✅ Measured comes from `scripts/lot_matching.py` — numpy + pandas, seed
20260909, one four-year single-name blotter, **under 1 s**. Everything marked ✅ source-verified
was read in **IRS Publication 550 (2025)**, *Investment Income and Expenses*, Cat. No. 15093R,
revised 2026-03-05, **read 2026-09-09**; page numbers are the printed page numbers of that
edition.

## 1. ✅ Measured — one blotter, four rules, four answers

Seeded blotter: 72 trades (43 buys, 29 sells) in one name, 2022-01-03 to 2025-10-20, marked at
107.2423. The strategy's **total** P&L, realised plus unrealised, is **+32,595.80** under every
rule.

| rule | realised | short-term | long-term | unrealised | **total** |
|---|---|---|---|---|---|
| **fifo** | 5,218.13 | −5,118.51 | 10,336.65 | 27,377.66 | **32,595.80** |
| **lifo** | 16,871.94 | 16,871.94 | **0.00** | 15,723.86 | **32,595.80** |
| **hifo** | **−752.43** | −4,425.57 | 3,673.14 | 33,348.23 | **32,595.80** |
| **lofo** | 19,341.30 | 15,339.32 | 4,001.98 | 13,254.50 | **32,595.80** |

🚨 **The realised gains differ by 20,093.73 — 61.6% of the entire strategy's P&L.** HIFO reports
a **realised loss** on a strategy that made money. LIFO reports **zero** long-term gain on a
four-year book. Every one of these is arithmetically correct.

✅ Measured — the same four rules at an **assumed** 37% short / 20% long rate (an input the
script prints, never a default it supplies):

| rule | tax on the realised column |
|---|---|
| fifo | 173.48 |
| lifo | 6,242.62 |
| hifo | **−902.83** |
| lofo | 6,475.94 |

**Spread 7,378.78 in tax, on one blotter, in one name, in one year.** This is the number a
pre-tax backtest silently sets to zero.

## 2. ✅ source-verified — what Publication 550 actually says

**FIFO is the default, and it is a default, not a preference.** Pub 550 (2025) p. 62, under
*Stocks and Bonds* → *Identification not possible*: *"If you buy and sell securities at various
times in varying quantities and you cannot adequately identify the shares you sell, the basis of
the securities you sell is the basis of the securities you acquired first."* The same paragraph
closes the average-price door for ordinary stock: *"Except for certain mutual fund shares,
discussed later, you cannot use the average price per share to figure gain or loss on the sale
of the shares."*

**Specific identification is an act, not a setting.** Pub 550 (2025) p. 62, *Adequate
identification* → *Broker holds stock*, verbatim — you make an adequate identification if you:

- *"Tell your broker or other agent the particular stock to be sold or transferred at the time
  of the sale or transfer, and"*
- *"Receive a written confirmation of this from your broker or other agent within a reasonable
  time."*

Two conditions, both contemporaneous with the sale. 🚨 **A spreadsheet that re-picks the lots in
April is not a specific identification of a trade made last August**, and Pub 550 says the
identified stock is what was sold *"even if stock certificates from a different lot are
delivered"* — the identification is the operative act, which is exactly why it has to have
happened.

**"LIFO" and "HIFO" are not statutory methods for securities.** There is no LIFO election for
stock in Pub 550's list. What a broker's *"LIFO"*, *"HIFO"* or *"tax lot optimizer"* setting
does is **stand in for the identification** — it instructs the broker at the time of the sale
and generates the confirmation. That is legitimate *as a specific identification*, and it is
legally a different thing from a method you may choose after the fact. Pub 550's only named
methods, and only for mutual funds (p. 67), are *"Specific share identification"* and
*"First-in first-out (FIFO)"*.

🔑 **The practical test: if the broker confirmations say FIFO and your P&L report says HIFO, the
report is not defensible.** Model whatever your account is actually configured to do, and record
which it was.

## 3. The invariant that makes the whole subject tractable

```
realised(rule) + unrealised(rule)  is IDENTICAL for every rule
realised(rule)                     is not
```

✅ Measured — the script computes `realised + unrealised` under all four rules and reports the
number of distinct values: **1**.

🔑 Lot selection **moves** P&L between this year and a later year, and between the short-term
and the long-term bucket. It does not create or destroy a dollar. A vendor claim that a lot
method "added N bps" is quoting the realised column and hiding the unrealised one; the gain a
HIFO sale did not book is still sitting in the basis of the lots you still own.

What it *is* worth is real and comes from three places, none of which is free money:

1. **Deferral** — tax paid later is worth less. The size of this is the discount rate, not the
   gain.
2. **Character** — long-term versus short-term. In the table above LIFO reports 0.00 long-term
   on a four-year book; FIFO reports 10,336.65. That is a rate difference on real dollars.
3. **Offset** — a realised loss this year can meet a realised gain this year.

And one place it is worth **less** than it looks: `../wash-sale-rules/SKILL.md`, because a
harvested loss followed by a repurchase inside 30 days is deferred rather than banked.

## 4. Holding period: one calendar year, not 365 days

Long-term means held **more than one year**. `is_long_term()` uses `pd.DateOffset(years=1)`, not
a day count: a lot bought 2024-02-29 and sold 2025-02-28 is **short**; sold 2025-03-01 it is
**long**. 🚨 A 365-day implementation gets leap years and month-ends wrong, and a year-end
harvesting rule lives precisely on that boundary — so the error is not random, it lands on the
trades the strategy cares about most.

## 5. Average basis is narrower than every tutorial implies

✅ source-verified, Pub 550 (2025) p. 67, *Average Basis*. You may use it only if the shares are
identical, were acquired at different times and prices, are left with a custodian or agent, and:

- *"They are shares in a mutual fund (or other regulated investment company);"*
- *"They are shares you hold in connection with a DRIP, and all the shares you hold in
  connection with the DRIP are treated as covered securities"*; or
- *"You acquired them after 2011 in connection with a DRIP."*

**Ordinary stock is not on that list.** `average_basis(blotter, share_class="stock")` raises
rather than returning a number, and quotes the restriction in the message. ✅ Measured: the demo
blotter under average basis, if it *were* a fund, realises **5,942.94** — a fifth number, and
one that is simply unavailable for the single-name equity book that produced it.

⚠️ Two details that get lost: the election must be made in writing to the custodian and is
**revocable only within a narrow window** (Pub 550 p. 68, *Revoking the average basis method
election*), and the **holding period still runs FIFO** — Pub 550 p. 68: *"To determine your
holding period, the shares disposed of are considered to be those acquired first."* So average
basis changes the size of the gain and not its short/long character. The script implements it
that way.

## 6. What this skill does not model, and where it goes wrong quietly

| Not modelled here | Where it belongs |
|---|---|
| A loss you are not allowed to book at all | `../wash-sale-rules/SKILL.md` — and it **re-writes the basis of the lots this skill tracks**, so run it *before* reading a realised number |
| Futures and broad-based index options, which have no lots at all | `../section-1256-and-derivatives-tax/SKILL.md` — year-end mark plus 60/40 |
| The rate, the jurisdiction, and reporting an after-tax Sharpe | `../after-tax-backtesting/SKILL.md`, whose guard refuses a number that does not state all three |
| A-share stamp duty and holding-period dividend tax | `../china-ashare-trading-taxes/SKILL.md` |
| Crypto: per-wallet basis is mandatory from 2025-01-01 (Rev. Proc. 2024-28) | ⚠️ not yet a skill in this library. Universal across-wallet basis is dead; do not reuse this module's single-pool model for a multi-wallet crypto book |

🚨 **Short sales are refused, not approximated.** `match_lots` raises if a sale exceeds
inventory. Short positions are IRC §1233, a different regime with its own holding-period rules,
and silently netting them into a long lot stack is how a plausible wrong number gets produced.

⚠️ **One symbol per blotter.** Lots are per security per account. Running two symbols through
one blotter would match a sale of A against a lot of B; the module does not check, so the caller
must split.

⚠️ **The 1099-B is not the answer either.** Brokers report basis only for **covered securities**
and their wash-sale adjustment (box 1g) is computed **within one account and by CUSIP**, so the
same-security purchase in your other account, or in your IRA, is not in it. The identity between
your model and the form is a thing to check, not a thing to assume.

## 7. Scripts and where this sits

`scripts/lot_matching.py` — `match_lots` (fifo / lifo / hifo / lofo), `is_long_term`,
`term_split`, `unrealised`, `compare_rules`, `spread_fraction`, `tax_due` (rates are arguments,
there is no default), `average_basis` (refuses ineligible share classes), and `make_blotter` for
the seeded demo. numpy + pandas, seed 20260909, under 1 s.

- The loss that gets disallowed and re-based — `../wash-sale-rules/SKILL.md`.
- Contracts with no lots: 60/40 and the December 31 mark —
  `../section-1256-and-derivatives-tax/SKILL.md`.
- Attaching all of this to a backtest, with the stated-assumptions guard —
  `../after-tax-backtesting/SKILL.md`.
- The US rulebook this sits inside — settlement, margin, Reg SHO, and the one-paragraph wash
  sale summary this plugin expands — `../../../fin-core/skills/us-market-rules/SKILL.md`.
- Turning a tax drag into a breakeven and a capacity limit, the same way a cost assumption is
  handled — `../../../fin-core/skills/backtest-validation/scripts/cost_curve.py`.
- Where the turnover that drives all of this comes from —
  `../../../fin-core/skills/execution-cost-analysis/SKILL.md`.
