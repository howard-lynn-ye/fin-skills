---
name: after-tax-backtesting
description: >-
  Attach lot matching, wash sales and section 1256 to an existing backtest and report after-tax
  Sharpe beside pre-tax - and refuse to report one that does not state its rate, jurisdiction and
  lot method. TRIGGER - after-tax return, after tax Sharpe, tax drag, tax-aware backtest, tax
  alpha, tax-managed strategy, turnover penalty, "what does this strategy return after tax", tax
  cost of rebalancing, taxable account backtest, tax-efficient turnover, pre-tax vs post-tax
  performance, capital gains netting, loss carryforward, "should I hold this in an IRA".
  Modelling assumptions for backtests, not tax advice. SKIP for the lot rules themselves
  (tax-lot-matching-and-cost-basis), for the wash-sale mechanics (wash-sale-rules), for futures
  and index options (section-1256-and-derivatives-tax), for A-share taxes
  (china-ashare-trading-taxes), and for transaction costs, which are a different drag entirely
  (execution-cost-analysis).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# After-tax backtesting

**An after-tax Sharpe without a rate, a jurisdiction and a lot method is not comparable to
anything — including the same strategy next quarter. So `report()` refuses to print one.**

These are **modelling assumptions for a backtest, not tax advice.** Confirm current law with a
qualified professional before relying on any of it.

This is the **assembly** skill. It owns nothing of its own: it wires
`../tax-lot-matching-and-cost-basis/SKILL.md`, `../wash-sale-rules/SKILL.md` and
`../section-1256-and-derivatives-tax/SKILL.md` onto a return series and reports both Sharpes.
Read those three for the rules and their sources; read this one for what happens when they meet
a backtest.

✅ Measured comes from `scripts/after_tax.py` — numpy + pandas, seed 20260909, one 4-year path,
**3.5 s**.

## 1. The guard, and why refusing is the feature

```python
TaxAssumptions(jurisdiction=..., short_rate=..., long_rate=..., lot_method=...)
```

All four are **required positional fields with no defaults**, and `require_assumptions()` raises
`MissingAssumptions` on anything less. ✅ Measured — the demo's two refusals, verbatim:

```
guard: after-tax results are not comparable without assumptions. Pass a TaxAssumptions
       with ['jurisdiction', 'short_rate', 'long_rate', 'lot_method'].
guard: after-tax result is missing ['long_rate', 'lot_method']. An after-tax Sharpe without
       a rate, a jurisdiction and a lot method is not comparable to anything.
```

🔑 **There is no default rate in this library and there never will be.** A rate depends on the
jurisdiction, the entity, the bracket and the year. A default turns an after-tax figure into an
unlabelled opinion — and the number that comes out is *lower* than the pre-tax one, which makes
it look conservative and therefore trustworthy. It is neither.

`report()` prints the assumption line above every result, plus the timing convention and any
`notes`, so two after-tax numbers can be compared by comparing their headers first.

## 2. ✅ Measured — the after-tax turnover penalty

One seeded 4-year path. **1,000 shares held constantly.** The only difference between rows is how
often the position is sold and bought straight back **at the same price with zero transaction
cost** — so the gross return series is identical by construction and *every* difference below is
tax. Assumptions: US federal individual, short 37%, long 20%, FIFO, wash sales on, losses carried
forward, no state tax, no NIIT.

| hold (days) | round trips/yr | gross Sharpe | **after-tax Sharpe** | retained | tax paid | wash-sale loss deferred |
|---|---|---|---|---|---|---|
| 1 | 252.0 | 0.488 | **0.312** | **63.9%** | 17,108 | −16,736,156 |
| 5 | 50.4 | 0.488 | 0.320 | 65.6% | 16,270 | −3,274,567 |
| 21 | 12.0 | 0.488 | 0.317 | 65.0% | 16,568 | −777,669 |
| 63 | 4.0 | 0.488 | 0.399 | 81.8% | 8,649 | −126,162 |
| 252 | 1.0 | 0.488 | 0.480 | 98.4% | 671 | −17,768 |
| **never** | 0.0 | 0.488 | **0.488** | **100.0%** | **0** | 0 |

✅ Measured: gross Sharpe has **exactly one distinct value** across all six rows, and pre-tax P&L
is **36,722.53** on every one of them.

🚨 **Churning daily costs 36.1% of the after-tax Sharpe of doing nothing, on identical
exposure.** That is not a commission — commissions are zero here by construction. It is the tax
code repricing a strategy whose gross behaviour never changed.

🔑 **The penalty saturates.** Monthly, weekly and daily churn all pay **16,270–17,108** of tax.
Once the basis is reset at every new high there is nothing left to accelerate, so the marginal
tax cost of trading more often than monthly is roughly zero here — the *first* increment of
turnover is what costs, and turnover beyond it is free of tax cost (though not of anything else).

## 3. 🚨 The asymmetry that makes the penalty bigger than it looks

Selling and rebuying immediately does **not** harvest losses. It only realises gains:

```
a round trip through a GAIN  ->  realises it, early, mostly at the short-term rate
a round trip through a LOSS  ->  is a wash sale, so the loss is DEFERRED into the new basis
```

So the basis becomes the **running maximum** of the price, realised gains are the sum of the
new-high increments, and the losses simply never arrive. ✅ Measured:

| | |
|---|---|
| daily churn, with wash sales modelled | tax **17,108.20**, after-tax Sharpe **0.312** |
| the same book with wash sales ignored | tax **0.00**, after-tax Sharpe **0.488** |

🚨 **A backtest that ignores wash sales reports this strategy as completely tax-free**, and its
after-tax Sharpe equals its pre-tax Sharpe to three decimals. Not "a little optimistic" — the
entire tax bill vanishes, because the disallowed losses net against the gains and produce a paper
loss year.

⚠️ **Read the `loss deferred` column as a running total, not as economic loss.** At daily churn
it reaches −16,736,156 on a book worth about 137,000. That is the ratchet documented in
`../wash-sale-rules/SKILL.md` §5: each disallowed dollar re-enters the basis and is disallowed
again the next day, so the column counts the same dollars hundreds of times. It is the right
number for the mechanism and the wrong number to put in a report.

## 4. ✅ Measured — the same exposure in a §1256 wrapper

Buy and hold, held instead as a §1256 contract (year-end marks, 60/40, 26.8% blended):

| year | recognised | tax | as stock |
|---|---|---|---|
| 2022 | 1,974.63 | 529.20 | 0.00 / 0.00 |
| 2023 | −15,537.93 | 0.00 | 0.00 / 0.00 |
| 2024 | 13,430.41 | 0.00 | 0.00 / 0.00 |

**Taxable income on a position nobody sold swings from −15,537.93 to +13,430.41** and costs
529.20 in total; the identical exposure held as stock recognises **0.00 every year and pays
0.00**. The 2023 loss carries forward and absorbs the 2024 gain, which is why 2024's tax is zero
— that is `net_and_tax()` doing character netting and carryforward, not a rounding artefact.

## 5. What the engine actually does, so you can disagree with it

| Step | Implementation | Where the rule lives |
|---|---|---|
| lot matching | `wash_sales.apply_wash_sales(rule=...)`, `fifo` or `lifo` | `../tax-lot-matching-and-cost-basis/SKILL.md` |
| wash sales | same call: disallow, re-base, carry the holding period | `../wash-sale-rules/SKILL.md` |
| short vs long | `lot_matching.is_long_term` — more than one **calendar** year | §4 of that skill |
| netting | `net_and_tax`: net within character, cross-net, carry the remainder forward **with its character**; a net loss produces **no refund** | US netting order |
| ordinary offset | `ordinary_offset_per_year`, **default 0.0** — set it to 3000 yourself if that is your jurisdiction | your choice, stated |
| §1256 | `section_1256.recognise_1256` on the year-end marks | `../section-1256-and-derivatives-tax/SKILL.md` |
| after-tax equity | `gross_equity − cumulative_tax`, tax debited on the **last trading day of each year** | a convention, printed by `report()` |

⚠️ **Every one of those rows is a modelling choice**, and the honest way to use this module is to
change the one you disagree with and re-run, not to accept the number. In particular:

- **Tax is debited at the year end from outside the book.** Real tax is paid in quarterly
  estimates and a filing months later. Moving the debit changes the after-tax *path* and hence
  the Sharpe, though not the total.
- **The equity curve is not rebalanced to pay the tax.** No shares are sold to raise cash, so the
  gross exposure stays exactly constant, which is what makes the gross Sharpe identical across
  rows and the comparison clean. A real account would have to fund it.
- **Sharpe uses a zero risk-free rate**, and `sharpe()` says so in its docstring. See
  `../../../fin-libraries/skills/lib-quantstats/scripts/rf_convention.py` for how much that one
  choice moves a Sharpe on its own.
- **If the tax bill exceeds the book, `after_tax_sharpe` is `nan`**, not a number. A Sharpe
  computed across a zero crossing is meaningless, and the result carries `solvent=False`.

🚨 **Not modelled, and each one is material:** state and local tax, the 3.8% net investment income
tax, the §475(f) mark-to-market election (which removes wash sales entirely and makes losses
ordinary), qualified-dividend versus ordinary treatment, foreign withholding, the §1256 loss
carryback election, straddles, constructive sales, and any account wrapper (IRA, 401(k), ISA,
SIPP, PEA) — inside which most of this file is irrelevant and the answer is simply the pre-tax
number.

## 6. Reporting it without misleading anyone

An after-tax backtest is **hypothetical performance**, and in the US that engages the Investment
Advisers Act Marketing Rule — see `../../../fin-core/skills/us-market-rules/SKILL.md` §6 for what
has to accompany it. `report()` labels the output hypothetical and prints the assumptions; it
does not print the trial count, the universe or the period, and
`../../../fin-core/skills/research-integrity-guards/scripts/result_manifest.py` is where those
belong.

🔑 **Report the pair, never the after-tax number alone.** A tax assumption is one more researcher
degree of freedom: a friendly rate and a friendly lot method can move an after-tax Sharpe more
than most signal changes, and unlike a signal change it leaves no trace in the equity curve. The
same discipline as a cost curve applies — see
`../../../fin-core/skills/backtest-validation/scripts/cost_curve.py`, and log the tax variant you
tried in the trial ledger at `../../../fin-core/skills/backtest-validation/SKILL.md` exactly as
you would log a parameter.

## 7. Scripts and where this sits

`scripts/after_tax.py` — `TaxAssumptions` (four required fields, no default rates),
`require_assumptions` / `MissingAssumptions` (the guard), `net_and_tax` (character netting and
carryforward), `realised_by_year`, `churn_blotter`, `after_tax_backtest`, `turnover_study`,
`report`, `sharpe`, and `demo_prices`. It imports the other three skills' modules through the
dual-mode idiom, so it works both as `python after_tax.py` and as
`fin_skills.tax_accounting.after_tax`. numpy + pandas, seed 20260909, 3.5 s.

- The three rule skills it assembles — `../tax-lot-matching-and-cost-basis/SKILL.md`,
  `../wash-sale-rules/SKILL.md`, `../section-1256-and-derivatives-tax/SKILL.md`.
- The same turnover, priced as execution cost rather than tax —
  `../../../fin-core/skills/execution-cost-analysis/SKILL.md` and
  `../../../fin-core/skills/backtest-validation/scripts/cost_curve.py`. **Add them; they are
  different drags on the same trades.**
- Presenting hypothetical performance — `../../../fin-core/skills/us-market-rules/SKILL.md` §6.
- Logging the tax variant as a trial — `../../../fin-core/skills/backtest-validation/SKILL.md`.
- The A-share equivalent, where the turnover penalty is written into the tax code itself —
  `../china-ashare-trading-taxes/SKILL.md`.
