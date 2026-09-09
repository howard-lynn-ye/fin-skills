---
name: execution-cost-analysis
description: >-
  Measure what your execution actually cost instead of assuming a number - implementation
  shortfall, benchmark choice, impact models, and the gap between the cost you assumed and the
  cost you paid. TRIGGER - transaction cost analysis, TCA, implementation shortfall, arrival
  price, decision price, slippage analysis, execution quality, fill quality, did I get a good
  fill; VWAP or TWAP benchmark, beat VWAP, participation rate, POV, percentage of volume, child
  orders, order slicing; market impact, temporary vs permanent impact, square-root law,
  Almgren-Chriss, price reversion after my order; "how much size can this strategy take",
  capacity, alpha decay with size; is my cost assumption realistic, "is 2 bps plausible", what
  book a stated cost supports. SKIP for a slippage assumption inside a backtest and for "works
  in backtest, loses live" with no measured fills (backtesting-engines), for whether the edge
  survives it (backtest-validation), and for broker order types (broker-execution-apis).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Execution cost analysis

**A backtest assumes a cost. TCA measures one.** Most research stacks do the first and never the
second, which is why "works in backtest, loses money live" is the most common complaint in the field
and the least often diagnosed.

This skill is about your own fills. Before you have any, a cost *assumption* is still checkable
two ways: `../backtest-validation/scripts/cost_curve.py` asks whether the edge survives it, and
§7 here asks whether the assumption was ever available at the size you claim to trade.
`../research-integrity-guards/SKILL.md` §4 is the checklist both sit inside.

## 1. 🚨 A perfect VWAP score is compatible with any amount of loss

✅ Measured with `scripts/benchmark_choice.py` — one 100,000-share buy through a session that drifts
**+120.7 bps against the buyer**:

| Schedule | vs decision price | vs interval VWAP | vs close |
|---|---|---|---|
| TWAP | +26.1 | −6.5 | −93.5 |
| **VWAP** | **+32.6** | **+0.0** | −87.1 |
| Opportunistic | −10.5 | −42.9 | −129.7 |
| Front-loaded (58m) | **−17.1** | −49.5 | −136.1 |

*(positive = cost; negative = beat that benchmark)*

🚨 **The VWAP schedule scores exactly 0.0 bps against VWAP — textbook-perfect — while costing the
fund 32.6 bps against the price at which the decision was made.** It matched the market's own
schedule through the whole adverse move, so it ate the move in full and the metric reported nothing.

🔑 **VWAP measures whether you looked like everyone else. It cannot tell you whether trading at that
pace was a good idea.** Finishing in 58 minutes instead was **49.6 bps better for the fund**.

⚠️ **What this does NOT show:** on this path both benchmarks pick the same winner, because the drift
is monotone and trading early wins on every measure. The benchmarks diverge on non-monotone paths.
The claim here is narrower and stronger — **a zero VWAP score carries no information about
shortfall**, which the VWAP row demonstrates by itself.

## 2. 🚨 VWAP is not exogenous — your prints are inside it

✅ Same script. Execution held **completely fixed**; only its size relative to the session changes:

| Participation | Your avg fill | VWAP incl. your prints | Measured cost |
|---|---|---|---|
| 1% | 99.8295 | 100.3208 | −49.0 |
| 5% | 99.8295 | 100.3009 | −47.0 |
| 20% | 99.8295 | 100.2265 | −39.6 |
| 50% | 99.8295 | 100.0776 | −24.8 |

**The fills never change. Only the benchmark moves toward them.**

🚨 **And if the schedule is proportional to volume, the measured cost is identically zero at every
participation** — `VWAP(base + own) = VWAP(base)` exactly. That is an algebraic identity, not an
achievement, and it is the trivial way to score well on this metric.

**A desk measured on VWAP is being paid to match the market, not to beat it.** Whether that is what
you want is a decision, not a default.

## 3. Implementation shortfall is the honest number

Perold's decomposition, against the price **when the decision was made**:

```
shortfall = (paper return of the ideal portfolio) − (actual return)
```

Decompose it, because the pieces have different owners:

| Component | Measured as | Whose problem |
|---|---|---|
| **Delay / decision cost** | decision price → order arrival at the desk | the research-to-trading handoff |
| **Market impact** | arrival price → average fill | the execution algorithm |
| **Timing** | drift over the execution window | the schedule |
| **Opportunity cost** | unfilled shares × (final price − decision price) | usually the biggest and usually ignored |
| **Fees, commissions, taxes** | explicit | accounting |

🚨 **Opportunity cost is the one that gets dropped**, because unfilled shares leave no fill record to
analyse. A strategy that "had low slippage" by filling 60% of its intended size has not established
anything — it has selected the easy 60%. **Report fill rate next to every cost number.**

## 4. Which benchmark answers which question

| Benchmark | Answers | Blind to |
|---|---|---|
| **Decision / arrival price** | Did the whole process cost the fund money? | nothing — this is the one to quote |
| Interval VWAP | Did we match the market's schedule? | whether trading then was wise; and it includes our own prints |
| Interval TWAP | Did we spread evenly? | volume, entirely |
| Close | Did we beat the mark? | ⚠️ **gameable and often the mark the P&L is struck at** — a conflict, not a benchmark |
| Previous close | — | 🚨 not a benchmark; it is a return, and it contains overnight news |

**Choose before the trade.** Choosing after seeing the fills is the same p-hacking as choosing a
backtest window after seeing the equity curve — and it counts as a trial. Record it in
`../backtest-validation/scripts/trial_ledger.py`.

## 5. Impact models and capacity

Two components with different lifetimes:

- **Temporary impact** — the price you push through to get filled; it decays after you stop.
- **Permanent impact** — the part that stays, because your trading told the market something.

**The reversion test separates them, and it is the one diagnostic worth running:** measure the price
`T` minutes after your last fill. If most of the move reverts, you paid temporary impact and could
trade slower. **If it does not revert, you moved the market permanently, and size is the problem.**

⚠️ The **square-root law** — impact ≈ `Y · σ · sqrt(Q / V)` — is the standard functional form, and
the empirical constant `Y` is order-1 but venue- and regime-dependent. ⚠️ **Almgren-Chriss** frames
the schedule as a trade-off between impact and timing risk under a risk-aversion parameter.

🚨 **A borrowed constant from a paper on a different market and decade is not a measurement.** §7
carries Almgren et al.'s published coefficients and checks them against that paper's own worked
example — a transcription, not a calibration of *your* market. Fit it on your own fills or state
that you assumed it.

**Capacity falls out of this.** If impact scales as `sqrt(Q)`, then alpha per share decays as size
grows, and the strategy has a size at which it stops paying. A backtest run at 100 shares says
nothing about that number.

## 6. What to record on every parent order

Without these fields you cannot compute anything above afterwards:

`decision_ts` · `decision_px` · `arrival_ts` · `arrival_px` · `side` · `target_qty` · `filled_qty` ·
`avg_fill_px` · `venue` per child · `fees` · `interval_vwap` · `interval_volume` ·
`px_at_+5m/+30m/+close` after the last fill

🔑 **`decision_ts` is the field everyone omits and the only one that makes shortfall computable.**
If the earliest timestamp you keep is when the order hit the broker, you have permanently lost the
delay component — and that is the component the research process controls.

## 7. 🚨 A cost assumption is a claim about order size, and nothing checks it

`../backtest-validation/scripts/cost_curve.py` asks whether the edge **survives** the cost you
stated. Nothing there asks whether that cost was ever **available** to you, and the two questions
come apart completely: in this repo's own defect benchmark (`benchmarks/RESULTS.md`) a strategy
whose breakeven is **45.9 bps** survives an assumption of **2.0 bps** without a murmur, and the
2.0 bps is still a fiction. `scripts/cost_plausibility.py` asks the second question, in four lines
of arithmetic anyone can redo by hand:

```
order per name = turnover x book / names traded
participation  = order per name / that name's ADV        (Almgren's X/V)
implied cost   = an impact model evaluated at that participation
plausible      = stated cost >= implied cost
```

### The model — and it is not the square root

✅ **Almgren, Thum, Hauptmann & Li, "Direct Estimation of Equity Market Impact", 10 May 2005** —
Citigroup US equity desks, Dec 2001–Jun 2003, 29,509 S&P 500 orders after filtering. Summary
equations, p.21:

```
I = gamma * sigma * (X/V) * (Theta/V)^(1/4)          permanent price move
J = I/2 + eta * sigma * (X/(V*T))^(3/5)              what the order actually costs

gamma = 0.314 +- 0.041 (t = 7.7)     eta = 0.142 +- 0.0062 (t = 23)
```

sigma is daily volatility, `V` average daily volume, `Theta` shares outstanding (so `Theta/V` is
the days it takes to turn the float), `T` the execution time as a fraction of a day.

🚨 **The temporary exponent is 3/5, not 1/2.** The square-root law is the version everyone quotes;
this paper rejects `beta = 1/2` at the 95% level and fits **0.600 ± 0.038**. ✅ The demo prints
both: at 0.10% of ADV the square root is **1.78×** the fitted number, at 1% **1.34×**, at 10%
**1.09×**. It moves the estimate and it never rescues an assumption that is 10× out.

✅ **Checked against the paper's own Table 3** — buying 10% of a day's volume in two large caps.
The demo prints this comparison every run:

| | I bps | paper | J, T=0.1 | paper | J, T=0.2 | paper | J, T=0.5 | paper |
|---|---|---|---|---|---|---|---|---|
| IBM | 19.85 | 20 | 32.22 | 32 | 24.63 | 25 | 18.41 | 18 |
| DRI | 21.67 | 22 | 42.93 | 43 | 32.01 | 32 | 23.05 | 23 |

Largest disagreement **0.41 bps** against a table printed in whole bps; normalised permanent impact
comes out 0.126 and 0.096 against their printed 0.126 and 0.096.

### The worked example

✅ Same script, seed 20260909 — a **25,000,000 USD** book turning over **8.93%** one way per day
across **6.5** names whose median ADV is **14,075,248 USD**, at **2.20%** daily volatility:

| | |
|---|---|
| order per name | 343,372 USD |
| **participation** | **2.44% of ADV per name per day** |
| permanent impact | 6.70 bps (half of it is paid) |
| temporary impact | 3.37 bps |
| **implied cost** | **6.72 bps** |
| stated cost | 2.00 bps |
| 2.00 bps becomes plausible at a book of | 5,168,782 USD — 0.21× the stated book |

🔑 **The implied number is a floor, not an estimate.** It is impact alone: no spread, no
commission, no borrow, no taxes. A stated cost below it is not optimistic, it is unavailable.

✅ Capacity is the same claim re-asked at every size, and it is the useful output even when the
verdict is `ok`:

| book USD | order/name | participation | impact bps | verdict at 2 bps |
|---|---|---|---|---|
| 1,000,000 | 13,735 | 0.10% | 0.62 | ok |
| 5,000,000 | 68,674 | 0.49% | 1.95 | ok |
| 25,000,000 | 343,372 | 2.44% | 6.72 | impact 6.7 bps |
| 100,000,000 | 1,373,488 | 9.76% | 21.13 | impact 21.1 bps |
| 500,000,000 | 6,867,439 | 48.79% | 87.32 | impact 87.3 bps + 49% of ADV |
| 2,000,000,000 | 27,469,754 | 195.16% | 314.70 | impact 314.7 bps + 195% of ADV |

🚨 **Past ~10% of ADV the model is not being extrapolated, it is being abandoned** — and a
backtest will happily print a Sharpe for the last row. Almgren et al. excluded orders below 0.25%
of ADV and declined to model beyond "a few percent"; the script warns when you leave that band
instead of quietly returning a number.

⚠️ **What this cannot do.** It is one desk's US large-cap fit from 2002, so treat the level as
indicative and the *direction* as reliable. Impact is **linear in sigma**, so a wrong volatility
scales the answer one for one — pass your own, do not take the 2%/day stand-in. `Theta/V` defaults
to 250 (their Table 3: IBM 263, DRI 87) and enters at the 1/4 power, so it barely moves anything.
The execution horizon defaults to a full session, the cheapest case, which makes the implied cost
a lower bound in that direction too.

🔑 **The check survives bad calibration.** The implied cost is linear in `gamma` and `eta`, so
halving both still leaves 6.72/2 = 3.36 bps against a stated 2.00 — the verdict does not move.
That is the point: it rejects numbers that were never available long before it can price anything
accurately.

## 8. Scripts

`scripts/benchmark_choice.py` — the benchmark table above, and the participation demonstration.
numpy only, fixed seed. The session drift is imposed deterministically rather than sampled, so the
sign of the result does not depend on the draw.

`scripts/cost_plausibility.py` — §7. Reproduces Almgren et al.'s Table 3, then prices a stated
cost against the participation the book implies. numpy/pandas, fixed seed. Also `fin_skills.api`'s
`cost_plausibility` guard, which is what catches `cost_too_low` in `benchmarks/leak_bench.py`.
