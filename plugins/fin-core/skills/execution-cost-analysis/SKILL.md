---
name: execution-cost-analysis
description: >-
  Measure what your execution actually cost instead of assuming a number - implementation shortfall,
  benchmark choice, impact models, and why a backtest that beat the market loses money live. TRIGGER
  - transaction cost analysis, TCA, implementation shortfall, arrival price, decision price, slippage
  analysis, execution quality, fill quality, did I get a good fill; VWAP or TWAP benchmark, beat
  VWAP, participation rate, POV, percentage of volume, child orders, order slicing; market impact,
  temporary vs permanent impact, square-root law, Almgren-Chriss, price reversion after my order;
  "my strategy works in backtest but loses money live"; "how much size can this strategy take",
  capacity, alpha decay with size. SKIP for assuming a cost inside a backtest, which is
  backtesting-engines and research-integrity-guards, and for broker order types
  (broker-execution-apis).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# Execution cost analysis

**A backtest assumes a cost. TCA measures one.** Most research stacks do the first and never the
second, which is why "works in backtest, loses money live" is the most common complaint in the field
and the least often diagnosed.

This skill is about your own fills. For choosing a cost *assumption* before you have fills, see
`../research-integrity-guards/SKILL.md` §4 and its
`../research-integrity-guards/scripts/cost_curve.py`.

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

🚨 **This library does not carry a calibrated `Y` or verified coefficients**, and a borrowed constant
from a paper on a different market and decade is not a measurement. Fit it on your own fills or
state that you assumed it.

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

## 7. Scripts

`scripts/benchmark_choice.py` — the benchmark table above, and the participation demonstration.
numpy only, fixed seed. The session drift is imposed deterministically rather than sampled, so the
sign of the result does not depend on the draw.
