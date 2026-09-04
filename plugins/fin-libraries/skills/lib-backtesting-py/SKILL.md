---
name: lib-backtesting-py
description: >-
  Single-asset bar-loop backtester with honest next-open fills, an AGPL-3.0 licence, and an
  indicator API that computes over the entire series before slicing. TRIGGER - from backtesting
  import Backtest, Strategy; pip install backtesting, bt = Backtest(df, MyStrategy), bt.run(),
  bt.optimize(), self.I(), self.buy(), self.sell(), self.data.Close, trade_on_close,
  exclusive_orders, finalize_trades, commission, spread, backtesting.lib crossover,
  _OutOfMoneyError, "kernc"; wanting a portfolio, a universe or a second instrument inside it.
  Memory is stale here: it is alive at 0.6.6 (2026-07-22), it is AGPL-3.0-or-later rather than
  MIT, and trade_on_close fills at data.Close[-2] rather than the current bar's close. SKIP for
  multi-asset or cross-sectional work and for engine choice generally (backtesting-engines). SKIP
  when the question is WHICH library to choose rather than how to use this one - that belongs to
  the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# backtesting.py

A small, honest bar loop with pessimistic fill logic — and two things people reliably get wrong: the
licence, and what `trade_on_close` actually does.

| | |
|---|---|
| pip / import | `pip install backtesting` · `import backtesting` |
| Version | **0.6.6 (2026-07-22)** · `>=3.9` · pure-Python `py3-none-any` wheel, no compiler |
| Licence | 🚨 **AGPL-3.0-or-later** (declared *and* classified) |
| Status | ✅ Active. 8,931★ `kernc/backtesting.py`, pushed 2026-08-05. 🚨 **Single asset. Full stop** |

## The trap that costs you money

🚨 **`Strategy.I` computes the indicator over the FULL series in `init()`, then slices it in
`next()`.** For a causal indicator (SMA, EMA, RSI) that is just an optimisation and is fine. For a
non-causal one it is a **silent, total leak**:

- a centred rolling window
- `scipy.signal.filtfilt` or any zero-phase filter
- `.shift(-1)`, `.rolling(...).mean().shift(-n)`, a forward-looking `argmax`
- a model **fitted on the whole series** and then evaluated pointwise

**The framework cannot detect any of these.** No warning, no assertion, no diagnostic — the equity curve simply comes out beautiful.

⚠️ Related residual risk: the loop slices to `i+1` *before* calling `next()`, so `self.data.High[-1]`, `Low[-1]` and `Close[-1]` are all readable inside `next()`. Market-order timing protects your *fills*; it does not stop you reading the current bar's high in a *signal*.

## 🚨 AGPL-3.0 — the most-missed fact about this library

Not MIT, not BSD. **AGPL is network copyleft**: if you build a *hosted service* on it — a web app, a Discord bot, an internal dashboard other people hit over a network — you must offer the complete corresponding source of your derived work to its users. Running it privately on your own machine is unaffected; shipping a SaaS on top of it is a licensing decision, not a detail.

## Run order per bar, and why the defaults are sound

For each bar `i`: `data._set_length(i+1)` → `broker.next()` (process the order queue) →
`strategy.next()` (documented as *"next tick, a moment before bar close"*). **Orders placed inside
`next()` are processed at the start of the following iteration** — that is what makes it causal.

✅ Fills, verified in `_process_orders`:

- **Market orders → the next bar's open.** Right by default.
- **Contingent (SL/TP) orders fill within the triggering bar** at the stop price, with gap-through handled: `price = max(price, stop_price)` for longs, `min(...)` for shorts. **A gap straight through your stop fills you at the open, not at the stop** — which is what actually happens.
- ✅ **SL is deliberately prioritized over TP** when one bar hits both; the source says *"Ensure SL orders are processed first"*. Pessimistic, which is correct.
- **Stop-limit orders are skipped** when the limit would have been hit before the stop triggered (*"pessimistically assume limit was hit before the stop"*).
- `commission` is folded into the fill price by `_adjusted_price` — longs marked up, shorts marked down. `spread` is a separate constant-bps parameter.

**Where the bar is ambiguous, it resolves against you.** That is why this library punches above its size.

## 🚨 `trade_on_close=True` does not mean "fill at this bar's close"

✅ Verified in `backtesting/backtesting.py`. The fill price is `data.Close[-2]`:

```python
prev_close = data.Close[-2]
price = prev_close if self._trade_on_close and not order.is_contingent else open
```

and the fill is timestamped at `self._i - 1`. Meanwhile the `Backtest` docstring says market orders
fill *"with respect to the current bar's closing price"*. **Both are true in different frames**:

- Inside `broker.next()` at iteration `i`, data is already sliced to `i+1`, so `Close[-1]` is bar
  `i` and `Close[-2]` is bar `i-1`.
- The order was queued during `strategy.next()` of iteration `i-1`.
- So the fill lands on **the close of the bar the strategy was looking at when it placed the
  order** — one bar back from the broker's frame, zero bars back from the strategy's.

Consequences: it is effectively a **market-on-close fill at the signal bar's close**, legitimate only
if you can actually trade that close (an MOC order, an auction). ✅ It applies **only to
non-contingent market orders** — SL/TP ignore it. **Leave it `False` unless you can name the live
order type that produces the fill.**

## What it does not model

Multiple assets · volume or liquidity limits · market impact · partial fills · corporate actions ·
borrow · funding · margin beyond a simple `margin` ratio and an `_OutOfMoneyError`. An order for a
million shares of a stock that traded 900 shares that bar fills instantly at one price.

## Minimal correct call

```python
from backtesting import Backtest, Strategy

bt = Backtest(
    df, MyStrategy,
    cash=10_000,
    commission=0.002,       # folded into the fill price, not charged separately
    spread=0.0,             # constant bps; the only other cost model that exists
    trade_on_close=False,   # 🚨 keep False: True fills at Close[-2], an MOC-style fill
    exclusive_orders=False,
    finalize_trades=True,   # else the last open trade stays unrealised
)
stats = bt.run()
# 🚨 bt.optimize(...) reports the grid MAXIMUM. That is not an expectation —
#    the grid size is your trial count.
```

## See also

- `../../../fin-core/skills/backtesting-engines/SKILL.md` — engine choice and timing table
- `../../../fin-core/skills/backtesting-engines/references/backtesting-py.md` — the source card
- `../../../fin-core/skills/backtesting-engines/references/execution-realism.md` — fill realism, ranked
- `../../../fin-core/skills/research-integrity-guards/references/leakage-checklist.md` — the portable defence against `Strategy.I` leaks

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`backtesting-engines`** (`../../../fin-core/skills/backtesting-engines/SKILL.md`).

