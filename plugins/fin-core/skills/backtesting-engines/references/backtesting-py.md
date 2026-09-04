# backtesting.py

A small, honest bar loop with pessimistic fill logic — and two things people get wrong about it: the
licence and what `trade_on_close` actually does.

| | |
|---|---|
| pip | `backtesting` · import `backtesting` |
| Version | **0.6.6 (2026-07-22)** |
| GitHub | `kernc/backtesting.py` — 8,931★, 80 open issues, pushed **2026-08-05** ✅ active |
| Licence | 🚨 **AGPL-3.0-or-later** (declared *and* classified) |
| Python | `>=3.9` · pure-Python `py3-none-any` wheel — installs anywhere, no compiler |
| Scope | 🚨 **Single asset. Full stop.** No portfolios, no cross-sectional anything |

## 🚨 AGPL-3.0 — the most-missed fact about this library

Not MIT, not BSD. **AGPL is network copyleft**: if you build a *hosted service* on it — a web app, a
Discord bot, an internal dashboard other people hit over a network — you must offer the complete
corresponding source of your derived work to its users. Running it privately on your own machine is
unaffected; shipping a SaaS on top of it is a licensing decision, not a detail.

Three different copyleft postures live in this one skill: AGPL here, LGPL in
`./nautilus-trader.md`, and Commons Clause (not OSI at all) in `./vectorbt.md`. Check before you
build, not after.

## 🚨 Single asset only

There is no universe, no portfolio, no cross-sectional ranking. One instrument per `Backtest`. If the
strategy compares names against each other, this is the wrong tool — go to `./zipline-reloaded.md`
(Pipeline) or `./vectorbt.md` (columns).

## Run order per bar — read from the source

For each bar `i`: `data._set_length(i+1)` → `broker.next()` (process the order queue) →
`strategy.next()` (documented as *"next tick, a moment before bar close"*).

The order of those last two is the whole design: **orders you place inside `next()` are processed at
the start of the *following* iteration.** That is what makes the default causally sound.

## ✅ Fills — `_process_orders`, verified in source

- **Market orders → the next bar's open.** Right by default, unlike `./vectorbt.md`.
- **Contingent (SL/TP) orders fill within the triggering bar** at the stop price, with gap-through
  handled: `price = max(price, stop_price)` for longs, `min(...)` for shorts. ✅ **A gap straight
  through your stop fills you at the open, not at the stop** — which is what actually happens, and
  is more honest than freqtrade, which fills stops exactly at the stop price even when the low was
  lower.
- ✅ **SL is deliberately prioritized over TP** when a single bar hits both. The source comment says
  it outright: *"Ensure SL orders are processed first"*. Pessimistic, which is correct — you cannot
  know which came first inside a bar.
- **Stop-limit orders are skipped** when the limit would have been hit before the stop triggered
  (*"pessimistically assume limit was hit before the stop"*).
- `commission` is folded into the fill price by `_adjusted_price` — longs marked up, shorts marked
  down. `spread` is a separate constant-bps parameter.

That cluster of choices is why this library punches above its size: **where the bar is ambiguous, it
resolves against you.**

## 🚨 `trade_on_close=True` does not mean "fill at this bar's close"

✅ From the source: with `trade_on_close=True` the fill price is **`data.Close[-2]`**, indexed at
`self._i - 1` — *the previous bar's close* as seen from inside `broker.next()`.

Because `broker.next()` runs one iteration after the `next()` that queued the order, that price is
the close of **the bar whose `next()` generated the signal**. Two consequences, and people usually
notice only one:

1. It is **not** the close of the bar currently being processed. Reading it that way and then
   reasoning about "the next bar" puts every subsequent timing argument off by one.
2. It is effectively a **market-on-close fill at the signal bar's close** — legitimate only if you
   can actually trade that close (an MOC order, an auction). If your live path sends a market order
   *after* the close prints, you will not get that price. Leave it `False` unless you can name the
   order type that produces the fill.

## 🚨 `Strategy.I` computes over the FULL series, then slices

`init()` runs the indicator across the entire dataset once; `next()` sees a slice of the result. For
a causal indicator (SMA, EMA, RSI) that is just an optimisation and is fine.

For a **non-causal** one it is a silent, total leak:

- a centred rolling window
- `scipy.signal.filtfilt` or any zero-phase filter
- `.shift(-1)`, `.rolling(...).mean().shift(-n)`, forward-looking `argmax`
- a model **fitted on the whole series** and then evaluated pointwise

**The framework cannot detect any of these.** There is no warning, no assertion, no diagnostic — the
equity curve simply comes out beautiful. Only freqtrade ships a detector for this class of bug
(`lookahead-analysis`); the portable defence is
`../../research-integrity-guards/references/leakage-checklist.md`.

⚠️ **Related residual risk:** the loop slices to `i+1` *before* calling `next()`, so
`self.data.High[-1]`, `Low[-1]` and `Close[-1]` are all readable inside `next()`. Market-order
timing protects your *fills*; it does not stop you reading the current bar's high in a *signal*.

```python
from backtesting import Backtest, Strategy

bt = Backtest(
    df, MyStrategy,
    cash=10_000,
    commission=0.002,       # folded into the fill price, not charged separately
    spread=0.0,             # constant bps; the only other cost model that exists
    trade_on_close=False,   # 🚨 keep False: True fills at Close[-2], an MOC-style fill
    exclusive_orders=False,
    finalize_trades=True,   # close open trades at the end, else the last trade is unrealised
)
stats = bt.run()
# 🚨 bt.optimize(...) reports the grid MAXIMUM. That number is not an expectation —
#    the grid size is your trial count. See ../../backtest-validation/SKILL.md
```

## What it does not model

Multiple assets · volume or liquidity limits · market impact · partial fills · corporate actions ·
borrow · funding · margin beyond a simple `margin` ratio and an `_OutOfMoneyError`. An order for a
million shares of a stock that traded 900 shares that bar fills instantly at one price. For a real
liquidity cap see `./zipline-reloaded.md` (2.5% of bar volume); for order-book realism,
`./nautilus-trader.md`.

## Where it fits

**The right tool for a single-instrument TA strategy you want an honest first read on**, especially
when the alternative is a hand-rolled pandas loop. It is the cheapest way to get next-open fills and
pessimistic stop handling without learning an event-driven framework. Just do not let
`optimize()`'s best row become the headline number — see `./_engine-matrix.md` for what to promote
the survivor into.
