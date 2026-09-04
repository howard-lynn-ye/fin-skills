---
name: lib-vectorbt
description: >-
  Vectorized Numba/Rust backtester built for parameter sweeps, whose from_signals fills at the
  signal's own bar close by default. TRIGGER - import vectorbt as vbt, pip install vectorbt,
  vbt.Portfolio.from_signals, from_orders, from_holding, ma_crossed_above, vbt.MA.run,
  vbt.IndicatorFactory, .vbt.signals.fshift, price=np.inf, reject_prob, allow_partial,
  stop_conflict_mode, cash_sharing, FlexArray, vectorbt[rust], VectorBT PRO, "this PRO example
  does not work"; a 10,000-combination grid, "my backtest looks too good", an equity curve that
  dies live. Memory is stale here: v1.0 (2026-04-22) was a breaking rewrite with an optional Rust
  engine, 1.1.0 shipped 2026-07-05, and the licence is Apache-2.0 plus Commons Clause - not OSI
  open source. SKIP for choosing among engines (backtesting-engines) and for judging a finished
  result (backtest-validation). SKIP when the question is WHICH library to choose rather than how
  to use this one - that belongs to the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# vectorbt

The fastest thing in its category for parameter sweeps — and the owner of the single most important
footgun in the domain, which is a **default, not a bug**.

| | |
|---|---|
| pip / import | `pip install vectorbt` · `import vectorbt as vbt` |
| Version | **1.1.0 (2026-07-05)**; **1.0.0 landed 2026-04-22**. `requires_python >=3.11,<3.15` |
| Licence | 🚨 **Apache-2.0 + Commons Clause** (read `LICENSE.md`) — **not OSI open source** |
| Status | ✅ Revived, not frozen. 8,978★ `polakowo/vectorbt`, pushed 2026-08-02 |

## The trap that costs you money

🚨 **`from_signals` fills at the signal's own bar close.** `Order.price` defaults to `np.inf`, and
vectorbt's own docstring in `vectorbt/portfolio/enums.py` says:

> *"If `-np.inf`, replaced by the current open (if available) or the previous close (≈ the current
> open in crypto). If `np.inf`, replaced by the current close."*

`from_orders`/`from_signals` resolve `if price is None: price = np.inf`. So
`vbt.Portfolio.from_signals(close, entries, exits)` fills at `close[t]` — the signal's own bar.
Combined with the equally default idiom `fast_ma.ma_crossed_above(slow_ma)` — also computed on
`close[t]` — **this is textbook same-bar execution.** The close is not knowable until the bar is
over. Free money in the backtest, nothing in live.

**Every vectorbt tutorial showing a beautiful equity curve without shifting is showing a biased
result.** The library will never warn you. Fixes, in order of preference:

```python
# 1. Fill at the next open, with signals shifted so the open is AFTER the signal bar
pf = vbt.Portfolio.from_signals(close, entries.vbt.signals.fshift(1),
                                exits.vbt.signals.fshift(1), price=open_)
# 2. Shift the signals only (still a close fill, but a later one)
# 3. price=-np.inf  -> current open; correct ONLY if the signal is also open-based
```

## 🚨 Commons Clause — you may not sell it

The Commons Clause is an addendum on top of Apache-2.0 that removes the right to **sell** the
software: **you may not sell a product or service whose value derives substantially from vectorbt,
including hosting or support services.** Internal research and personal use are unaffected; a SaaS
is not. Compare `backtesting.py` (AGPL, network copyleft) and `nautilus_trader` (LGPL) — three
different copyleft postures in one problem space.

**VectorBT PRO** is closed source and subscription-based. ⚠️ Circa-2026 pricing of **$25/month, $20 billed annually, lifetime from a $150 floor** is **[UNVERIFIED]** — from a search result, and pricing pages move. PRO adds chunking, data ingest, richer records and pattern search, and **most PRO tutorials assume it** — a frequent source of "this example doesn't work" on the OSS build.

## 🚨 0.28.x → 1.0.0 is a breaking rewrite

Pinned 0.x code does not carry over unchanged. v1.0 added an optional **Rust engine** (`pip install vectorbt[rust]`, auto-dispatching between Numba and Rust per call) plus `FlexArray` zero-copy broadcast; 1.1.0 added Python 3.14 / pandas 3 / NumPy 2.4 support.

## What it models, and what it fakes

**Modelled:** percentage and fixed fees · percentage slippage · min/max size · size granularity ·
**partial fills by cash** (`allow_partial`) · **order rejection probability** (`reject_prob`) · cash
sharing across a group · long/short/both · SL/TP/trailing stops with configurable
`stop_entry_price` / `stop_exit_price`.

**Absent:** no order book, no queue position, no latency, **no partial-fill-by-liquidity** (only by cash), no corporate actions, no borrow cost, no margin call, and **no concept of an asset's lifetime** — a frame built from today's index members is silently survivorship-biased and vectorbt cannot tell. Stops are resolved against intrabar OHLC with a crude, configurable conflict mode (`stop_conflict_mode="exit"` by default); intrabar high/low ordering is guessed, as in every bar engine.

## Where it is the right tool

**Searching a parameter space.** A whole 2-D grid simulates as one array operation, orders of
magnitude faster than any event-driven engine. **The discipline that makes it safe:** sweep with
vectorbt, then **re-run the survivor in an event-driven engine** (nautilus_trader, zipline-reloaded,
LEAN) and check the numbers agree. When they disagree, vectorbt is usually the optimistic one — and
the gap is your bias estimate.

🚨 The sweep is itself a **trial count**. A 10,000-combination grid means `n_trials = 10,000` for
Deflated Sharpe purposes.

## Minimal correct call

```python
import vectorbt as vbt

fast, slow = vbt.MA.run(close, 10), vbt.MA.run(close, 50)
entries, exits = fast.ma_crossed_above(slow), fast.ma_crossed_below(slow)

pf = vbt.Portfolio.from_signals(
    close,
    entries.vbt.signals.fshift(1),   # 🚨 shift: the signal used close[t]
    exits.vbt.signals.fshift(1),
    price=open_,                     # 🚨 explicit: default np.inf == close[t]
    fees=0.0005, slippage=0.0005, freq="1D",
)
# n_trials for this grid = number of (fast, slow) combinations you swept.
```

## See also

- `../../../fin-core/skills/backtesting-engines/SKILL.md` §2.1 — signal→order timing across engines
- `../../../fin-core/skills/backtesting-engines/references/vectorbt.md` — the source card
- `../../../fin-core/skills/backtesting-engines/references/_engine-matrix.md` — licences and what each engine omits
- `../../../fin-core/skills/backtest-validation/SKILL.md` — trial counts and Deflated Sharpe

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`backtesting-engines`** (`../../../fin-core/skills/backtesting-engines/SKILL.md`).

