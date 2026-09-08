# Execution realism — what each engine actually simulates

Ordered from most to least realistic. Read this before believing a fill.

## nautilus_trader — the strongest execution modelling in OSS

Rust core (mimalloc allocator, tokio async networking) with Python as the control plane. Two
precision modes: **high-precision (128-bit, 16 decimals — the default in official wheels)** and
standard (64-bit, 9 decimals). 🚨 **Python 3.12–3.14 only**; Linux x86_64/ARM64, macOS ARM64,
Windows x86_64. Bi-weekly releases across `develop`/`nightly`/`master`. **LGPL-3.0-or-later**, CLA
for contributors.

**Data granularity:** order-book deltas (L1/L2/L3), quote ticks, trade ticks, bars, and custom data at
**nanosecond resolution**, across multiple venues and instruments in one backtest.

**The models that do not exist elsewhere:**
- **`FillModel`** — `prob_fill_on_limit` (default **1.0**): probability a limit order fills when the
  market *touches but does not cross* its price. `prob_slippage` (default **0.0**): probability of a
  one-tick adverse move, **L1 only**. Optional synthetic liquidity beyond best bid/ask.
- **`LatencyModel`** — `base_latency_nanos`, `insert_latency_nanos`, `update_latency_nanos`
  (e.g. 5 ms base + 2 ms insert + 3 ms update). **Real order-arrival latency in a backtest is rare
  and valuable.**
- **`OmsType`** NETTING vs HEDGING, with the ExecutionEngine reconciling when strategy and venue types
  differ.
- **`RiskEngine`** validates price precision, quantity bounds, notional limits, margin and trading
  state (**ACTIVE / HALTED / REDUCING**) **before anything reaches the venue** — in backtest *and*
  live. `REDUCING` is the correct state for a controlled wind-down.
- Contingency orders (OCO, OUO, OTO), post-only, reduce-only, iceberg; a native **TWAP** execution
  algorithm with child-order tracking.

**Bar-based backtests are explicitly a heuristic** (the docs say so): bars become *"synthetic market
updates for an L1 order book"*. Bar volume is split evenly across four price points, visited
**Open → High → Low → Close** by default; `bar_adaptive_high_low_ordering=True` visits whichever
extreme is nearer the open first. The docs state this is *"a deterministic heuristic, not a
reconstruction of the actual trade sequence"*. Requires venue `bar_execution=True` and
`BookType.L1_MBP`.

🚨 **`ts_init` is the single biggest nautilus footgun.** Look-ahead protection is structural — orders
submitted inside `on_bar()` arrive *after* that bar finishes processing — **but `ts_init` must be the
bar's close.** For data timestamped at the open you must set `ts_init = ts_event + interval_ns`.
Getting it wrong **silently makes bars visible early.**

**What it does not model**, stated plainly in its docs: it cannot know how your order would have
changed other participants' behaviour. Slippage modelling does not apply to L2/L3 — there the recorded
book determines impact. Consumed liquidity per level is not tracked unless explicitly configured.

**Cost:** steep learning curve — nanosecond clocks, instrument definitions, venue configs, precision
modes, and a Rust build if you go off the wheels.

## QuantConnect LEAN — the broadest reality modelling

**Engine is Apache-2.0**, C# core with Python algorithm support, event-driven and modular.

Per-security: **fill models, slippage models, fee/transaction models, brokerage models** (simulate a
*specific* broker's rules), **buying-power/margin models, settlement models, short-availability
(borrow) models, margin-interest-rate models, dividend-yield models**, plus full option
pricing/volatility/exercise/assignment. Portfolio-level: **margin-call model** and risk-free-rate
model.

🔑 **Nothing else in this list models borrow availability or settlement.** Its map files and factor
files handle ticker changes and delistings, so QC's own equity data is survivorship-bias-free.

🚨 **The cost reality, flagged loudly:**
- The **engine** is Apache-2.0 and runs standalone.
- ⚠️ **Conflicting official sources on the CLI.** The LEAN CLI *docs* state: *"To use the CLI, you must
  be a member in an organization on a paid tier."* The CLI's **own GitHub README does not mention any
  paid tier** and just says `pip install lean` + Docker. **[PARTIALLY VERIFIED — check your own tier
  before promising free local use.]**
- **Data is the real cost.** `lean init` pulls a *tiny* sample dataset. Anything real comes from the QC
  Data Library (paid, per-dataset), `lean data generate` (fake data), or your own files converted into
  LEAN's bespoke on-disk format — **the conversion is the main friction.**

## backtesting.py — small, honest, single-asset

A bar loop with a small `_Broker`. Run order per bar `i` (read in source): `data._set_length(i+1)` →
`broker.next()` (process orders) → `strategy.next()` ("next tick, a moment before bar close").

**Fills, read from `_process_orders`:**
- Market orders → **next bar's open** (default). With `trade_on_close=True` → **`data.Close[-2]`**,
  i.e. the *previous* bar's close. 🚨 Both are causally consistent, but `trade_on_close` is widely
  misread as "fill at this bar's close" — it is not.
- Contingent (SL/TP) orders fill **within the triggering bar** at the stop price, with gap-through
  handled: `price = max(price, stop_price)` for longs, `min` for shorts. **A gap past your stop fills
  you at the open, not the stop.** Correct.
- ✅ **SL is deliberately prioritized over TP** when both are hit in one bar — the source literally
  says `# Ensure SL orders are processed first`. Pessimistic, which is right.
- Stop-limit: if the limit would have been hit before the stop triggered, the order is **skipped**
  ("pessimistically assume limit was hit before the stop").
- `commission` is folded into the fill price (`_adjusted_price`) — longs marked up, shorts marked
  down. `spread` is a separate constant-bps parameter.

🚨 **Residual look-ahead risk:** the loop slices to `i+1` *before* calling `next()`, so
`self.data.High[-1]` / `Low[-1]` / `Close[-1]` are readable inside `next()`. Market-order timing
protects you; **reading the current bar's high in a signal does not.** And `Strategy.I` computes
indicators over the **full series** in `init()` then slices — a non-causal indicator leaks the future
and **the framework cannot detect it**.

**Does not model:** multiple assets (single instrument, full stop), volume/liquidity limits, market
impact, corporate actions, borrow, funding, or margin beyond a simple `margin` ratio and an
`_OutOfMoneyError`.

🚨 **AGPL-3.0** — network-use copyleft. A hosted service built on it must offer source. **The single
most-missed licensing fact about this library.**

## zipline-reloaded — the equity-correctness benchmark

Apache-2.0, C/Cython extensions (so wheels matter). Maintenance-only — the only 2025 commits on `main`
are Dependabot bumps.

**What it gets right and almost nothing else does:** an asset database with `start_date`, `end_date`
and `auto_close_date`, so **positions in delisted names are auto-closed**; real **volume-limited
partial fills** (2.5% of bar volume by default); splits and dividends applied through an adjustments
database; and Pipeline for cross-sectional factor computation over a point-in-time universe.

**Use it when** you want a frozen, correct US-equity backtester. **Do not use it** if you need new
asset classes — it is a compatibility-maintenance project, not a feature project.

## Qlib

An **AI-oriented quant research platform**, not primarily an execution simulator. Layers: data (with a
**point-in-time database** and Arctic backend), a 20+ model zoo, `qrun` YAML orchestration,
backtest/analysis, **nested decision execution**, and an RL framework for order execution (PPO, OPDS).

**Backtest realism:** configurable `open_cost` / `close_cost` / `min_cost` (docs example:
0.0005 / 0.0015 / ¥5) and a **`limit_threshold` (e.g. 0.095) that models China's ±10% daily price
limit** — orders are blocked when the limit is hit. Reports `with_cost` vs `without_cost` side by side.
Ships `TopkDropoutStrategy` and `EnhancedIndexingStrategy`.

🚨 **Do not promise a turnkey dataset.** The docs state *"The official dataset is currently
disabled."* Data now comes from community Yahoo dumps in GitHub releases with an explicit quality
warning. See `../../../fin-china/skills/china-trading-stack/SKILL.md` §2 for the working replacement.
⚠️ `pyqlib` 0.9.7 dates to 2025-08-15 even though commits continue into 2026-07 — **packaging lags the
repo.**

## The summary that matters

| | partial fills | volume cap | margin | borrow | settlement | corp actions | delistings | latency |
|---|---|---|---|---|---|---|---|---|
| nautilus_trader | ✅ | ✅ book | ✅ | ⚠️ | ❌ | ❌ | ❌ | ✅ |
| LEAN | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| zipline-reloaded | ✅ | ✅ 2.5% | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |
| backtesting.py | ❌ | ❌ | ⚠️ ratio | ❌ | ❌ | ❌ | ❌ | ❌ |
| vectorbt | ⚠️ by cash | ❌ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Qlib | ⚠️ | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ⚠️ | ❌ |

**A blank column is not a small approximation.** Borrow cost decides whether a short strategy is
viable; settlement decides whether a cash-account strategy is legal; delistings decide whether your
return series is fiction.
