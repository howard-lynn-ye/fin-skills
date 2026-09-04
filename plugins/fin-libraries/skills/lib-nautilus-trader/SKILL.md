---
name: lib-nautilus-trader
description: >-
  Event-driven Rust-core engine with the strongest execution modelling in open source, gated to
  Python 3.12-3.14, where a wrong ts_init silently makes every bar visible one interval early.
  TRIGGER - import nautilus_trader, pip install nautilus_trader, BacktestEngine, BacktestNode,
  TradingNode, Strategy.on_bar, ts_init, ts_event, FillModel, prob_fill_on_limit, prob_slippage,
  LatencyModel, base_latency_nanos, RiskEngine, OmsType NETTING HEDGING, BookType.L1_MBP,
  bar_execution, bar_adaptive_high_low_ordering, high-precision build, ClientOrderId; "could not
  find a version that satisfies nautilus_trader", a Rust source build on an Intel Mac or Alpine.
  Memory is stale here: 1.231.0 shipped 2026-08-02, a 2.0 line is in release candidates that moves
  the fill model to nautilus_trader.execution, and the licence is LGPL-3.0-or-later. SKIP for
  choosing among engines generally (backtesting-engines). SKIP when the question is WHICH library
  to choose rather than how to use this one - that belongs to the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# nautilus_trader

The strongest execution modelling in open source, and the only engine where the **same strategy
object** runs the backtest and the live session — but it will not even install on Python 3.11.

| | |
|---|---|
| pip / import | `pip install nautilus_trader` · `import nautilus_trader` |
| Version | **1.231.0 (2026-08-02)** · ⚠️ a **2.0.0** line is in RC (`v2.0.0rc4`, 2026-09-02) |
| Licence | 🚨 **LGPL-3.0-or-later** (declared and classified), CLA required for contributors |
| Status | ✅ Very active, bi-weekly across `develop`/`nightly`/`master`. 28,367★, pushed 2026-09-04 |

## The trap that costs you money

🚨 **`ts_init` must be the bar's CLOSE.** Look-ahead protection here is *structural* — orders
submitted inside `on_bar()` arrive after that bar finishes processing — and that protection depends
entirely on this one field.

**Vendor bars are very often timestamped at their open.** Load them unchanged and every bar becomes visible one interval early. The engine does not complain; it **silently makes bars visible early**, and every metric looks better for it.

```python
INTERVAL_NS = 60 * 1_000_000_000                 # 1-minute bars, in nanoseconds
bar = Bar(
    bar_type, Price(o, 2), Price(h, 2), Price(l, 2), Price(c, 2), Quantity(v, 0),
    ts_event=ts_open,                            # when the bar's period began
    ts_init=ts_open + INTERVAL_NS,               # 🚨 the CLOSE — not ts_open
)
```

## 🚨 The install gate nobody reads

`requires_python` is **`>=3.12,<3.15`** — wheels are cp312/cp313/cp314 only. On Python 3.11 `pip install nautilus_trader` does not warn, downgrade or degrade; it **fails to resolve.** Its neighbours are far more permissive (`zipline-reloaded` `>=3.10`, `backtesting.py` `>=3.9`), so an existing 3.11 research environment cannot simply add nautilus. Packaging edges verified from the file list:

- **No macOS x86_64 wheel** — Intel Macs get a Rust source build.
- **`manylinux_2_35`** is a high glibc floor (Ubuntu 22.04-era); older distros fall back to the sdist.
- **No musllinux wheel** — Alpine containers build from source.

Two precision builds: **high-precision (128-bit, 16 decimals — the default in the official wheels)** and standard (64-bit, 9 decimals). High precision suits crypto quantities and tiny tick sizes; it is slower, and the two modes are **not interchangeable at the ABI level**.

## 🚨 `FillModel` defaults are optimistic

✅ Read from `nautilus_trader/backtest/models/fill.pyx` at tag `v1.231.0`:
`def __init__(self, double prob_fill_on_limit = 1.0, double prob_slippage = 0.0, random_seed: int | None = None)`

| Parameter | Default | Meaning |
|---|---|---|
| `prob_fill_on_limit` | 🚨 **1.0** | Fill probability when the market *touches but does not cross* your limit. **1.0 means you are always at the front of the queue.** |
| `prob_slippage` | 🚨 **0.0** | Probability of a one-tick adverse move — **L1 only**, and off by default. |
| `random_seed` | 🚨 **`None`** | With `prob_slippage > 0` fills are **random**; unseeded runs are **not reproducible** and two "identical" backtests disagree. |

**It is a model you must configure, not one you inherit.** ✅ 1.231.0's `backtest.models` already exports `BestPriceFillModel`, `OneTickSlippageFillModel`, `ProbabilisticFillModel`, `LimitOrderPartialFillModel`, `SizeAwareFillModel`, `VolumeSensitiveFillModel`, `CompetitionAwareFillModel`, `MarketHoursFillModel`, `TwoTierFillModel`, `ThreeTierFillModel`, plus `FixedFeeModel` / `MakerTakerFeeModel` / `PerContractFeeModel` — check before subclassing.

⚠️ **2.0 renames it.** The base model becomes `DefaultFillModel`, imported from `nautilus_trader.execution`, and `CompetitionAwareFillModel` gains `liquidity_factor` (default `0.3`). Parameter names and defaults are unchanged; 1.x imports will need updating.

## The models that do not exist elsewhere

**`LatencyModel`** — `base_latency_nanos`, `insert_latency_nanos`, `update_latency_nanos` (e.g. 5 ms base + 2 ms insert + 3 ms update). ✅ Real order-arrival latency inside a backtest is rare and valuable; nothing comparable exists in the other OSS engines.

**`RiskEngine`** validates price precision, quantity bounds, notional limits, margin and the trading state — **ACTIVE / HALTED / REDUCING** — *before* anything reaches the venue, in backtest **and** live. `REDUCING` (only position-reducing orders permitted) is the closest thing to a real kill switch any OSS engine ships.

**`OmsType`** NETTING vs HEDGING with ExecutionEngine reconciliation; contingency orders (OCO, OUO, OTO), post-only, reduce-only, iceberg, and a native TWAP algorithm with child-order tracking.

## 🚨 Bar backtests are a documented heuristic

The docs say so outright: bars become *"synthetic market updates for an L1 order book"*. Bar volume is **split evenly across four price points**, visited **Open → High → Low → Close**; `bar_adaptive_high_low_ordering=True` visits whichever extreme is nearer the open first. Requires venue `bar_execution=True` and `BookType.L1_MBP`. The docs call it *"a deterministic heuristic, not a reconstruction of the actual trade sequence"*.

So a bar backtest here is **better documented, not more truthful**. The engine earns its reputation on tick and L2/L3 data. With OHLCV only, you are paying its learning curve for an approximation you could get elsewhere.

## Minimal correct call

```python
from nautilus_trader.backtest.models import FillModel      # 2.0: DefaultFillModel, from ...execution

fill_model = FillModel(
    prob_fill_on_limit=0.2,   # default 1.0 = always at the front of the queue
    prob_slippage=0.5,        # default 0.0 = no slippage at all; L1 only
    random_seed=42,           # default None -> unseeded -> not reproducible
)
```

## What it does not model

It cannot know how your order would have changed other participants' behaviour. Slippage modelling **does not apply to L2/L3** — there the recorded book determines impact. Consumed liquidity per level is not tracked unless configured. **No corporate actions and no settlement model.**

Take it on when the strategy is latency- or microstructure-sensitive, spans venues, or is genuinely going live — the backtest→live code path and `ClientOrderId` reconciliation are the payoff. Do not take it on to backtest a daily moving-average cross.

## See also

- `../../../fin-core/skills/backtesting-engines/SKILL.md` — engine choice
- `../../../fin-core/skills/backtesting-engines/references/nautilus-trader.md` — the source card
- `../../../fin-core/skills/backtesting-engines/references/execution-realism.md` — fill realism, ranked
- `../../../fin-core/skills/broker-execution-apis/SKILL.md` — why order-ID reconciliation matters

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`backtesting-engines`** (`../../../fin-core/skills/backtesting-engines/SKILL.md`).

