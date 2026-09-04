# nautilus_trader

The strongest execution modelling in open source, and the only engine here where the **same strategy
object** runs the backtest and the live session — but it will not even install on Python 3.11.

| | |
|---|---|
| pip | `nautilus_trader` · import `nautilus_trader` |
| Version | **1.231.0 (2026-08-02)** · ⚠️ a **2.0.0** line is in release candidates (`v2.0.0rc4`, 2026-09-02) |
| GitHub | `nautechsystems/nautilus_trader` — 28,367★, 107 open issues, pushed **2026-09-04** |
| Licence | 🚨 **LGPL-3.0-or-later** (declared and classified), CLA required for contributors |
| Python | 🚨 **`>=3.12,<3.15`** — wheels are cp312 / cp313 / cp314 only |
| Wheels | macOS **arm64 only**, `manylinux_2_35` x86_64 + aarch64, `win_amd64`, plus an sdist |
| Status | ✅ Very active, bi-weekly releases across `develop` / `nightly` / `master` |

## 🚨 The install gate nobody reads

`requires_python` is **`<3.15,>=3.12`**. On Python 3.11 `pip install nautilus_trader` does not
warn, downgrade, or degrade — it **fails to resolve**. Its neighbours in this skill are far more
permissive (`./zipline-reloaded.md` is `>=3.10`, `./backtesting-py.md` is `>=3.9`), so an existing
3.11 research environment cannot simply add nautilus. Check your interpreter before you plan the
project.

Two further packaging edges verified from the file list:
- **No macOS x86_64 wheel.** Intel Macs get a Rust source build, not a wheel.
- **`manylinux_2_35`** is a high glibc floor (Ubuntu 22.04-era). Older distros fall back to the sdist,
  which means compiling the Rust core.
- There is **no musllinux wheel** — Alpine containers build from source.

🚨 **LGPL-3.0-or-later.** Dynamic linking to an unmodified library keeps your strategy code your own,
but modifications to nautilus itself must be released, and you must let users relink. Compare
`./vectorbt.md` (Commons Clause, not OSI) and `./backtesting-py.md` (AGPL, network copyleft) —
three different copyleft postures in one skill.

## Precision modes

Two builds: **high-precision (128-bit, 16 decimals — the default in the official wheels)** and
standard (64-bit, 9 decimals). High precision is what you want for crypto quantities and for venues
with tiny tick sizes; it is also slower and the two modes are not interchangeable at the ABI level.

## The models that do not exist elsewhere

**`FillModel`** — ✅ read from `nautilus_trader/backtest/models/fill.pyx` at tag `v1.231.0`:
`def __init__(self, double prob_fill_on_limit = 1.0, double prob_slippage = 0.0, random_seed: int | None = None)`

| Parameter | Default | Meaning |
|---|---|---|
| `prob_fill_on_limit` | 🚨 **1.0** | Probability a limit order fills when the market *touches but does not cross* its price. **1.0 means you are always at the front of the queue.** |
| `prob_slippage` | 🚨 **0.0** | Probability of a one-tick adverse move — **L1 only** (on L2/L3 the recorded book determines impact), and off by default. |
| `random_seed` | 🚨 **`None`** | With `prob_slippage > 0` fills are **random**. Unseeded runs are **not reproducible** and two "identical" backtests will disagree. |

Out of the box the fill model is therefore *optimistic*: perfect queue position, zero slippage.
It is a model you must configure, not a model you inherit.

✅ The 1.231.0 `backtest.models` package exports far more than the research literature suggests:
`BestPriceFillModel`, `OneTickSlippageFillModel`, `ProbabilisticFillModel`,
`LimitOrderPartialFillModel`, `SizeAwareFillModel`, `VolumeSensitiveFillModel`,
`CompetitionAwareFillModel`, `MarketHoursFillModel`, `TwoTierFillModel`, `ThreeTierFillModel`, plus
`FixedFeeModel` / `MakerTakerFeeModel` / `PerContractFeeModel`. Subclass only after checking whether
one of these already does the job.

⚠️ **The 2.0 line renames it.** In the current default branch the base model is `DefaultFillModel`,
imported from `nautilus_trader.execution`, and `CompetitionAwareFillModel` gains a `liquidity_factor`
(default `0.3`). The parameter names and defaults are unchanged. Code written against 1.x imports
will need the import updated when 2.0 ships.

**`LatencyModel`** — `base_latency_nanos`, `insert_latency_nanos`, `update_latency_nanos`
(e.g. 5 ms base + 2 ms insert + 3 ms update). ✅ **Real order-arrival latency inside a backtest is
rare and valuable**; nothing else in `./_engine-matrix.md` offers it.

**`RiskEngine`** validates price precision, quantity bounds, notional limits, margin, and the trading
state — **ACTIVE / HALTED / REDUCING** — *before* anything reaches the venue, in backtest **and**
live. `REDUCING` (only position-reducing orders permitted) is the correct state for a controlled
wind-down, and it is the closest thing to a real kill switch any OSS engine ships.

**`OmsType`** NETTING vs HEDGING, with the ExecutionEngine reconciling when strategy and venue types
disagree. Contingency orders (OCO, OUO, OTO), post-only, reduce-only, iceberg, and a native TWAP
algorithm with child-order tracking.

## 🚨 Bar backtests are a documented heuristic, not a simulation

The docs say so outright: bars become *"synthetic market updates for an L1 order book"*.

- Bar volume is **split evenly across four price points**, visited **Open → High → Low → Close**.
- `bar_adaptive_high_low_ordering=True` instead visits whichever extreme is nearer the open first.
- Requires venue `bar_execution=True` and `BookType.L1_MBP`.
- The docs call it *"a deterministic heuristic, not a reconstruction of the actual trade sequence"*.

So a bar backtest here is *better documented* than a bar backtest elsewhere, not fundamentally more
truthful. The engine earns its reputation on **tick and L2/L3 data**. If all you have is OHLCV bars,
you are paying nautilus's learning curve for an approximation you could get elsewhere.

## 🚨 `ts_init` — the single biggest footgun

Look-ahead protection is **structural**: orders submitted inside `on_bar()` arrive *after* that bar
finishes processing. That protection depends entirely on one field.

**`ts_init` must be the bar's CLOSE.** Vendor bars are very often timestamped at their **open**. If
you load them unchanged, every bar becomes visible one interval early and the engine will not
complain — **it silently makes bars visible early**, and every metric looks better for it.

```python
from nautilus_trader.backtest.models import FillModel      # 2.0: DefaultFillModel, from ...execution
from nautilus_trader.model.data import Bar

# Bars timestamped at the OPEN must be shifted before they are valid nautilus data.
INTERVAL_NS = 60 * 1_000_000_000                 # 1-minute bars, in nanoseconds
bar = Bar(
    bar_type, Price(o, 2), Price(h, 2), Price(l, 2), Price(c, 2), Quantity(v, 0),
    ts_event=ts_open,                            # when the bar's period began
    ts_init=ts_open + INTERVAL_NS,               # 🚨 the CLOSE — not ts_open
)

fill_model = FillModel(
    prob_fill_on_limit=0.2,   # default 1.0 = always at the front of the queue
    prob_slippage=0.5,        # default 0.0 = no slippage at all; L1 only
    random_seed=42,           # default None -> unseeded -> not reproducible
)
```

## What it does not model — stated plainly in its own docs

It cannot know how your order would have changed other participants' behaviour. Slippage modelling
**does not apply to L2/L3** — there the recorded book determines impact. Consumed liquidity per level
is not tracked unless explicitly configured. There is no corporate-actions layer and no settlement
model: for US equity survivorship and adjustments see `./zipline-reloaded.md`.

## When it is worth the cost

Steep learning curve: nanosecond clocks, instrument definitions, venue configs, precision modes, and
a Rust build the moment you step off the wheel matrix. Take it on when the strategy is
**latency- or microstructure-sensitive**, spans multiple venues, or is genuinely going live — the
backtest→live code path is the payoff, along with first-class `ClientOrderId` reconciliation
(see `../../broker-execution-apis/references/interactive-brokers.md` for why that matters).

Do not take it on to backtest a daily moving-average cross. Sweep in `./vectorbt.md`, validate the
survivor here, and treat multiplicity with `../../backtest-validation/SKILL.md`.
