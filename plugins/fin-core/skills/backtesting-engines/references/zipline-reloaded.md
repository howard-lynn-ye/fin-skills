# zipline-reloaded

The US-equity correctness benchmark: the only free engine that models **delistings, real
volume-limited partial fills, and point-in-time corporate actions** at once — and it is frozen.

| | |
|---|---|
| pip | `zipline-reloaded` · 🚨 import **`zipline`** (the PyPI name and the module name differ) |
| Version | **3.1.1 (2025-07-19)** — over a year old |
| GitHub | `stefan-jansen/zipline-reloaded` — 1,933★, 44 open issues |
| Licence | ✅ **Apache-2.0** (declared as a licence expression) |
| Python | `>=3.10`; wheels **cp310–cp313 only — no 3.14** |
| Wheels | macOS x86_64 + arm64, `manylinux2014` **x86_64 only**, `win_amd64` · ⚠️ **no Linux aarch64** |
| Status | ⚠️ **Maintenance-only** — see below |

Stefan Jansen's fork of Quantopian's zipline (he wrote *ML for Trading*). C/Cython extensions, so
**wheels matter**: off the matrix above you are compiling Cython against your own NumPy.

⚠️ **Interpreter squeeze:** zipline-reloaded caps at 3.13 while `./nautilus-trader.md` requires
`>=3.12`. **3.12 and 3.13 are the only interpreters where both install.**

## ⚠️ The maintenance verdict, stated precisely

✅ Verified from the commit log: **every commit on `main` since the 3.1.1 release (2025-07-19) is a
Dependabot CI bump**, the newest dated **2025-11-13** — `actions/checkout`, `cibuildwheel`,
`upload-artifact`. The last substantive commits are from May–July 2025 (a Cython 3.1 fix, pandas 2.1
warning cleanup).

This is a **compatibility-maintenance project, not a feature project**. That is the right frame for
deciding: fine if you want a *frozen, correct* US-equity backtester; wrong if you need a new asset
class, a new data source, or a bug fixed this year.

## ✅ What it gets right that almost nothing else does

### Asset lifetimes — the survivorship answer

The asset database carries **`start_date`, `end_date` and `auto_close_date`** per asset. A position
in a name that delists is **automatically closed** at `auto_close_date`; the asset simply stops being
tradeable after `end_date`. See `./vectorbt.md` and `./backtesting-py.md` — neither has any concept
of an asset ceasing to exist, so a frame built from today's index members is silently biased and the
engine cannot tell.

🚨 **But survivorship is a property of YOUR bundle, not of zipline.** The engine can represent a
delisted asset; it cannot invent one. A bundle ingested from today's ticker list is survivorship-
biased no matter how correct the simulator is. See
`../../research-integrity-guards/references/leakage-checklist.md`.

### Volume-limited partial fills — the real one

`VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)` `[V-SRC]`:

- Fills **at most 2.5% of the bar's volume**, per bar.
- Prices the fill as `price * (1 ± price_impact * volume_share**2)` — quadratic in participation.
- **The unfilled remainder carries to the next bar** rather than vanishing or filling anyway.

Futures default to a `0.05` volume limit; `VolatilityVolumeShare` is the futures impact model. Also
shipped: `NoSlippage`, `FixedSlippage`, `FixedBasisPointsSlippage`.

**This is the best free liquidity model in the category.** Everything else in `./_engine-matrix.md`
except LEAN and nautilus will happily fill an order larger than the entire bar's volume, instantly,
at one price.

**Commission defaults** `[V-SRC]`: `PerShare` **$0.001/share with a $0.00 minimum**, `PerContract`
**$0.85/contract**, `PerDollar` **0.0015**. The `$0.00` minimum is the one to override — most retail
US brokers still have a floor, and a strategy that trades 3-share lots is free here and not in life.

### Adjustments applied as of the simulation date

Raw prices are stored; splits and dividends live in a separate **adjustments database** and are
applied **as of the simulation date**, so history is adjusted only for events that had already
happened. Dividends are paid into cash.

🚨 This is the fix for the trap in `../../research-integrity-guards/references/adjustment-conventions.md`:
a back-adjusted `Adj Close` series restates history every time a dividend is paid, so today's
adjusted 2015 price **was not the 2015 price**. Engines with no adjustments DB inherit that silently.

### Pipeline

Cross-sectional factor computation with **explicit windowing** over a point-in-time universe. The
window length is a declared parameter rather than an implicit pandas slice, which structurally
discourages look-ahead. Pairs with `../../factor-and-timeseries-research/references/alphalens-reloaded.md`.

## 🚨 The catch: there is no turnkey bundle any more

The Quandl and Quantopian bundles that made zipline a one-liner are **gone**. You must ingest your
own with `zipline ingest`, and the point-in-time and survivorship properties of the result are
yours to guarantee. Budget for the bundle, not for the engine.

```python
def initialize(context):
    # 🚨 Both defaults are worth setting explicitly — the $0 commission minimum especially.
    set_slippage(us_equities=slippage.VolumeShareSlippage(
        volume_limit=0.025,     # ≤2.5% of bar volume; remainder carries to the next bar
        price_impact=0.1,       # fill price scales with volume_share ** 2
    ))
    set_commission(us_equities=commission.PerShare(cost=0.001, min_trade_cost=1.00))
    context.asset = symbol("AAPL")

def handle_data(context, data):
    # Orders placed here execute on SUBSEQUENT bars via the blotter — no same-bar fill.
    order_target_percent(context.asset, 0.10)
```

## Where it fits

Use it as the **arbiter**, not the workhorse: sweep parameters in `./vectorbt.md` (seconds), then
re-run the survivor here and check the numbers agree. When they disagree, vectorbt is almost always
the optimistic one, and the gap is your bias estimate. For the execution-realism ranking across all
engines see `./execution-realism.md`; for the multiplicity correction the sweep created, see
`../../backtest-validation/SKILL.md`.
