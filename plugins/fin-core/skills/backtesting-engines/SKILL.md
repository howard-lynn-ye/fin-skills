---
name: backtesting-engines
description: >-
  Choose a backtesting engine and know what it silently models wrong. TRIGGER - "backtest this",
  backtest a crossover or a moving-average strategy, simulate a strategy, walk-forward, parameter
  sweep, "test this trading idea"; comparing or choosing backtest frameworks; vectorbt,
  backtesting.py, backtrader, zipline, PyBroker, bt, nautilus_trader, LEAN, freqtrade, jesse; how
  an engine models fills, slippage, commissions; what slippage to assume, partial fills, margin,
  shorting or delistings; taking a strategy from backtest to live; "my backtest looks too good";
  "works in backtest but loses money live". Several popular engines fill at the signal's own bar
  close by default. SKIP for judging whether a finished result is real (backtest-validation) for
  A-share rules (china-trading-stack), for crypto funding and perpetuals
  (crypto-data-and-execution), for options, which need assignment and settlement no engine here
  models (options-backtesting), and for measuring fills you already have
  (execution-cost-analysis).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# Backtesting engines

Two things decide this choice: **what the engine models honestly**, and **what licence you can
live with**. Speed is almost never the binding constraint — a fast wrong answer is worse than a
slow right one.

Before trusting any result from any engine, run the audit in `research-integrity-guards`.

## 1. Pick an engine

| Task | Engine | Why |
|---|---|---|
| Sweep 10k parameter combos, one or few assets | **vectorbt** | Vectorized/Numba, seconds not hours. 🚨 **Same-bar fill by default — see §2.1** |
| Single-asset TA strategy, honest by default | **backtesting.py** | Next-open market fills, pessimistic SL-before-TP. **AGPL-3.0** |
| Serious event-driven, multi-venue, backtest→live | **nautilus_trader** | Rust core, L2/L3 book, latency + fill models, same code both ways. **LGPL-3.0. Needs Python ≥3.12** |
| US equity cross-sectional / factor research | **zipline-reloaded** | Real volume-limited partial fills, splits/divs/**delistings**, Pipeline. Maintenance-only |
| ML strategies with walk-forward + bootstrap | **PyBroker** | Walk-forward and bootstrapped metrics built in |
| Asset-allocation / weight strategies | **bt** | Tree-of-algos rebalancing. Not an order-level simulator. MIT |
| Institutional all-asset, willing to pay | **QuantConnect LEAN** | Deepest reality modelling in OSS; map/factor files handle ticker changes + delistings |
| Crypto retail bot, live-first | **freqtrade** | 🥇 **Best bias-detection tooling in the field** (§4). GPL-3.0 |

**Do not start new work on:** `backtrader` (last commit 2023-04-19 — the most over-recommended dead
library in the domain, 23k stars notwithstanding), `fastquant` (2023), `blankly` (2024),
`qstrader` (dormant 2024-06), `catalyst` (archived), `mlfinlab` (off PyPI).

## 2. 🚨 The correctness section — ordered by how much the mistake costs

### 2.1 Signal→order timing (look-ahead)

A signal computed from bar *t*'s close cannot be filled at bar *t*'s close.

| Engine | Default | Verdict |
|---|---|---|
| **vectorbt** | `from_signals(close, entries, exits)` fills at the **same bar's close** | 🚨 **SILENTLY WRONG by default** |
| backtesting.py | market orders → next bar's open | ✅ right |
| backtrader | market orders → next bar's open | ✅ right (`cheat_on_open/close` opt into danger) |
| zipline-reloaded | orders execute on subsequent bars via the blotter | ✅ right |
| nautilus_trader | orders from `on_bar()` arrive **after** that bar finishes | ✅ right by construction |
| LEAN | event-driven, fills on subsequent data | ✅ right |
| freqtrade | entries at open; **exit signals at the NEXT candle's open** | ✅ right |
| PyBroker | bar loop with explicit next-bar execution | ✅ right |

**The vectorbt footgun — verified in source.** `Order.price` defaults to `np.inf`, and vectorbt's
own docstring (`vectorbt/portfolio/enums.py`) says: *"If `np.inf`, replaced by the current close."*
`from_signals`/`from_orders` resolve `if price is None: price = np.inf`. So:

```python
pf = vbt.Portfolio.from_signals(close, entries, exits)   # fills at close[t] — the signal's own bar
```

Combined with the equally default `fast_ma.ma_crossed_above(slow_ma)` (also computed on `close[t]`),
this is textbook same-bar execution. **Every vectorbt tutorial showing a beautiful equity curve
without shifting is showing a biased result.** It is a default, not a bug, and it will never warn you.

Fixes, in order of preference:
```python
# 1. fill at the next open, with signals shifted so the open is AFTER the signal bar
pf = vbt.Portfolio.from_signals(close, entries.vbt.signals.fshift(1),
                                exits.vbt.signals.fshift(1), price=open_)
# 2. shift signals only
# 3. price=-np.inf  -> current open; correct ONLY if the signal is also open-based
```

**backtesting.py's residual risk:** the run loop slices data to `i+1` before calling
`Strategy.next()`, so `self.data.High[-1]` / `Low[-1]` / `Close[-1]` are readable inside `next()`.
Market-order timing protects you; reading the current bar's high in a signal does not. Separately,
`Strategy.I` computes indicators over the **full series** in `init()` and then slices — so a
**non-causal indicator** (centered window, `filtfilt`, `.shift(-1)`, any zero-phase filter, a model
fitted on all data) leaks the future and **the framework cannot detect it.**

### 2.2 Survivorship

**No engine fixes this — it is a property of your data.** What differs is whether the engine can
*represent* an asset's lifetime at all.

| Engine | Can represent delisting | Note |
|---|---|---|
| zipline-reloaded | ✅ asset DB has `start_date`, `end_date`, `auto_close_date`; delisted positions auto-close | Best-structured — but only as good as the bundle you ingest |
| LEAN | ✅ map files + factor files handle ticker changes and delistings | Strongest turnkey answer if you pay for the data |
| Qlib | ⚠️ has a PIT database concept, but the *shipped* dataset is a community Yahoo dump with an explicit quality warning | Don't trust the free bundle for survivorship |
| vectorbt / backtesting.py / bt / PyBroker | ❌ no concept of it | A frame built from today's index members is silently biased |
| freqtrade / jesse / OctoBot / hummingbot | ❌ **chronically biased in practice** | A pair list of today's top-volume coins over 2021 excludes every coin that died. **Worse in crypto than equities.** |

### 2.3 Recursive-indicator warm-up divergence

EMA, RSI, ADX — anything with state — converge to different values depending on how much history
preceded them. Backtest sees 5,000 candles; live sees whatever one API call returns (~1,000). **The
EMA differs, so the signal differs, so backtest and live disagree for reasons unrelated to the
strategy.** This affects **every engine**, and only freqtrade ships a detector
(`freqtrade recursive-analysis`, §4).

### 2.4 Consolidated footgun list

| # | Footgun | Bites hardest in |
|---|---|---|
| 1 | Same-bar fill on the signal's own close | **vectorbt `from_signals` default** |
| 2 | Non-causal indicators computed over the full series then sliced | backtesting.py `Strategy.I`, vectorbt, PyBroker, any pandas pipeline |
| 3 | Recursive-indicator warm-up divergence backtest vs live | every engine; only freqtrade detects it |
| 4 | Survivorship-biased universe from today's ticker list | vectorbt / backtesting.py / bt / PyBroker / freqtrade / jesse |
| 5 | `Adj Close` used for signals *and* execution | any engine without an adjustments DB |
| 6 | Restated (non-PIT) fundamentals | everything except LEAN and Qlib's PIT DB |
| 7 | Stops assumed to fill exactly at the stop price | freqtrade (documented), most bar engines |
| 8 | Intrabar high/low ordering guessed | all bar engines |
| 9 | Orders larger than the bar's liquidity filling instantly | everything except zipline (2.5% cap), LEAN, nautilus |
| 10 | Reporting the `optimize()` / hyperopt grid maximum as "the result" | backtesting.py, freqtrade hyperopt, LEAN optimizer, Optuna sweeps |
| 11 | Wrong `ts_init` making bars visible at their open | nautilus_trader |
| 12 | `cheat_on_open` / `cheat_on_close` | backtrader |
| 13 | `trade_on_close=True` misread as "fill at this bar's close" (it uses `Close[-2]`) | backtesting.py |
| 14 | Trusting Qlib's shipped community dataset | Qlib |

Footgun #10 is not an engine bug — it is `backtest-validation`'s territory, and it is the one that
most often survives all the way to a production decision.

## 3. 🚨 Licences — several are not what people assume

| Engine | Licence | Consequence |
|---|---|---|
| **vectorbt** (open) | **Apache-2.0 + Commons Clause** | Cannot sell it, or a service whose value derives substantially from it |
| **PyBroker** (`lib-pybroker`) | **Apache-2.0 + Commons Clause** | same |
| **backtesting.py** | **AGPL-3.0** | Network copyleft — serving it obliges source disclosure |
| **backtrader** | GPL-3.0 | + unmaintained |
| **freqtrade**, **OctoBot**, **lumibot** | GPL-3.0 | |
| **nautilus_trader** | LGPL-3.0-or-later | Dynamic linking generally fine |
| **RQAlpha** | **Custom — non-commercial only** | |
| zipline-reloaded, LEAN, hummingbot | Apache-2.0 | Clean |
| bt, jesse, vnpy, Qlib, wondertrader | MIT | Clean |

## 4. freqtrade's bias detectors — portable thinking, even if you never use freqtrade

`freqtrade lookahead-analysis` re-runs a backtest on progressively truncated data and flags
indicators whose historical values change. Limitations it states honestly: only checks **triggered**
signals (untriggered ones escape); can false-positive on limit orders with custom pricing callbacks;
useless for rarely-signalling strategies.

Its stated mechanism is the general lesson: *"Backtesting initializes all timestamps (loads the whole
dataframe into memory)"* while live processes candles sequentially. **That sentence describes
vectorbt, PyBroker and backtesting.py's `Strategy.I` equally well.**

`freqtrade recursive-analysis` varies `startup_candle_count` and reports each indicator's
last-row variance versus the base calculation. `-` = converged; large % = raise your startup candles.

Both ideas are worth reimplementing against whatever engine you actually use. See
`plugins/fin-core/skills/signal-construction/scripts/assert_causal.py`.

## 5. The reference engine in this repo — `fin_skills.engine`

This library ships a twelfth engine. Not because the eleven above are missing a feature —
they are not — but because none of them produces an object this repo's guards can read.
`benchmarks/leak_bench.py` has to hand-write **twelve `_adapt_*` functions** to marshal one
pipeline's artefacts into guard keywords. `fin_skills.engine` does that marshalling once:

```python
from fin_skills.api import check
from fin_skills.engine import Sessions, Panel, Universe, Execution, Costs, run, rank_long_short

res = run(panel, momentum, execution=Execution(fill_at="next_open", signal_lag=1),
          costs=Costs(commission_bps=0.5, spread_bps=1.0, cash_rate=0.05),
          universe=Universe(rebalance="ME", min_adv=1e6), benchmark=spy)
print(check(res.to_bundle()).summary())
```

**Measured on a 40-name, 1,260-session synthetic panel with 6 splits and 7 delistings
(2026-09-09):** 14,325 fills, 850 order slices deferred by the 5% participation cap, 58
rebalances, and `check(res.to_bundle())` runs **six guards with no adapter written** —
`adjustment_check`, `assert_causal`, `cost_curve`, `pit_universe`, `rf_convention`,
`survivorship_audit`. Five passed; `cost_curve` failed, because the demo's 40-5 momentum
rule scores Sharpe 0.63 gross, 0.13 net at 5 bps and **−0.37 at 10 bps round-trip**.
Ending a run with a failing guard is the intended default, not a defect.

`coverage().unlocks()` then names the next slot in the library's own vocabulary:
`+ regime_labels` unlocks `regime_coverage`, `+ model_returns` unlocks `spa_test`,
`+ card` unlocks `result_manifest`.

### 5.1 What it is

**724 lines of code in ten modules** (1,252 physical lines; `tests/test_engine_size.py`
holds the budget at 900 / 1,400 and prints the per-module table). numpy and pandas only —
no new dependency, asserted by a test. One bar loop, in the only causal order: fill what
earlier bars ordered → close out what delisted → mark the book → *then* decide.

| Module | What it owns |
|---|---|
| `spec.py` | `Sessions`, `Execution`, `Costs` — frozen, validated at construction |
| `panel.py` | OHLCV whose **adjustment convention is a field**, plus `fingerprint()` |
| `actions.py` | corporate actions → a factor series; back / forward / raw+factors |
| `costs.py` | four slippage models: fixed, spread-share, zipline's quadratic, Almgren sqrt |
| `sizing.py` | five sizers, incl. `from_weights()` for a PyPortfolioOpt/skfolio result |
| `universe.py` | point-in-time membership → `{rebalance date: [tickers]}` |
| `execute.py` | the bar loop |
| `result.py` | `BacktestResult`, `to_bundle()`, `check()`, `card()` |
| `invariants.py` | the twelve claims, each tagged with the test that proves it |

### 5.2 🚨 What it refuses to do, and why

Most of these raise at construction or at `run()`; the rest are arguments with no default
at all. Every one of them is some other engine's silent default.

| Refusal | The default it exists to refuse |
|---|---|
| `Execution(signal_lag=0)` raises | vectorbt's `Order.price=np.inf` → the signal's own bar close (§2.1) |
| `Sessions` has no default calendar | `exchange_calendars` computes its bounds from `Timestamp.now()` at import, so a run anchored on them is not reproducible |
| `Panel(adjustment=...)` has no default | every vendor's default differs; `readjust()` refuses without an `actions` table, because a convention with no event table cannot be checked |
| a bar dated outside the calendar raises | dropping it silently is how a holiday bar becomes a return |
| a delisting is a **booked closeout** | a NaN that vanishes is the survivorship bias, not a data gap; `on_delist="raise"` refuses to guess a recovery at all |
| an over-cap order **carries**, in `result.unfilled` | filling 10× a bar's volume instantly is every vectorized engine's default |
| `run(strict=True)` raises on a declared convention the action table contradicts | `adjustment_check`'s `expected=` argument, applied to the engine's own input |

### 5.3 Twelve invariants, seven of them falsified by a guard this repo already ships

`tests/test_engine.py` (28 tests) proves each. **Seven** run an existing guard on the
engine's own output rather than a bespoke assertion — `assert_causal` is run on `run()`
itself (`fn=lambda close: run(rebuild(panel, close), sig).weights`), and `fold_leak_test`
takes `run` as its `run_fold`. Six is the number `check(to_bundle())` reaches unaided;
`fold_leak_test` needs folds, which are not artefacts of one run.

| | Invariant | Proven by |
|---|---|---|
| I1 | no output cell before bar *k* moves when rows ≥ *k* are perturbed | `assert_causal` |
| I2 | `signal_lag=0` raises; a bar-*t* signal fills at *t+lag* | bespoke |
| I3 | every fill is dated strictly after its decision bar | bespoke (sentinel open) |
| I4 | the adjustment convention is declared **and** checked | `adjustment_check` |
| I5 | a delisting is a booked closeout; the name never reappears | `survivorship_audit` |
| I6 | `universe[d]` uses only rows ≤ *d* | `pit_universe` |
| I7 | `turnover[t] == (w[t] - w[t-1]).abs().sum()`, to 1e-12 | bespoke |
| I8 | `net(0)` **is** gross; `net(b)` non-increasing in *b* | `cost_curve` |
| I9 | every index is a subset of the declared sessions; windows count **bars** | bespoke |
| I10 | no fill exceeds `cap × volume`; the remainder carries | bespoke |
| I11 | cash is a position: held weights + cash weight = 1 exactly | `rf_convention` |
| I12 | two runs are identical; folds share no state | `fold_leak_test` |

The point of I2's second half: a perfect oracle (`close.pct_change().shift(-1)`) passes
**every engine invariant** and posts a gross Sharpe above 5 — and the same
`res.check(guards=["assert_causal"])` reports `LOOK-AHEAD`. The engine does not stop you;
it hands the guard enough to convict you.

### 5.4 When to use a real engine instead

**It is not fast, it is not event-driven, and it has no live path.** The demo above took
29 s and 39 s on two runs — call it 30 ms per session for 40 names. A 10,000-combination
vectorbt sweep is not this. Reach for the engines in §1 when you need:

| You need | Use |
|---|---|
| A parameter sweep, or anything where wall-clock decides the design | **vectorbt** — then re-run the survivor here |
| Intrabar stop/target semantics, gap-through handling | **backtesting.py** |
| Minute or tick bars, an L2/L3 book, latency and fill probability | **nautilus_trader** |
| Borrow availability, settlement, margin interest, option assignment | **LEAN** |
| An adjustments database, Pipeline, a battle-tested blotter | **zipline-reloaded** |
| To trade | none of the above from this engine — it has no broker path at all |

Also absent by design: order types beyond market, an intrabar price path (`next_vwap` is
the typical price (H+L+C)/3 and says so in its own docstring), a margin model, a
short-locate, currency conversion, and any option or futures lifecycle. Its intended use
is the arbiter role §1 already recommends — sweep elsewhere, re-run the survivor here, and
let `check()` say what is wrong.

## 6. Reference files

`references/<engine>.md` — architecture, what it models, licence, maintenance verdict, ingestion
story, and its specific footguns.

```bash
grep -ril "same-bar\|look-ahead" plugins/fin-core/skills/backtesting-engines/references/
grep -i -A8 "FOOTGUN" plugins/fin-core/skills/backtesting-engines/references/vectorbt.md
```

## Per-library deep dives

The optional `fin-libraries` plugin carries a dedicated skill for each library below. Load one
only after this skill has told you which library you want:

- **`lib-freqtrade`** — freqtrade
- **`lib-vectorbt`** — vectorbt

## Constraints no engine models for you

Two US rules bind hardest exactly where a backtest books its best trades, and no engine in the
matrix above models either: **Reg SHO Rule 201** blocks shorting into a 10% intraday drop at
the price you assumed, and a short leg needs a **locate and a borrow fee** that moves with the
same demand your signal is reading. See `../us-market-rules/SKILL.md` §2-3.

## Measuring the cost, not assuming it

This skill and `../research-integrity-guards/SKILL.md` both help you pick a cost ASSUMPTION
before you have fills. Once you have real fills, the assumption is testable — and
"works in backtest, loses money live" is an answerable question rather than a complaint.
See `../execution-cost-analysis/SKILL.md`.
