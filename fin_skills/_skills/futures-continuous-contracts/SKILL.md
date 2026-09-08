---
name: futures-continuous-contracts
description: >-
  Build and use a futures price series correctly — a continuous contract does not exist in the
  market, it is stitched, and the stitching method changes your answer. TRIGGER - futures, continuous
  contract, back-adjusted, Panama adjustment, ratio adjustment, roll, roll yield, contango,
  backwardation, front month, expiry, first notice day, open interest roll, CME, Globex, ES, CL, NG,
  VX, GC, ZN; joining futures bars to an equity calendar; "my futures backtest returns look wrong";
  negative prices in a price series; norgatedata, databento continuous symbols, yfinance CL=F or
  ES=F. SKIP for Chinese futures and 夜盘 (china-trading-stack) and for crypto perpetuals, which have
  funding rather than rolls (crypto-data-and-execution).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# Futures continuous contracts

**There is no such thing as "the price of crude oil".** There is a sequence of expiring contracts,
and any long series is a construction you chose. Most of what follows was **executed**, not read.

## 1. 🚨 The adjustment method and the return operator must match

This is a 2×2, not a preference. ✅ Measured over 180 rolls against true dollar P&L:

| Series | `diff()` | `pct_change()` |
|---|---|---|
| **difference** (Panama / back-adjusted) | ✅ **exact — $0 error** on true dollar P&L | 🚨 corr **0.97** in mild contango, **−0.005** in backwardation |
| **ratio** (proportional) | 🚨 off by **36.9%** of total P&L | ✅ **exact — corr 1.000000, max err 0.000000** |
| **unadjusted** (spliced) | 🚨 wrong at every roll | 🚨 wrong at every roll |

> **difference + `diff()` → dollar P&L. ratio + `pct_change()` → percentage returns.
> unadjusted → levels only (margin, tick value, limit moves). Never cross the pairs.**

✅ Reproduced in `scripts/continuous_contract.py`: over 8 synthetic years the unadjusted series
implies **+6.2%/yr against a true −12.9%/yr — 19 percentage points wrong**, while ratio matches the
truth to **+0.0 pp**.

## 2. 🚨 Back-adjusted prices go negative, and you can predict when

A difference-adjusted series subtracts the cumulative roll gap from all history. Under sustained
backwardation the subtraction eventually exceeds the price level.

🔑 **Crossing time = `1 / (annual roll yield)` years — independent of the price level.** At crude's
current front-month backwardation the series crosses zero in **well under 5 years of history**.

✅ Measured on one run: **1,141 of 3,780 bars ≤ 0**, 51 sign changes, `np.log()` returns NaN,
`pct_change()` produced a **+16,247% "return"**, and annualised "vol" came out **55.5 against a true
4.17**. The demo in `scripts/continuous_contract.py` shows **1,084 of 2,014 days negative**.

**A negative price is not a bug in the data.** It is the correct output of the construction. It means
`pct_change()`, `log()`, and every volatility estimate built on them are meaningless on that series.

## 3. 🚨 Back-adjusted history is rewritten at every roll

✅ From pysystemtrade's source, `_roll_in_panama`:

```python
adjusted_prices_values = [adj_price + roll_differential for adj_price in adjusted_prices_values]
```

**Every prior value.** And its incremental updater `update_with_multiple_prices_no_roll` **returns an
empty series if a roll occurred** — the reference implementation *refuses* to update across a roll
and forces a full rebuild. That is the honest design.

**How far history moves per year of new data** (✅ arithmetic from live roll yields):

| Contract | History moves |
|---|---|
| ES | ≈ **+2.0%** of level per year |
| NG | ≈ **+25%/yr** |
| VX front | ≈ **+60%/yr** |
| CL at 8% backwardation | ≈ **−8%/yr** |

Re-pull a back-adjusted series a year later and **every historical level has moved by that much**.
This is the same reproducibility failure as forward-adjusted equities
(`../../../fin-core/skills/research-integrity-guards/references/adjustment-conventions.md`) — snapshot
and hash what you used.

## 4. 🚨 yfinance futures are not continuous contracts

✅ Verified against Yahoo's own API: `CL=F` returns `shortName = "Crude Oil Oct 26"` and `ES=F`
returns `"E-Mini S&P 500 Sep 26"`. **They are an unadjusted front-month splice.**

`CL=F` contains **close = −37.63 on 2020-04-20 and +10.01 the next day** — a raw roll discontinuity
of **$47.64** sitting inside what people treat as a price series.

🚨 **And you cannot rebuild it correctly, because Yahoo deletes expired contracts.** `CLV26.NYM`,
`ESZ26.CME`, `6EZ26.CME` all return data; **`CLZ20.NYM` returns "Not Found… symbol may be
delisted"**. Live contracts only — so there is no path from yfinance to a correct historical
continuous series.

## 5. Roll yield is most of the return

✅ Live, 2026-09-04: ES Sep26 7733.75 → Dec26 7801.25 = **+0.873%/quarter ≈ +3.54%/yr contango**.
CL Oct26 90.84 → Nov26 87.97 = **−3.16%/month ≈ −32%/yr backwardation**, and CLV26 → CLZ27 is
90.84 → 70.89, **−22% across 15 months**.

🚨 **A spot-price backtest of a long crude strategy misses roughly +30%/yr of roll return.**
"The index returned X" and "a futures position returned X" are different numbers, and for
commodities the gap is usually larger than the alpha being claimed.

### ✅ Measured on real ETFs — the roll cost is not a rounding error

| Rolling ETF | vs the underlying | Gap |
|---|---|---|
| **UNG** −23.51%/yr | NG=F −0.26%/yr | **−23.25 pp/yr** |
| **VIXY** −47.74%/yr | ^VIX −1.84%/yr | **−45.9 pp/yr** |
| GLD −0.50 pp/yr vs GC=F | — | ✅ **the control** — gold is near-zero carry, so the method is not manufacturing the gap |

**A strategy backtested on `^VIX` and traded through `VIXY` loses ~46 percentage points a year to
something that never appears in the price series.** The GLD row is what makes the other two
credible: the same method finds almost no gap where there should be none.

## 6. Roll rules are a second researcher degree of freedom

Calendar (N business days before expiry) · open-interest crossover · volume crossover · first notice
day. ✅ `scripts/roll_schedule.py` shows **a Sharpe spread of 0.71 across four rules on one dataset**
(volume +1.43 vs first_notice +0.72).

🚨 **Each rule is a trial.** Picking the one that backtests best is p-hacking, and first notice day
is a *hard* constraint for physically-settled contracts, not a preference — miss it and you are
liable for delivery. Record the rule in
`../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`.

## 7. 🚨 The CME calendar is not the equity calendar

✅ All executed against `exchange_calendars` 4.13.2:

- **CMES sessions open on the PREVIOUS calendar day.** Session `2026-01-02` has
  `open = 2026-01-01 23:00 UTC`, `close = 2026-01-02 23:00 UTC` — 24 hours spanning two dates.
  **Group intraday futures bars by `.date()` and every evening bar is mis-assigned.**
- **CME trades on 21 days in 2024–2026 when NYSE is shut** (774 vs 753 sessions); the reverse count
  is **0**. **Reindexing futures onto an equity calendar silently deletes 21 sessions of P&L.**
- ⚠️ **`exchange_calendars` does not model the CME daily maintenance halt** — `break_start`/`break_end`
  are `NaT` on every CMES session checked.

Also see the four verified `exchange_calendars` defects in
`../../../fin-asia/skills/asia-pacific-markets/SKILL.md` §3 — including that its default date bounds
move with `Timestamp.now()`.

## 8. Data sources

| Source | Gives you | The catch |
|---|---|---|
| **Databento** | Individual contracts **and** roll rules (`.c.` calendar, `.n.` open interest, `.v.` volume, rank `.0/.1/…`) | ✅ prices are the raw prices of whichever instrument is current — **you do the stitching**. That is the right division of labour |
| **Norgate** | Pre-built continuous series | 🚨 **The Python API cannot see or set the adjustment method.** `price_timeseries()` exposes only `stock_price_adjustment_setting` (an *equity* enum) and `padding_setting`. **The construction is configured in the desktop app (a local service on `localhost:38889`) — outside your code, your requirements file and your git history.** |
| **yfinance** | `CL=F`, `ES=F` | 🚨 unadjusted front-month splice; expired contracts deleted (§4) |
| **openbb** futures | — | 🚨 ✅ source-verified: **it is literally yfinance** under the hood. Same splice, plus AGPL-3.0 |
| 🔴 `quandl` / `nasdaq-data-link` | continuous series | repo **archived**; last release **2022-08-29** |
| IB / `ib_async` | Individual contracts, live | Contract definitions and pacing limits — see `../../../fin-core/skills/broker-execution-apis/references/interactive-brokers.md` |

🔑 **Prefer a source that gives you individual contracts.** A pre-stitched series whose method you
cannot see or record is not reproducible research, however good the data is.

## 9. 🚨 Contract specs — the multiplier does NOT determine tick value

The intuitive shortcut `tick_value = multiplier x tick_size` is wrong often enough to matter.
✅ Measured across 219 futures: **47 distinct multipliers, 28 tick sizes, 13 currencies.**

- **ES and RTY both have a x50 multiplier — but tick values of $12.50 and $5.00.**
- **M6E ticks 2x coarser than 6E**, despite being the micro version.
- **Treasuries tick in 1/32 to 1/256**, not decimals.
- 🚨 **11 contracts carry a `price_magnifier` of 100** — the quoted number is not the price.

**A P&L computed in "points" is meaningless across contracts.** Look the spec up per contract;
do not derive it.

⚠️ **`cmegroup.com` returns HTTP 403 to all programmatic access**, so the specs cannot be scraped —
get them from your broker's contract details (`ib_async` `reqContractDetails`) or your data vendor.

Margin is **not a cost** — it is collateral, and initial ≠ maintenance. Settlement is physical or
cash, and **first notice day binds before last trading day** for physically-settled contracts.
Limit moves halt trading. None of this is in a price series.

## 10. Scripts

- `scripts/continuous_contract.py` — all three stitchings plus `true_roll_return()` ground truth;
  the demo shows the difference series going negative and unadjusted being 19 pp wrong.
- `scripts/roll_schedule.py` — four roll rules, and the Sharpe spread they produce on one dataset.
