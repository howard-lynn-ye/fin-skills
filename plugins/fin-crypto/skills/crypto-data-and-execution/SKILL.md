---
name: crypto-data-and-execution
description: >-
  Crypto market data and execution, and how a 24/7 market breaks equity tooling. TRIGGER - crypto,
  Bitcoin, BTC, Ethereum, ETH, digital assets, perpetuals, perps, funding rate, basis or
  cash-and-carry, liquidation price, maintenance margin, inverse or coin-margined contract,
  Deribit, crypto options, crypto order book; ccxt, cryptofeed, python-binance, freqtrade, jesse,
  hummingbot, OctoBot; a crypto exchange, testnet or sandbox; annualizing crypto returns, 365 vs
  252; porting an equity strategy to crypto. Annualization is 365 not 252, funding is a carry that
  spot backtests omit, liquidation is not a stop-loss, and exchange pair lists are chronically
  survivorship-biased. SKIP for equity and futures brokers (broker-execution-apis), for RL or
  deep-learning agents even on crypto (rl-and-ml-trading), and for ccxt or freqtrade specifics
  once the library is named (lib-ccxt, lib-freqtrade).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-08"
---

# Crypto data and execution

Crypto inverts several assumptions baked into equity tooling: there is **no session, no holiday
calendar, no consolidated tape, no corporate actions — and a far worse survivorship problem.**

## 1. Pick a tool

| Task | Use | Note |
|---|---|---|
| **Connectivity to any venue** | **`ccxt` 4.5.78** (MIT, 43,917★, ~daily releases) | 100+ venues. **Not a backtester — no simulation layer at all** |
| Live L2/L3 feeds | `cryptofeed` 2.5.0 | 🚨 **Python ≥3.12 and no Windows wheel** (sdist only) |
| Binance-specific | `python-binance` 1.0.37 | MIT |
| **Retail bot, live-first** | **`freqtrade` 2026.8** (GPL-3.0, 54,160★) | 🥇 **best bias-detection tooling in the whole field** — see §4 |
| Backtest + live, MIT | `jesse` 3.1.1 (MIT, 8,435★) | Very active |
| Market making / CEX-DEX | `hummingbot` (Apache-2.0, 19,918★) | 🚨 **no Windows wheel**, py≥3.10.12 |
| GUI-first bot | `OctoBot` | GPL-3.0 |
| Serious event-driven, multi-venue | `nautilus_trader` | LGPL-3.0, **needs Python ≥3.12** |

✅ **CCXT Pro is free.** It was merged into the MIT `ccxt` package at v1.95 (2022) — WebSocket
methods (`watchTicker`, `watchOrderBook`, `watchTrades`, `watchOrders`…) are **no longer a paid
product**. Ignore any subscription-expiry notice or tutorial telling you to buy it. Today:
`ccxt` (sync REST) · `ccxt.async_support` · `ccxt.pro` (async + WS) — one package.
Since v4.5.66 it also covers **prediction markets** (Polymarket, Kalshi, Hyperliquid) through the
same unified API.

## 2. 🚨 Survivorship is worse here than in equities

**A pair list built from today's top-volume coins excludes every coin that died** — and in crypto
that is most of them. Exchanges delist aggressively and silently; a token that went to zero simply
stops appearing in `fetch_markets()`.

`freqtrade`, `jesse`, `OctoBot` and `hummingbot` are all **chronically survivorship-biased in
practice** because their pair lists are constructed live. Backtesting 2021 on today's pairs is not
a backtest of 2021.

**Guards:** snapshot `fetch_markets()` periodically and version it; reconstruct historical pair lists
from your own archived snapshots; state explicitly when you could not, and treat the result as an
upper bound.

## 3. What crypto breaks in equity-derived tooling

1. **No session, no calendar.** `exchange_calendars` and `pandas_market_calendars` do not apply.
   Annualization is **365**, not 252. A library defaulting to 252 **understates** Sharpe and vol by
   16.9% on a calendar-day series, and a calendar-CAGR-over-√252-vol ratio **overstates** by 20.4% (§8).
2. **No corporate actions, but plenty of discontinuities** — token migrations, redenominations,
   chain splits, ticker reuse. Price series are stitched by the exchange with no adjustment record.
3. **No consolidated tape.** Every venue has its own book and its own price. "The" BTC price does not
   exist; cross-venue backtests need explicit venue attribution.
4. **Perpetuals carry funding.** Funding is paid/received every 8h, 4h or 1h — per symbol (§9) — and
   can be larger than the alpha being measured. **A perp backtest without funding is not a backtest.**
5. **Fees are large and tier-dependent.** In Alpha Arena S1, fees alone consumed **13–17% of
   capital** in ~2 weeks. Maker/taker asymmetry changes strategy viability outright.
6. **24/7 means no overnight gap** — strategies keyed to opens/closes have no analogue.

## 4. freqtrade's bias detectors — worth using even from another framework

- **`freqtrade lookahead-analysis`** re-runs a backtest on progressively truncated data and flags
  indicators whose historical values change. Limits it states honestly: only checks **triggered**
  signals; can false-positive on limit orders with custom pricing callbacks.
- **`freqtrade recursive-analysis`** varies `startup_candle_count` and reports each indicator's
  last-row variance — **the only off-the-shelf tool measuring the recursive warm-up problem** (EMA/RSI/
  ADX converge differently depending on how much history preceded them; backtest sees 5,000 candles,
  live sees ~1,000).

Both ideas are portable — see `signal-construction` §5 and `plugins/fin-core/skills/signal-construction/scripts/assert_causal.py`.

## 5. ccxt gotchas

- The unified API normalizes **common fields only**; venue-specific behaviour goes through `params`
  passthrough and implicit methods that are **not portable across exchanges**.
- **`enableRateLimit` must be on.**
- `fetchOHLCV` has per-exchange `limit` caps and inconsistent pagination — always loop and dedupe on
  timestamp.
- **`set_sandbox_mode(True)` exists but only some exchanges have testnets.** Verify
  `exchange.urls['api']` actually points at a testnet host before assuming you are safe.
- Precision and rounding rules differ per venue and **silently reject orders**.
- Order-status vocabularies are only loosely unified.

## 6. Order safety

Everything in `broker-execution-apis` §3 applies. Crypto-specific:
- 🚨 **Create API keys with trade scope only — never withdraw.** Withdraw-enabled keys are how
  accounts get drained.
- **`stoploss_on_exchange`** (freqtrade) places the stop **at the exchange** so it survives your
  process dying. **Off by default; the most important line in retail crypto risk management.**
- IP-allowlist the key where the venue supports it.
- Deterministic client order IDs — retries on a timed-out order are routine on crypto venues.

## 7. Reference files

`references/<library>.md` for versions, licences, venue coverage and quirks.

## 8. 🚨 Annualisation: the factor is rows per year, and the error runs both ways

✅ `scripts/perp_mechanics.py` §1, one synthetic calendar-day series (1,460 rows, seed 0):

| Statistic | √365 (right) | √252 (wrong) | Error |
|---|---|---|---|
| Sharpe (mean/std·√P) | 0.6903 | 0.5736 | **understated 16.9%** — √(252/365) = 0.8309 |
| Annual vol | 0.4733 | 0.3933 | understated 16.9% |
| Mean × periods | 32.67%/yr | 22.56%/yr | understated 31.0% |
| Calendar CAGR (23.94%) ÷ annual vol | 0.5057 | 0.6086 | **overstated 20.4%** |

**Which way the 252 error goes depends on the formula.** `mean/std·√252` understates; a ratio whose
numerator is annualised by calendar time and whose denominator is √252 vol overstates by √(365/252)
− 1 = 20.4%. ✅ Live: quantstats 0.0.81 and empyrical 0.5.5 both default to 252 and reproduce the
understated 0.5736 — pass `periods=365` / `annualization=365`. ✅ quantstats 0.0.81's `cagr()` uses
`years = len(returns) / periods`, not calendar time: a 365-row year is 1.45 "years" at the default,
and it returns 15.97% for a series whose calendar CAGR is 23.94%.

**The mistake in the other direction.** A strategy flat at weekends, kept as weekday rows only
(1,042 of them), annualised with √365 overstates its Sharpe by 20.4% (0.9407 vs 0.7816). The same
P&L padded with weekend zeros (1,460 rows) and annualised with √365 gives 0.7946 — the weekday/√252
answer to 0.0129. **The factor is the number of rows per year in the series you hand the function:**
365 for calendar days, 252 for weekday-only rows, 8,760 for hours, 1,095 for 8h funding intervals.

## 9. 🚨 Funding is the carry a spot backtest omits

The crypto twin of `../../../fin-futures-fx/skills/fx-markets/SKILL.md` §1: a spot series has no
carry, a position does. For a perp the carry is funding, exchanged between longs and shorts at every
settlement. `payment = notional × rate` per interval. ✅ OKX docs (`public/funding-rate`): **positive
rate = longs pay shorts; negative = shorts pay longs.** It is charged on **notional**, so leverage
multiplies it against margin: $100,000 at 0.01% per 8h is $10 per settlement, $30/day, $900 (0.90%)
per 30 days — **9.0% of margin per month at 10x**.

🚨 **The interval is per symbol, not "8h".** ✅ OKX, 2026-09-08, all 644 live swaps: 357 settle every
8h, 286 every 4h, 1 every 1h. OKX's docs say to read `nextFundingTime − fundingTime`; Bybit's docs:
*each symbol has a different funding interval*, read `instruments-info`. ⚠️ Binance's docs and API
answered HTTP 202/451 from here — nothing about Binance is verified in this file.

✅ **A real year** — Deribit BTC-PERPETUAL, hourly `interest_1h` summed per UTC day, 2025-09-08 →
2026-09-07: a long paid **2.77% of notional** while the index fell 29.0%; 280 of 365 days positive;
worst day 7.36 bps, best −3.39 bps. The largest 30-day bill was **0.825%** (window ending 2025-10-28)
against a +1.15% price move; |funding| exceeded |price move| in 10 of 336 windows and flipped a
long's sign in 3. Median 30-day |move| 7.97% vs median 30-day funding 0.128%: **in this low-funding
year funding was a tax, rarely the whole story — and still 27.7% of margin at 10x.** ✅ OKX
BTC-USDT-SWAP (linear), last 92 days: 1.11% of notional = 4.40%/yr, price +23.1%; 29 of 277
settlements printed exactly 0.0100%, the instrument's `interestRate` field, and none exceeded it (⚠️
the premium-plus-clamped-interest formula predicts exactly that mass point). ✅ OKX caps each
settlement at ±0.375% — pinned there, 33.75% of notional per month, a hypothetical. ⚠️ Illustrative,
not observed in either sample: a bull-market path at 0.05% per 8h costs a long 4.5% of notional in
30 days, 45% of margin at 10x. Fetch the history; do not assume the regime.

**A perp backtest is price return minus funding paid (long) or plus funding received (short),
settlement by settlement, on the notional at each settlement.** ✅ `fetch_funding_rate_history` is
`True` in ccxt 4.5.78's `has` for binance, bybit, okx and deribit, `False` for coinbase and kraken;
the base class raises `NotSupported`. Depth served: ✅ OKX 3 months (documented), Deribit ≥ 420
days hourly (pulled today), Bybit 200 rows/call (documented). **Archive it yourself —
the venue's window is shorter than your backtest.**

## 10. Basis and cash-and-carry

A dated future trades away from spot by the **basis**, which is what a perp's funding stream
approximates. `annualised = (F/S − 1) × 365/days` (simple) or `(F/S)^(365/days) − 1`. ✅ Deribit
marks, 2026-09-08 13:25 UTC, index 78,449.67:

| Contract | Mark | Days | Basis | Simple /yr |
|---|---|---|---|---|
| BTC-25SEP26 | 78,561.31 | 16.8 | 0.142% | 3.10% |
| BTC-25DEC26 | 79,513.65 | 107.8 | 1.356% | 4.59% |
| BTC-26MAR27 | 80,458.36 | 198.8 | 2.560% | 4.70% |
| BTC-25JUN27 | 81,460.24 | 289.8 | 3.838% | 4.83% |

✅ OKX BTC-USD-261225 minutes later: 1.444% → 4.89%/yr. Two venues, one curve, both near the
2.77%/yr the perp actually paid (§9).

**The trade:** buy spot, sell the dated future, hold to expiry, collect the basis whatever the price
does. On $100,000 via BTC-25DEC26: $1,356 locked, $35 futures taker fee (✅ 0.035%), 4.47%/yr net of
that fee. The risks are not in the arithmetic:

- **Margin on the short leg.** A 30% rally marks the short −$30,000 and a linear venue wants it in
  cash now; the spot gain is a coin, not cash. At 3x isolated (mmr 0.4%) the short is liquidated at
  +32.8%, at 5x at +19.5% — the hedge becomes a naked long. On an **inverse** future with BTC as
  collateral a 1x short is a synthetic dollar and cannot be liquidated (§12).
- **Exchange risk** — both legs and the collateral sit on venues (§13).
- **A perp instead of the dated future** does not lock anything: the carry is the funding stream,
  sign-changing, never expiring into the basis. This is the roll analogue in
  `../../../fin-futures-fx/skills/futures-continuous-contracts/SKILL.md` §5 — a dated future rolls
  and earns roll yield, a perp never rolls and pays funding. Neither is in a spot series.

## 11. 🚨 Liquidation is not a stop-loss

Maintenance margin is the equity floor below which the venue closes you. ✅ OKX BTC-USDT-SWAP tier 1
(positions up to 10 BTC = $784,622 at the fetched index): mmr 0.4%, initial 1%, max 100x; tiers rise
with size. Isolated linear position, fees ignored: long liquidates at
`(1 − 1/L)/(1 − mmr)`, short at `(1 + 1/L)/(1 + mmr)`, bankruptcy at ∓1/L. ✅ Computed:

| Leverage | Long liq | Short liq | Bankruptcy | Margin left at liquidation |
|---|---|---|---|---|
| 3x | −33.07% | +32.80% | ∓33.33% | 0.8% |
| 5x | −19.68% | +19.52% | ∓20.00% | 1.6% |
| 10x | −9.64% | +9.56% | ∓10.00% | 3.6% |

The simplified `entry × (1 − 1/L + mmr)` differs by a few bps; funding and fees move it more.

Why it is not a stop: **(1) you keep almost nothing** — at 10x it fires with 3.6% of margin left.
**(2) A liquidation fee** — ✅ Deribit's instrument spec carries `max_liquidation_commission = 0.01`:
1% of position value = 10% of margin at 10x, more than remains, so the loss is 100% of margin; an
on-exchange stop at the same price pays 0.035% × 10 = 0.35% and keeps 3.3%. **(3) The price is not
yours** — ✅ OKX docs: `liqPx` is the *estimated mark price at which this position would be forcibly
liquidated*, and it *can change quickly due to funding rate accrual*: the trigger is the mark, not
the last trade, and the engine's market order fills wherever the book is. **(4) Funding walks the
line toward you** — a 10x long paying 0.01% per 8h for 30 days sees its liquidation move from −9.64%
to −8.73% with the price unchanged. **(5) Cascades** — every liquidation is a market order into the
next cluster of liquidation levels; the mechanism is impact,
`../../../fin-core/skills/execution-cost-analysis/SKILL.md` §5, and a fixed-slippage backtest cannot
see it. `stoploss_on_exchange` (§6) is the retail defence; sizing so the liquidation price sits far
outside the stop is the professional one.

## 12. Crypto options and inverse contracts

`../../../fin-core/skills/derivatives-pricing/SKILL.md` prices them,
`../../../fin-core/skills/options-backtesting/SKILL.md` owns the lifecycle; this is only what is
crypto-specific. ✅ Deribit `public/get_instruments`, 2026-09-08:

- **BTC-PERPETUAL**: `contract_size` 10 USD, settled in BTC, `instrument_type: reversed`, max
  leverage 50x, maker 0.015% / taker 0.035%.
- **BTC options**: 908 listed, every one settled in BTC, contract 1 BTC, tick 0.0001 BTC, 11
  expiries all at 08:00 UTC. ⚠️ European, cash-settled to the index — not fetched today.
- **USDC-settled linear options exist too** — 3,538 listed across SOL, BTC, ETH, HYPE, XRP, AVAX and
  TRX — and `BTC_USDC-PERPETUAL` is linear. "Deribit is inverse" is no longer the whole story.

**Inverse P&L lives in the coin.** Long N USD of an inverse contract: P&L = N(1/P₀ − 1/P₁) BTC (✅ the
formula OKX's docs print for inverse `upl`). ✅ $100,000 at 78,449.67, 1x:

| Move | Linear, USD | Inverse, BTC | Inverse + BTC collateral, USD | 1x inverse short + BTC |
|---|---|---|---|---|
| −20% | −20,000 | −0.3187 | −40,000 | 0 |
| +20% | +20,000 | +0.2125 | +40,000 | 0 |

The USD column is exactly **2× linear**: the collateral is itself long BTC, so a "1x long" on an
inverse venue is a 2x long in dollars. The BTC column is asymmetric — gains cap at N/P₀ = 1.2747 BTC,
losses are unbounded as the price falls. A **1x inverse short with BTC collateral is a synthetic
dollar**: equity constant at every price, no liquidation level at any maintenance rate. A BTC-settled
call pays `(S − K)/S` BTC: 0.2 BTC at S = 100,000 for K = 80,000, approaching 1 BTC and never
exceeding it, and the premium you paid in BTC is worth more dollars exactly when the call wins.

## 13. 🚨 Venue and counterparty risk is a first-order cost

- ⚠️ **Exchanges fail with client assets on them**: Mt. Gox (2014), QuadrigaCX (2019), FTX (2022 —
  withdrawals halted, then bankruptcy), Bybit's hot-wallet theft (2025). Short withdrawal pauses at
  solvent venues are routine. A backtest with all its capital on one venue has an unmodelled tail.
- ⚠️ **Proof-of-reserves shows assets at a snapshot, not liabilities**, and the accounting firm that
  produced several of the 2022 attestations withdrew from the work the same year.
- ✅ **Access is jurisdictional.** From this location on 2026-09-08 Binance's API answered HTTP 451
  and Bybit's 403; OKX, Deribit, Coinbase, Kraken, Bitstamp and Gemini answered 200. A strategy that
  assumes a venue is assuming it will serve you.
- **Survivorship has two layers** — dead tokens (§2) and dead venues. Every bar is one venue's book;
  attribute it, and expect the venue with the best 2019 data not to exist.
- Keys: §6. Collateral: keep on-venue only what the strategy needs; the rest is counterparty exposure
  with no coupon.

## 14. Data: there is no official close, and "daily" means three things

✅ The close of 2026-09-07 as each venue's 1D candle labels it: Coinbase 79,091.97, Kraken 79,090.30,
Bitstamp 79,090.41, Gemini 79,101.21 — 00:00 UTC bars, 1.4 bps apart. **OKX 78,834.10**: its `1D` bar
opens at 00:00 UTC+8, i.e. 16:00 UTC the previous day (✅ OKX docs — `1D` is UTC+8, `1Dutc` is UTC+0).
**Deribit 78,453.50**: its 1D chart bars open at 08:00 UTC, its settlement hour. That is −32.6 and
−80.7 bps from Coinbase's for the same date, and none of it is a venue difference. ✅ Across 60 closed
UTC days (Coinbase/Kraken/Gemini) the cross-venue close range had median 1.77 bps, max 4.30 bps —
small, non-zero, USD venues only; USDT pairs add the stablecoin premium. **"00:00 UTC" is a
convention you choose; a backtest of "the daily close" must say whose and when** — the crypto
restatement of `fx-markets` §4.

Funding history: ✅ OKX `GET /api/v5/public/funding-rate-history` (3 months), Deribit
`public/get_funding_rate_history` (hourly, ≥ 420 days), Bybit `GET /v5/market/funding/history` (200
rows/call, docs). Binance `GET /fapi/v1/fundingRate` ⚠️ unverifiable from here. Candles: every
venue's last bar is unclosed (✅ OKX marks it `confirm=0`) — `lib-ccxt` covers `fetch_ohlcv`.

## 15. Scripts and cross-links

`scripts/perp_mechanics.py` — every number in §8–§14 from constants fetched 2026-09-08 (endpoint
named beside each) and one seeded synthetic series; verifies §8 against quantstats and empyrical
when they are installed. No network at run time.

- `../../../fin-futures-fx/skills/fx-markets/SKILL.md` — §1 the carry a spot backtest omits (FX twin
  of §9), §4 no official close
- `../../../fin-futures-fx/skills/futures-continuous-contracts/SKILL.md` §5 — roll yield; a dated
  future rolls, a perp funds
- `../../../fin-core/skills/derivatives-pricing/SKILL.md` — pricing, Greek scaling, vol surfaces
- `../../../fin-core/skills/options-backtesting/SKILL.md` — assignment, expiry and margin lifecycle
- `../../../fin-core/skills/execution-cost-analysis/SKILL.md` §5 — impact, the cascade mechanism
- `../../../fin-core/skills/broker-execution-apis/SKILL.md` §3 — order safety (§6 here)
- `../../../fin-core/skills/signal-construction/SKILL.md` §5 — causal indicators (§4 here)
- `../../../fin-libraries/skills/lib-ccxt/SKILL.md` and
  `../../../fin-libraries/skills/lib-freqtrade/SKILL.md` — the per-library deep dives
