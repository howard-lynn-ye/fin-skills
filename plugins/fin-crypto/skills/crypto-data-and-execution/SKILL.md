---
name: crypto-data-and-execution
description: >-
  Get crypto market data and trade it, and correct for the ways crypto breaks tooling built for
  equities. Covers ccxt (and why ccxt.pro is now free), cryptofeed, python-binance, freqtrade,
  jesse, hummingbot, OctoBot, and exchange/testnet realities. TRIGGER — use for any task involving
  crypto, digital assets, Bitcoin, Ethereum, perpetual futures, funding rates, order books, or
  exchange connectivity; when choosing a crypto data source, backtester or trading bot; or when
  porting an equity strategy to a 24/7 market.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# Crypto data and execution

Crypto inverts several assumptions baked into equity tooling: there is **no session, no holiday
calendar, no consolidated tape, no corporate actions — and a far worse survivorship problem.**

## 1. Pick a tool

| Task | Use | Note |
|---|---|---|
| **Connectivity to any venue** | **`ccxt` 4.5.77** (MIT, 43,850★, ~daily releases) | 100+ venues. **Not a backtester — no simulation layer at all** |
| Live L2/L3 feeds | `cryptofeed` 2.5.0 | 🚨 **Python ≥3.12 and no Windows wheel** (sdist only) |
| Binance-specific | `python-binance` 1.0.37 | MIT |
| **Retail bot, live-first** | **`freqtrade` 2026.8** (GPL-3.0, 53,985★) | 🥇 **best bias-detection tooling in the whole field** — see §4 |
| Backtest + live, MIT | `jesse` 3.1.0 (MIT, 8,413★) | Very active |
| Market making / CEX-DEX | `hummingbot` (Apache-2.0, 19,781★) | 🚨 **no Windows wheel**, py≥3.10.12 |
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
   Annualization is **365**, not 252. A library defaulting to 252 will overstate your Sharpe by
   ~20%.
2. **No corporate actions, but plenty of discontinuities** — token migrations, redenominations,
   chain splits, ticker reuse. Price series are stitched by the exchange with no adjustment record.
3. **No consolidated tape.** Every venue has its own book and its own price. "The" BTC price does not
   exist; cross-venue backtests need explicit venue attribution.
4. **Perpetuals carry funding.** Funding is paid/received every 8h (venue-dependent) and is often
   larger than the alpha being measured. **A perp backtest without funding is not a backtest.**
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
