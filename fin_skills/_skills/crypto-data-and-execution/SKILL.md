---
name: crypto-data-and-execution
description: >-
  TRIGGER - choose a crypto data or exchange client, Bitcoin or Ethereum OHLCV, crypto
  order book feeds, ccxt, cryptofeed, python-binance, freqtrade, jesse, hummingbot,
  OctoBot; exchange API pagination, testnet, sandbox, precision, retries, or order safety.
  SKIP for token migrations, rebases and delisted universes (crypto-token-events), perpetual
  funding and mark-price liquidation (perpetuals-and-funding), AMM pools and impermanent loss
  (defi-and-amm-mechanics), 365-day annualisation, venue outages and stablecoin depegs
  (crypto-market-structure), equity brokers (broker-execution-apis), RL agents
  (rl-and-ml-trading), and named-library implementation details (lib-ccxt, lib-freqtrade).
license: MIT
compatibility: Python 3.10+; offline examples use numpy and pandas; optional exchange clients.
metadata:
  version: "0.1.0"
  verified_on: "2026-09-08"
allowed-tools: Read Bash(python:*)
---

# Crypto data and execution

Use this as the entry point for sourcing crypto data and selecting an execution client.
The library/version table and venue observations below retain their **2026-09-08** verification
 date; they are historical observations, not current package or venue guarantees.

✅ Routing and the separation of REST/WebSocket connectivity were reviewed on 2026-09-14 against
[CCXT's own README](https://raw.githubusercontent.com/ccxt/ccxt/master/README.md).
Installing a client does not itself start a collector or subscribe to a feed.

| When the task concerns | Continue with |
|---|---|
| Rebases, token swaps, forks, wrapped assets, historical pairs | `../crypto-token-events/SKILL.md` |
| Funding, mark/index/last, margin, liquidation, open interest | `../perpetuals-and-funding/SKILL.md` |
| AMM impact, LP fees, concentrated ranges, impermanent loss | `../defi-and-amm-mechanics/SKILL.md` |
| Calendar rows, venue dispersion, outages, ADL, quote depegs | `../crypto-market-structure/SKILL.md` |

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

## Scripts and references

`scripts/perp_mechanics.py` is retained for existing imports, including
`fin_skills.api.conventions`; its constants remain dated historical examples. New focused
mechanics are in the sibling skills' scripts. Do not equate these offline demonstrations with
a live market-data collector.

- `references/ccxt.md` and `references/freqtrade.md` retain the library research.
- `references/_venue-notes.md` retains venue-specific observations.
- `references/2026-09-08-research-snapshot.md` preserves the former full entrypoint, including
  its original tables, calculations and cross-references. Current sibling guidance supersedes
  the broad claims corrected during the split.
- `../../../fin-core/skills/broker-execution-apis/SKILL.md` — execution safety.
- `../../../fin-core/skills/signal-construction/SKILL.md` — causal indicators.
- `../../../fin-libraries/skills/lib-ccxt/SKILL.md` and
  `../../../fin-libraries/skills/lib-freqtrade/SKILL.md` — named-library details.
