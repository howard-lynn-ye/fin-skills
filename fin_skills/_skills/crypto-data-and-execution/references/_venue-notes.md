# Crypto libraries and venue realities

Verified 2026-09-03 against the PyPI JSON API and the GitHub REST API.

## Metadata

| Package | Version | Released | ★ | Licence | Python / wheels |
|---|---|---|---|---|---|
| **`ccxt`** | 4.5.77 | 2026-09-01 | **43,850** | **MIT** | pure python. ~daily releases |
| `cryptofeed` | 2.5.0 | 2026-08-09 | — | — | 🚨 **>=3.12, and NO Windows wheel** (2 wheels, sdist only) |
| `python-binance` | 1.0.37 | 2026-06-08 | — | MIT | pure python |
| **`freqtrade`** | 2026.8 | 2026-08-31 | **53,985** | **GPL-3.0** | pure python, monthly releases |
| `jesse` | 3.1.0 | 2026-09-02 | 8,413 | MIT | pure python |
| `hummingbot` | 20260729 | 2026-07-29 | 19,781 | Apache-2.0 | 🚨 **>=3.10.12, NO Windows wheel** |
| `OctoBot` | 2.1.1 | 2026-03-29 | 6,510 | GPL-3.0 | |
| `nautilus_trader` | 1.231.0 | 2026-08-02 | 28,351 | LGPL-3.0-or-later | 🚨 **>=3.12,<3.15** — win wheels cp312/313/314 only |

**Install reality on Windows + Python 3.11:** `ccxt`, `python-binance`, `freqtrade` and `jesse`
install cleanly. `cryptofeed`, `hummingbot` and `nautilus_trader` **will not**.

## ✅ CCXT Pro is free — and most docs are wrong about it

CCXT Pro was **merged into the MIT `ccxt` package at v1.95 (2022)**. WebSocket methods
(`watchTicker`, `watchOrderBook`, `watchTrades`, `watchOrders`, `watchMyTrades`, …) are **no longer
a paid product**. Any subscription-expiry notice or tutorial telling you to buy it is out of date.

Today, one package, three import surfaces:
```python
import ccxt                  # sync REST
import ccxt.async_support    # async REST
import ccxt.pro              # async + WebSockets
```

Since **v4.5.66** it also covers **prediction markets** (Polymarket, Kalshi, Hyperliquid, Limitless,
Myriad) through the same unified API.

## ccxt gotchas

- **The unified API normalizes common fields only.** Venue-specific behaviour goes through `params`
  passthrough and **implicit (auto-generated) methods**, which are **not portable** — code written
  against `binance.fapiPrivateGetPositionRisk` runs nowhere else.
- **`enableRateLimit` must be on.** It is the only thing between you and a ban.
- **`fetchOHLCV` has per-exchange `limit` caps and inconsistent pagination.** Always loop, pass
  `since`, and **dedupe on timestamp** — several venues return overlapping windows.
- **`set_sandbox_mode(True)` exists but only some exchanges have testnets.** Verify
  `exchange.urls['api']` actually points at a testnet host before assuming you are safe.
- **Precision and rounding rules differ per venue and silently reject orders.** Use
  `exchange.amount_to_precision()` / `price_to_precision()` rather than rounding yourself.
- **Order-status vocabularies are only loosely unified** — `'open'`/`'closed'`/`'canceled'` map
  differently per venue for partially-filled and expired orders.
- 🚨 **ccxt is connectivity, not a backtester.** It has **no simulation layer at all**.

## What crypto breaks in equity-derived tooling

1. **No session, no holiday calendar.** `exchange_calendars` and `pandas_market_calendars` do not
   apply. **Annualization is 365, not 252** — a library defaulting to 252 overstates Sharpe by ~20%.
2. **No corporate actions, but plenty of discontinuities** — token migrations, redenominations,
   chain splits, ticker reuse. The exchange stitches the series with no adjustment record, so there
   is no equivalent of a split factor to check against.
3. **No consolidated tape.** Every venue has its own book and its own price. "The" BTC price does not
   exist; a cross-venue backtest needs explicit venue attribution, and cross-venue spreads are real
   P&L, not data error.
4. **Perpetuals carry funding**, paid or received every 8h (venue-dependent), and it is frequently
   **larger than the alpha being measured**. A perp backtest without funding is not a backtest.
5. **Fees are large and tier-dependent.** In Alpha Arena S1, fees alone consumed **13–17% of capital
   in ~2 weeks**. Maker/taker asymmetry changes strategy viability outright, and tier discounts mean
   your realized fee depends on 30-day volume you have to model.
6. **24/7 means no overnight gap** — strategies keyed to opens, closes or gap statistics have no
   analogue.

## 🚨 Survivorship is worse here than in equities

A pair list built from today's top-volume coins **excludes every coin that died** — and in crypto
that is most of them. Exchanges delist aggressively and silently; a token that went to zero simply
stops appearing in `fetch_markets()`.

`freqtrade`, `jesse`, `OctoBot` and `hummingbot` are all **chronically survivorship-biased in
practice**, because their pair lists are constructed live. Backtesting 2021 on today's pairs is not
a backtest of 2021.

**Guards:** snapshot `fetch_markets()` on a schedule and version the snapshots; reconstruct
historical pair lists from your own archive; when you cannot, say so and treat the result as an
upper bound.

## freqtrade's bias detectors — portable ideas

- **`freqtrade lookahead-analysis`** re-runs a backtest on progressively truncated data and flags
  indicators whose historical values change. Stated limits: only checks **triggered** signals
  (untriggered ones escape as false negatives); can false-positive on limit orders with custom
  pricing callbacks; may falsely flag FreqAI target indicators; useless for rarely-signalling
  strategies.
- **`freqtrade recursive-analysis`** varies `startup_candle_count` and reports each indicator's
  last-row variance versus the base calculation. `-` = converged; `nan%` = insufficient data; large
  percentages mean raise your startup candles. **The only off-the-shelf tool that measures the
  recursive warm-up problem directly.**

Its stated mechanism is the general lesson: *"Backtesting initializes all timestamps (loads the whole
dataframe into memory)"* while live processes candles sequentially. That sentence describes vectorbt,
PyBroker and backtesting.py's `Strategy.I` equally well.

## Order safety

- 🚨 **API keys with trade scope only — never withdraw.** Withdraw-enabled keys are how accounts get
  drained. IP-allowlist where the venue supports it.
- **`stoploss_on_exchange`** (freqtrade) places the stop **at the exchange**, so it survives your
  process dying. **Off by default — the most important line in retail crypto risk management.**
- **Deterministic client order IDs.** Retries on a timed-out order are routine on crypto venues;
  a UUID retry doubles your position, a deterministic ID is rejected as a duplicate.
- Verify a testnet is really a testnet before trusting `set_sandbox_mode`.
