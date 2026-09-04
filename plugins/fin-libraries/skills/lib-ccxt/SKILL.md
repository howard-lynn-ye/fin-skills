---
name: lib-ccxt
description: >-
  The unified MIT client for 100+ crypto venues - and not a backtester, with an OHLCV endpoint
  that silently truncates and returns an unclosed final bar. TRIGGER - import ccxt, import
  ccxt.pro, import ccxt.async_support, pip install ccxt, fetch_ohlcv, fetchOHLCV, load_markets,
  fetch_markets, create_order, watchOrderBook, watchTicker, watchMyTrades, set_sandbox_mode,
  enableRateLimit, amount_to_precision, price_to_precision, exchange.has, options defaultType,
  parse8601, implicit methods like fapiPrivateGetPositionRisk, CCXT Pro subscription expiry,
  funding rate history; an order rejected on precision or min-notional, fewer candles returned
  than requested. Memory is stale here: CCXT Pro was merged into the free MIT package at v1.95,
  prediction markets landed at 4.5.66, and 4.5.77 shipped 2026-09-01. SKIP for equity and futures
  brokers (broker-execution-apis). SKIP when the question is WHICH library to choose rather than
  how to use this one - that belongs to the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# ccxt

The universal connectivity layer for crypto — one unified API over 100+ venues, MIT, and **not a
backtester**. It has no simulation layer at all; everything it returns is live venue state.

| | |
|---|---|
| pip / import | `pip install ccxt` · `import ccxt` / `ccxt.async_support` / `ccxt.pro` |
| Version | **4.5.77 (2026-09-01)** · **1,617 releases** — roughly daily · `>=3.10`, pure-python wheel |
| Licence | **MIT** ✅ (GitHub `license.spdx_id`). ⚠️ PyPI `info.license` is null with no classifier — do not read PyPI metadata as the answer |
| Status | ✅ Extremely active. 43,863★ `ccxt/ccxt`, pushed 2026-09-04. Multi-language repo generated from one source |

## The trap that costs you money

🚨 **`set_sandbox_mode(True)` is not a guarantee.** Only some exchanges have testnets; **the call
succeeds regardless.** Verify the host the client actually resolved, never your own config flag:

```python
ex.set_sandbox_mode(True)
assert "test" in ex.urls["api"]["public"], f"NOT a testnet: {ex.urls['api']}"
```

🚨 **And it is connectivity, not a backtester.** No fills, no fees applied, no portfolio, no
simulation layer. Anything that looks like a ccxt backtest is code someone wrote around it.

## ✅ CCXT Pro is free — most tutorials are wrong about this

CCXT Pro was **merged into the MIT `ccxt` package at v1.95 (2022)**. WebSocket methods
(`watchTicker`, `watchOrderBook`, `watchTrades`, `watchOrders`, `watchMyTrades` …) are **no longer a
paid product**. Ignore any subscription-expiry notice or tutorial telling you to buy it.

```python
import ccxt                  # sync REST
import ccxt.async_support    # async REST
import ccxt.pro              # async + WebSockets  (the ex-"Pro" surface)
```

Since **v4.5.66** the same unified API also covers **prediction markets** — Polymarket, Kalshi,
Hyperliquid, Limitless, Myriad.

## 🚨 `fetch_ohlcv` has three silent data bugs

1. **It returns fewer candles than you asked for.** Every venue caps `limit` differently (commonly 500 or 1000) and paginates inconsistently. Ask for 5,000 daily candles and you get the cap, with **no error and no warning** — a truncated series that quietly becomes a shorter backtest. Loop on `since`, and **dedupe on timestamp**: several venues re-return overlapping windows, and duplicate bars inflate bar counts and break `pct_change`.
2. **The last candle is unclosed.** The in-progress bar comes back as the final row. Use it in a signal and you are reading a partially-formed close that will change — a live-only look-ahead that never appears in a backtest. **Drop the last row unless you have verified it is closed.**
3. **Perp funding is not in OHLCV.** Funding is paid or received every ~8h and is frequently larger than the alpha being measured. Fetch it separately; ❓ the unified funding-history method is not implemented on every venue — check `exchange.has` first.

## 🚨 Precision, listings, and status vocabularies

**Precision and rounding are venue rules and silently reject orders.** Amount step, price tick and min-notional differ per venue; rounding yourself produces a rejection you may only notice as a missing fill. Use `exchange.amount_to_precision(sym, amt)` and `exchange.price_to_precision(sym, px)` **after `load_markets()`**.

**`fetch_markets()` is today's listings.** Every delisted or dead token is absent, so any pair list built from it is **survivorship-biased** — and in crypto most tokens die. Snapshot and version `fetch_markets()` on a schedule; reconstruct historical universes from your own archive.

⚠️ **Order-status vocabularies are only loosely unified.** `'open'` / `'closed'` / `'canceled'` map differently per venue for **partially filled** and **expired** orders. **Reconcile on filled quantity, not on the status string.**

## ⚠️ Unified only where it says it is

The unified API normalizes common fields only. Venue-specific behaviour goes through `params` passthrough and **implicit (auto-generated) methods** — code written against `binance.fapiPrivateGetPositionRisk` runs on exactly one exchange. **Portable code checks `exchange.has[...]` first.**

⚠️ **`enableRateLimit` is the only thing between you and a ban.** Set it explicitly rather than trusting the default, and note the built-in limiter is a client-side sleep — it does not know about your other processes sharing the same key.

## Minimal correct call

```python
import ccxt

ex = ccxt.binance({
    "enableRateLimit": True,        # explicit: never rely on the default
    "options": {"defaultType": "spot"},
})
ex.set_sandbox_mode(True)
assert "test" in ex.urls["api"]["public"], f"NOT a testnet: {ex.urls['api']}"
ex.load_markets()                    # required before any precision helper

since, rows = ex.parse8601("2024-01-01T00:00:00Z"), []
while True:
    batch = ex.fetch_ohlcv("BTC/USDT", timeframe="1h", since=since, limit=1000)
    if not batch:
        break
    rows += batch
    since = batch[-1][0] + 1         # +1ms: several venues re-return the `since` bar
    if len(batch) < 1000:
        break
rows = list({r[0]: r for r in rows}.values())   # dedupe on timestamp
rows = rows[:-1]                                # drop the unclosed final candle
```

## See also

- `../../../fin-crypto/skills/crypto-data-and-execution/SKILL.md` — crypto data and execution overall
- `../../../fin-crypto/skills/crypto-data-and-execution/references/ccxt.md` — the source card
- `../../../fin-crypto/skills/crypto-data-and-execution/references/_venue-notes.md` — venue, licence and install matrix
- `../../../fin-core/skills/broker-execution-apis/SKILL.md` §3 — order-safety patterns (scoped keys, idempotent IDs)

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`broker-execution-apis`** (`../../../fin-core/skills/broker-execution-apis/SKILL.md`).

