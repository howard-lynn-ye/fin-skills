# ccxt

The universal connectivity layer for crypto — one unified API over 100+ venues, MIT, and **not a
backtester**. It has no simulation layer at all; everything it returns is live venue state.

| | |
|---|---|
| pip | `ccxt` · **4.5.77 (2026-09-01)** · **1,617 releases** — roughly daily |
| GitHub | `ccxt/ccxt` — **43,863★**, 846 open issues, pushed 2026-09-04 ✅ |
| Licence | **MIT** ✅ (GitHub API `license.spdx_id`). ⚠️ PyPI `info.license` is null and there is **no licence classifier** — do not read PyPI metadata as the answer |
| Python | `requires_python >=3.10` ✅ · pure-python wheel (`py3-none-any`) + sdist — installs anywhere |
| Maintenance | ✅ Extremely active. Multi-language repo (JS/TS/Python/C#/PHP/Go) generated from one source |

## ✅ CCXT Pro is free — most tutorials are wrong about this

CCXT Pro was **merged into the MIT `ccxt` package at v1.95 (2022)**. WebSocket methods
(`watchTicker`, `watchOrderBook`, `watchTrades`, `watchOrders`, `watchMyTrades`…) are **no longer a
paid product**. Ignore any subscription-expiry notice or tutorial telling you to buy it.

```python
import ccxt                  # sync REST
import ccxt.async_support    # async REST
import ccxt.pro              # async + WebSockets  (the ex-"Pro" surface)
```

Since **v4.5.66** the same unified API also covers **prediction markets** — Polymarket, Kalshi,
Hyperliquid, Limitless, Myriad.

## 🚨 Traps

**1. 🚨 It is connectivity, not a backtester.** There is **no simulation layer**, no fills, no fees
applied, no portfolio. Anything that looks like a ccxt backtest is code someone wrote around it.
Pair it with an engine — see `../../../../fin-core/skills/backtesting-engines/SKILL.md`.

**2. 🚨 `fetchOHLCV` silently returns fewer candles than you asked for.** Every venue caps `limit`
differently (commonly 500 or 1000) and paginates inconsistently. Ask for 5,000 daily candles and you
get the cap, with **no error and no warning** — a truncated series that quietly becomes a shorter
backtest. Always loop on `since`, and **dedupe on timestamp**: several venues return overlapping
windows, and duplicate bars inflate bar counts and break `pct_change`.

**3. 🚨 The last candle is unclosed.** `fetchOHLCV` returns the in-progress bar as the final row. Use
it in a signal and you are reading a partially-formed close that will change — a live-only look-ahead
that never appears in a backtest. **Drop the last row unless you have verified it is closed.**

**4. 🚨 `set_sandbox_mode(True)` is not a guarantee.** Only some exchanges have testnets. The call
succeeds regardless. **Verify the resolved host before trading:**
`assert "test" in exchange.urls["api"]["public"]` — assert on the URL the client actually resolved,
never on your own config flag.

**5. 🚨 Precision and rounding differ per venue and silently reject orders.** Amount step, price tick
and min-notional are venue rules. Rounding yourself produces a rejection you may only notice as a
missing fill. Use `exchange.amount_to_precision(sym, amt)` and `exchange.price_to_precision(sym, px)`
after `load_markets()`.

**6. 🚨 `fetch_markets()` is today's listings.** Every delisted or dead token is absent, so any pair
list built from it is **survivorship-biased** — and in crypto most tokens die. Snapshot and version
`fetch_markets()` on a schedule; reconstruct historical universes from your own archive. See
`_venue-notes.md`.

**7. 🚨 Perp funding is not in OHLCV.** Funding is paid or received every ~8h and is frequently larger
than the alpha being measured. It must be fetched separately; ❓ the unified funding-history method is
not implemented on every venue — check `exchange.has` before relying on it.

**8. ⚠️ The unified API normalizes common fields only.** Venue-specific behaviour goes through
`params` passthrough and **implicit (auto-generated) methods** — code written against
`binance.fapiPrivateGetPositionRisk` runs on exactly one exchange. Portable code checks
`exchange.has[...]` first.

**9. ⚠️ Order-status vocabularies are only loosely unified.** `'open'` / `'closed'` / `'canceled'` map
differently per venue for **partially filled** and **expired** orders. Reconcile on filled quantity,
not on the status string.

**10. ⚠️ `enableRateLimit` is the only thing between you and a ban.** Set it explicitly rather than
trusting the default, and note the built-in limiter is a client-side sleep — it does not know about
your other processes sharing the same key.

## Minimal correct usage — dangerous defaults set explicitly

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

## Where it fits

- Data + execution for anything crypto: `../SKILL.md`
- Venue/licence/install matrix and the crypto-vs-equities differences: `_venue-notes.md`
- `freqtrade` uses ccxt underneath — its exchange quirks are ccxt quirks: `freqtrade.md`
- Order-safety patterns (trade-scope-only keys, idempotent client order IDs, paper assertions):
  `../../../../fin-core/skills/broker-execution-apis/SKILL.md` §3
