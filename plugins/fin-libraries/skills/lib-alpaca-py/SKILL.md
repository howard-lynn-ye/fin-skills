---
name: lib-alpaca-py
description: >-
  Alpaca's current Python SDK, which defaults to the paper host but lets url_override silently
  send live orders from a client that believes it is in the sandbox. TRIGGER - import alpaca, pip
  install alpaca-py, TradingClient, StockHistoricalDataClient, CryptoHistoricalDataClient,
  submit_order, LimitOrderRequest, MarketOrderRequest, OrderSide, TimeInForce, client_order_id,
  url_override, paper=True, BaseURL.TRADING_PAPER, paper-api.alpaca.markets, bracket OCO OTO
  orders, trail_percent, IEX vs SIP feed, Algo Trader Plus, alpaca-trade-api, APCA_API_BASE_URL;
  an order rejected asynchronously for time-in-force or price precision. Memory is stale here:
  alpaca-trade-api was deprecated in 2024 and defaulted to LIVE, whereas alpaca-py 0.44.0
  (2026-08-11) declares paper=True - the widely repeated warning is inverted. SKIP for Interactive
  Brokers and for the general order-safety patterns (broker-execution-apis). SKIP for choosing
  between libraries - that is the domain skill's job.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# alpaca-py

The easiest US broker to automate — REST, real paper trading, free data — and the one whose rules
change silently depending on **which asset class** you point them at.

| | |
|---|---|
| pip / import | `pip install alpaca-py` · `import alpaca` |
| Version | **0.44.0 (2026-08-11)** · `>=3.10,<4.0`, classifiers through 3.14 · pure-Python wheel |
| Licence | ✅ **Apache-2.0** |
| Status | ✅ Active. 1,481★ `alpacahq/alpaca-py`, pushed 2026-09-03. 🔴 Legacy `alpaca-trade-api` 3.2.0 (2024-01-12) is deprecated |

## The trap that costs you money

🚨 **`url_override` bypasses the `paper` flag.** It replaces the base URL outright while `sandbox=paper` is still set from the flag — so `url_override=<live host>, paper=True` sends **live orders from a client that believes it is in the sandbox.** Never combine the two.

🚨 **And the advice you remember is inverted.** Paper and live are separate key pairs *and* separate hosts (`paper-api.alpaca.markets` vs `api.alpaca.markets`); keys are not portable.

- ✅ Verified in `alpaca/trading/client.py`: `TradingClient.__init__` declares **`paper: bool = True`**, resolving `url_override if url_override else BaseURL.TRADING_PAPER if paper else BaseURL.TRADING_LIVE`. **Omitting `paper` in `alpaca-py` gives you PAPER.**
- ✅ Verified in `alpaca_trade_api/common.py`: the legacy library's `get_base_url()` returns `os.environ.get('APCA_API_BASE_URL', 'https://api.alpaca.markets')` — **live by default, with no `paper` argument anywhere.** `REST(key, secret)` with live keys is a live session.

**The warning "leaving `paper` unset trades for real" is true of the LEGACY library only.** If you are porting old code or old blog snippets, that inversion is the bug. Assert on a server-returned fact, not on the flag — see the minimal call below.

## 🚨 Time-in-force varies by asset class

The full menu is `day`, `gtc`, `opg` (opening auction), `cls` (closing auction), `ioc`, `fok` — but
almost none of it applies to every product:

| Product | Accepted TIF |
|---|---|
| **Crypto** | 🚨 **`gtc` and `ioc` only** |
| **Options, OTC** | 🚨 **`gtc` and `day` only** |
| **Fractional shares** | 🚨 **`day` only** |
| **Extended hours** | 🚨 Requires a **limit** order with `day` or `gtc` — market orders not accepted |
| Whole-share equities | Full menu; ⚠️ `ioc`/`fok` gated ("contact sales") |

A `TimeInForce.DAY` crypto order and a `TimeInForce.IOC` fractional order are both rejections, not fills — **and the rejection arrives asynchronously**, so a naive loop just sees nothing happen. `gtc` additionally carries a **90-day expiration policy**. **Trailing stops** (`trail_price` / `trail_percent`) 🚨 **trigger only during regular market hours** — an overnight gap does not move them.

## 🚨 Price precision

**2 decimals at or above $1.00, 4 decimals below.** Violating orders are **rejected**. Anything that computes a limit from a float — a percentage offset, an ATR multiple, a mid-price — needs an explicit round for that price band, and **the band is per-order, not per-symbol**.

## `client_order_id` is the idempotency primitive

Caller-supplied and unique; auto-generated if omitted. **Omitting it removes your only defence against the duplicate-order failure mode:** you submit, the response times out, you retry, and you now hold 2× the position. Networks and broker gateways make that routine.

Derive it from strategy state, never from a clock or a UUID:

```python
coid = f"{strategy_id}-{symbol}-{bar_ts.isoformat()}-{side}"   # replay-safe
```

A retry then either succeeds or is rejected as a duplicate — both outcomes are safe. A UUID retry doubles your position. Reconcile on `client_order_id` against the list-orders endpoint.

## 🚨 Free data is IEX only — your backtest will not match your fills

- The **free tier is the IEX feed**, a single venue carrying a small fraction of consolidated volume.
- **SIP (the full consolidated tape) requires the paid Algo Trader Plus subscription.**
- Data older than 15 minutes is available on all feeds.

A strategy researched on free IEX bars and traded against SIP-priced execution is validated on a different price series than it trades. IEX bars have thinner volume, different highs and lows, and gaps where IEX simply did not print. **This is a data problem masquerading as a strategy problem.**

## Advanced order types and limits

**Bracket** (entry + take-profit limit + stop-loss, OCO-linked) — `day`/`gtc` only, no extended hours, DNR/DNC mandatory. **OCO** is exit-only, same side, limit + stop/stop-limit. **OTO** is entry plus one leg. Extended sessions: overnight 8pm–4am ET, pre-market 4am–9:30am ET, after-hours 4pm–8pm ET. ⚠️ **Rate limit commonly cited as 200 requests/minute** on the free tier — a secondary source, not confirmed against current docs. Build a limiter regardless.

## Minimal correct call

```python
from alpaca.common.enums import BaseURL
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

client = TradingClient(KEY, SECRET, paper=True)   # default is True; state it anyway
assert client._base_url == BaseURL.TRADING_PAPER, "REFUSING: not the paper host"
assert client.get_account().status == "ACTIVE"    # a server-returned fact

client.submit_order(LimitOrderRequest(
    symbol="AAPL", qty=1, side=OrderSide.BUY,
    limit_price=190.25,                    # 🚨 2dp at/above $1, 4dp below — else rejected
    time_in_force=TimeInForce.DAY,         # 🚨 varies by asset class; see the table
    client_order_id="ma20-AAPL-2026-09-04-BUY",   # deterministic → retry-safe
))
```

## See also

- `../../../fin-core/skills/broker-execution-apis/SKILL.md` §3 — order-safety patterns to emit by default
- `../../../fin-core/skills/broker-execution-apis/references/alpaca.md` — the source card
- `../../../fin-core/skills/broker-execution-apis/references/_broker-matrix.md` — the full TIF matrix
- `../../../fin-core/skills/market-data-sourcing/references/_decision-table.md` — when IEX is not enough

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`broker-execution-apis`** (`../../../fin-core/skills/broker-execution-apis/SKILL.md`).

