# Alpaca

The easiest US broker to automate — REST, real paper trading, free data — and the one whose rules
change silently depending on **which asset class** you point them at.

| | |
|---|---|
| pip | `alpaca-py` · import `alpaca` |
| Version | **0.44.0 (2026-08-11)** |
| GitHub | `alpacahq/alpaca-py` — 1,481★, 83 open issues, pushed **2026-09-03** ✅ active |
| Licence | ✅ **Apache-2.0** |
| Python | `>=3.10,<4.0` · classifiers through **3.14** · pure-Python wheel |
| 🔴 Legacy | `alpaca-trade-api` **3.2.0 (2024-01-12)** — deprecated, and its default is dangerous (below) |

⚠️ Recent commits on `main` are largely dependency bumps; feature velocity is moderate, not stalled.

## 🚨 Paper vs live — and the advice that is now inverted

Paper and live are **separate API key pairs AND separate hosts**:
`paper-api.alpaca.markets` vs `api.alpaca.markets`. Keys are not portable between them.

✅ **Verified in `alpaca/trading/client.py`:** `TradingClient.__init__` declares
**`paper: bool = True`**, and the base URL resolves as
`url_override if url_override else BaseURL.TRADING_PAPER if paper else BaseURL.TRADING_LIVE`.

**So in `alpaca-py`, omitting `paper` gives you PAPER.** You must pass `paper=False` deliberately to
reach the live host. That is a good default — the same posture as IB's Read-Only API
(`./interactive-brokers.md`).

🚨 **The widely-repeated warning "leaving `paper` unset trades for real" is about the LEGACY
library, and it is true there.** ✅ Verified in `alpaca_trade_api/common.py`: `get_base_url()`
returns `os.environ.get('APCA_API_BASE_URL', 'https://api.alpaca.markets')` — **live by default**,
with no `paper` argument anywhere. `REST(key, secret)` with live keys and no `APCA_API_BASE_URL`
set is a live session. If you are porting old code or old blog snippets, that inversion is the bug.

🚨 **`url_override` bypasses the flag.** It replaces the base URL outright while `sandbox=paper` is
still set from the flag — so `url_override=<live host>, paper=True` sends **live orders from a
client that believes it is in the sandbox.** Never combine the two.

**Assert on a server-returned fact, not on the flag:**

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
    time_in_force=TimeInForce.DAY,         # 🚨 varies by asset class; see the table below
    client_order_id="ma20-AAPL-2026-09-04-BUY",   # deterministic → retry-safe
))
```

## 🚨 Time-in-force varies by asset class

The full menu is `day`, `gtc`, `opg` (opening auction), `cls` (closing auction), `ioc`, `fok` — but
almost none of it applies to every product:

| Product | Accepted TIF |
|---|---|
| **Crypto** | 🚨 **`gtc` and `ioc` only** |
| **Options, OTC** | 🚨 **`gtc` and `day` only** |
| **Fractional shares** | 🚨 **`day` only** |
| **Extended hours** | 🚨 Requires a **limit** order with `day` or `gtc` — market orders are not accepted |
| Whole-share equities | Full menu; ⚠️ `ioc`/`fok` are gated ("contact sales") |

A `TimeInForce.DAY` crypto order and a `TimeInForce.IOC` fractional order are both rejections, not
fills — and the rejection arrives asynchronously, so a naive loop just sees nothing happen. `gtc`
additionally carries a **90-day expiration policy**.

**Trailing stops** (`trail_price` / `trail_percent`) 🚨 **trigger only during regular market hours** —
an overnight gap does not move them.

## 🚨 Price precision

**2 decimals at or above $1.00, 4 decimals below.** Orders violating this are **rejected**. Anything
that computes a limit from a float — a percentage offset, an ATR multiple, a mid-price — needs an
explicit round to the right precision for that price band, and the band is per-order, not per-symbol.

## `client_order_id` is the idempotency primitive

Caller-supplied and unique; auto-generated if you omit it. **Omitting it removes your only defence
against the duplicate-order failure mode:** you submit, the response times out, you retry, and you
now hold 2× the position. Networks and broker gateways make that routine.

Derive it from strategy state, never from a clock or a UUID:

```python
coid = f"{strategy_id}-{symbol}-{bar_ts.isoformat()}-{side}"   # replay-safe
```

A retry then either succeeds or is rejected as a duplicate — both outcomes are safe. A UUID retry
doubles your position. Reconcile on `client_order_id` against the list-orders endpoint; contrast
`./interactive-brokers.md`, where `orderId` is per-`clientId` and per-session and `orderRef` is
**not** an idempotency key.

## 🚨 Free data is IEX only — your backtest will not match your fills

- The **free tier is the IEX feed**, a single venue carrying a small fraction of consolidated volume.
- **SIP (full consolidated tape) requires the paid Algo Trader Plus subscription.**
- Data older than 15 minutes is available on all feeds.

So a strategy researched on free IEX bars and traded against SIP-priced live execution is being
validated on a different price series than it trades. IEX bars have thinner volume, different highs
and lows, and gaps where IEX simply did not print. **This is a data problem masquerading as a
strategy problem** — see `../../backtesting-engines/references/execution-realism.md` and
`../../market-data-sourcing/references/_decision-table.md`.

## Advanced order types

**Bracket** (entry + take-profit limit + stop-loss, OCO-linked) — `day`/`gtc` only, no extended
hours, DNR/DNC mandatory. **OCO** is exit-only, same side, limit + stop/stop-limit. **OTO** is entry
plus one leg.

**Extended sessions:** overnight 8pm–4am ET, pre-market 4am–9:30am ET, after-hours 4pm–8pm ET.

⚠️ **Rate limit: commonly cited as 200 requests/minute** on the free tier — secondary source, not
confirmed against current docs. Build a limiter regardless; confirm the number before relying on it.

## Where it fits

The best first live-trading target in US equities: real paper accounts on a separate host, a safe
client default, and a proper idempotency key. Its weaknesses are **data** (IEX unless you pay) and
**asset-class asymmetry** (the TIF table). For a broker with no sandbox at all and a 7-day token
wall, see `./schwab-and-others.md`; for the safest defaults in the industry, `./interactive-brokers.md`.
