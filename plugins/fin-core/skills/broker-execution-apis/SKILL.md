---
name: broker-execution-apis
description: >-
  Connect to a broker and place orders without accidentally trading live money. TRIGGER - connect
  to Interactive Brokers, TWS, IB Gateway, ib_async, ib_insync, ibapi, Alpaca, Schwab, schwab-py,
  Tastytrade, Tradier or Robinhood; place, modify or cancel an order; read positions or balances;
  set up paper trading; order types, time-in-force, bracket or OCO orders, client order ID; FIX,
  quickfix, simplefix; "make sure I don't send a live order"; a broker connection being refused.
  Load before any code that can transmit an order. SKIP for crypto exchanges and ccxt
  (crypto-data-and-execution), and for vnpy, CTP, QMT or any Chinese broker gateway
  (china-trading-stack). This skill answers TWS/Gateway port and connection failures on its own -
  the fin-libraries deep dive is optional and most installs will not have it.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# Broker and execution APIs

Order-sending code is the one place in quant work where a bug spends real money instantly. The
patterns in §3 are not optional extras — emit them by default in any code that can transmit an
order, regardless of broker.

## 1. Which client

| Broker | Use | Status |
|---|---|---|
| **Interactive Brokers** | **`ib_async` 2.1.0** | ✅ the successor. ⚠️ `main` last commit 2025-12-06; real work sits unreleased on `next` |
| IB, vendor code | `ibapi` **10.x from IB's own site** | PyPI's 9.81.1 (2020-12-06) is badly stale. IB non-commercial licence |
| IB, reconciliation | `ibflex` | Parses IB Flex XML activity/trade reports — end-of-day truth |
| **Alpaca** | **`alpaca-py` 0.44.0** | ✅ active. `alpaca-trade-api` is deprecated |
| **Schwab** | `schwab-py` 1.5.1 | ⚠️ works, but `main` stalled ~13 months, 39 open issues. See the 7-day token limit below |
| Tastytrade | `tastytrade` 13.2.3 | ✅ active, small |
| **Crypto, any venue** | **`ccxt` 4.5.77** | ✅ MIT, ~daily releases, 100+ venues |
| FIX, full session | `quickfix` | The reference open FIX engine. Heavyweight, C++ toolchain |
| FIX, messages only | `simplefix` | **No session layer** — correct only when the venue handles the session |

🚨 **`ib_insync` is dead and archived** (2024-03-14). The repo's archive notice is from the author's
family: Ewald de Wit died on 2024-03-11. Last release 2023-07-02. **Do not start new work on it and
do not fork it.** `ib_async` moved to the `ib-api-reloaded` org because the maintainers could not get
access to the original repo, PyPI project, or docs.

🚨 **`ib_fut` does not exist** — no PyPI package, no plausible repo. If a doc or model mentions it,
that is an error; the real adjacent tool is `ibflex`.

**Recommend against:** `robin_stocks` (reverse-engineered private endpoints, **no paper trading**,
MFA/device-token friction, 322 open issues, account lockouts are the reported failure mode), `pyrh`
(dead 2024-08), `tda-api` (dead — TDA absorbed into Schwab).

✅ **CCXT Pro is free.** It was merged into the MIT `ccxt` package at v1.95 (2022). WebSocket methods
(`watchTicker`, `watchOrderBook`, `watchOrders`…) are no longer a paid product. **Any tutorial
telling you to buy ccxt.pro is out of date.** Today: `ccxt` (sync REST), `ccxt.async_support`,
`ccxt.pro` (async + WS) — all one package.

## 2. Broker-specific facts that decide your architecture

### Interactive Brokers
- **TWS vs Gateway** are functionally identical to the API. **Gateway uses ~40% fewer resources** and
  has no GUI — the right headless choice. **Both require a daily restart**; 974+ supports autorestart.
- **Ports: TWS live 7496 / TWS paper 7497; Gateway live 4001 / paper 4002.** API connections are
  **blocked by default** (*Edit → Global Configuration → API → Settings → Enable ActiveX and Socket Clients*).
- 🥇 **`Read-Only API` is ON by default** — *"an additional precautionary measure"*; while active,
  **API orders are prevented**. IB is the one broker whose default is safe. **Leave it on until the
  strategy is proven; turning it off is a deployment step with its own checklist.**
- **`ib_async` does NOT wrap `ibapi`** — it implements the IBKR binary protocol internally. Strength:
  one dependency, async-native. Risk: it must track IB's protocol changes itself (note the
  `next-protobuf` branch — IB is migrating to protobuf).
- **Market data is the #1 reason IB pipelines fail:** paid per-exchange subscriptions tied to the
  account, concurrent-session limits (running TWS and CP Gateway together competes for the same
  session), ~100 simultaneous streaming lines without boosters, and heavily **pacing-limited**
  historical data — violate pacing and you are throttled or disconnected.
- **Web API / Client Portal is a different API** — REST+WebSocket, **not interoperable with the TWS
  API**. Either the local CP Gateway (**requires a manual browser login**, so it cannot be fully
  automated) or **OAuth 1.0a**, which IB gates to institutional clients. **No official Python SDK.**

🚨 **What accidentally sends a live order at IB:**
1. **Connecting to 7496/4001 instead of 7497/4002 — one character, no other confirmation.**
2. Unchecking Read-Only API and forgetting to re-check it.
3. Reusing a `clientId` that has resting orders from another process.
4. `placeOrder()` with `MKT` outside RTH on an illiquid contract.
5. Precautionary-settings violations normally raise a TWS popup — **a headless Gateway has nobody to
   click it**, so orders either sit blocked or, with relaxed presets, go through unchecked.

### Alpaca
- **TIF varies by asset class and this is a real footgun:** crypto supports **only `gtc` and `ioc`**;
  options/OTC only `gtc`/`day`; **fractional orders only `day`**; extended hours requires
  **limit + day/gtc**. `ioc`/`fok` on whole shares is gated.
- Price precision: 2 decimals ≥ $1.00, 4 below. Violations are rejected.
- `client_order_id` is caller-supplied — **this is your idempotency primitive**.
- **Free data is IEX only** (a small fraction of consolidated volume); **SIP needs the paid Algo
  Trader Plus tier**. **Backtests built on free IEX will not match live SIP execution.**
- Paper and live have **separate key pairs and separate hosts**.
  ✅ **`alpaca-py`'s default is SAFE** — verified in the 0.44.0 wheel:
  `TradingClient.__init__(..., paper: bool = True, ...)`, so **omitting the flag gives you paper.**
  🚨 **The legacy `alpaca-trade-api` is the dangerous one** — verified in the 3.2.0 wheel:
  `get_base_url()` returns **`https://api.alpaca.markets`, the LIVE host**, unless
  `APCA_API_BASE_URL` is set. Old tutorials and old code default to production.
  🚨 In `alpaca-py`, **`url_override` bypasses the paper→URL mapping** while still setting
  `sandbox=paper` — so the flag and the host can disagree. Assert on the account, not the flag.

### Schwab
🚨 **The refresh token hard-expires after 7 days.** Access tokens last 30 minutes and `schwab-py`
auto-refreshes them, **but nothing can extend the refresh token** — after ~7 days you must complete
an interactive browser login again, and Schwab enforces this server-side. **Fully unattended
long-running Schwab automation is not possible under the current OAuth model.** There is also **no
sandbox** — treat all Schwab access as live.

### ccxt
The unified API normalizes **common fields only**; anything venue-specific goes through `params`
passthrough and implicit methods that are **not portable**. `enableRateLimit` must be on.
`fetchOHLCV` has per-exchange `limit` caps and inconsistent pagination. `set_sandbox_mode(True)`
exists but **only some exchanges have testnets**. Precision/rounding rules differ per venue and
silently reject orders. **CCXT is connectivity, not a backtester — it has no simulation layer.**

## 3. Order-safety patterns — emit these by default

### 3.1 Assert the paper account from a server-returned fact, never a local flag

| Broker | Server-side proof |
|---|---|
| IB | Account ID prefix from `managedAccounts` — `DU`/`DF` = paper, `U` = live. **Assert on the account ID, not the port** |
| Alpaca | `TradingClient.get_account()` plus the base URL in use; paper keys cannot reach the live host |
| Schwab | Account hash from the accounts endpoint — **no sandbox exists** |
| freqtrade | `dry_run: true` **and** confirm no trade-scoped exchange keys are present |
| ccxt venues | `set_sandbox_mode(True)` **and** verify `exchange.urls['api']` points at a testnet host |

```python
# Fail closed, on a server-returned fact
acct = ib.managedAccounts()[0]
assert acct.startswith("DU"), f"REFUSING TO TRADE: {acct} is not a paper account"
```

### 3.2 Read-only gates
- Keep IB's Read-Only API on through development.
- Create broker/exchange keys with **trade** scope only when trading, **never with withdraw**. On
  crypto venues, withdraw-enabled keys are how accounts get drained.
- Application gate: `LIVE = os.environ.get("TRADING_LIVE") == "1"`, defaulting to False, checked
  inside the submission function. **Environment beats config, because config files get committed.**

### 3.3 Idempotent client order IDs
The failure: you submit, the response times out, you retry, you now hold 2× the position. Networks
and broker gateways make this routine.

```python
# Derive deterministically from strategy state — never from a clock or a UUID
coid = f"{strategy_id}-{symbol}-{bar_ts.isoformat()}-{side}"   # replay-safe
```
A retry then either succeeds or is rejected as a duplicate — both safe. **A UUID retry doubles your
position.** Note IB's `orderId` is per-`clientId` and per-session (reusing a `clientId` across
processes is how duplicate/orphan orders happen), and IB's `orderRef` is a free-text tag, **not** an
idempotency key.

### 3.4 Kill switches
- **freqtrade:** `stoploss_on_exchange` places the stop **at the exchange**, so it survives your
  process dying. **The most important line in retail crypto risk management — and it is off by default.**
- **nautilus_trader:** `RiskEngine` states **ACTIVE / HALTED / REDUCING**, enforced in backtest *and*
  live. `REDUCING` (position-reducing orders only) is the right state for a controlled wind-down.
- **IB has no API kill switch** — the equivalents are re-enabling Read-Only, killing the Gateway
  process, or account-level restrictions in Account Management.
- **Build your own regardless:** a heartbeat file/key checked every loop, plus hard
  `max_orders_per_minute` and `max_notional_per_day` counters that flip the process to
  flatten-and-halt. Neither depends on the broker cooperating.

### 3.5 Position reconciliation
**Never trust in-process state as the source of truth** — restarts, partial fills, manual
intervention and liquidations all desynchronize it.
- On startup and on a timer, pull **positions + open orders** from the broker and diff against local
  state. **Halt on mismatch — do not "fix".**
- IB: set the **Master Client ID** so one client receives updates on *all* open orders and commission
  reports across clients; otherwise a second process's orders are invisible. Primitives:
  `reqPositions`, `reqOpenOrders`, `reqExecutions`; `ibflex` for end-of-day truth.
- 🚨 **Treat any unexplained position as halt-and-alert, never as something to auto-flatten** —
  auto-flattening a position you mis-parsed turns a reconciliation bug into a realized loss.

## 4. Pre-live checklist

1. Assert paper account from a **server-returned** identifier.
2. Confirm backtest fills are next-bar, not same-bar (`backtesting-engines` §2.1).
3. Run paper/dry-run for **at least one full market cycle** and diff its fills against the backtest's
   — divergence here is the cheapest bug you will ever find.
4. Verify recursive indicators have converged at your live warm-up length.
5. Set exchange-side stops where supported.
6. Deterministic client order IDs.
7. Hard daily notional and order-count caps **in code, not config**.
8. Reconciliation on startup + heartbeat, halting on mismatch.
9. A kill switch **you have actually tested by triggering it**.
10. Log every submitted order with its client ID and the exact market data that produced it.

## 5. Reference files

`references/<broker>.md` — auth model, paper availability, order types, TIF matrix, rate limits,
market-data terms, and the specific ways that broker lets you trade by accident.

## Per-library deep dives

The optional `fin-libraries` plugin carries a dedicated skill for each library below. Load one
only after this skill has told you which library you want:

- **`lib-alpaca-py`** — alpaca-py
- **`lib-ccxt`** — ccxt

## US rules that decide what the account can do

Settlement is **T+1 since 2024-05-28** and the **Pattern Day Trader rule was eliminated on
2026-06-04** (SEC 34-105226) — a model answering from its training prior will get both wrong.
Reg T initial margin and the broker's own maintenance number decide the leverage this account
can actually reach. See `../us-market-rules/SKILL.md`.
