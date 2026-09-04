# Interactive Brokers

The one broker whose **default is safe** — and the one where a single character switches paper for live.

## Which client

| Client | Status |
|---|---|
| **`ib_async` 2.1.0** (2025-12-08) | ✅ **the successor.** BSD-2, 1,730★ |
| 🔴 `ib_insync` 0.9.86 (2023-07-02) | **ARCHIVED 2024-03-14.** The repo's notice is from the author's family: **Ewald de Wit died on 2024-03-11.** Do not start new work on it and do not fork it |
| `ibapi` (official) | PyPI shows **9.81.1.post1 from 2020-12-06** — badly stale. The real 10.x ships from **IB's own site** as a versioned installer, under IB's **non-commercial** API licence |
| `ibflex` 1.1 (2026-05-23) | Parses IB **Flex XML** activity/trade reports — the tool for reconciliation and tax |
| 🚨 `ib_fut` | **Does not exist.** No PyPI package, no plausible repo. If a doc names it, that is an error |

`ib_async` moved to the `ib-api-reloaded` org because the maintainers **could not get access** to the
original repo, PyPI project, or docs infrastructure. Maintainer **Matt Stancliff**, explicitly "open
to adding more committers" — **single-maintainer risk, acknowledged.**

✅ **It does NOT wrap `ibapi`:** *"The ibapi package from IB is not needed. ib_async implements the
full IBKR API binary protocol internally."* Strength: one dependency, async-native. Risk: **it must
track IB's protocol changes itself** — note the `next-protobuf` branch, since **IB is migrating to
protobuf messages**.

⚠️ **Maintenance caveat to disclose:** `main`'s last commit is **2025-12-06** (= release 2.1.0). Work
continues on `next` (through 2026-07-14: implied-vol field fixes, logging scoping, snapshot Ticker
fixes) but **has not merged or shipped in ~9 months**. Open issues describe real bugs: gateway
connections dropping after a successful connect, combo/BAG market data going NaN after long uptime,
`KeyError` in `contractDetails` on warning 1102, stale `reqTickersAsync` cache reads.
**Pin your version and read `next`.**

## TWS vs Gateway

Functionally identical from the API's perspective. **IB Gateway consumes ~40% fewer resources** and
has no trading GUI — the right choice for headless deployment.

- **Both require a daily restart** to refresh contract definitions; versions 974+ support autorestart.
- API connections are **blocked by default**: *Edit → Global Configuration → API → Settings → "Enable
  ActiveX and Socket Clients"*.
- **Ports: TWS live 7496, TWS paper 7497** (✅ documented); **Gateway live 4001, Gateway paper 4002**
  (⚠️ not on the page I verified, but universally documented elsewhere). Configurable; client and TWS
  must match.
- **Master Client ID:** setting it makes that client automatically receive updates on **all** open
  orders and commission reports across clients — **essential for reconciliation.**

## 🥇 Read-Only API is on by default

Verbatim from IB's docs: **"Read-Only API" is enabled by DEFAULT in TWS "as an additional
precautionary measure". When active, API orders are prevented** (and order info is not exposed to the
API).

**IB is the one broker where the default is safe and you must deliberately opt into danger.** Leave it
on until the strategy is proven; treat turning it off as a deployment step with its own checklist.

## 🚨 What accidentally sends a live order

1. **Connecting to 7496/4001 instead of 7497/4002.** One character. **There is no other confirmation.**
2. Unchecking Read-Only API and forgetting to re-check it.
3. Reusing a `clientId` that has resting orders from another process.
4. `ib.placeOrder()` with `orderType='MKT'` outside RTH on an illiquid contract.
5. TWS "Precautionary Settings" normally raise a popup or return an API error on size/price
   violations — **but a headless Gateway has nobody to click the popup**, so orders either sit blocked
   or, with relaxed presets, go through unchecked.

**Assert on the account ID, not the port:**

```python
acct = ib.managedAccounts()[0]
assert acct.startswith("DU"), f"REFUSING TO TRADE: {acct} is not a paper account"
```

IB paper accounts are prefixed `DU` (or `DF`); live accounts start with `U`. The account ID is a
**server-returned fact**; the port is a local config value you can typo.

## 🚨 Market data is the #1 reason IB pipelines fail

- Market data requires **paid subscriptions per exchange/bundle**, tied to the account.
- IB enforces **concurrent-session limits** — running TWS and the Client Portal Gateway simultaneously
  **competes for the same market data session.**
- Simultaneous streaming lines are capped (commonly ~100 without boosters).
- **Historical data is heavily pacing-limited.** Violating pacing gets you throttled or disconnected.

Budget for this before designing a data pipeline around IB. For research history, a dedicated vendor
is usually cheaper than fighting pacing limits.

## IBKR Web API / Client Portal — a different API

REST + WebSocket, **architecturally distinct from the socket-based TWS API**, and *"The Web API does
not work with legacy TWS API."* Two access modes:

- **Client Portal Gateway** — a local Java process that **requires a manual browser login with
  username and password**, so it **cannot be fully automated**.
- **OAuth 1.0a** — for headless automation, but IB **gates it to institutional clients**.

Endpoints are the same across both; only authentication differs. IB discourages running CP Gateway
alongside TWS because of competing market data sessions. **Python support is third-party only — IB
ships no official Python SDK for the Web API.**

## Reconciliation primitives

`reqPositions` · `reqOpenOrders` · `reqExecutions` · the **Master Client ID** so one client sees all
clients' orders · `ibflex` for end-of-day Flex-report truth.

🚨 IB's `orderId` is **per-`clientId` and per-session** — reusing a `clientId` across processes is how
duplicate and orphan orders happen. IB's `orderRef` is a free-text tag, **not an idempotency key**.

**Treat any unexplained position as halt-and-alert, never as something to auto-flatten.**
