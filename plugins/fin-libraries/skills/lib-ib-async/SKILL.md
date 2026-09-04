---
name: lib-ib-async
description: >-
  The maintained Interactive Brokers Python client - successor to the archived ib_insync - where
  one digit of the port number is all that separates paper from live. TRIGGER - import ib_async,
  from ib_async import IB, pip install ib_async, ib.connect, clientId, ports 7496 7497 4001 4002,
  TWS, IB Gateway, reqHistoricalData, reqMktData, reqTickersAsync, placeOrder, managedAccounts,
  reqPositions, reqOpenOrders, reqExecutions, Master Client ID, Read-Only API, orderRef, ibflex,
  ibapi, ib_insync; "Enable ActiveX and Socket Clients", pacing violations, error 1102, a DU or U
  account prefix. Memory is stale here: ib_insync was archived 2024-03-14 after its author died,
  ib_async 2.1.0 (2025-12-08) is the successor and does not wrap ibapi, its main branch has been
  static about nine months, and ib_fut does not exist. SKIP for non-IB brokers and for the general
  order-safety patterns (broker-execution-apis). SKIP for choosing between libraries - that is the
  domain skill's job.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# ib_async

The one broker whose **default is safe** — and the one where a single character switches paper for
live.

| | |
|---|---|
| pip / import | `pip install ib_async` · `import ib_async` |
| Version | **2.1.0 (2025-12-08)** · `requires_python >=3.10` |
| Licence | **BSD-2** |
| Status | ✅ The successor. 1,730★ `ib-api-reloaded/ib_async`. ⚠️ `main` static since 2025-12-06 |

## The trap that costs you money

🚨 **Connecting to 7496 (TWS live) or 4001 (Gateway live) instead of 7497 / 4002 sends real orders.
One character, and there is no other confirmation.** The other four ways to send a live order by
accident:

- Unchecking Read-Only API and forgetting to re-check it.
- Reusing a `clientId` that has resting orders from another process.
- `ib.placeOrder()` with `orderType='MKT'` outside RTH on an illiquid contract.
- TWS "Precautionary Settings" normally raise a popup or return an API error on size/price
  violations — **but a headless Gateway has nobody to click the popup**, so orders either sit
  blocked or, with relaxed presets, go through unchecked.

**Assert on the account ID, not the port.** IB paper accounts are prefixed `DU` (or `DF`); live accounts start with `U`. The account ID is a **server-returned fact**; the port is a local config value you can typo.

## 🥇 Read-Only API is on by default

Verbatim from IB's docs: **"Read-Only API" is enabled by DEFAULT in TWS "as an additional precautionary measure". When active, API orders are prevented** (and order info is not exposed to the API). **IB is the one broker where the default is safe and you must deliberately opt into danger.** Leave it on until the strategy is proven; treat turning it off as a deployment step with its own checklist.

## Which client — and which are dead

| Client | Status |
|---|---|
| **`ib_async` 2.1.0** (2025-12-08) | ✅ **the successor.** BSD-2, 1,730★ |
| 🔴 `ib_insync` 0.9.86 (2023-07-02) | **ARCHIVED 2024-03-14.** The repo notice is from the author's family: **Ewald de Wit died 2024-03-11.** Do not start new work on it and do not fork it |
| `ibapi` (official) | PyPI shows **9.81.1.post1 from 2020-12-06** — badly stale. Real 10.x ships from **IB's own site** under IB's **non-commercial** API licence |
| `ibflex` 1.1 (2026-05-23) | Parses IB **Flex XML** activity/trade reports — the tool for reconciliation and tax |
| 🚨 `ib_fut` | **Does not exist.** No PyPI package, no plausible repo. If a doc names it, that is an error |

✅ **`ib_async` does NOT wrap `ibapi`:** *"The ibapi package from IB is not needed. ib_async implements the full IBKR API binary protocol internally."* Strength: one dependency, async-native. Risk: **it must track IB's protocol changes itself** — note the `next-protobuf` branch, since **IB is migrating to protobuf messages**.

⚠️ **Maintenance caveat to disclose.** `main`'s last commit is **2025-12-06** (= release 2.1.0). Work continues on `next` (through 2026-07-14: implied-vol field fixes, logging scoping, snapshot Ticker fixes) but **has not merged or shipped in ~9 months**. Open issues describe real bugs: gateway connections dropping after a successful connect, combo/BAG market data going NaN after long uptime, `KeyError` in `contractDetails` on warning 1102, stale `reqTickersAsync` cache reads. It moved to the `ib-api-reloaded` org because the maintainers **could not get access** to the original repo, PyPI project or docs. Single maintainer, acknowledged. **Pin your version and read `next`.**

## TWS vs Gateway

Functionally identical from the API's perspective. **IB Gateway consumes ~40% fewer resources** and
has no trading GUI — the right choice for headless deployment.

- **Both require a daily restart** to refresh contract definitions; 974+ support autorestart.
- API connections are **blocked by default**: *Edit → Global Configuration → API → Settings →
  "Enable ActiveX and Socket Clients"*.
- **Ports: TWS live 7496, TWS paper 7497** ✅ documented; **Gateway live 4001, Gateway paper 4002**
  ⚠️ universally documented elsewhere but not on the page verified. Configurable; client and TWS
  must match.

## 🚨 Market data is the #1 reason IB pipelines fail

- Market data requires **paid subscriptions per exchange/bundle**, tied to the account.
- IB enforces **concurrent-session limits** — running TWS and the Client Portal Gateway at once
  **competes for the same market data session**.
- Simultaneous streaming lines are capped (commonly ~100 without boosters).
- **Historical data is heavily pacing-limited.** Violating pacing gets you throttled or disconnected.

Budget for this before designing a pipeline around IB; for research history a dedicated vendor is usually cheaper than fighting pacing limits.

## Reconciliation

`reqPositions` · `reqOpenOrders` · `reqExecutions` · **Master Client ID** (that client automatically receives updates on **all** open orders and commission reports across clients — essential for reconciliation) · `ibflex` for end-of-day Flex-report truth.

🚨 IB's `orderId` is **per-`clientId` and per-session** — reusing a `clientId` across processes is how duplicate and orphan orders happen. IB's `orderRef` is a free-text tag, **not an idempotency key**. **Treat any unexplained position as halt-and-alert, never as something to auto-flatten.**

## Minimal correct call

```python
from ib_async import IB

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=17)      # 7497 = TWS paper; 7496 = TWS LIVE

acct = ib.managedAccounts()[0]
assert acct.startswith("DU"), f"REFUSING TO TRADE: {acct} is not a paper account"
```

## See also

- `../../../fin-core/skills/broker-execution-apis/SKILL.md` §3 — order-safety patterns to emit by default
- `../../../fin-core/skills/broker-execution-apis/references/interactive-brokers.md` — the source card
- `../../../fin-core/skills/broker-execution-apis/references/_broker-matrix.md` — client status across brokers

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`broker-execution-apis`** (`../../../fin-core/skills/broker-execution-apis/SKILL.md`).

