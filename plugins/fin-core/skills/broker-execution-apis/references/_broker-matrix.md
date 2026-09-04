# Broker and execution clients — verified metadata

Verified 2026-09-03 against the PyPI JSON API, the GitHub REST API, and official broker docs.

## Metadata

| Project | PyPI | Version | Released | ★ | Licence | Last commit | Status |
|---|---|---|---|---:|---|---|---|
| erdewit/ib_insync | `ib_insync` | 0.9.86 | 2023-07-02 | 3,285 | BSD-2 | 2024-03-14 | 🔴 **ARCHIVED — author deceased** |
| ib-api-reloaded/ib_async | `ib_async` | 2.1.0 | 2025-12-08 | 1,730 | BSD-2 | main **2025-12-06**; `next` 2026-07-14 | ✅ successor; ⚠️ main stale ~9 mo |
| IB official | `ibapi` | 9.81.1.post1 | **2020-12-06** | — | IB non-commercial | — | ⚠️ PyPI copy stale; get 10.x from IB |
| alpacahq/alpaca-py | `alpaca-py` | 0.44.0 | 2026-08-11 | 1,481 | Apache-2.0 | 2026-09-02 | ✅ active |
| (legacy) | `alpaca-trade-api` | 3.2.0 | 2024-01-12 | — | — | — | 🔴 deprecated |
| tastyware/tastytrade | `tastytrade` | 13.2.3 | 2026-08-07 | 256 | MIT | 2026-08-07 | ✅ active, small |
| alexgolec/schwab-py | `schwab-py` | 1.5.1 | 2025-06-30 | 468 | MIT | **2025-08-04** | ⚠️ **stalled**, 39 open issues |
| alexgolec/tda-api | `tda-api` | 1.6.0 | 2022-06-07 | 1,323 | MIT | 2024-06-16 | 🔴 **DEAD — TDA absorbed by Schwab** |
| jmfernandes/robin_stocks | `robin_stocks` | 3.4.0 | 2025-05-18 | 2,120 | MIT | 2026-02-11 | ⚠️ limping, **322 open issues** |
| robinhood-unofficial/pyrh | — | — | — | 1,789 | MIT | 2024-08-08 | 🔴 dead |
| ccxt/ccxt | `ccxt` | 4.5.77 | 2026-09-01 | 43,850 | MIT | 2026-09-03 | ✅ extremely active (daily) |
| quickfix/quickfix | `quickfix` | 1.16.0 | 2026-05-09 | 1,987 | Custom (QuickFIX) | 2026-05-20 | Active-ish |
| da4089/simplefix | `simplefix` | 1.0.17 | 2023-09-12 | 255 | MIT | 2026-06-01 | Low activity, stable |
| shidenggui/easytrader | `easytrader` | 0.23.7 | 2025-04-16 | 10,124 | MIT | 2026-02-28 | Slowing |
| openctp/openctp | — | — | — | 2,921 | BSD-3 | 2026-07-29 | Active |
| Futu | `futu-api` | 10.10.7008 | 2026-08-13 | — | Apache-2.0 | — | Vendor-maintained. 🚨 **sdist only, no Windows wheel** |
| 掘金 GoldMiner | `gm` | 3.0.186 | 2026-07-28 | — | Apache-2.0 | — | Vendor-maintained |

## ib_async — the caveat to disclose

Same API lineage as `ib_insync`, moved to the `ib-api-reloaded` org because the maintainers **could
not get access** to the original repo, PyPI project, or docs infrastructure. Maintainer Matt
Stancliff, explicitly "open to adding more committers" — **single-maintainer risk, acknowledged**.

✅ **It does NOT wrap `ibapi`**: *"The ibapi package from IB is not needed. ib_async implements the
full IBKR API binary protocol internally."* Strength: one dependency, async-native. Risk: it must
track IB's protocol changes itself — note the `next-protobuf` branch, since **IB is migrating to
protobuf messages**.

⚠️ `main`'s last commit is 2025-12-06 (= release 2.1.0). Real work continues on `next` (through
2026-07-14: implied-vol field fixes, logging scoping, snapshot Ticker fixes) but **has not been
merged or released in ~9 months**. Open issues describe real bugs: gateway connections dropping
after a successful connect, combo/BAG market data going NaN after long uptime, `KeyError` in
`contractDetails` on warning 1102, stale `reqTickersAsync` cache reads. **Pin your version and read
`next`.**

## Schwab — the constraint that decides your architecture

🚨 **The refresh token hard-expires after 7 days.** Access tokens last 30 minutes and `schwab-py`
auto-refreshes them into `token_path`, **but nothing can extend the refresh token** — after ~7 days
you must complete an interactive browser login again. Schwab enforces this server-side, and it may
fire sooner or later than exactly 7 days.

**Fully unattended long-running Schwab automation is not possible under the current OAuth model.**
OAuth requires a loopback callback (commonly `https://127.0.0.1:8182`). **There is no sandbox** —
treat all Schwab access as live.

## Alpaca order matrix

**Order types:** market, limit, stop, stop_limit, trailing_stop (`trail_price` or `trail_percent`;
**triggers only during regular market hours**).

**Time in force by asset class — a real footgun:**

| Asset | Allowed TIF |
|---|---|
| Whole-share equities | `day`, `gtc`, `opg`, `cls`; `ioc`/`fok` gated ("contact sales") |
| **Crypto** | 🚨 **only `gtc` and `ioc`** |
| Options, OTC | only `gtc`, `day` |
| **Fractional** | 🚨 **only `day`** |
| Extended hours | requires **limit + day/gtc** |

`gtc` carries a 90-day expiration policy. **Price precision:** 2 decimals ≥ $1.00, 4 below —
violations are rejected.

**Advanced:** bracket (entry + TP limit + SL, OCO-linked, day/gtc only, no extended hours, DNR/DNC
mandatory), OCO (exit-only, same side), OTO (entry + one leg).

**Sessions:** overnight 8pm–4am ET, pre-market 4am–9:30am ET, after-hours 4pm–8pm ET.

**Data:** free tier is **IEX only** (a small fraction of consolidated volume); **SIP requires the
paid Algo Trader Plus subscription**. Data older than 15 minutes is on all feeds.
🚨 **Backtests built on the free IEX feed will not match live SIP-based execution.**

⚠️ Rate limit commonly cited as **200 requests/minute** free — secondhand, confirm against current docs.

## FIX

- **`quickfix`** — the C++ QuickFIX engine with Python bindings. Full session-layer state machine,
  message store, logging, sequence-number recovery. Heavyweight, needs a C++ toolchain, non-OSI
  QuickFIX licence. It is *the* reference open FIX engine.
- **`simplefix`** — **message parsing/encoding only. No session layer, no sequence management, no
  reconnection.** Correct when the venue or a broker gateway owns the session; **wrong if you must
  be a FIX initiator.**
- Reality check: FIX access at retail brokers is essentially nonexistent — this is
  institutional/prime-broker territory.

## Recommend against

**Robinhood** (`robin_stocks`, `pyrh`) — reverse-engineered private endpoints, **no paper trading**,
MFA/device-token friction, ToS risk. `pyrh` dead (2024-08); `robin_stocks` limps with 322 open
issues. **Account lockouts are the reported failure mode.** Personal read-only tinkering at most.
