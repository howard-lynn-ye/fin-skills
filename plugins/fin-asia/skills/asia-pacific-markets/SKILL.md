---
name: asia-pacific-markets
description: >-
  Get data and trade in Asia-Pacific markets outside mainland China — Hong Kong, Taiwan, Japan,
  Korea, India, Singapore and Australia. Covers futu-api, tigeropen, shioaji, twstock, FinMind,
  jquants-api-client, pykrx, FinanceDataReader, nsepython, jugaad-data, kiteconnect, upstox,
  breeze-connect, python-kis, tejapi and trading-ig, plus the exchange_calendars defects that affect
  these venues. TRIGGER — use for HKEX, TWSE, TSE/JPX, KRX/KOSPI/KOSDAQ, NSE/BSE, SGX or ASX data or
  trading; for Hong Kong, Taiwan, Japanese, Korean, Indian, Singaporean or Australian stocks; when
  the task mentions Stock Connect, VCM, Muhurat, CSAT, STT, or lot sizes that vary per stock; or when
  porting a US strategy to an Asian market.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# Asia-Pacific markets (ex-mainland China)

Mainland A-shares are a separate skill (`fin-china`). Everything here is the rest of the region, where
the libraries are thinner and the market-structure differences are larger.

**Two verdicts up front:** outside Korea, **no free Asian source gives a survivorship-bias-free
universe**, and **only J-Quants and TEJ offer genuine point-in-time fundamentals**. `ib_async` is the
practical cross-market execution answer.

## 1. Libraries — verified 2026-09-04

| pip | Version | Released | ★ | Licence | Dist | Verdict |
|---|---|---|---|---|---|---|
| `futu-api` | 10.10.7008 | 2026-08-13 | 1,313 | Apache-2.0 | **sdist only** | ✅ active, vendor. **But see §2** |
| `tigeropen` | 3.7.2 | 2026-09-03 | 146 | Apache-2.0 | wheel+sdist | ✅ active |
| `shioaji` | 1.7.4 | 2026-08-27 | 511 | 🚨 **none declared** | binary wheels | ✅ active, vendor, **closed source** |
| `twstock` | 1.5.1 | 2026-04-23 | 1,515 | MIT | wheel+sdist | ⚠️ alive but scraper-fragile |
| `FinMind` | 2.0.9 | 2026-08-19 | **2,780** | Apache-2.0 | wheel+sdist | ✅ most-starred Taiwan library |
| `jquants-api-client` | 2.6.0 | 2026-08-27 | 198 | Apache-2.0 | wheel+sdist | ✅ **official JPX** |
| `pykrx` | 1.2.8 | 2026-05-04 | 1,086 | MIT | wheel+sdist | ⚠️ **now needs credentials — §5** |
| `finance-datareader` | 0.9.202 | 2026-05-13 | 1,538 | MIT | wheel+sdist | ✅ **best free delisted story anywhere here** |
| `nsepython` | 2.97 | 2025-05-26 | 366 | 🚨 **GPL-3.0** | wheel+sdist | semi-active |
| `jugaad-data` | 0.35.5 | 2026-08-25 | 568 | ⚠️ **"YOLO"** — effectively public domain, not OSI | wheel+sdist | active |
| `kiteconnect` | 5.2.1 | 2026-07-23 | 1,303 | MIT | wheel+sdist | ✅ official Zerodha |
| `upstox-python-sdk` | 2.29.0 | 2026-08-19 | 186 | MIT | sdist only | ✅ official |
| `breeze-connect` | 1.0.69 | 2026-04-14 | 87 | MIT | sdist only | ⚠️ 128 open issues |
| `python-kis` | 2.1.6 | 2025-10-13 | 288 | MIT | wheel+sdist | best 3rd-party Korea Investment |
| `tejapi` | 0.1.31 | 2025-01-08 | — | MIT | wheel+sdist | thin wrapper, paid TEJ backend |
| `trading-ig` | 0.0.25 | 2026-08-25 | 374 | BSD-3 | wheel+sdist | community IG wrapper |

🔴 **Dead but still widely recommended:** `nsepy` (2020 — its last commit is literally titled
"Deprecation notice"), `investpy` (2022), **`ib-insync` (repo `archived: true`)**, `asx` (2022),
`mojito2` (2023), `korea-investment-stock` (self-deprecated on PyPI), and **Angel One
`smartapi-python`** (stale ~19 months, **and its PyPI homepage repo 404s**).

🚨 **Do not exist on PyPI** (all verified 404): `kabusapi`, `kabusapi-python`, `kabuslib`,
`jquantsapi`, `TejToolAPI`, `SGXData`. **There is no maintained Python SDK for Kabu Station.**

🚨 **Name collisions:** `jpx` on PyPI is a **JMESPath engine**, not Japan Exchange. `sgx` is an
**evolutionary-algorithm** package, not Singapore Exchange. `XBSE` in `exchange_calendars` is
**Bucharest**, not Bombay.

## 2. Two licence/packaging findings worth correcting

✅ **`futu-api` is sdist-only — but the usual conclusion is wrong.** All 84 files across every release
since 2018 are sdists, **yet the package is pure Python** (no `ext_modules`, zero `.c/.pyx/.so/.pyd`
in the archive), **so it installs fine on Windows.** The real traps are: it **requires the FutuOpenD
daemon** running separately, `python_requires` is unset, and `protobuf>=3.20.0` is unpinned.

🚨 **`shioaji` has no licence at all.** PyPI `license` and `license_expression` are both `None`, and
the **GitHub repo root has no LICENSE file and no source** — it ships **closed-source binary wheels**.
Windows `win_amd64` is supported; **there are no musllinux wheels, so it fails on Alpine.**

## 3. 🚨 exchange_calendars — four verified defects

`exchange_calendars` 4.13.2 is genuinely well maintained (regular per-venue holiday commits). These
are real defects anyway, all reproduced by executing it locally:

1. 🔴 **Korean CSAT late opens stop in 2021.** On the national university entrance exam day KRX
   **opens one hour late**. The released list contains 30 dates ending **2020-12-03**, and even
   `master` ends at **2021-11-18**. ✅ Executed: `XKRX.session_open("2024-11-14")` returns **09:00**;
   the true open was 10:00. **CSAT days for 2022–2026 are all wrong.** Any intraday model keyed to
   "first bar of the day" is an hour off on those dates.
2. 🔴 **`XSES` models no lunch break ever** (`has_break=False`), though SGX had one until 2011 — the
   source admits it.
3. 🔴 **The default date bounds are a moving target.** ✅ `GLOBAL_DEFAULT_START` / `_END` are computed
   from `pd.Timestamp.now()` at import as **today − 20 years** / **today + 1 year**. **The same code
   returns a different calendar tomorrow.** → **always pass explicit `start=` and `end=`.**
4. 🔴 **`XBOM` has 245 sessions in 2026 and ZERO in 2027** — Indian holidays are announced annually
   and hand-maintained. Same for `XSES`. By contrast `XHKG`, `XTAI`, `XTKS`, `XKRX`, `XASX` extend to
   today+1y because their holidays are rule-generated.

🚨 **There is no `XNSE`.** ✅ `"XNSE" in get_calendar_names()` → **False**. NSE handles the
overwhelming majority of Indian volume and **has no calendar** — you must use `XBOM` (BSE) as a
proxy. Holidays are near-identical in practice but not guaranteed, and **special sessions like
Muhurat trading (the ~1-hour Diwali session) are exchange-specific.** Using XBOM for NSE is a
reasonable approximation, **silently made**.

✅ **What it gets right** (also executed): Japan's close moving 15:00 → **15:30 on 2024-11-05**;
Hong Kong's open 10:00 → **09:30 on 2011-03-07**; Korea's 2025-01-27 temporary holiday; Taiwan's LNY
closure.

🔴 **Do not trust it for:** Korean CSAT opens 2022+; Singapore pre-2011 intraday; Indian dates after
2026; anything NSE-specific; **Stock Connect trading days (no such calendar exists)**; HK typhoon
suspensions (same-day events no static calendar can carry).

## 4. 🚨 Date-dependent regimes that silently break backtests

**Key every rule to a date.** These changed within most backtest windows:

| Market | Change |
|---|---|
| **Korea** | Daily price limit was **±15% before 2015-06-15**, ±30% after |
| **Korea** | **Short selling banned in roughly 5 of the last 17 years** — the most recent ban ran Nov 2023 → **2025-03-31**. A long/short backtest across it is fiction |
| **India** | T+2 → phased **T+1** (Feb 2022 → Jan 2023, ordered by market cap) → optional **T+0** |
| **India** | 🚨 **Oct/Nov 2024 SEBI overhaul tripled F&O lot sizes to ₹15 lakh** and **collapsed weekly expiries to one per exchange**; STT on futures/options rose **~60%** |
| **Japan** | **TSE restructuring 2022-04-04** (Prime / Standard / Growth) — historical index membership and segment labels do not carry across it |
| **Japan** | Close moved 15:00 → **15:30 on 2024-11-05** |
| **Hong Kong** | Open moved 10:00 → **09:30 on 2011-03-07** |

## 5. Per-market notes

**Hong Kong.** 🚨 **Lot sizes vary per stock** — there is no universal round lot, and a backtest
assuming 100 shares is wrong for most names. T+2 settlement, **no daily price limits**, but a
**VCM (Volatility Control Mechanism)**: ±5%–50% thresholds by tier with a **5-minute cooling-off**,
covering HSCI Large/Mid/SmallCap constituents plus SPACs and ETFs. Stock Connect adds quota and
eligibility constraints, and **no Stock Connect trading calendar exists in any library.**

**Taiwan.** ±10% daily limit; **09:00–13:30 with no lunch break**; a **14:00–14:30 after-hours
fixed-price session**; six tick-size tiers. `FinMind` (2,780★) is the most-starred library;
`shioaji` is the broker path but is closed-source and unlicensed.

**Japan.** **J-Quants is confirmed official JPX.** ✅ Its **free tier gives 2 years of data with a
12-week embargo** — usable for research, not for anything near-live. `jquants-api-client` is the
client. **No maintained Kabu Station SDK exists.** Lunch break 11:30–12:30.

**Korea — the bright spot.** 🔑 **`FinanceDataReader` provides not just the delisted list but price
history for delisted names** (`DataReader('KRX-DELISTING:036360')`), plus administrative-issue lists.
**This is the best free survivorship-bias story in this entire domain.**
⚠️ **`pykrx` is no longer credential-free** — its README (May 2026) now *requires* `KRX_ID`/`KRX_PW`,
actual Korea Exchange member credentials, for authenticated datasets. **This will break existing CI.**

**India.** 🚨 **STT (securities transaction tax) materially changes intraday economics** — a cost
model ported from US equities will overstate net returns. Circuit breakers are index-level and
per-stock. `kiteconnect` (Zerodha) is the most usable broker API; `nsepython` is **GPL-3.0**;
`jugaad-data` ships `LICENSE.YOLO.md`, which is effectively public domain but **not OSI-approved** —
flag it if you have a licence allowlist. The official KIS repo (1,590★) **declares no licence at all**.

**Singapore / Australia.** Thin library coverage. `yfinance` for ASX, `trading-ig` for IG Markets,
and **Interactive Brokers as the practical answer** for both data and execution.

## 6. Cross-cutting

| | Survivorship-free universe | PIT fundamentals |
|---|---|---|
| **Korea** | ✅ FinanceDataReader (free, incl. delisted price history) | ❌ |
| **Japan** | ❌ | ✅ **J-Quants** |
| **Taiwan** | ❌ | ✅ **TEJ** (paid) |
| HK / India / SG / AU | ❌ | ❌ |

Everything in `../../fin-core/skills/research-integrity-guards/SKILL.md` applies, and the settlement,
price-limit and lot-size rules above are the local equivalents of the A-share rules in
`../../fin-china/skills/china-ashare-data/SKILL.md`.
