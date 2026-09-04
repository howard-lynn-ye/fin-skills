---
name: china-trading-stack
description: >-
  Backtest and execute Chinese-market strategies. Covers vnpy (and its CTP/gateway/datafeed
  ecosystem), Microsoft Qlib and its China dataset situation, RQAlpha, wondertrader/wtpy, tqsdk for
  futures, QMT/miniQMT, easytrader, and the Futu and Tiger broker SDKs — plus the settlement,
  price-limit and session rules that make Western backtest engines wrong on A-shares. TRIGGER — use
  when backtesting or trading A-shares, Chinese futures, options or convertible bonds; when the task
  mentions vnpy, qlib, RQAlpha, CTP, QMT, miniQMT, easytrader, tqsdk, 掘金, 聚宽 or 米筐; when
  connecting to a Chinese broker; or when porting a US-market strategy to China.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# China trading stack

A backtest engine written for US equities is **wrong on A-shares by default** — it will let you sell
what you bought this morning, fill you at limit-up, and trade through a suspension. §3 is the list
of rules your engine must encode.

Data sourcing is a separate skill: `china-ashare-data`.

## 1. Pick a framework

| Task | Use | Note |
|---|---|---|
| Live trading, broadest gateway coverage | **vnpy** 4.4.0 (MIT, **45,089★**, pushed 2026-09-01) | The de-facto standard. Ships **no data of its own** |
| Futures research + live, incl. 夜盘 | **tqsdk** 3.10.2 (Apache-2.0, 4,991★, **only 4 open issues**) | Best-maintained here. Free tier covers all futures + options |
| Quant research pipeline / factor ML | **Qlib** `pyqlib` 0.9.7 (MIT, 48,253★) | 🚨 official China dataset is **disabled** — see §2 |
| Event-driven A-share backtest | **RQAlpha** 6.3.0 | 🚨 **custom licence: non-commercial only**; sdist-only, no Windows wheel |
| C++-core multi-asset | wondertrader / wtpy (MIT) | wtpy slowing (last push 2025-08) |
| Retail broker automation | easytrader 0.23.7 (MIT, 10,124★) | Slowing; GUI-automation fragility |
| HK/US via a Chinese broker | `futu-api`, `tigeropen` | Vendor-maintained. ⚠️ `futu-api` is sdist-only **but pure Python, so it installs fine on Windows**; it does require the separate **FutuOpenD daemon** |

## 2. 🚨 Qlib's China dataset is disabled

Qlib's README states verbatim: *"Due to more restrict data security policy. The official dataset is
disabled temporarily."* The documented `qlib_data --region cn` command still appears in the README
but the backing dataset is down.

✅ **The live replacement is community-run: `chenditc/investment_data`** — 1,440★, Apache-2.0, latest
release tag **`2026-09-03` published the same day** (it updates **daily**), asset `qlib_bin.tar.gz`
at **563.8 MB**. It merges Tushare + akshare + baostock + Yahoo + historical Wind/Caihui dumps,
cross-validates them, **rescales adjustment factors to a common basis**, and explicitly backfills
*"delist company's data"* → **delisted names are included**, which the official bundle never handled well.

```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
```

⚠️ Qlib's `cn_data` stores a `$factor` field **and** adjusted prices — know which convention a dump
uses before mixing it with any other source. Qlib's own `check_data_health` script exists precisely because
these dumps have gaps. Qlib itself provides **no PIT fundamentals**; `instruments/csi300.txt` encodes
membership date ranges whose quality is entirely inherited from the dump you installed.

## 3. 🚨 Rules a Western engine will get wrong

Encode all of these, or your A-share backtest is fiction.

1. **T+1 settlement.** Shares bought today cannot be sold until the next session (cash is T+0).
   **Any intraday-reversal strategy on A-shares is unimplementable.** Futures are T+0 — a mixed
   equity/futures backtest needs **two settlement models**.
2. **Price limits, which vary by board and by ST status.** 主板 ±10% (ST/*ST **±5%**), 创业板
   `300xxx`/`301xxx` ±20%, 科创板 `688xxx` ±20%, 北交所 ±30%; new listings differ again on day 1.
   **A bar closing at limit-up generally could not have been bought** — filling there is the second
   largest source of fake alpha in A-share research.
3. **Suspensions run for months.** Sources variously omit the bar, emit `volume=0` with a
   forward-filled price, or emit NaN. Forward-filled bars create spurious zero returns that
   **deflate measured volatility**.
4. **The 90-minute lunch break.** Sessions are 09:30–11:30 and 13:00–15:00 — exactly **240 minute
   bars**. `resample('1min')` or any wall-clock rolling window silently spans 90 minutes of
   non-existent time. **Roll over bar index, not wall-clock.**
5. **Futures 夜盘** runs ~21:00 to 23:00/01:00/02:30, so **a futures trading day starts the previous
   calendar evening.** tqsdk and rqdatac handle the mapping; scrapers do not.
6. **Forward-adjusted (前复权) prices are look-ahead contaminated** — see `china-ashare-data` §2.
7. **Fees and 印花税:** stamp duty is charged on the **sell side only**; add 过户费 and broker
   commission with a per-order minimum. A cost model ported from US equities understates round-trip
   cost asymmetrically.

## 4. vnpy specifics

vnpy is a **framework, not a data source** — every datafeed adapter wraps a third-party, mostly paid
service. ✅ Verified on PyPI:

| Adapter | Provider | Cost | Latest |
|---|---|---|---|
| `vnpy_xt` | 迅投研 | Paid | 1.4.6 (2025-10-18) |
| `vnpy_rqdata` | 米筐 | Paid | 3.4.7.8 (2026-04-16) |
| `vnpy_tushare` | Tushare | 积分 | 1.4.21.0 (2025-06-11) |
| `vnpy_tqsdk` | 天勤 | Paid | 3.8.6.0 (2025-10-02) |
| `vnpy_wind` / `vnpy_ifind` | 万得 / 同花顺 | Paid | 1.1.0 (2025-06-11) |
| `vnpy_udata` / `vnpy_tinysoft` | 恒生 / 天软 | Paid | ⚠️ **2023 — stale** |

✅ **`vnpy_datayes` (通联) does NOT exist on PyPI** — if a doc lists it, that is wrong.

Config is uniform: `SETTINGS["datafeed.name"/"username"/"password"]`. ⚠️ For RQData these are **not**
your ricequant.com website login.

🔑 **`vnpy_datarecorder` records live tick/bar from a connected trading gateway into a local DB** —
so with a broker CTP account you can **build your own tick history for free**, bypassing vendor fees.
The most under-appreciated capability in this ecosystem.

## 5. Order safety

Everything in `broker-execution-apis` §3 applies, plus:
- **CTP simulation (SimNow) vs production** differ by broker ID and front address — assert on a
  server-returned account field, not a config flag.
- **easytrader drives a GUI**; a UI change or a popup silently breaks it mid-session. Never leave it
  unattended without an external position reconciliation loop.
- **QMT / miniQMT** ships from the broker, not PyPI, and its API surface varies by broker build.
  Pin the build you tested against.

## 6. Reference files

`references/<framework>.md` for architecture, licence, gateway coverage and setup;
`references/_ashare-rules.md` for the machine-checkable version of §3.
