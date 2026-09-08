---
name: china-ashare-data
description: >-
  Get China A-share and Greater China market data without the ecosystem's silent traps. TRIGGER -
  A股, 沪深, 北交所, 科创板, 创业板; akshare, tushare, baostock, efinance, adata, qstock, mootdx,
  easyquotation, jqdatasdk, 聚宽, rqdatac, 米筐, Wind, 万得, Choice, 东方财富; 复权, qfq, hfq, 前复权, 后复权; ST,
  退市, delisted A-share tickers, 退市股票列表; 停牌 suspension; 公告日 versus 报告期; CSI300, HS300, 中证 index
  membership. Three popular libraries default to forward-adjusted prices, which are rewritten
  retroactively and are therefore look-ahead contaminated. SKIP for backtesting or trading
  A-shares (china-trading-stack) and for Hong Kong, Taiwan, Japan or Korea (asia-pacific-markets).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# China A-share data

The Chinese ecosystem's defaults are more dangerous than the Western one's: **three of the most
popular libraries default to forward-adjusted (前复权) prices**, which are retroactively rewritten
and therefore look-ahead contaminated. Read §2 before anything else.

Most facts below are ✅ verified by downloading the wheel and reading the source, not from docs.

## 1. Pick a source

| Constraint | Use | Note |
|---|---|---|
| Widest free coverage, any asset | **akshare** (MIT, 1,103 interfaces) | Scraper. Safe default `adjust=""` |
| Free + **delisted list** + safe defaults | **baostock** 0.9.3 | **Revived in 2026** after 5 dormant years |
| **Point-in-time fundamentals**, cheap | **tushare pro** | `f_ann_date` + `update_flag` |
| **Best PIT + index history**, paid | **rqdatac** | `get_pit_financials_ex` with `if_adjusted` |
| PIT + limit-up/suspension fields, paid | **jqdatasdk** | 🚨 **geo-blocked outside mainland China** |
| **Raw prices + raw corporate actions** | **mootdx** (`xdxr`) | The most look-ahead-safe free path |
| Chinese futures, incl. 夜盘 | **tqsdk** (Apache-2.0, 4 open issues) | Best-maintained here |
| Live trading framework | **vnpy** → `china-trading-stack` | Ships no data itself |
| Realtime whole-market snapshot | **easyquotation** | Quotes only, no history |
| Institutional, index history, delisted | **Wind** / **Choice** | Terminal-bound, unpublished pricing |

**Do not use:** `pytdx` (archived 2020, **no licence**), `Ashare` (**no licence** — you cannot
legally ship it), `adata` for backtests (§2), `qstock` for anything you build on (18 months stale).

## 2. 🚨 复权 — the qfq look-ahead bias

Three conventions:
- **不复权 (raw)** — the actually traded price. Correct for limit-up/down logic, tick sizes, lot sizing.
- **后复权 hfq** — anchored at the **listing date**: `adj = raw × factor`. A new dividend only appends
  a factor; **history never changes.** Append-only, stable, reproducible.
- **前复权 qfq** — anchored at **now**: `adj = raw × factor / factor(anchor)`.

**Why qfq is poison:** the anchor moves. Every new corporate action rescales the *entire* history.
So (a) the same query run a month later returns different historical prices — non-reproducible; and
(b) a qfq price at *t* depends on events after *t* — it could not have been computed at *t*.

**Defaults, all ✅ verified in source:**

| Library | Default | Safe? |
|---|---|---|
| **efinance** `get_quote_history` | `fqt=1` = **qfq** | ❌ |
| **adata** `get_market` | `adjust_type=1` = **qfq**, and its docstring says **only qfq is implemented** | ❌❌ |
| **jqdatasdk** `get_price` | `fq='pre'` = **qfq** | ❌ |
| **akshare** `stock_zh_a_hist` | `adjust=""` = raw | ✅ |
| **baostock** `query_history_k_data_plus` | `adjustflag='3'` = raw | ✅ |
| **tushare** `pro_bar` | `adj=None` = raw | ✅ |
| **mootdx** | raw + `xdxr` factors | ✅✅ |

🚨 **tushare's qfq is worse than the standard kind.** Verified in `tushare/pro/data_pro.py`:

```python
if adj == 'hfq': data[col] = data[col] * data['adj_factor']
if adj == 'qfq': data[col] = data[col] * data['adj_factor'] / float(fcts['adj_factor'][0])
```

`fcts` is returned newest-first, so `[0]` is the factor on **the last date of your query window** —
**the same bar has different values depending on what `end_date` you asked for.** Two further
subtleties in that function: `adj_factor` is `bfill()`-ed, and `pre_close = close.shift(-1)` is only
correct because rows arrive **descending** — sort ascending first and `pre_close`, `change` and
`pct_chg` all silently break.

**Rule: hfq, or raw + factors. Never qfq.** Render qfq at display time only; never persist it.

## 3. 涨跌停, ST, T+1 — fills that could not have happened

A bar closing at limit-up generally could not have been bought. Filling at close on limit-up days is
**the second-largest source of fake alpha in A-share research**, after survivorship.

| Board | Code prefix | Daily limit |
|---|---|---|
| 主板 Main | `60xxxx` / `000xxx` / `001xxx` | **±10%** — but **ST/\*ST: ±5%** |
| 创业板 ChiNext | `300xxx` / `301xxx` | **±20%** (since Aug 2020) |
| 科创板 STAR | `688xxx` | **±20%** |
| 北交所 BSE | `8xxxxx` / `43xxxx` / `87xxxx` | **±30%** |

`ST` is a **name** change, not a code change — a ticker's limit rule changes mid-history with no
identifier change. Logic that assumes ±10% mislabels every ST bar.

Only **jqdatasdk** (`high_limit`/`low_limit`/`paused` fields, `get_preopen_infos`) and the
commercial vendors give exact bounds. Free substitute: akshare's limit-up pools
(`stock_zt_pool_em` plus 跌停/炸板/强势 variants). Deriving limits from `prev_close × (1±pct)` is
error-prone because of the board matrix, ST's ±5%, and tick rounding.

**T+1 settlement:** shares bought today cannot be sold until the next session (cash is T+0).
**Any intraday-reversal strategy on A-shares is unimplementable.** Futures are T+0 — a mixed
equity/futures backtest needs two settlement models.

**北交所 is the weakest coverage link.** `rqdatac` needs `enable_bjse=True` — **default False**, so
BSE is silently absent. Verify BSE explicitly before claiming a "full A-share" universe.

## 4. 停牌 (suspension) → zero-volume bars

Chinese suspensions run for **months**. Sources handle them three incompatible ways: omit the bar /
emit `volume=0` with a forward-filled price / emit NaN. Forward-filled bars create spurious zero
returns that **deflate measured volatility**; omitted bars misalign multi-asset panels.

Explicit control: **jqdatasdk** (`skip_paused`, `fill_paused`, `paused`), **baostock**
(`tradestatus` per bar + `query_suspended_stocks()`), **akshare** (`stock_zh_a_stop_em`). The
scrapers give you nothing — inferring from `volume == 0` also catches genuinely illiquid bars.

## 5. 公告日 vs 报告期 — the fundamentals look-ahead

Every filing has **报告期 / `end_date`** (the period described) and **公告日 / `ann_date` /
`info_date`** (when it was published). Annual reports appear up to **four months** after period end
(deadline 30 April). **Joining on 报告期 leaks up to four months of future information** — alone
enough to turn a worthless value factor into a spectacular backtest.

Filings are also aggressively restated: 业绩预告 → 业绩快报 → 正式财报 → later restatements.

| Source | PIT fundamentals | Index constituent history | Delisted |
|---|---|---|---|
| **rqdatac** | ✅ best — `get_pit_financials_ex`, `info_date` + **`if_adjusted`** restatement flag | ✅ + `return_create_tm` insertion timestamps | ✅ |
| **jqdatasdk** | ✅ `get_history_fundamentals(watch_date=)` | ✅ `get_index_stocks(date=)` | ✅ |
| **tushare** | ✅ partial — build it from `f_ann_date` + `update_flag` | ✅ `index_weight`, `index_member` | ✅ `list_status='D'` |
| **Wind / Choice** | ✅ | ✅ | ✅ |
| **baostock** | ❌ | ⚠️ HS300/SSE50/CSI500 only, but **dated** | ✅ `query_terminated_stocks` |
| **akshare** | ❌ | 🚨 ❌ **current only** — `index_stock_cons_csindex(symbol)` has **no date param** | ⚠️ lists yes, price history ❓ |
| efinance / adata / qstock / easyquotation / Ashare | ❌ | ❌ | ❌ |

🚨 **rqdatac's PIT call has a trap of its own:** default `statements='latest'`, `date=None` returns
the **most recently restated** record. Its own docstring example returns a 2019 restatement for a
2018 quarter. **You must pass `date=<as-of>`.**

🚨 **tushare's `stock_basic` defaults to `list_status='L'`** — omit it and your universe is silently
survivorship-biased. Delisting in China went from rare to common after the 2020 退市新规, so **the
bias is time-varying**, which is worse than a constant one.

## 6. Sessions and timezone

**Asia/Shanghai (UTC+8), no DST.** Equity session **09:30–11:30, 13:00–15:00**, plus 09:15–09:25
集合竞价 and 14:57–15:00 closing auction. Exactly **240 minute bars** in a normal session.

🚨 **The 90-minute lunch break breaks naive resampling.** `pandas.resample('1min')` or a wall-clock
rolling window silently spans 90 minutes of non-existent time. **Compute rolling windows over bar
index, never wall-clock.**

**Futures 夜盘** runs ~21:00 to 23:00/01:00/02:30 → **a futures trading day starts the previous
calendar evening.** A classic off-by-one. `tqsdk` and `rqdatac` handle it; scrapers do not.

## 7. Licence and legality — the data, not the code

**All code licences here are permissive** (MIT / BSD-3 / Apache-2.0). Two exceptions that are
*stricter* than GPL: **`pytdx` and `Ashare` have NO licence file** = all rights reserved under
Berne. You may not legally redistribute or ship them.

🚨 **The real risk is the data.** akshare, efinance, adata, qstock, easyquotation and Ashare scrape
东方财富 / 新浪 / 同花顺 / 腾讯, whose ToS prohibit automated bulk extraction and commercial
redistribution. **The MIT licence on the scraper grants nothing regarding the scraped data.**
akshare's own docs say the data is *"only for academic research purposes"* and warn of *"commercial
risks"*. 反不正当竞争法 Art. 12 and 数据安全法 have both been applied to systematic financial
scraping — and Microsoft's own qlib pulled its official China dataset *"due to more restrict data
security policy"*, a live example of the risk materializing.

**Rule:** free scrapers for research, prototyping and personal use. Anything commercial,
client-facing or redistributed needs a licensed vendor (Wind, Choice, RiceQuant, JQData, or
tushare's paid tier).

**Operational:** these get IP-banned. akshare's own source warns `大量抓取容易封 IP`; adata ships a
proxy hook; akshare depends on `curl_cffi` for TLS-fingerprint impersonation. Plan rate limiting
and caching from the start.

## 8. Two more things that will bite

- 🚨 **akshare purges its own PyPI history.** 219 releases are listed but the oldest surviving is
  **1.16.72 (2025-04-05)** — you *cannot* `pip install akshare==1.12.x`. Combined with ~2.3
  releases/week (the cadence *is* the maintenance model, since ~85% of changes are scraper fixes),
  **vendor the data you need rather than pinning a version.**
- 🚨 **tushare sends your token over plaintext HTTP.** ✅ verified in 1.4.29 source:
  `__http_url = 'http://api.waditu.com/dataapi'`. No HTTPS option is shipped.
- ⚠️ **tushare's GitHub repo last pushed 2024-03-13, but PyPI shipped releases in March 2026** —
  the public repo is not the source of truth for the package.
- 🚨 **baostock's new 0.9.x functions are not re-exported in `__init__.py`.** Import the submodule:
  `from baostock.security.sectorinfo import query_terminated_stocks`.

## 9. Canonical safe snippets

```python
# akshare — safe default (raw), or explicit hfq
import akshare as ak
raw = ak.stock_zh_a_hist("000001", "daily", "20200101", "20260901", adjust="")
hfq = ak.stock_zh_a_hist("000001", "daily", "20200101", "20260901", adjust="hfq")

# tushare — universe INCLUDING delisted, and PIT fundamentals
import tushare as ts
pro = ts.pro_api("TOKEN")
live = pro.stock_basic(list_status="L", fields="ts_code,name,list_date,delist_date")
dead = pro.stock_basic(list_status="D", fields="ts_code,name,list_date,delist_date")  # required!
bars = ts.pro_bar(ts_code="000001.SZ", adj="hfq", start_date="20200101")  # never adj="qfq"
inc  = pro.income(ts_code="000001.SZ", fields="ts_code,ann_date,f_ann_date,end_date,update_flag,revenue")
# filter f_ann_date <= as_of for a genuine point-in-time view

# baostock — free delisted list + raw/hfq bars
import baostock as bs
from baostock.security.sectorinfo import query_terminated_stocks
bs.login()
rs = bs.query_history_k_data_plus(
    "sh.600000", "date,code,open,high,low,close,volume,tradestatus,isST",
    start_date="2020-01-01", end_date="2026-09-01", frequency="d", adjustflag="1")  # 1 = hfq
delisted = query_terminated_stocks()
bs.logout()
```

## 10. Reference files

`references/` holds one file per library with exact versions, source-verified signatures, traps and
snippets. Grep before guessing:

```bash
grep -ril "复权\|adjust" plugins/fin-china/skills/china-ashare-data/references/
grep -i -A6 "TRAP" plugins/fin-china/skills/china-ashare-data/references/tushare.md
```

## ❓ Not verified — state as unknown, do not guess
tushare 积分 per-minute rate table and RMB prices (thresholds ✅, frequencies ❓) · commercial pricing
for Wind / JQData / rqdatac / TqSdk Pro / Choice (none publish prices; every figure in circulation
is secondhand) · JoinQuant's post-2021 regulatory status (site geo-blocked) · whether akshare returns
*price history* for delisted tickers · what baostock's new API key unlocks · Futu / Tiger HK SDKs.

## Per-library deep dives

The optional `fin-libraries` plugin carries a dedicated skill for each library below. Load one
only after this skill has told you which library you want:

- **`lib-akshare`** — akshare
- **`lib-tushare`** — tushare
