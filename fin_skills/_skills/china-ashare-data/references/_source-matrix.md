# China A-share data sources — verified metadata

Verified 2026-09-03. Most claims come from **downloading the wheel and reading the source** — those
are the strongest here and are marked ✅ source.

## Metadata

| Package | Version | Released | ★ | Licence | Python | Status |
|---|---|---|---|---|---|---|
| **akshare** | 1.18.94 | 2026-08-21 | **22,394** | MIT | **>=3.11** | ✅ firehose: 29 releases in 90 days |
| **tushare** | 1.4.29 | 2026-03-25 | 15,382 | BSD-3 | — | ⚠️ repo last pushed **2024-03-13** yet PyPI shipped in 2026 |
| **baostock** | 0.9.3 | 2026-07-10 | — | BSD | — | ✅ **revived 2026** after 5 dormant years. **No GitHub repo** |
| efinance | 0.5.9 | 2026-07-17 | 3,978 | MIT | — | active, 152 open issues |
| adata | 2.9.5 | 2025-12-26 | 5,152 | Apache-2.0 | — | ⚠️ ~8 months stale |
| qstock | 1.3.8 | 2025-03-16 | 1,932 | MIT | — | ⚠️ ~18 months stale |
| mootdx | 0.11.7 | 2024-05-04 | 2,245 | MIT | — | ⚠️ ~26 months stale, 99 open issues |
| easyquotation | 0.7.7 | 2025-03-25 | 5,378 | ⚠️ GitHub MIT vs PyPI BSD | — | realtime only |
| **pytdx** | 1.72 | **2019-08-26** | 1,554 | 🚨 **NONE** | — | 🔴 **archived; branch renamed to `archive`** |
| **Ashare** | not on PyPI | — | 3,787 | 🚨 **NONE** | — | single file, no licence |
| jqdatasdk | 1.9.8 | 2026-01-29 | 1,388 | MIT | — | 🚨 **site geo-blocked outside mainland China** |
| rqdatac | 3.6.4 | 2026-08-28 | — | 🚨 **none declared — proprietary** | — | ✅ most active commercial SDK (197 releases) |
| tqsdk | 3.10.2 | 2026-08-18 | 4,991 | Apache-2.0 | — | ✅ **only 4 open issues** |

🚨 **`pytdx` and `Ashare` have NO licence file** — under Berne that means all rights reserved, which
is **more restrictive than GPL**. You cannot legally redistribute or ship them. `mootdx` is pytdx's
living successor.

⚠️ **adata's PyPI classifier says MIT but its repo LICENSE is Apache-2.0** — trust the repo.
⚠️ **easyquotation: GitHub says MIT, PyPI classifier says BSD** — trust the repo.
⚠️ **rqdatac ships a compiled binary** (`connection.cpython-310-darwin.so`) beside 48 `.py` files,
built per-CPython/platform. Not open source.

## Capability matrix

| Source | Delisted | Index history | PIT fundamentals | Limit bounds | Suspension flag |
|---|---|---|---|---|---|
| **rqdatac** | ✅ `de_listed_date` | ✅ + insertion timestamps | ✅ **best** — `info_date` + `if_adjusted` | ✅ | ✅ |
| **jqdatasdk** | ✅ `get_all_securities(date=None)` | ✅ `get_index_stocks(date=)` | ✅ `get_history_fundamentals(watch_date=)` | ✅ `high_limit`/`low_limit` | ✅ `paused` |
| **tushare** | ✅ `list_status='D'` | ✅ `index_weight`, `index_member` | ✅ partial — build from `f_ann_date` | ❌ | ❌ |
| **baostock** | ✅ `query_terminated_stocks` | ⚠️ HS300/SSE50/CSI500 only, **dated** | ❌ | ❌ | ✅ `tradestatus` |
| **akshare** | ⚠️ lists + financials yes; **price history ❓** | 🚨 ❌ **current only, no date param** | ❌ | ⚠️ via limit pools | ⚠️ `stock_zh_a_stop_em` |
| Wind / Choice | ✅ | ✅ | ✅ | ✅ | ✅ |
| efinance / adata / qstock / easyquotation / Ashare | ❌ | ❌ | ❌ | ❌ | ❌ |

## ✅ Source-verified adjustment defaults

| Library | Call | Default | Safe? |
|---|---|---|---|
| **efinance** | `get_quote_history(..., fqt=1)` | **qfq** | 🚨 |
| **adata** | `get_market(..., adjust_type=1)` | **qfq**, and docstring says *only qfq is implemented* | 🚨🚨 |
| **jqdatasdk** | `get_price(..., fq='pre')` | **qfq** | 🚨 |
| akshare | `stock_zh_a_hist(..., adjust="")` | raw | ✅ |
| baostock | `query_history_k_data_plus(..., adjustflag='3')` | raw | ✅ |
| tushare | `pro_bar(..., adj=None)` | raw | ✅ |
| mootdx | `bars` + `xdxr` | raw + raw corporate actions | ✅✅ |

🚨 **tushare's qfq is anchored to your query's `end_date`**, not to today — verified in
`tushare/pro/data_pro.py`: `data['adj_factor'] / float(fcts['adj_factor'][0])` where `fcts` is
newest-first. **The same bar has different values depending on what `end_date` you asked for.**
Two further traps in that function: `adj_factor` is `bfill()`-ed, and `pre_close = close.shift(-1)`
is only correct because rows arrive **descending** — sort ascending first and `pre_close`, `change`
and `pct_chg` all break silently.

## Per-source notes

### akshare
✅ `ast`-parsed `__init__.py`: **1,103 public interfaces** — nothing else is close. Scraper tells:
`curl_cffi` (TLS-fingerprint impersonation), `mini-racer` (embedded JS engine), bs4/lxml/html5lib.
Sources: 东方财富 / 新浪 / 同花顺 / 金十 / 和讯 + exchange sites.

🚨 **It purges its own PyPI history.** 219 releases listed; the **oldest surviving is 1.16.72
(2025-04-05)**. You cannot `pip install akshare==1.12.x`. With ~85% of changes being scraper fixes,
the release cadence *is* the maintenance model — **vendor the data you need rather than pinning.**

Sina path uniquely offers `adjust="qfq-factor"` / `"hfq-factor"` to fetch raw factors — the right way
to roll your own. Its docstring warns `大量抓取容易封 IP`.

### tushare
🚨 ✅ **Plaintext HTTP**: `__http_url = 'http://api.waditu.com/dataapi'`. Token and data unencrypted;
no HTTPS option shipped. Also `tushare/util/verify_token.py` and `stock/cons.py`.

✅ `DataApi.__getattr__` returns `partial(self.query, name)` — **any** `api_name` is accepted
client-side; all permissioning is server-side against 积分. The client tells you nothing about access.

✅ 积分 thresholds (from `tushare.pro/document/1?doc_id=108`): 120 = 日线 · 1000 = 港股日线 ·
2000 = most 财务/基金/期货/指数 · 3000 = 限售股解禁 · 5000+ = fund portfolios, option daily, higher
frequency. ❓ The per-minute rate table and RMB prices are **not verified** — "200元→2000分" is secondhand.

### baostock
✅ Verified by reading the 0.9.3 wheel; author metadata unchanged (`baostock@163.com`) → not a hijack.
New in 0.9.x: `query_terminated_stocks` (**退市**), `query_suspended_stocks`, `query_st_stocks`,
`query_starst_stocks`, `query_stocks_in_risk`, `query_gem_stocks`, `query_shhk_stocks`/
`query_szhk_stocks`, `query_stock_concept`, `query_cpi_data`/`query_ppi_data`/`query_pmi_data`,
bulk `query_daily_history_k_AStock`/`_ETF`, `query_daily_adjust_factor(date)`.

🚨 **Most of these are NOT re-exported in `__init__.py`.** Import the submodule directly:
```python
from baostock.security.sectorinfo import query_terminated_stocks
```
0.9.x also added `bs.set_API_key()`; anonymous login still works, so it remains free. ❓ What the key
unlocks is unverified — the site is fully JS-rendered.

Stateful socket protocol: `login()`/`logout()` required, **not thread-safe**, no public repo or issue
tracker, single anonymous maintainer, single server.

### rqdatac — best PIT, with a trap
✅ `get_pit_financials_ex(order_book_ids, fields, start_quarter, end_quarter, date=None,
statements='latest', market='cn')` returns `quarter`, **`info_date`** (公告发布日) and
**`if_adjusted`** (0 = as originally reported, 1 = a later restatement). Almost nothing else exposes
the restatement flag.

🚨 **Defaults are contaminated**: `statements='latest'`, `date=None` returns the most recently
restated record — its own docstring example returns a **2019 restatement for a 2018 quarter**.
**You must pass `date=<as-of>`.**

`index_components(id, date=, start_date=, end_date=, return_create_tm=False)` — `return_create_tm=True`
gives the **DB insertion timestamp**, letting you detect retroactively backfilled membership.

🚨 `enable_bjse` defaults to **False**, so 北交所 is **silently absent** unless you opt in.
Auth: `init(username, password, addr=("rqdatad-pro.ricequant.com",16011))`; credentials are **not**
your ricequant.com website login. `get_quota()` exists → metered.

### jqdatasdk — geo-blocked
🚨 ✅ Verified: `joinquant.com` returns HTTP 200 with a full-page block —
「很抱歉，当前网站暂不支持来自非中国大陆地区 IP 地址的访问。」 **You cannot read the docs, register, or
authenticate from outside mainland China.** API surface verified by reading the 1.9.8 wheel
(74 exported functions) instead.

🔑 Best-in-class suspension/limit handling: `get_price` accepts `skip_paused` and `fill_paused`, and
`fields` include `high_limit`, `low_limit`, `paused`, `factor` — so you can correctly refuse fills
at limit-up. Also `get_call_auction` (集合竞价), `get_mtss` (融资融券), bundled `alpha101`/`alpha191`.

### mootdx — the most look-ahead-safe free path
Two modes: **`Reader`** parses **local 通达信 `.day`/`lc1`/`lc5` binaries** off your disk (zero
network, zero rate limit, zero ToS exposure); **`Quotes`** speaks the TDX binary protocol
(`bars`, `minute`, `transaction(s)` historical tick, `stock_list`, `finance`, **`xdxr`**).

🔑 **`xdxr` gives raw unadjusted prices plus raw corporate-action records** → compute your own hfq
factors and avoid every vendor's qfq normalization entirely.

⚠️ Reverse-engineered servers rotate and the bundled list is 2 years stale; `transactions` is paged
at `offset=800`.

## ❓ Not verified
tushare 积分 per-minute rate table and RMB prices · commercial pricing for Wind / JQData / rqdatac /
TqSdk Pro / Choice (none publish prices) · JoinQuant's post-2021 regulatory status · whether akshare
returns *price history* for delisted tickers · what baostock's API key unlocks · baostock history
start dates per frequency · Futu / Tiger HK SDKs.
