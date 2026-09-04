---
name: lib-akshare
description: >-
  akshare is the widest free Chinese-market scraper (1,103 public interfaces) and it purges its
  own PyPI history, so you cannot pin it. TRIGGER - akshare, "import akshare as ak", pip install
  akshare, stock_zh_a_hist, stock_zh_a_daily, index_stock_cons_csindex, stock_zt_pool_em,
  stock_zh_a_stop_em, adjust="qfq"/"hfq", 复权, 前复权, 后复权, 涨跌停, 东方财富, 新浪财经, A股数据, 沪深300成分股, "No
  matching distribution found for akshare==", akshare 报错, akshare 封 IP. akshare ships roughly 2.3
  releases a week and deletes the old ones, so any signature, column name or version pin you
  remember is probably already gone. SKIP for lib-tushare, which is the skill for point-in-time
  fundamentals and dated index membership. SKIP when the question is WHICH library to choose
  rather than how to use this one - that belongs to the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# akshare

The widest free Chinese-market coverage by a large margin, and a pure scraper you cannot pin, cannot
use for point-in-time anything, and cannot legally redistribute the data from.

| | |
|---|---|
| pip / import | `akshare` / `import akshare as ak` |
| Version | 1.18.94 (2026-08-21) · 219 releases · ~29 releases in the last 90 days |
| Licence | MIT — **code only**, not the scraped data |
| Python | `requires_python >=3.11` — hard-blocked on 3.10 and below; pure-python wheel + sdist |
| Status | Firehose maintenance. ~85% of changes are scraper fixes; the cadence *is* the model |

Surface verified by `ast`-parsing `__init__.py`: **1,103 public interfaces**. The scraper tells are in
the dependency list — `curl_cffi` (TLS-fingerprint impersonation), `mini-racer` (embedded JS engine),
bs4/lxml/html5lib. Upstreams: 东方财富 · 新浪 · 同花顺 · 金十 · 和讯 plus the exchange sites.

## The trap that costs you money

**You cannot pin a version, because akshare deletes its own PyPI history.** 219 releases are listed
but the **oldest surviving release is 1.16.72 (2025-04-05)**. `pip install akshare==1.12.x` fails
outright, so a `requirements.txt` written a year ago no longer resolves. Combined with ~2.3
releases/week, function signatures and returned column names move under you between installs.

There is no fix inside akshare. **Vendor the data you need to Parquet.** Do not plan to reproduce a
result by pinning the library — plan to reproduce it from the bytes you saved.

## Index constituents are current-only — every index backtest on them is biased

`ak.index_stock_cons_csindex(symbol=...)` **has no date parameter.** It returns *today's*
HS300/CSI500 membership and nothing else. Backtesting "the HS300" with it means holding, throughout
history, exactly the names that were promoted into the index later — textbook inclusion bias, and
akshare cannot tell you it happened.

Use tushare `index_weight` / `index_member`, baostock's dated (HS300/SSE50/CSI500-only) lists, or a
licensed vendor.

## Adjustment: the default is safe — keep it that way

`stock_zh_a_hist(..., adjust="")` returns **raw (不复权)** prices, verified in source. That is the
correct base for limit-up, tick-size and lot-size logic.

- `adjust="hfq"` — anchored at listing, append-only, reproducible. **Use this for signals.**
- `adjust="qfq"` — anchor is *now*, so re-running the same query next month returns different
  history. **Never persist it.**
- `stock_zh_a_daily(..., adjust="hfq-factor")` — the Sina path uniquely returns the **adjustment
  factors themselves**, which is the right way to keep raw prices and roll your own convention. Its
  own docstring warns 大量抓取容易封 IP.

## No point-in-time fundamentals, and no unified schema

Financial statements come back keyed on **报告期** with no usable 公告日. Joining on 报告期 leaks up
to four months of future information (the annual-report deadline is 30 April).

There is also **no common return type across the 1,103 functions** — column names, dtypes and row
ordering differ per upstream page and change when that page changes. Normalize at the boundary and
never index positionally. Suspension and limits are indirect: `stock_zh_a_stop_em` for 停牌, and the
limit pools (`stock_zt_pool_em` plus the 跌停/炸板/强势 variants) substitute for real
`high_limit`/`low_limit` fields. Delisted *price* history is unverified — test it for your universe
before claiming survivorship-free coverage.

## MIT covers the scraper, not the data

akshare scrapes 东方财富 / 新浪 / 同花顺 / 腾讯, whose ToS prohibit automated bulk extraction and
commercial redistribution. **The MIT licence grants nothing regarding the scraped data.** akshare's
own docs describe the data as for academic research purposes and warn of commercial risk.
反不正当竞争法 Art. 12 and 数据安全法 have both been applied to systematic financial scraping — and
Microsoft's own qlib pulled its official China dataset citing data-security policy.

Free scrapers for research, prototyping and personal use. Anything commercial, client-facing or
redistributed needs Wind, Choice, RiceQuant, JQData or tushare's paid tier. And rate-limit from the
first line of code: `curl_cffi` is in the tree precisely to impersonate a browser, and akshare's own
source warns you will get your IP banned.

## Minimal correct call

```python
import akshare as ak

# raw is the default and the correct base for limit-up / tick-size / lot-size logic
raw = ak.stock_zh_a_hist(symbol="000001", period="daily",
                         start_date="20200101", end_date="20260901", adjust="")

# hfq for signals: anchored at listing, append-only, reproducible
hfq = ak.stock_zh_a_hist(symbol="000001", period="daily",
                         start_date="20200101", end_date="20260901", adjust="hfq")

# best of both: raw prices + the factors, so you own the convention (Sina path only)
fct = ak.stock_zh_a_daily(symbol="sz000001", adjust="hfq-factor")

# never do this for anything you store or backtest on:
# ak.stock_zh_a_hist(..., adjust="qfq")   # anchor = today; history is rewritten every dividend

# and never build a historical universe from this: it is TODAY's membership, no date parameter
# ak.index_stock_cons_csindex(symbol="000300")
```

## See also

- `../../../fin-china/skills/china-ashare-data/SKILL.md` — 复权, 涨跌停, 停牌, 公告日 vs 报告期
- `../../../fin-china/skills/china-ashare-data/references/akshare.md` — the verified reference card
- `../../../fin-china/skills/china-ashare-data/references/_source-matrix.md` — capability matrix
- `../lib-tushare/SKILL.md` — point-in-time fundamentals and dated index membership

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`china-ashare-data`** (`../../../fin-china/skills/china-ashare-data/SKILL.md`).

