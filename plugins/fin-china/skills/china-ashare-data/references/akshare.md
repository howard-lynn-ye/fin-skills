# akshare

The widest free Chinese-market coverage by a large margin — **1,103 public interfaces** — and a pure
scraper you cannot pin, cannot rely on for point-in-time anything, and cannot legally redistribute the
data from.

| | |
|---|---|
| pip | `akshare` · **1.18.94 (2026-08-21)** · 219 releases · ✅ ~29 releases in 90 days |
| GitHub | `akfamily/akshare` — **22,409★**, 6 open issues, pushed 2026-09-02 ✅ |
| Licence | **MIT** ✅ (PyPI classifier + repo agree) — **code only**, see §Licence |
| Python | 🚨 **`requires_python >=3.11`** ✅ — hard-blocked on 3.10 and below |
| Wheels | pure-python `py3-none-any` + sdist — installs anywhere that has 3.11+ |
| Maintenance | ✅ Firehose. ~85% of changes are scraper fixes; **the cadence *is* the maintenance model** |

Surface verified by `ast`-parsing `__init__.py`: **1,103 public interfaces** ✅. Scraper tells in the
dependency list: `curl_cffi` (TLS-fingerprint impersonation), `mini-racer` (embedded JS engine),
bs4/lxml/html5lib. Upstreams: 东方财富 · 新浪 · 同花顺 · 金十 · 和讯 + the exchange sites.

## 🚨 Traps

**1. 🚨 You cannot pin a version — akshare purges its own PyPI history.** 219 releases are listed but
the **oldest surviving is 1.16.72 (2025-04-05)** ✅. `pip install akshare==1.12.x` fails outright, so a
`requirements.txt` written a year ago will not resolve. **Vendor the data you need to Parquet; do not
try to reproduce a result by pinning the library.** Combined with ~2.3 releases/week, function
signatures and returned column names move under you.

**2. 🚨 Index constituents are current-only — there is no date parameter.**
`ak.index_stock_cons_csindex(symbol=...)` returns **today's** HS300/CSI500 membership and nothing
else ✅. Backtesting "the HS300" with it means holding, throughout history, exactly the names that
were promoted into the index later. This is textbook survivorship/inclusion bias and akshare cannot
tell you it happened. Use tushare `index_weight` / `index_member`, baostock's dated (but
HS300/SSE50/CSI500-only) lists, or a licensed vendor — see `_source-matrix.md`.

**3. 🚨 No point-in-time fundamentals.** Financial statements come back keyed on **报告期** with no
usable 公告日. Joining on 报告期 leaks up to four months of future information (annual-report deadline
is 30 April). See `../SKILL.md` §5.

**4. ✅ The adjustment default is safe — keep it that way.** `stock_zh_a_hist(..., adjust="")` returns
**raw (不复权)** prices ✅ verified in source. `adjust="hfq"` is the correct choice for signals.
**Never persist `adjust="qfq"`** — its anchor is *now*, so the same query re-run next month returns
different history. `../SKILL.md` §2.

**5. 🔑 The Sina path uniquely exposes raw factors.** `adjust="qfq-factor"` / `"hfq-factor"` return the
adjustment factors themselves ✅, which is the right way to roll your own adjustment and keep raw
prices for limit/tick logic. Its own docstring warns 大量抓取容易封 IP.

**6. 🚨 It will get your IP banned.** akshare's own source carries that warning; `curl_cffi` exists in
the dependency tree precisely to impersonate a browser TLS fingerprint. Rate-limit and cache from the
first line of code, not after the first ban.

**7. ⚠️ Every interface has its own schema, and the columns are Chinese.** There is no unified return
type across the 1,103 functions — column names, dtypes and even row ordering differ per upstream page
and change when that page changes. Normalize at the boundary; never index positionally.

**8. ⚠️ Suspension and limit handling is indirect.** `stock_zh_a_stop_em` for 停牌 and the limit-up
pools (`stock_zt_pool_em` plus 跌停/炸板/强势 variants) are the free substitutes for real
`high_limit`/`low_limit` fields. Deriving limits from `prev_close × (1±pct)` is error-prone: the board
matrix, ST's ±5% and tick rounding all bite. `../SKILL.md` §3.

**9. ❓ Delisted price history is unverified.** akshare lists delisted names and returns their
financials; whether `stock_zh_a_hist` returns *price history* for a delisted ticker was **not
confirmed**. Test it for your universe before claiming survivorship-free coverage.

## 🚨 Licence — MIT covers the scraper, not the data

akshare scrapes 东方财富 / 新浪 / 同花顺 / 腾讯, whose ToS prohibit automated bulk extraction and
commercial redistribution. **The MIT licence grants nothing regarding the scraped data.** akshare's
own docs describe the data as for *"academic research purposes"* and warn of commercial risk.
反不正当竞争法 Art. 12 and 数据安全法 have both been applied to systematic financial scraping — and
Microsoft's own qlib pulled its official China dataset citing data-security policy
(`../../china-trading-stack/references/qlib.md`).

**Rule:** free scrapers for research, prototyping and personal use. Anything commercial,
client-facing or redistributed needs Wind, Choice, RiceQuant, JQData or tushare's paid tier.

## Minimal correct usage

```python
import akshare as ak

# raw is the default and the correct base for limit-up / tick-size / lot-size logic
raw = ak.stock_zh_a_hist(symbol="000001", period="daily",
                         start_date="20200101", end_date="20260901", adjust="")

# hfq for signals: anchored at listing, append-only, reproducible
hfq = ak.stock_zh_a_hist(symbol="000001", period="daily",
                         start_date="20200101", end_date="20260901", adjust="hfq")

# best of both: raw prices + the factors, so you control the convention
fct = ak.stock_zh_a_daily(symbol="sz000001", adjust="hfq-factor")   # Sina path only

# never do this for anything you store or backtest on:
# ak.stock_zh_a_hist(..., adjust="qfq")   # anchor = today; history is rewritten every dividend
```

## Where it fits

- Source selection, 复权, 涨跌停, 停牌, 公告日 vs 报告期: `../SKILL.md`
- Full cross-source capability matrix and per-library defaults: `_source-matrix.md`
- Safe-default alternatives: `baostock.md` (free, dated index lists, delisted list) · `tushare.md`
  (PIT-capable, but read its qfq trap first)
- Backtesting A-shares on this data: `../../china-trading-stack/SKILL.md`
