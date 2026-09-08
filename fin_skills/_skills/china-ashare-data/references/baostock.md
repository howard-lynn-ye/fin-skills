# baostock

Free, no registration, **safe adjustment default**, and the only free source with a real 退市 list —
revived in 2026 after five dormant years. Also: no repo, no issue tracker, one anonymous maintainer,
one server, and a `get_data()` that breaks on modern pandas.

| | |
|---|---|
| pip | `baostock` · **0.9.3 (2026-07-10)** · only 13 releases ever |
| GitHub | 🚨 **none.** No public repo, no issue tracker, no changelog. Site is `baostock.com` |
| Licence | **BSD** ✅ (PyPI classifier "License :: OSI Approved :: BSD License") |
| Python | ⚠️ `requires_python` null · pure-python wheel + sdist · needs `pandas` |
| Transport | 🚨 raw TCP to **`public-api.baostock.com:10030`** ✅ — custom binary protocol, single host, non-standard port |
| Maintenance | ✅ Revived. Author metadata unchanged (`baostock@163.com`) → not a package hijack |

All findings below marked ✅ come from reading the 0.9.3 wheel source.

## 🚨 Traps

**1. 🚨 `ResultData.get_data()` crashes on pandas ≥ 2.0 — but only for large results.** ✅ Verified in
`baostock/data/resultset.py`:

```python
df = pd.DataFrame(self.data, columns=self.fields)
while (self.error_code == '0') & self.next():
    temp_df = pd.DataFrame(self.data, columns=self.fields)
    df = df.append(temp_df, ignore_index=True)    # DataFrame.append REMOVED in pandas 2.0
```

Page size is **2,000 rows** (`BAOSTOCK_PER_PAGE_COUNT`) ✅. Under 2,000 rows `next()` returns False,
the loop never runs, and `get_data()` works fine. Cross 2,000 rows — i.e. any real query — and it
raises `AttributeError: 'DataFrame' object has no attribute 'append'`. **It works in your notebook and
fails on your backtest.** Use the manual `next()` / `get_row_data()` loop below.

**2. 🚨 `start_date` defaults to 2015-01-01, not to the listing date.** ✅
`DEFAULT_START_DATE = "2015-01-01"`. Omit `start_date` and you silently get a truncated history with
no warning — a decade of pre-2015 data missing from a series that looks complete.

**3. 🚨 Errors are returned, not raised.** ✅ Every function returns a `ResultData` whose `error_code`
starts at `BSERR_NO_LOGIN`; parameter validation `print()`s a Chinese message to stdout and returns an
error-coded object. **A failed call yields an empty DataFrame, not an exception.** Assert
`rs.error_code == '0'` after every call or empty frames propagate into your pipeline as "no data".

**4. 🚨 Most of the new 0.9.x functions are NOT re-exported from `__init__.py`.** ✅ Confirmed by
reading it — `__init__` exports only the legacy set (`query_history_k_data_plus`,
`query_hs300_stocks`, `query_sz50_stocks`, `query_zz500_stocks`, `query_stock_industry`, the
evaluation/macro families, `query_daily_history_k_AStock` / `_ETF`, `query_daily_adjust_factor`,
`set_API_key`). `bs.query_terminated_stocks` raises `AttributeError`. Import the submodule:

```python
from baostock.security.sectorinfo import query_terminated_stocks
```

✅ Full `sectorinfo` list in 0.9.3: `query_stock_industry`, `query_hs300_stocks`, `query_sz50_stocks`,
`query_zz500_stocks`, **`query_terminated_stocks`** (退市), **`query_suspended_stocks`** (停牌),
`query_st_stocks`, `query_starst_stocks`, `query_stock_concept`, `query_stock_area`,
`query_ame_stocks`, `query_gem_stocks`, `query_shhk_stocks`, `query_szhk_stocks`,
`query_stocks_in_risk`.

**5. 🚨 Stateful socket session — `login()`/`logout()` required, and NOT thread-safe.** ✅ Module-level
connection state means concurrent calls from threads corrupt each other's paging cursors. One process,
one session, serial calls. Retries must re-login.

**6. ✅ The adjustment default is safe.** `query_history_k_data_plus(..., adjustflag='3')` ✅ —
**`'3'` = raw (不复权)**, **`'1'` = 后复权 hfq**; ❓ `'2'` is the remaining qfq value, unverified in
source. Raw or hfq only. `../SKILL.md` §2.

**7. 🚨 No point-in-time fundamentals.** The `query_profit_data` / `query_balance_data` /
`query_cash_flow_data` family is keyed on 报告期 with no usable 公告日. Joining on it leaks up to four
months. Use tushare's `f_ann_date` or rqdatac instead — `tushare.md`, `_source-matrix.md`.

**8. ⚠️ Index history is dated but narrow.** `query_hs300_stocks(date=)` / `sz50` / `zz500` accept a
date ✅ — genuinely better than akshare's undated constituents — but those three indices are all you
get. No CSI1000, no sector indices, no weights.

**9. ❓ Unverified.** What `set_API_key()` unlocks (`login(user_id='anonymous', password='123456')` ✅
is the default and still works free) · per-frequency history start dates · whether the single server
has any availability guarantee. The site is fully JS-rendered and there is no repo to read.

## Minimal correct usage

```python
import baostock as bs
import pandas as pd
from baostock.security.sectorinfo import query_terminated_stocks   # not on the bs namespace

bs.login()                        # anonymous; free
rs = bs.query_history_k_data_plus(
    "sh.600000",
    "date,code,open,high,low,close,volume,amount,turn,tradestatus,isST",
    start_date="2010-01-01",      # explicit: the default is 2015-01-01
    end_date="2026-09-01",
    frequency="d",
    adjustflag="1",               # 1 = hfq. '3' (raw) is the default; never persist qfq
)
assert rs.error_code == "0", rs.error_msg        # errors are returned, not raised

rows = []
while (rs.error_code == "0") & rs.next():        # NOT rs.get_data() — it uses DataFrame.append
    rows.append(rs.get_row_data())
df = pd.DataFrame(rows, columns=rs.fields)

dead = query_terminated_stocks()                 # the free survivorship fix
bs.logout()
```

`tradestatus` is the per-bar 停牌 flag ✅ — use it to drop suspended bars rather than inferring from
`volume == 0`, which also catches genuinely illiquid names. `../SKILL.md` §4.

## Where it fits

- 复权, 涨跌停, 停牌, PIT fundamentals in full: `../SKILL.md`
- Cross-source capability matrix: `_source-matrix.md`
- Wider coverage, no dated index history: `akshare.md` · PIT fundamentals: `tushare.md`
- Backtesting on this data: `../../china-trading-stack/SKILL.md`
