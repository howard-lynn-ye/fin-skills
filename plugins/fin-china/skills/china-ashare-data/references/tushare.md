# tushare

The cheapest route to genuinely point-in-time A-share fundamentals — and the owner of the most
dangerous adjustment implementation in the Chinese ecosystem, which is **worse than ordinary qfq**.

| | |
|---|---|
| pip | `tushare` · **1.4.29 (2026-03-25)** · 229 releases |
| GitHub | `waditu/tushare` — 15,383★, **758 open issues**, ⚠️ **pushed 2024-03-13** |
| Licence | **BSD-3-Clause** ✅ (GitHub SPDX; PyPI classifier says "BSD License") |
| Python | ⚠️ `requires_python` is **null** — declares nothing; pure-python wheel + sdist |
| Access | Token + 积分 (points). Client-side permission checks: **none** |
| Maintenance | ⚠️ **The public repo is not the source of truth** — it has been idle since 2024-03 while PyPI shipped 1.4.29 in 2026-03 |

## 🚨 Traps

**1. 🚨 `pro_bar(adj="qfq")` is anchored to YOUR `end_date`, not to today.** ✅ verified in
`tushare/pro/data_pro.py`:

```python
if adj == 'hfq': data[col] = data[col] * data['adj_factor']
if adj == 'qfq': data[col] = data[col] * data['adj_factor'] / float(fcts['adj_factor'][0])
```

`fcts` arrives **newest-first**, so `[0]` is the factor on **the last date of your query window**.
**The same bar returns different prices depending on what `end_date` you asked for.** Two queries that
overlap disagree; concatenating them produces a series with a discontinuity at the seam and no error.
Ordinary qfq is merely non-reproducible over time — this is non-reproducible over *query parameters*.

**2. 🚨 `pre_close = close.shift(-1)` in that same function is only correct because rows arrive
descending.** ✅ Sort the frame ascending before letting tushare compute, and `pre_close`, `change`
and `pct_chg` all break **silently** — plausible-looking numbers, off by one row, wrong sign of
correction. `adj_factor` is also `bfill()`-ed, so a missing factor inherits a *later* one.

**Rule: `adj=None` (raw) or `adj="hfq"`. Never `adj="qfq"`.** `../SKILL.md` §2.

**3. 🚨 `stock_basic` defaults to `list_status='L'`** — live names only. Omit it and your universe is
silently survivorship-biased. Worse, delisting in China went from rare to common after the 2020
退市新规, so **the bias is time-varying**, which is harder to reason about than a constant one. You must
issue a second call with `list_status='D'` and union the results.

**4. 🚨 Your token travels in plaintext HTTP.** ✅ verified in 1.4.29 source:
`__http_url = 'http://api.waditu.com/dataapi'`. **No HTTPS option ships.** Token and returned data are
both readable on any hop. Treat the token as compromised-by-default: never reuse it as a password
elsewhere, and do not run it over untrusted networks.

**5. 🚨 The client validates nothing.** ✅ `DataApi.__getattr__` returns `partial(self.query, name)` —
**any** `api_name` is accepted client-side. A typo, a renamed endpoint and an endpoint you lack 积分
for are indistinguishable until the server answers. There is no `has`-style capability map; wrap every
call and inspect the server error.

**6. ⚠️ 积分 thresholds ✅, rate limits ❓.** Verified from `tushare.pro/document/1?doc_id=108`:
**120** = 日线 · **1000** = 港股日线 · **2000** = most 财务/基金/期货/指数 · **3000** = 限售股解禁 ·
**5000+** = fund portfolios, option daily, higher frequency. ❓ The per-minute call table and the RMB
prices circulating online are **secondhand — do not state them as fact.**

**7. ⚠️ No price limits, no suspension flag.** tushare gives you neither `high_limit`/`low_limit` nor a
per-bar 停牌 flag ✅. You cannot refuse a limit-up fill from tushare data alone — cross-reference
akshare's limit pools or a licensed vendor. `../SKILL.md` §3.

## 🔑 What it is genuinely best at: point-in-time fundamentals, cheaply

Every filing carries **报告期 `end_date`** (the period) and **公告日 `ann_date` / `f_ann_date`** (when
it was published). Annual reports land up to **four months** after period end. **Joining on `end_date`
leaks up to four months of future information** — on its own enough to turn a worthless value factor
into a spectacular backtest.

- `f_ann_date` = the *actual* announcement date of that version of the filing. Filter on it.
- `update_flag` distinguishes an original report from a later restatement — keep the version whose
  `f_ann_date <= as_of`, not the latest one.
- The 业绩预告 → 业绩快报 → 正式财报 → restatement chain means the same quarter has several rows.

This is the cheapest source in the ecosystem with a real PIT story. `rqdatac` is better
(`get_pit_financials_ex` carries an `if_adjusted` restatement flag) but is paid and metered;
`jqdatasdk` is geo-blocked outside mainland China. `_source-matrix.md` has the full comparison.

## Minimal correct usage

```python
import tushare as ts
pro = ts.pro_api("TOKEN")            # 🚨 sent over plaintext HTTP

# universe: BOTH halves, or you are survivorship-biased
live = pro.stock_basic(list_status="L", fields="ts_code,name,list_date,delist_date")
dead = pro.stock_basic(list_status="D", fields="ts_code,name,list_date,delist_date")
universe = pd.concat([live, dead], ignore_index=True)

# prices: raw or hfq, never qfq
bars = ts.pro_bar(ts_code="000001.SZ", adj="hfq",
                  start_date="20200101", end_date="20260901")   # leave rows DESCENDING

# point-in-time fundamentals
inc = pro.income(ts_code="000001.SZ",
                 fields="ts_code,ann_date,f_ann_date,end_date,update_flag,revenue")
pit = inc[inc["f_ann_date"] <= as_of]                     # publication date, not 报告期
pit = pit.sort_values("f_ann_date").groupby("end_date").tail(1)   # latest version KNOWN at as_of
```

## Where it fits

- 复权, 涨跌停, T+1, 停牌, 公告日 vs 报告期 in full: `../SKILL.md`
- Cross-source capability and defaults matrix: `_source-matrix.md`
- Free companions: `akshare.md` (widest coverage, no PIT) · `baostock.md` (delisted list, safe
  defaults, no PIT)
- The `vnpy_tushare` datafeed adapter: `../../china-trading-stack/references/vnpy.md`
