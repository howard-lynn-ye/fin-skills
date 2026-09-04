---
name: lib-tushare
description: >-
  tushare is the cheapest source of genuinely point-in-time A-share fundamentals, and it sends
  your token over plaintext HTTP. TRIGGER - tushare, tushare pro, "import tushare as ts",
  ts.pro_api, pro_bar, adj="qfq", stock_basic, list_status, daily_basic, adj_factor, income,
  balancesheet, f_ann_date, ann_date, update_flag, 报告期, 公告日, tushare token, 积分, waditu,
  api.waditu.com, "抱歉，您没有接口访问权限", tushare 权限不够. The public GitHub repo has been idle since 2024-03
  while PyPI kept shipping through 2026, so recalled behaviour does not match the installed wheel.
  SKIP for lib-akshare, which is the skill for breadth of free Chinese coverage rather than PIT.
  SKIP when the question is WHICH library to choose rather than how to use this one - that belongs
  to the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# tushare

The cheapest route to real point-in-time A-share fundamentals — and the owner of the most dangerous
adjustment implementation in the Chinese ecosystem, which is **worse than ordinary qfq**.

| | |
|---|---|
| pip / import | `tushare` / `import tushare as ts` |
| Version | 1.4.29 (2026-03-25) · 229 releases |
| Licence | BSD-3-Clause (GitHub SPDX; PyPI classifier says "BSD License") |
| Python | `requires_python` is **null** — declares nothing; pure-python wheel + sdist |
| Status | The public repo is **not** the source of truth: idle since 2024-03-13, PyPI shipped 2026-03 |

Access is token + 积分 (points). There are **no client-side permission checks** at all.

## The trap that costs you money

**`pro_bar(adj="qfq")` is anchored to YOUR `end_date`, not to today.** Verified in
`tushare/pro/data_pro.py`:

```python
if adj == 'hfq': data[col] = data[col] * data['adj_factor']
if adj == 'qfq': data[col] = data[col] * data['adj_factor'] / float(fcts['adj_factor'][0])
```

`fcts` arrives **newest-first**, so `[0]` is the factor on **the last date of your query window**.
The same bar returns different prices depending on what `end_date` you asked for. Two overlapping
queries disagree; concatenating them produces a series with a silent discontinuity at the seam.
Ordinary qfq is merely non-reproducible over time — this is non-reproducible over *query parameters*.

In the same function, `pre_close = close.shift(-1)` is correct **only because rows arrive
descending**. Sort the frame ascending before letting tushare compute and `pre_close`, `change` and
`pct_chg` all break silently — plausible numbers, off by one row, wrong sign of correction.
`adj_factor` is also `bfill()`-ed, so a missing factor inherits a *later* one.

**Rule: `adj=None` (raw) or `adj="hfq"`. Never `adj="qfq"`.** Leave the rows descending.

## Your token travels in plaintext HTTP

Verified in the 1.4.29 source: `__http_url = 'http://api.waditu.com/dataapi'`. **No HTTPS option
ships.** Token and returned data are both readable on any hop. Treat the token as
compromised-by-default: never reuse it as a password elsewhere, and do not run it over untrusted
networks.

The client also validates nothing. `DataApi.__getattr__` returns `partial(self.query, name)`, so
**any** `api_name` is accepted client-side. A typo, a renamed endpoint and an endpoint you lack 积分
for are indistinguishable until the server answers. There is no capability map — wrap every call and
inspect the server error.

## `stock_basic` is survivorship-biased by default

`stock_basic` defaults to **`list_status='L'`** — live names only. Omit the argument and your
universe is silently survivorship-biased. Worse, delisting in China went from rare to common after
the 2020 退市新规, so **the bias is time-varying**, which is harder to reason about than a constant
one. Issue a second call with `list_status='D'` and union the results.

tushare also gives you **no `high_limit`/`low_limit` and no per-bar 停牌 flag**. You cannot refuse a
limit-up fill from tushare data alone — cross-reference akshare's limit pools or a licensed vendor.

## What it is genuinely best at: point-in-time fundamentals, cheaply

Every filing carries **报告期 `end_date`** (the period) and **公告日 `ann_date` / `f_ann_date`** (when
it was published). Annual reports land up to **four months** after period end, so **joining on
`end_date` leaks up to four months of future information** — on its own enough to turn a worthless
value factor into a spectacular backtest.

- **`f_ann_date` is what makes point-in-time possible** — the actual announcement date of *that
  version* of the filing. Filter on it.
- `update_flag` distinguishes an original report from a later restatement. Keep the version whose
  `f_ann_date <= as_of`, not the latest one.
- The 业绩预告 → 业绩快报 → 正式财报 → restatement chain means one quarter has several rows.

积分 thresholds are verified: **120** = 日线 · **1000** = 港股日线 · **2000** = most 财务/基金/期货/指数
· **3000** = 限售股解禁 · **5000+** = fund portfolios, option daily, higher frequency. The per-minute
call-rate table and the RMB prices circulating online are secondhand — do not treat them as fact.

## Minimal correct call

```python
import pandas as pd, tushare as ts
pro = ts.pro_api("TOKEN")            # token is sent over PLAINTEXT HTTP

# universe: BOTH halves, or you are survivorship-biased
live = pro.stock_basic(list_status="L", fields="ts_code,name,list_date,delist_date")
dead = pro.stock_basic(list_status="D", fields="ts_code,name,list_date,delist_date")
universe = pd.concat([live, dead], ignore_index=True)

# prices: raw or hfq, never qfq; leave rows DESCENDING
bars = ts.pro_bar(ts_code="000001.SZ", adj="hfq",
                  start_date="20200101", end_date="20260901")

# point-in-time fundamentals: filter on the publication date, not 报告期
inc = pro.income(ts_code="000001.SZ",
                 fields="ts_code,ann_date,f_ann_date,end_date,update_flag,revenue")
pit = inc[inc["f_ann_date"] <= as_of]
pit = pit.sort_values("f_ann_date").groupby("end_date").tail(1)   # latest version KNOWN at as_of
```

## See also

- `../../../fin-china/skills/china-ashare-data/SKILL.md` — 复权, 涨跌停, T+1, 停牌, 公告日 vs 报告期
- `../../../fin-china/skills/china-ashare-data/references/tushare.md` — the verified reference card
- `../../../fin-china/skills/china-ashare-data/references/_source-matrix.md` — capability matrix
- `../lib-akshare/SKILL.md` — widest free coverage, limit pools, but no PIT

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`china-ashare-data`** (`../../../fin-china/skills/china-ashare-data/SKILL.md`).

