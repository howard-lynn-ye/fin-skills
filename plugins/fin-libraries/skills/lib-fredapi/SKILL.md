---
name: lib-fredapi
description: >-
  fredapi wraps FRED/ALFRED and is the primary anti-look-ahead tool in macro - and three of its
  four vintage methods are buggy in source. TRIGGER - fredapi, "from fredapi import Fred",
  Fred(api_key=), FRED_API_KEY, get_series, get_series_all_releases, get_series_as_of_date,
  get_series_first_release, get_series_vintage_dates, realtime_start, realtime_end, ALFRED,
  vintage, data revision, revised GDP, CPI or payrolls, Philadelphia Fed Real-Time Data Set,
  full-fred, DBnomics, "Bad Request. The value for variable api_key is not registered". Frozen at
  0.5.2 since 2024-05, so these bugs are current behaviour, not history you remember from an old
  version. SKIP for lib-edgartools, which is the skill for company filings and fundamentals. SKIP
  when the question is WHICH library to choose rather than how to use this one - that belongs to
  the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# fredapi

A thin client for FRED and ALFRED. **This is the primary anti-look-ahead tool in macro** — and its
vintage helpers do not do what their docstrings say.

| | |
|---|---|
| pip / import | `fredapi` / `from fredapi import Fred` |
| Version | 0.5.2 (2024-05-05) · 16 releases · last code change 2024; a doc tweak in 2026-01 |
| Licence | Apache-2.0 |
| Status | Stable / low activity, 1,655★. Root URL `https://api.stlouisfed.org/fred`. A **free API key is required** — pass `api_key=`, `api_key_file=`, or set `$FRED_API_KEY` |

Everything below comes from **reading the 0.5.2 source**, not from executing it.

## The trap that costs you money

**GDP, payrolls and CPI are revised for years, and `get_series` returns only today's numbers.**
FRED's own documented example: 2013Q4 GDP was 17102.5 (2014-01-30) → 17080.7 (2014-02-28) → 17089.6
(2014-03-27). Backtesting on the current series trades on figures published up to a decade later.

The second half of the same trap: **use `realtime_start` as your timestamp, not the observation
date.** A January figure is published in February. Indexing the January observation at January is a
one-to-three-month look-ahead. `realtime_start` is effectively the release date, and that is the
correct timestamp for a backtest. (It is also the second, independent reason `pandas_datareader`'s
FRED reader is not research-grade — verified: no `realtime`, `vintage`, `alfred` or `as_of` anywhere
in its source.)

## The four vintage methods

| Method | Returns |
|---|---|
| `get_series_all_releases(id, realtime_start=, realtime_end=)` | DataFrame with `date`, `realtime_start`, `value` — **every revision**. The only one that behaves |
| `get_series_as_of_date(id, as_of_date)` | Revisions known on/before a date — **see bug 1** |
| `get_series_first_release(id)` | First print only, ignoring all revisions |
| `get_series_vintage_dates(id)` | The dates on which the series was revised |

## Three bugs, verified in source

**1. `get_series_as_of_date` returns DUPLICATE date rows, not the latest per date.** The docstring
promises "a Series where each index is the observation date". The implementation is:

```python
df = self.get_series_all_releases(series_id)
data = df[df['realtime_start'] <= as_of_date]
return data
```

That is a **DataFrame with one row per revision** up to `as_of_date`. Treating it as a series
double-counts observations. **You must add `.groupby('date').last()` yourself.**

**2. `realtime_end` is silently dropped.** In `get_series_all_releases` the parse line is commented
out:

```python
# realtime_end = self._parse(child.get('realtime_end'))
```

So you cannot directly tell when a vintage was superseded — infer it from the next `realtime_start`.

**3. Both convenience methods download the ENTIRE revision history, unbounded.**
`get_series_as_of_date` and `get_series_first_release` both call `get_series_all_releases(series_id)`
with **no realtime bounds** — the defaults are `earliest_realtime_start = '1776-07-04'` and
`latest_realtime_end = '9999-12-31'`. For a heavily revised series that is a large payload every
call. **Cache it once rather than calling per-date in a loop.**

## Alternatives, and the licence traps among them

- **Philadelphia Fed Real-Time Data Set** — deeper history than ALFRED for core NIPA series: a
  **244-vintage matrix, 1965Q4 → 2026Q3**, plain XLSX, no key, no wrapper. Read it with
  `pd.read_excel(url, index_col=0)`; rows are observation periods, columns are vintages
  (`ROUTPUT99Q1` = real GDP as published in 1999Q1).
- **`full-fred`** — **GPL-3.0** and alpha (0.2). Prefer `fredapi` (Apache-2.0).
- **DBnomics** — vintages only at **dataset release level** (`WEO:2024-10`), not per-observation, and
  the client is **AGPL-3.0**.

Not verified: which FRED series actually carry ALFRED vintages (not all do), the runtime behaviour of
any of the above, and whether the three bugs persist on an unreleased branch.

## Minimal correct call

```python
from fredapi import Fred
fred = Fred(api_key='...')

# WRONG — current vintage; severe look-ahead in any backtest
gdp_now = fred.get_series('GDP')

# RIGHT — what was actually known on 2014-06-30
allr = fred.get_series_all_releases('GDP')          # date | realtime_start | value  (cache this)
pit = (allr[allr.realtime_start <= '2014-06-30']
       .sort_values('realtime_start')
       .groupby('date')['value'].last())            # REQUIRED: as_of_date does not do this

first = fred.get_series_first_release('GDP')        # initial prints only
print(fred.get_series_vintage_dates('GDP')[:5])
```

## See also

- `../../../fin-core/skills/fundamental-and-macro-data/SKILL.md` §5 — macro vintages
- `../../../fin-core/skills/fundamental-and-macro-data/references/fredapi.md` — the reference card
- `../lib-edgartools/SKILL.md` — the point-in-time source for company fundamentals

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`fundamental-and-macro-data`** (`../../../fin-core/skills/fundamental-and-macro-data/SKILL.md`).

