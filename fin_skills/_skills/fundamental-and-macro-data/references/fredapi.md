# fredapi — FRED/ALFRED vintages

`fredapi` **0.5.2 (2024-05-05)** · **Apache-2.0** · 1,655★ · stable/low-activity (last code change
2024; a doc tweak in 2026-01). Root URL `https://api.stlouisfed.org/fred`. **A free API key is
required** — pass `api_key=`, `api_key_file=`, or set `$FRED_API_KEY`.

⚠️ Everything below is from **reading the 0.5.2 source**, not from executing it (no key available).

## Why it matters

**This is the primary anti-look-ahead tool in macro.** GDP, payrolls and CPI are revised for years.
FRED's own documented example: 2013Q4 GDP was 17102.5 (2014-01-30) → 17080.7 (2014-02-28) →
17089.6 (2014-03-27). Backtesting on the current series trades on numbers published up to a decade
later.

## The four vintage methods

| Method | Returns |
|---|---|
| `get_series_all_releases(id, realtime_start=, realtime_end=)` | DataFrame with `date`, `realtime_start`, `value` — **every revision** |
| `get_series_as_of_date(id, as_of_date)` | Revisions known on/before a date — ⚠️ **see bug 1** |
| `get_series_first_release(id)` | First print only, ignoring all revisions |
| `get_series_vintage_dates(id)` | The dates on which the series was revised |

## 🚨 Three bugs, verified in source

**1. `get_series_as_of_date` does not do what its docstring says.** The docstring promises "a Series
where each index is the observation date". The implementation is:

```python
df = self.get_series_all_releases(series_id)
data = df[df['realtime_start'] <= as_of_date]
return data
```

It returns a **DataFrame with duplicate `date` rows** — every revision up to `as_of_date`, not the
latest one per date. Treating it as a series double-counts observations. **You must add
`.groupby('date').last()` yourself.**

**2. `realtime_end` is silently dropped.** In `get_series_all_releases` the parse line is commented
out:

```python
# realtime_end = self._parse(child.get('realtime_end'))
```

So you cannot directly tell when a vintage was superseded — infer it from the next `realtime_start`.

**3. Unbounded downloads.** Both `get_series_as_of_date` and `get_series_first_release` call
`get_series_all_releases(series_id)` with **no realtime bounds** — the defaults are
`earliest_realtime_start = '1776-07-04'` and `latest_realtime_end = '9999-12-31'`. Every call pulls
the entire revision history. For heavily revised series this is a large payload. **Cache it once
rather than calling per-date in a loop.**

## Correct usage

```python
from fredapi import Fred
fred = Fred(api_key='...')

# WRONG — current vintage; severe look-ahead in any backtest
gdp_now = fred.get_series('GDP')

# RIGHT — what was actually known on 2014-06-30
allr = fred.get_series_all_releases('GDP')          # date | realtime_start | value
pit = (allr[allr.realtime_start <= '2014-06-30']
       .sort_values('realtime_start')
       .groupby('date')['value'].last())            # REQUIRED: as_of_date does not do this

first = fred.get_series_first_release('GDP')        # initial prints only
print(fred.get_series_vintage_dates('GDP')[:5])
```

## 🚨 Use `realtime_start` as the timestamp

A January figure is published in February. Indexing the January observation at January is a 1–3
month look-ahead. **`realtime_start` is effectively the release date** — that is the correct
timestamp for a backtest, and it is a second, independent reason `pandas_datareader`'s FRED reader
is not research-grade (✅ verified: no `realtime|vintage|alfred|as_of` anywhere in its source).

## Alternatives and complements

- **Philadelphia Fed Real-Time Data Set** — deeper history than ALFRED for core NIPA series: a
  **244-vintage matrix, 1965Q4 → 2026Q3**, plain XLSX, no key, no wrapper. Read it with
  `pd.read_excel(url, index_col=0)`; rows are observation periods, columns are vintages
  (`ROUTPUT99Q1` = real GDP as published in 1999Q1).
- **`full-fred`** — 🚨 **GPL-3.0**, alpha (0.2). Prefer `fredapi` (Apache-2.0).
- **DBnomics** — vintages only at **dataset release level** (`WEO:2024-10`), not per-observation,
  and the client is 🚨 **AGPL-3.0**.

## ❓ Not verified
Which FRED series actually have ALFRED vintages (not all do) · runtime behaviour of any of the above
· whether the three bugs persist in an unreleased branch.
