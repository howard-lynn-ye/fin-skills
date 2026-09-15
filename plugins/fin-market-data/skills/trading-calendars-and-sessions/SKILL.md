---
name: trading-calendars-and-sessions
description: >-
  TRIGGER - trading calendar, market sessions, early close, lunch break, holidays.US,
  exchange_calendars, pandas_market_calendars, DateOutOfBounds, resample has empty bars,
  aligning Tokyo to New York, 交易日历, 休市. Pin session bounds, compare venue calendars,
  and aggregate against explicit sessions. SKIP for vendor choice and price adjustments
  (market-data-sourcing), stale closes and bad OHLC (data-quality-validation), corporate
  events (corporate-actions-processing), storage and as-of joins (market-data-engineering),
  and settlement or lot-size rules (asia-pacific-markets).
license: MIT
compatibility: Python with numpy and pandas; optional calendar libraries are checked when installed.
metadata:
  version: "0.1.0"
  verified_on: "2026-09-14"
allowed-tools: Bash(python:*)
---

# Trading calendars and sessions

Use an explicit date window and a versioned calendar. Keep session labels separate from
observation timestamps. An overnight session's date need not be its opening civil date.

## Reproducibility trap

✅ Source-verified 2026-09-14: installed `exchange_calendars 4.13.2`,
`exchange_calendars/exchange_calendar.py`, lines 59 and 62, derives default bounds from
`pd.Timestamp.now()` at import, minus 20 years and plus 1 year. Calendar-specific bounds
can narrow this range. Supply `start=` and `end=` and record the package version.

✅ Measured 2026-09-14 by [calendars.py](scripts/calendars.py), with the library installed:

| Simulated import date | Default XNYS first session | Last session | Sessions |
|---|---|---|---|
| 2026-06-15 | 2006-06-15 | 2027-06-15 | 5282 |
| 2026-06-16 | 2006-06-16 | 2027-06-16 | 5282 |

The second index loses 2006-06-15 and gains 2027-06-16: symmetric difference 2. The test
runs imports in child processes so it does not modify a live application's pandas clock.
Without the library, the demo labels its weekday-only fallback; it is not an exchange calendar.

```python
import exchange_calendars as xc
cal = xc.get_calendar("XNYS", start="2000-01-01", end="2026-12-31")
```

## Aggregate observations using the supplied schedule

✅ Executed reference implementation, 2026-09-14: `schedule_bars(px, schedule, bar_minutes)`
accepts a price Series and explicit `open`, `close`, optional `break_start`, `break_end`
columns. The schedule index is the session label. It uses left-closed/right-open intervals,
starts a new bucket after lunch, retains empty buckets as missing values with count zero,
and rejects observations outside the schedule. Missing observations do not shift boundaries.
Use timezone-aware instants for DST and overnight sessions. Half-days use their actual close.

The `minutes` output counts nonmissing price observations; it equals trading minutes only
for complete one-minute input. It does not impute missing observations or validate whether
an externally supplied schedule matches an exchange announcement.

🚨 `session_bars` is a separate synthetic comparison: it counts available rows and compresses
non-trading time, so a bucket can span lunch. It requires complete one-minute observations.
Do not use it to hide holes in a delivered dataset.

✅ Measured 2026-09-14: on the demo's fixed Tokyo-like schedule, wall-clock half-hour resampling
creates 16764 buckets, only 2500 containing observations. At hourly resolution, wall-clock
counts are `[60, 60, 30, 30, 60, 60]`, while compressed trading-time counts are
`[60, 60, 60, 60, 60]`. The synthetic process has no lunch shock; partial buckets have
0.4768 times the variance of full buckets in this seed. This is a bucket-duration comparison,
not evidence that actual lunch or overnight volatility is zero. The synthetic weekdays and
fixed historical shape are not an up-to-date Tokyo schedule.

## Cross-venue holes and library disagreements

✅ Measured 2026-09-14: bundled 2024 closure tables agree with installed
`exchange_calendars 4.13.2`. XNYS has 252 sessions and XTKS 245; 16 occur only on XNYS and
9 only on XTKS. Reindexing Tokyo onto New York adds holes and drops Tokyo-only observations.
Keep the holes and a venue-open mask rather than treating carried prices as fresh trades.

✅ Measured with `pandas_market_calendars 5.4.0` and `holidays 0.104`, 2026-09-14:

- `holidays.US(years=2024)` disagrees with XNYS on 3 dates. Its federal-holiday list is not
  the financial calendar; `financial_holidays("NYSE", years=2024)` agrees in this test.
- The HKEX/XHKG libraries disagree on 2024-09-06. Compare against an exchange notice before
  choosing one; library agreement alone is not exchange verification.
- The installed NSE calendar returns all 261 weekdays for 2027, while XBOM rejects that
  requested window. This flags incomplete forward coverage; neither result verifies the
  actual future exchange schedule. NSE and BSE are different venues, not interchangeable.

Run `python scripts/calendars.py` from this skill directory to reproduce the table and
optional library comparisons. Re-run after a calendar dependency changes. The bundled closures
cover only 2024 and intentionally reject other years; use a maintained schedule in production.
