---
name: data-quality-validation
description: >-
  TRIGGER - stale closes, zero volume, missing trading sessions, duplicate timestamps,
  negative prices, invalid OHLC, outlier returns, validate downloaded bars, 数据质量, 缺失行情.
  Validate a daily OHLCV panel and report defects without filling, clipping or deleting data.
  SKIP for selecting a vendor or detecting price-adjustment convention (market-data-sourcing),
  session schedules (trading-calendars-and-sessions), corporate event accounting
  (corporate-actions-processing), and storage/as-of joins (market-data-engineering).
license: MIT
compatibility: Python with numpy and pandas; optional calendar libraries are checked when installed.
metadata:
  version: "0.1.0"
  verified_on: "2026-09-14"
allowed-tools: Bash(python:*)
---

# Data quality validation

Validate one instrument's delivered daily OHLCV panel before computing signals. Report bad
rows and missing sessions; leave the original values intact. A diagnostic is not an imputation
policy and an outlier is not automatically an error.

## Executable contract

✅ Implemented and executed 2026-09-14 in [data_quality.py](scripts/data_quality.py):

```python
from fin_skills.market_data.data_quality import validate_panel
report = validate_panel(bars, sessions=declared_sessions)
print(report.summary())
if not report.passed:
    # Preserve the delivery and investigate the reported rows before calculating signals.
    raise ValueError("bar delivery failed its declared quality contract")
```

The report has `checks`, per-check counts, severity and example dates, plus `stats` and
`passed`. Schema, timestamp, nonfinite-value, positive-price and OHLC-order defects fail.
Negative volume fails; zero volume is a warning because some markets or instruments have
valid no-trade sessions. Use `require_volume=False` for a source without volume.

Repeated-close and return thresholds are declared acceptance limits, not universal market
rules. Defaults are exposed as `max_stale_run`, `max_stale_frac`, `max_abs_return` and
`outlier_sigma`. The absolute-return cap is applied to simple returns symmetrically; robust
outlier screening uses centred log-return deviations. Invalid prices are not bridged to
manufacture a return. Zero-variance samples report an undefined volatility ratio.

`fin_skills.api.get("data_quality").run(bars=bars, dates=declared_sessions)` provides the same
check through the guard interface. `Bundle` uses the existing `bars` and `dates` slots.
Direct `validate_panel` also reports an unsorted input; Bundle rejects unsorted time-series
slots before the guard runs. Neither path sorts or repairs data for the caller.

## Calendar coverage is an explicit input

The session list must cover exactly the requested sample and venue. Dates are local session
labels. Convert observation instants to the exchange timezone before supplying them; UTC
midnight is not a universal exchange session boundary. Intraday missing-minute coverage is
outside this daily contract; use an explicit schedule with trading-calendars-and-sessions.

Missing `sessions` produces a warning that calendar coverage was not checked. With a declared
calendar, every absent session and every extra local date is counted. A calendar mismatch may
mean a wrong venue, timezone or session-label convention rather than a fabricated quote.

✅ Measured 2026-09-14: the seeded `fill_cost()` removes 24 of 504 declared sessions. The
comparison fill creates 23 repeated-close runs. Its retained volatility is 0.9768 times the
complete synthetic panel's volatility, while the moving average differs by over one basis
point on 306 sessions, with maximum relative error 37.01 basis points. These are synthetic
measurements, not general bounds. The validator itself never performs that fill.

## Repeated prices do not identify a repair

🚨 An identical close can be a real unchanged trade, a stale quote or an inserted non-session
row. Removing it changes the observation clock. A fresh quote following no prints can include
several sessions' accumulated return, so dropping stale days is not generally unbiased.

✅ Measured 2026-09-14: `mechanism_table()` shows identical delivered values under two possible
calendar interpretations. The sample has 156 repeated-close runs, with 35.02% of rows repeated.
Volatility is 0.3211 with rows retained and 0.3985 with them removed; their ratio is 1.2408.
For the latent real-session process, volatility is 0.3240: deleting rows overshoots. If the extra
rows are known non-sessions, the fresh-session comparison uses a different annualization clock.
The ratio is a sensitivity diagnostic; it does not reveal which interpretation is correct.

For a fixed volatility target with no other constraints, position size is inversely proportional
to estimated volatility. Caps, leverage limits, changing risk and execution can alter realized
exposure. Do not promise that a sample ratio equals future realized risk.

Run `python scripts/data_quality.py` from this skill directory. The demo is seeded, offline,
ASCII-only and writes no files. Tests cover missing sessions, malformed deliveries, constant
prices and guard integration without a live data source.
