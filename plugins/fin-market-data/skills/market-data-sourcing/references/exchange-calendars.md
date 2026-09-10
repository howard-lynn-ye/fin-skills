# exchange_calendars

102 exchange session calendars — the de-facto standard, genuinely well-maintained, and still wrong in
four specific ways that produce silently bad intraday numbers. It models **sessions, not tradability**.

| | |
|---|---|
| pip | `exchange_calendars` · **4.13.2 (2026-03-10)** |
| GitHub | `gerrymanoim/exchange_calendars` — **666★**, 176 forks, 22 open issues, pushed 2026-09-02 |
| Licence | **Apache-2.0** ✅ (PyPI + GitHub agree) |
| Python | `>=3.10,<4` · wheel + sdist · classifiers through 3.14 |
| Maintenance | ✅ **Active.** Per-venue holiday commits land regularly ("XKRX: … CSAT updates" 2026-06-04, "XKLS: 2026 trading calendar" 2026-03-02, "XIDX: add 2026 holidays" 2026-07-22) |

## 🚨 Trap 1 — the default date bounds are computed from the wall clock

The single most damaging default, because it breaks **reproducibility** rather than raising.

✅ Verified by execution on 2026-09-04: `GLOBAL_DEFAULT_START = 2006-09-04`,
`GLOBAL_DEFAULT_END = 2027-09-04` — i.e. **today − 20 years** and **today + 1 year**, evaluated from
`pd.Timestamp.now()` **at import time**. `xc.get_calendar("XTKS")` returns
`first_session=2006-09-04`, `last_session=2027-09-03`.

Two distinct failures:

1. **Silent truncation.** A backtest starting in 2000 finds its calendar simply *begins* in 2006. No
   error, no warning — just a shorter sample and a survivorship-flavoured start date.
2. **The same code returns a different calendar tomorrow.** Anything cached, hashed, diffed, or
   compared against a calendar built on another day disagrees. A calendar hash in a reproducibility
   manifest changes every night.

**Always pass explicit `start=` and `end=`, and pin the version.** There is no way to make the
default deterministic.

## 🚨 Trap 2 — there is no XNSE

✅ Verified: `"XNSE" in xc.get_calendar_names()` → **False**.

- **India is `XBOM` only** (BSE, aliased `BSE`). NSE — which carries the overwhelming majority of
  Indian cash and derivatives volume — **has no calendar**.
- Using XBOM as an NSE proxy is a *reasonable approximation, silently made*. ⚠️ NSE and BSE holidays
  are near-identical in practice but not guaranteed identical, and special sessions (notably
  **Muhurat trading**, the ~1-hour Diwali session) are exchange-specific.
- 🚨 **`XBSE` is Bucharest, not Bombay.** The four-letter similarity has burned people. If you typed
  `XBSE` meaning India, you have been running a Romanian calendar.

## 🚨 Trap 3 — Korean CSAT late opens stop in 2021

KRX opens **one hour late (10:00)** on the national university entrance exam day (수능). The library
models this properly via `precomputed_csat_days` — and then stops updating it.

✅ Verified: the released 4.13.2 list holds 30 dates **ending 2020-12-03**. **Even current `master`
ends at 2021-11-18** — the 2026-06-04 "CSAT updates" commit added exactly one date.

```python
XKRX.session_open("2024-11-14")   # -> 09:00   🚨 WRONG; 2024-11-14 was a CSAT day, true open 10:00
```

**CSAT days for 2022, 2023, 2024, 2025 and 2026 are all wrong**, in the release *and* on master.
One day per year the Korean open is off by an hour: fatal for opening-auction and first-bar intraday
logic, harmless for daily bars.

## 🚨 Trap 4 — Singapore's lunch break is not modelled, ever

✅ Verified in source — `exchange_calendar_xses.py` states plainly: *"NOTE: For now, we are skipping
the intra-day break from 12:00 to 13:00."*

✅ Verified by execution on a **2010** session: `session_break_start` / `session_break_end` are
**`NaT`**, `has_break` is **`False`**.

SGX had a lunch break **until 2011**. So XSES reports a continuous session that did not exist for
every pre-2011 date. Any intraday VWAP, minutes-since-open feature, or session-shape vol estimator
built on pre-2011 Singapore data is computed over a fictional session.

## 🚨 Trap 5 — forward coverage is uneven; XBOM has zero sessions in 2027

✅ Executed: **`XBOM` has 245 sessions in 2026 and ZERO in 2027.** Its `bound_max` stops at
2026-12-31 because Indian holidays are announced annually and hand-maintained (a manual
"Update xbom.csv" commit on 2026-02-26). `XSES` likewise ends 2026-12-31.

By contrast `XHKG`, `XTAI`, `XTKS`, `XKRX`, `XASX` all run to 2027-09-03 (today + 1y) because their
holidays are rule-generated.

⚠️ **But rule-generated forward sessions are predictions, not facts.** Lunar holidays in
Korea/Taiwan/HK, Korean 임시공휴일, Taiwanese typhoon closures, Indian election days and HK
typhoon/black-rainstorm suspensions are all announced weeks or days ahead. **Treat any forward date
beyond the currently announced year as ❓ estimate.**

## ✅ What it gets right — each confirmed by executing the library

| Fact | Verified |
|---|---|
| **Japan's close moved 15:00 → 15:30 on 2024-11-05** | ✅ `2024-11-01` → close 15:00; `2024-11-05` → close 15:30. Lunch 11:30–12:30 on both. Unusually well tracked. |
| **Hong Kong's open moved 10:00 → 09:30 on 2011-03-07** | ✅ `2011-03-04` → open 10:00; `2011-03-07` → open 09:30. |
| Korea's ad-hoc 임시공휴일 | ✅ 2025-01-27 is correctly a non-session (sessions jump 01-24 → 01-31). |
| Taiwan's long Lunar New Year closure | ✅ 2025 sessions jump 01-22 → 02-03. |
| Korea's session-structure history | ✅ lunch break abolished 2000-05-22; close extended 15:00→15:30 on 2016-08-01; open changes 1978/1986/1995/1998. The 1998-12-07 transition independently corroborates the ±15% price-limit regime start. |
| Japan's lunch break | ✅ `break_start_times=11:30`, `break_end_times=12:30`. |

## Minimal correct call

```python
import exchange_calendars as xc

# NEVER call get_calendar() without start/end: the defaults are today-20y .. today+1y,
# recomputed from pd.Timestamp.now() at import, so the same code differs tomorrow.
cal = xc.get_calendar("XTKS", start="2005-01-01", end="2026-12-31")

sessions = cal.sessions_in_range("2024-10-28", "2024-11-08")
cal.session_open("2024-11-05")        # 09:00 JST
cal.session_close("2024-11-05")       # 15:30 JST  (was 15:00 before this date)
cal.session_break_start("2024-11-05") # 11:30 -- two sessions, not one

# India: XBOM only. "XNSE" in xc.get_calendar_names() is False. XBSE is Bucharest.
ind = xc.get_calendar("XBOM", start="2015-01-01", end="2026-12-31")  # no 2027 sessions exist
```

Pin it: `exchange_calendars==4.13.2` **plus** explicit `start=`/`end=`. Without both, your calendar is
a function of the wall clock and of whenever you last upgraded.

## Do not trust it for

- Korean CSAT-day opens 2022+ · Singapore pre-2011 intraday · Indian dates after 2026 · anything
  NSE-specific.
- 🚨 **Stock Connect trading days — no such calendar exists.** Connect's calendar is the
  *intersection* of HKEX and SSE/SZSE, minus settlement-mismatch days. It is shut on days HKEX itself
  is open. A Connect backtest built on `XHKG` trades on days the channel was closed.
- **Typhoon / black-rainstorm suspensions in HK** and other same-day events. No static calendar can
  carry these.
- ⚠️ **Tradability.** A day being a session says nothing about price limits, halts, VCM/VI
  cooling-offs, or short-selling bans. Those are separate gates you apply yourself — see
  `_decision-table.md` for the per-market data sources.

**Cross-check trick:** a session with **zero volume across the entire market** is a suspension your
calendar missed. Run it as an assertion over any Asian panel.

## Related

- `_decision-table.md` — which vendor covers which market, and the delisted-security axis.
- `yfinance.md` — the free source whose intraday timezone convention changed in 1.4.0; calendar joins
  across that boundary shift bars.
- `../../market-data-engineering/references/asof-joins.md` — joining a signal to a session grid is an
  as-of join, with all the exact-match hazards that implies.
