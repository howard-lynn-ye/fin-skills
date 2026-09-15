#!/usr/bin/env python3
"""Trading sessions: reproducibility of the calendar, session-aware bars, cross-venue holes.

WHY: three separate failures live behind the word "calendar", and none of them raises.

  1. exchange_calendars derives its default date bounds from `pd.Timestamp.now()` AT
     IMPORT (`GLOBAL_DEFAULT_START = now - 20y`, `GLOBAL_DEFAULT_END = now + 1y`), so
     `get_calendar("XNYS")` with no start/end returns a DIFFERENT session index tomorrow.
     Anything anchored on it - a sample window, a calendar hash in a manifest, a cached
     session grid - is a function of the wall clock.
  2. Wall-clock resampling ignores the session. On a market with a lunch break the
     buckets that straddle the break carry half the trading minutes, so an intraday
     volatility profile grows a midday trough that is pure artefact; and a `resample`
     across a multi-session span fabricates a bucket for every overnight and weekend
     half-hour.
  3. Two venues do not share a session index. Reindexing one onto the other's calendar
     makes a hole on every date the second trades and the first does not, and ffill turns
     each hole into a stale price that looks like a real observation.

Everything here is a pure function over numpy/pandas. `exchange_calendars`,
`pandas_market_calendars` and `holidays` are imported INSIDE the functions that use them:
the module imports and the demo runs without any of them, using the bundled reference
tables, and cross-checks against each library when it is installed.

Usage:
    from calendars import clock_drift, resample_comparison, venue_overlap

    print(clock_drift("2026-06-15", "2026-06-16"))          # the reproducibility number
    print(venue_overlap(["XNYS", "XTKS"]))                   # the cross-venue holes
    print(resample_comparison())                             # session-aware vs wall clock
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

SEED = 20260910

# --------------------------------------------------------------------------------------
# Reference tables
#
# Weekday closures for calendar year 2024, transcribed from exchange_calendars 4.13.2 on
# 2026-09-10. They exist so the demo produces the SAME numbers with or without the library
# installed; `verify_closures_against_library()` re-derives them when it is present.
# None of these five venues had a weekend session in 2024.
# --------------------------------------------------------------------------------------
CLOSURES_2024: dict[str, tuple[str, ...]] = {
    "XNYS": ("2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
             "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25"),
    "XLON": ("2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06", "2024-05-27",
             "2024-08-26", "2024-12-25", "2024-12-26"),
    "XETR": ("2024-01-01", "2024-03-29", "2024-04-01", "2024-05-01", "2024-12-24",
             "2024-12-25", "2024-12-26", "2024-12-31"),
    "XTKS": ("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-08", "2024-02-12",
             "2024-02-23", "2024-03-20", "2024-04-29", "2024-05-03", "2024-05-06",
             "2024-07-15", "2024-08-12", "2024-09-16", "2024-09-23", "2024-10-14",
             "2024-11-04", "2024-12-31"),
    "XHKG": ("2024-01-01", "2024-02-12", "2024-02-13", "2024-03-29", "2024-04-01",
             "2024-04-04", "2024-05-01", "2024-05-15", "2024-06-10", "2024-07-01",
             "2024-09-06", "2024-09-18", "2024-10-01", "2024-10-11", "2024-12-25",
             "2024-12-26"),
}

# Early closes in 2024 (half-days), same source and date. A venue's close is not one time.
EARLY_CLOSES_2024: dict[str, tuple[str, ...]] = {
    "XNYS": ("2024-07-03", "2024-11-29", "2024-12-24"),
    "XLON": ("2024-12-24", "2024-12-31"),
    "XETR": ("2024-12-30",),
    "XTKS": (),
    "XHKG": ("2024-02-09", "2024-12-24", "2024-12-31"),
}

# Session shape in LOCAL minutes past midnight: (open, break_start, break_end, close).
# break_start/break_end are None for a continuous session. XTKS moved its close from
# 15:00 to 15:30 on 2024-11-05; this is the pre-change shape, which is what 2024 mostly is.
SESSION_SHAPES: dict[str, tuple[int, int | None, int | None, int]] = {
    "XNYS": (9 * 60 + 30, None, None, 16 * 60),
    "XLON": (8 * 60, None, None, 16 * 60 + 30),
    "XETR": (9 * 60, None, None, 17 * 60 + 30),
    "XTKS": (9 * 60, 11 * 60 + 30, 12 * 60 + 30, 15 * 60),
    "XHKG": (9 * 60 + 30, 12 * 60, 13 * 60, 16 * 60),
}

# US FEDERAL weekday holidays in 2024 - what `holidays.US(years=2024)` yields. Two of them
# are NYSE trading days and one NYSE closure is missing from the list entirely.
US_FEDERAL_WEEKDAY_2024: tuple[str, ...] = (
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19", "2024-07-04",
    "2024-09-02", "2024-10-14", "2024-11-11", "2024-11-28", "2024-12-25",
)


# --------------------------------------------------------------------------------------
# Sessions from the reference tables
# --------------------------------------------------------------------------------------
def reference_sessions(code: str, year: int = 2024) -> pd.DatetimeIndex:
    """Weekdays of `year` minus that venue's bundled closures. No library needed."""
    if code not in CLOSURES_2024:
        raise KeyError(f"no bundled closures for {code!r}; have {sorted(CLOSURES_2024)}")
    if year != 2024:
        raise ValueError("the bundled closure table covers 2024 only")
    days = pd.bdate_range(f"{year}-01-01", f"{year}-12-31")
    closed = pd.DatetimeIndex(pd.to_datetime(list(CLOSURES_2024[code])))
    return days.difference(closed)


def library_sessions(code: str, year: int = 2024) -> pd.DatetimeIndex | None:
    """The same sessions from exchange_calendars, or None when it is not installed.

    Always passes explicit start/end - the whole point of section 1.
    """
    try:
        import exchange_calendars as xc
    except ImportError:
        return None
    cal = xc.get_calendar(code, start=f"{year - 1}-12-01", end=f"{year + 1}-01-31")
    return pd.DatetimeIndex(cal.sessions_in_range(f"{year}-01-01", f"{year}-12-31"))


def verify_closures_against_library(year: int = 2024) -> dict[str, str]:
    """{code: 'agrees' | 'DIFFERS: ...' | 'library not installed'} for the bundled table."""
    out: dict[str, str] = {}
    for code in sorted(CLOSURES_2024):
        lib = library_sessions(code, year)
        if lib is None:
            out[code] = "library not installed"
            continue
        ref = reference_sessions(code, year)
        diff = ref.symmetric_difference(lib)
        out[code] = "agrees" if len(diff) == 0 else (
            "DIFFERS: " + ", ".join(str(d.date()) for d in diff[:6]))
    return out


# --------------------------------------------------------------------------------------
# 1. The wall-clock default bounds
# --------------------------------------------------------------------------------------
def default_bounds(today) -> tuple[pd.Timestamp, pd.Timestamp]:
    """exchange_calendars' own rule: (today - 20 years, today + 1 year), floored to a day.

    Source-verified in exchange_calendars/exchange_calendar.py lines 59 and 62 (4.13.2,
    read 2026-09-10) - both are module-level constants, so they freeze at IMPORT time.
    """
    t = pd.Timestamp(today).floor("D")
    return t - pd.DateOffset(years=20), t + pd.DateOffset(years=1)


_CLOCK_CACHE: dict[tuple[str, str], object] = {}


def sessions_under_clock(fake_today, code: str = "XNYS"):
    """Measure default bounds in a child process, without changing the caller's clock.

    Fakes only the child's pandas Timestamp.now during the library import. No modules
    are purged or patched in the caller; the function is safe to call in a live library.
    Returns None when the optional library is absent.
    """
    if importlib.util.find_spec("exchange_calendars") is None:
        return None
    key = (str(pd.Timestamp(fake_today).date()), code)
    if key not in _CLOCK_CACHE:
        program = r'''
import json, sys
import pandas as pd
real = pd.Timestamp
class FakeTimestamp(real):
    @classmethod
    def now(cls, tz=None):
        return real(sys.argv[1], tz=tz)
pd.Timestamp = FakeTimestamp
try:
    import exchange_calendars as xc
    from exchange_calendars import exchange_calendar as ecm
    result = [str(ecm.GLOBAL_DEFAULT_START), str(ecm.GLOBAL_DEFAULT_END),
              [str(d) for d in xc.get_calendar(sys.argv[2]).sessions]]
finally:
    pd.Timestamp = real
print(json.dumps(result))
'''
        proc = subprocess.run([sys.executable, "-c", program, *key], capture_output=True,
                              text=True, check=True, timeout=30)
        start, end, dates = json.loads(proc.stdout)
        _CLOCK_CACHE[key] = (pd.Timestamp(start), pd.Timestamp(end), pd.DatetimeIndex(dates))
    return _CLOCK_CACHE[key]


def _weekday_sessions(start, end) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end)))


def clock_drift(day_a, day_b, code: str = "XNYS") -> dict:
    """Diff the DEFAULT session index of two imports whose wall clocks differ.

    With exchange_calendars installed this imports it twice under faked clocks and diffs
    the real session indices. Without it, the same arithmetic runs on a Mon-Fri reference
    venue, which shows the identical structure: sessions fall off the front and appear at
    the back, so the two indices are not the same object.
    """
    a_start, a_end = default_bounds(day_a)
    b_start, b_end = default_bounds(day_b)
    got_a = sessions_under_clock(day_a, code)
    got_b = sessions_under_clock(day_b, code)
    if got_a is not None and got_b is not None:
        a_start, a_end, sa = got_a
        b_start, b_end, sb = got_b
        source = "exchange_calendars"
    else:
        sa = _weekday_sessions(a_start, a_end)
        sb = _weekday_sessions(b_start, b_end)
        source = "reference Mon-Fri venue (exchange_calendars not installed)"
    lost = sa.difference(sb)
    gained = sb.difference(sa)
    return {
        "source": source, "code": code,
        "bounds_a": (a_start.date().isoformat(), a_end.date().isoformat()),
        "bounds_b": (b_start.date().isoformat(), b_end.date().isoformat()),
        "n_a": len(sa), "n_b": len(sb),
        "first_a": sa[0].date().isoformat(), "last_a": sa[-1].date().isoformat(),
        "first_b": sb[0].date().isoformat(), "last_b": sb[-1].date().isoformat(),
        "lost": len(lost), "gained": len(gained),
        "symmetric_difference": len(lost) + len(gained),
        "lost_dates": [d.date().isoformat() for d in lost[:4]],
        "gained_dates": [d.date().isoformat() for d in gained[-4:]],
        "reproducible": len(lost) + len(gained) == 0,
    }


def truncation_cost(panel_start: str, panel_end: str, today, code: str = "XNYS") -> dict:
    """Rows a panel loses by being reindexed onto the DEFAULT (clock-derived) calendar."""
    start, end = default_bounds(today)
    panel = pd.DatetimeIndex(pd.bdate_range(panel_start, panel_end))
    kept = panel[(panel >= start) & (panel <= end)]
    return {"panel_rows": len(panel), "kept": len(kept), "dropped": len(panel) - len(kept),
            "default_start": start.date().isoformat(), "code": code,
            "note": "date-bound filter on a weekday panel only; excludes exchange-holiday effects"}


# --------------------------------------------------------------------------------------
# 2. Session-aware bars vs wall-clock resampling
# --------------------------------------------------------------------------------------
def session_minutes(day: pd.Timestamp, shape: Sequence) -> pd.DatetimeIndex:
    """Left-labelled one-minute stamps of one session, lunch break excluded."""
    o, bs, be, c = shape
    day = pd.Timestamp(day).normalize()
    if bs is None:
        return pd.date_range(day + pd.Timedelta(minutes=o), periods=c - o, freq="1min")
    am = pd.date_range(day + pd.Timedelta(minutes=o), periods=bs - o, freq="1min")
    pm = pd.date_range(day + pd.Timedelta(minutes=be), periods=c - be, freq="1min")
    return am.append(pm)


def minutes_per_session(shape: Sequence) -> int:
    o, bs, be, c = shape
    return (c - o) if bs is None else (bs - o) + (c - be)


def synthetic_minute_bars(code: str = "XTKS", n_sessions: int = 250,
                          annual_vol: float = 0.30, seed: int = SEED) -> pd.Series:
    """Synthetic weekday sessions with a fixed historical intraday shape, not a real calendar."""
    shape = SESSION_SHAPES[code]
    days = pd.bdate_range("2024-01-02", periods=n_sessions)
    idx = pd.DatetimeIndex(np.concatenate(
        [session_minutes(d, shape).values for d in days]))
    m = minutes_per_session(shape)
    sig = annual_vol / np.sqrt(252 * m)
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, sig, len(idx)))), index=idx)


def wall_clock_bars(px: pd.Series, freq: str = "30min") -> pd.DataFrame:
    """`px.resample(freq)` - what everybody writes. Keeps the empty buckets."""
    g = px.resample(freq)
    return pd.DataFrame({"last": g.last(), "minutes": g.count()})


def session_bars(px: pd.Series, bar_minutes: int = 30) -> pd.DataFrame:
    """Trading-time buckets over COMPLETE one-minute inputs; may cross a lunch break.

    This synthetic comparison intentionally compresses non-trading time. For delivered
    observations, missing minutes must not shift later boundaries: use schedule_bars.
    """
    if not isinstance(bar_minutes, int) or isinstance(bar_minutes, bool) or bar_minutes < 1:
        raise ValueError("bar_minutes must be a positive integer")
    if not isinstance(px.index, pd.DatetimeIndex) or not px.index.is_monotonic_increasing:
        raise ValueError("px needs a sorted DatetimeIndex")
    if px.index.has_duplicates:
        raise ValueError("px contains duplicate timestamps")
    sess = px.index.normalize().to_numpy()
    pos = pd.Series(np.arange(len(px)), index=px.index).groupby(sess).cumcount().to_numpy()
    bucket = pos // bar_minutes
    g = px.groupby([sess, bucket])
    return pd.DataFrame({"last": g.last(), "minutes": g.count()})



def schedule_bars(px: pd.Series, schedule: pd.DataFrame, bar_minutes: int = 30) -> pd.DataFrame:
    """Aggregate observations against explicit session open/close and optional lunch times.

    Schedule columns: open, close, optionally break_start/break_end; index: session label.
    Use timezone-aware instants for DST/overnight markets. All instants must be aware or
    all naive. Intervals are left-closed/right-open. A bucket resets at each segment open,
    so neither lunch nor missing observations move its boundaries. Empty buckets remain
    NaN, with count zero. No observations may fall outside the supplied schedule.
    """
    if not isinstance(px, pd.Series) or not isinstance(px.index, pd.DatetimeIndex):
        raise TypeError("px must be a Series with a DatetimeIndex")
    if not pd.api.types.is_numeric_dtype(px) or pd.api.types.is_bool_dtype(px) or pd.api.types.is_complex_dtype(px):
        raise TypeError("prices must have a real numeric dtype")
    if np.isinf(px.astype(float)).any():
        raise ValueError("prices must be finite or missing")
    if px.index.has_duplicates or px.index.hasnans or not px.index.is_monotonic_increasing:
        raise ValueError("px timestamps must be unique, nonmissing and sorted")
    if not isinstance(bar_minutes, int) or isinstance(bar_minutes, bool) or bar_minutes < 1:
        raise ValueError("bar_minutes must be a positive integer")
    if not isinstance(schedule, pd.DataFrame) or not {"open", "close"} <= set(schedule):
        raise TypeError("schedule needs open and close columns")
    if schedule.index.has_duplicates:
        raise ValueError("schedule labels must be unique")
    if ("break_start" in schedule) != ("break_end" in schedule):
        raise ValueError("supply both break columns or neither")
    rows, keys = [], []
    assigned = np.zeros(len(px), dtype=int)
    previous_close = None
    step = pd.Timedelta(minutes=bar_minutes)
    for label, row in schedule.iterrows():
        op, cl = pd.Timestamp(row["open"]), pd.Timestamp(row["close"])
        bounds = [op, cl]
        bs, be = row.get("break_start", pd.NaT), row.get("break_end", pd.NaT)
        if pd.isna(bs) != pd.isna(be):
            raise ValueError("a break needs both start and end")
        if pd.notna(bs):
            bounds = [op, pd.Timestamp(bs), pd.Timestamp(be), cl]
        if any(pd.isna(t) or (t.tzinfo is None) != (px.index.tz is None) for t in bounds):
            raise ValueError("schedule and observations need matching timezone awareness")
        if any(a >= b for a, b in zip(bounds, bounds[1:])):
            raise ValueError("schedule times must be strictly increasing")
        if previous_close is not None and op < previous_close:
            raise ValueError("sessions must be sorted and nonoverlapping")
        previous_close = cl
        for segment, (start, end) in enumerate(zip(bounds[::2], bounds[1::2])):
            for left in pd.date_range(start, end, freq=step, inclusive="left"):
                right = min(left + step, end)
                mask = (px.index >= left) & (px.index < right)
                assigned += mask
                values = px[mask].dropna()
                keys.append((label, segment, left))
                rows.append({"open": values.iloc[0] if len(values) else np.nan,
                             "high": values.max(), "low": values.min(),
                             "last": values.iloc[-1] if len(values) else np.nan,
                             "minutes": int(values.size), "end": right})
    if np.any(assigned != 1):
        raise ValueError("observations outside the schedule or assigned more than once")
    index = pd.MultiIndex.from_tuples(keys, names=["session", "segment", "start"])
    return pd.DataFrame(rows, index=index,
                        columns=["open", "high", "low", "last", "minutes", "end"])


def _within_session_returns(bars: pd.DataFrame) -> pd.Series:
    """Bar-to-bar log returns, never crossing a session boundary."""
    lvl = bars.index.get_level_values(0)
    return np.log(bars["last"]).groupby(lvl).diff().dropna()


def resample_comparison(code: str = "XTKS", n_sessions: int = 250,
                        bar_minutes: int = 30, profile_minutes: int = 60,
                        seed: int = SEED) -> dict:
    """The two measured resampling numbers: fabricated buckets, and the fake midday trough."""
    px = synthetic_minute_bars(code, n_sessions, seed=seed)
    shape = SESSION_SHAPES[code]
    m = minutes_per_session(shape)

    wall = wall_clock_bars(px, f"{bar_minutes}min")
    nonempty = int((wall["minutes"] > 0).sum())
    sess = session_bars(px, bar_minutes)

    # the intraday profile, at a bar size that does NOT divide the session cleanly
    wp = px.groupby([px.index.normalize().to_numpy(),
                     (px.index.hour * 60 + px.index.minute) // profile_minutes])
    wall_prof = pd.DataFrame({"last": wp.last(), "minutes": wp.count()})
    sess_prof = session_bars(px, profile_minutes)

    wall_var = (_within_session_returns(wall_prof)
                .groupby(level=1).apply(lambda s: float(np.var(s, ddof=1))))
    sess_var = (_within_session_returns(sess_prof)
                .groupby(level=1).apply(lambda s: float(np.var(s, ddof=1))))
    first_day = wall_prof.index.get_level_values(0)[0]
    wall_minutes = [int(v) for v in wall_prof.loc[first_day, "minutes"].to_numpy()]
    sess_minutes = [int(v) for v in sess_prof.loc[first_day, "minutes"].to_numpy()]

    straddle = [b for b, mm in zip(wall_prof.loc[first_day].index, wall_minutes)
                if mm != profile_minutes and b in wall_var.index]
    rest = [b for b in wall_var.index if b not in straddle]
    trough = (float(wall_var.loc[straddle].mean() / wall_var.loc[rest].mean())
              if straddle and rest else float("nan"))
    return {
        "code": code, "n_sessions": n_sessions, "minutes_per_session": m,
        "bar_minutes": bar_minutes,
        "wall_buckets": int(len(wall)), "wall_nonempty": nonempty,
        "wall_fabricated": int(len(wall)) - nonempty,
        "wall_nonempty_pct": round(100.0 * nonempty / len(wall), 1),
        "session_bars": int(len(sess)),
        "session_bars_per_session": int(len(sess) / n_sessions),
        "calendar_days_spanned": int((px.index[-1].normalize()
                                      - px.index[0].normalize()).days) + 1,
        "profile_minutes": profile_minutes,
        "wall_minutes_per_bar": wall_minutes,
        "session_minutes_per_bar": sess_minutes,
        "wall_var_spread": round(float(wall_var.max() / wall_var.min()), 3),
        "session_var_spread": round(float(sess_var.max() / sess_var.min()), 3),
        "midday_trough": round(trough, 4),
    }


# --------------------------------------------------------------------------------------
# 3. Cross-venue session holes
# --------------------------------------------------------------------------------------
def venue_overlap(codes: Iterable[str] = ("XNYS", "XLON", "XETR", "XTKS", "XHKG"),
                  year: int = 2024) -> pd.DataFrame:
    """Pairwise: sessions on A that are not sessions on B, both directions."""
    codes = list(codes)
    sets = {c: set(reference_sessions(c, year)) for c in codes}
    rows = []
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            rows.append({"a": a, "b": b, "n_a": len(sets[a]), "n_b": len(sets[b]),
                         "only_a": len(sets[a] - sets[b]),
                         "only_b": len(sets[b] - sets[a]),
                         "both": len(sets[a] & sets[b])})
    return pd.DataFrame(rows)


def panel_holes(codes: Iterable[str] = ("XNYS", "XLON", "XETR", "XTKS", "XHKG"),
                year: int = 2024) -> dict:
    """What a global panel on the UNION calendar actually contains."""
    codes = list(codes)
    sets = {c: set(reference_sessions(c, year)) for c in codes}
    union = set().union(*sets.values())
    inter = set.intersection(*sets.values())
    per = {c: len(union - sets[c]) for c in codes}
    return {"codes": codes, "union": len(union), "intersection": len(inter),
            "rows_not_a_session_everywhere": len(union) - len(inter),
            "holes_per_venue": per,
            "worst": max(per, key=lambda c: per[c])}


def reindex_report(source: str, target: str, year: int = 2024) -> dict:
    """Reindexing a `source`-venue series onto the `target` calendar: what ffill invents."""
    s = reference_sessions(source, year)
    t = reference_sessions(target, year)
    series = pd.Series(np.arange(len(s), dtype=float), index=s)
    onto = series.reindex(t)
    holes = int(onto.isna().sum())
    dropped = int(len(s) - len(s.intersection(t)))
    filled = onto.ffill()
    stale = int((filled.diff() == 0).sum())
    return {"source": source, "target": target, "n_source": len(s), "n_target": len(t),
            "holes": holes, "source_sessions_dropped": dropped,
            "stale_rows_after_ffill": stale,
            "pct_target_rows_fabricated": round(100.0 * holes / len(t), 2)}


# --------------------------------------------------------------------------------------
# 4. Which library lies about which market
# --------------------------------------------------------------------------------------
def federal_holidays_vs_sessions(year: int = 2024) -> dict:
    """`holidays.US` is a PUBLIC holiday list. NYSE is not a public-holiday calendar."""
    try:
        import holidays as _h
        fed = sorted(d for d in _h.US(years=year) if d.weekday() < 5)
        source = "holidays"
        try:
            fin = sorted(d for d in _h.financial_holidays("NYSE", years=year)
                         if d.weekday() < 5)
        except Exception:                                     # noqa: BLE001 - optional path
            fin = None
        lazy_len = len(_h.US())
    except ImportError:
        fed = [pd.Timestamp(d).date() for d in US_FEDERAL_WEEKDAY_2024]
        source = "bundled US federal table (holidays not installed)"
        fin, lazy_len = None, None
    weekdays = set(d.date() for d in pd.bdate_range(f"{year}-01-01", f"{year}-12-31"))
    sessions = set(d.date() for d in reference_sessions("XNYS", year))
    closed = weekdays - sessions
    open_but_federal = sorted(d for d in fed if d in sessions)
    closed_but_not_federal = sorted(d for d in closed if d not in set(fed))
    return {
        "source": source, "n_federal_weekday": len(fed), "n_nyse_closures": len(closed),
        "federal_but_nyse_open": [str(d) for d in open_but_federal],
        "nyse_closed_but_not_federal": [str(d) for d in closed_but_not_federal],
        "wrong_days": len(open_but_federal) + len(closed_but_not_federal),
        "financial_holidays_exact": (None if fin is None
                                     else sorted(fin) == sorted(closed)),
        "empty_until_first_lookup": lazy_len,
    }


def forward_coverage(year: int = 2027) -> list[dict]:
    """Past its holiday table, does the library say so or invent a Mon-Fri year?"""
    weekdays = len(pd.bdate_range(f"{year}-01-01", f"{year}-12-31"))
    rows: list[dict] = []
    try:
        import pandas_market_calendars as mcal
        for name in ("NYSE", "NSE"):
            try:
                n = len(mcal.get_calendar(name).valid_days(f"{year}-01-01", f"{year}-12-31"))
                rows.append({"library": "pandas_market_calendars", "calendar": name,
                             "days": n, "weekdays": weekdays,
                             "verdict": "ALL WEEKDAYS - fabricated" if n == weekdays
                                        else "holidays applied"})
            except Exception as exc:                          # noqa: BLE001 - report it
                rows.append({"library": "pandas_market_calendars", "calendar": name,
                             "days": None, "weekdays": weekdays,
                             "verdict": f"raised {type(exc).__name__}"})
    except ImportError:
        rows.append({"library": "pandas_market_calendars", "calendar": "-", "days": None,
                     "weekdays": weekdays, "verdict": "not installed"})
    try:
        import exchange_calendars as xc
        for name in ("XNYS", "XBOM"):
            try:
                cal = xc.get_calendar(name, start=f"{year}-01-01", end=f"{year}-12-31")
                n = len(cal.sessions)
                rows.append({"library": "exchange_calendars", "calendar": name, "days": n,
                             "weekdays": weekdays,
                             "verdict": "ALL WEEKDAYS - fabricated" if n == weekdays
                                        else "holidays applied"})
            except Exception as exc:                          # noqa: BLE001 - report it
                rows.append({"library": "exchange_calendars", "calendar": name,
                             "days": None, "weekdays": weekdays,
                             "verdict": f"raised {type(exc).__name__}"})
    except ImportError:
        rows.append({"library": "exchange_calendars", "calendar": "-", "days": None,
                     "weekdays": weekdays, "verdict": "not installed"})
    return rows


def india_disagreement(year: int = 2024) -> dict:
    """exchange_calendars has no NSE at all; pandas_market_calendars has one. They differ."""
    out: dict = {"xnse_in_exchange_calendars": None, "nse_in_pandas_market_calendars": None,
                 "n_mcal_nse": None, "n_ec_xbom": None, "disagreements": None,
                 "dates": []}
    try:
        import exchange_calendars as xc
        names = xc.get_calendar_names()
        out["xnse_in_exchange_calendars"] = "XNSE" in names
        cal = xc.get_calendar("XBOM", start=f"{year - 1}-12-01", end=f"{year + 1}-01-31")
        bom = set(d.date() for d in cal.sessions_in_range(f"{year}-01-01", f"{year}-12-31"))
        out["n_ec_xbom"] = len(bom)
    except ImportError:
        bom = None
    try:
        import pandas_market_calendars as mcal
        out["nse_in_pandas_market_calendars"] = "NSE" in mcal.get_calendar_names()
        nse = set(d.date() for d in
                  mcal.get_calendar("NSE").valid_days(f"{year}-01-01", f"{year}-12-31"))
        out["n_mcal_nse"] = len(nse)
    except ImportError:
        nse = None
    if bom is not None and nse is not None:
        diff = sorted(nse ^ bom)
        out["disagreements"] = len(diff)
        out["dates"] = [str(d) for d in diff]
    return out


def library_agreement(year: int = 2024) -> list[dict]:
    """Do the two calendar libraries agree on the venues they both claim to model?"""
    pairs = [("XNYS", "NYSE"), ("XLON", "LSE"), ("XTKS", "JPX"), ("XHKG", "HKEX")]
    rows = []
    for ec_name, mcal_name in pairs:
        ec = library_sessions(ec_name, year)
        try:
            import pandas_market_calendars as mcal
            mc = pd.DatetimeIndex(mcal.get_calendar(mcal_name)
                                  .valid_days(f"{year}-01-01", f"{year}-12-31")).normalize()
            mc = mc.tz_localize(None) if mc.tz is not None else mc
        except ImportError:
            mc = None
        if ec is None or mc is None:
            rows.append({"venue": f"{ec_name}/{mcal_name}", "ec": None if ec is None
                         else len(ec), "mcal": None if mc is None else len(mc),
                         "symmetric_difference": None, "dates": ""})
            continue
        diff = ec.symmetric_difference(mc)
        rows.append({"venue": f"{ec_name}/{mcal_name}", "ec": len(ec), "mcal": len(mc),
                     "symmetric_difference": len(diff),
                     "dates": ", ".join(str(d.date()) for d in diff[:4])})
    return rows


# --------------------------------------------------------------------------------------
def _fmt(rows: list[dict]) -> str:
    return pd.DataFrame(rows).to_string(index=False) if rows else "(none)"


if __name__ == "__main__":
    print("=== 1. exchange_calendars default bounds move with the wall clock ===")
    print("    the rule, verified 2026-09-14 in exchange_calendar.py lines 59/62 (4.13.2):")
    for day in ("2026-06-15", "2026-06-16"):
        s, e = default_bounds(day)
        print(f"      clock {day} -> GLOBAL_DEFAULT_START {s.date()}  "
              f"GLOBAL_DEFAULT_END {e.date()}")
    d1 = clock_drift("2026-06-15", "2026-06-16")
    print(f"    measured via {d1['source']}, calendar {d1['code']}:")
    print(f"      import on 2026-06-15: {d1['n_a']} sessions, "
          f"{d1['first_a']} .. {d1['last_a']}")
    print(f"      import on 2026-06-16: {d1['n_b']} sessions, "
          f"{d1['first_b']} .. {d1['last_b']}")
    print(f"      lost {d1['lost']} {d1['lost_dates']}, gained {d1['gained']} "
          f"{d1['gained_dates']}, symmetric difference {d1['symmetric_difference']}")
    print("      ONE DAY of wall clock and the session index is not the same object.")
    print("    the same diff at longer horizons:")
    for other in ("2026-06-22", "2026-07-15"):
        d = clock_drift("2026-06-15", other)
        print(f"      vs {other}: lost {d['lost']:>3}  gained {d['gained']:>3}  "
              f"symmetric difference {d['symmetric_difference']:>3}")
    t = truncation_cost("2000-01-03", "2026-09-09", "2026-09-10")
    print(f"    silent truncation: a {t['panel_rows']}-business-day 2000-2026 panel "
          f"reindexed onto the")
    print(f"      default date bounds (start {t['default_start']}) keep {t['kept']} rows and "
          f"drops {t['dropped']}.")
    print(f"      {t['note']}.")
    print("    FIX: xc.get_calendar(code, start=..., end=...), always, and pin the version.")

    print("\n=== 2. session-aware bars vs wall-clock resampling (lunch-break venue) ===")
    r = resample_comparison()
    print(f"    {r['code']}: {r['n_sessions']} sessions x {r['minutes_per_session']} "
          f"trading minutes, spanning {r['calendar_days_spanned']} calendar days")
    print(f"    px.resample('{r['bar_minutes']}min') -> {r['wall_buckets']} buckets, "
          f"{r['wall_nonempty']} with data ({r['wall_nonempty_pct']}%), "
          f"{r['wall_fabricated']} fabricated")
    print(f"    session bucketing   -> {r['session_bars']} bars, "
          f"{r['session_bars_per_session']} per session, none empty")
    print(f"    at {r['profile_minutes']}-minute bars, minutes of trading per bar in one "
          f"session:")
    print(f"      wall clock : {r['wall_minutes_per_bar']}   <- the two that straddle lunch")
    print(f"      session    : {r['session_minutes_per_bar']}")
    print(f"    intraday variance spread across buckets: wall clock "
          f"{r['wall_var_spread']}x, session {r['session_var_spread']}x")
    print(f"    the lunch-straddling bars carry {r['midday_trough']}x the variance of the "
          f"rest -")
    print("      a duration artefact here; this synthetic process has no lunch shock.")

    print("\n=== 3. two venues do not share a session index (2024) ===")
    print(_fmt(venue_overlap().to_dict("records")))
    h = panel_holes()
    print(f"    union of the five calendars: {h['union']} dates; a session on ALL five: "
          f"{h['intersection']}")
    print(f"    -> {h['rows_not_a_session_everywhere']} rows of a global panel are not a "
          f"session somewhere")
    print(f"    holes per venue on the union grid: {h['holes_per_venue']}")
    rr = reindex_report("XTKS", "XNYS")
    print(f"    reindex {rr['source']} onto {rr['target']}: {rr['holes']} holes in "
          f"{rr['n_target']} rows "
          f"({rr['pct_target_rows_fabricated']}%), and {rr['source_sessions_dropped']} "
          f"{rr['source']} sessions vanish")
    print(f"    ffill turns those holes into {rr['stale_rows_after_ffill']} stale rows that "
          f"read as real observations")

    print("\n=== 4. calendar library coverage and disagreements ===")
    f = federal_holidays_vs_sessions()
    print(f"    holidays.US vs NYSE 2024 [{f['source']}]: {f['n_federal_weekday']} federal "
          f"weekday holidays, {f['n_nyse_closures']} NYSE closures")
    print(f"      federal holiday but NYSE OPEN : {f['federal_but_nyse_open']}")
    print(f"      NYSE CLOSED but not federal   : {f['nyse_closed_but_not_federal']}")
    print(f"      -> wrong on {f['wrong_days']} of {f['n_nyse_closures']} closures")
    if f["financial_holidays_exact"] is not None:
        print(f"      holidays.financial_holidays('NYSE') exactly right: "
              f"{f['financial_holidays_exact']}")
    if f["empty_until_first_lookup"] is not None:
        print(f"      len(holidays.US()) with no years= and no lookup yet: "
              f"{f['empty_until_first_lookup']}  <- set() of it is EMPTY")
    print("    same venue, both calendar libraries, 2024 sessions:")
    print(_fmt(library_agreement()))
    print("    past the end of the holiday table (2027):")
    print(_fmt(forward_coverage()))
    ind = india_disagreement()
    print(f"    India: XNSE in exchange_calendars = "
          f"{ind['xnse_in_exchange_calendars']}, NSE in pandas_market_calendars = "
          f"{ind['nse_in_pandas_market_calendars']}")
    print(f"      mcal NSE {ind['n_mcal_nse']} sessions vs ec XBOM {ind['n_ec_xbom']}; "
          f"they disagree on {ind['disagreements']} dates {ind['dates']}")
    print("    bundled 2024 closure table vs exchange_calendars:",
          verify_closures_against_library())

    print("\nThe rule: pin the calendar (explicit start/end, a pinned version, its hash in "
          "the manifest), bucket delivered bars against explicit session segments, and never reindex "
          "one venue onto another's sessions without reporting the holes.")
