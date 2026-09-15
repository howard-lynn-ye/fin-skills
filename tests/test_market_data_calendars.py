"""fin_skills.market_data.calendars - session index reproducibility, session-aware bars, venues.

The clock tests fake `pd.Timestamp.now()` at two FIXED dates, so they assert the same thing
whatever today is.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from conftest import requires
from fin_skills.market_data.calendars import (CLOSURES_2024, EARLY_CLOSES_2024,
                                              SESSION_SHAPES, clock_drift, default_bounds,
                                              federal_holidays_vs_sessions, forward_coverage,
                                              india_disagreement, library_agreement,
                                              library_sessions, minutes_per_session,
                                              panel_holes, reference_sessions, reindex_report,
                                              resample_comparison, schedule_bars, session_bars,
                                              session_minutes, sessions_under_clock,
                                              synthetic_minute_bars, truncation_cost,
                                              venue_overlap, verify_closures_against_library,
                                              wall_clock_bars)

DAY_A, DAY_B = "2021-03-01", "2021-03-02"


# ------------------------------------------------------------------ 1. the wall-clock bounds
def test_default_bounds_are_today_minus_20y_and_plus_1y():
    lo, hi = default_bounds("2026-06-15")
    assert lo == pd.Timestamp("2006-06-15") and hi == pd.Timestamp("2027-06-15")
    # a leap day: the DateOffset, not 365*20 days
    lo, _ = default_bounds("2024-02-29")
    assert lo == pd.Timestamp("2004-02-29")


def test_bounds_move_by_exactly_one_day_when_the_clock_does():
    a_lo, a_hi = default_bounds(DAY_A)
    b_lo, b_hi = default_bounds(DAY_B)
    assert (b_lo - a_lo) == pd.Timedelta(days=1)
    assert (b_hi - a_hi) == pd.Timedelta(days=1)


def test_two_clocks_one_day_apart_give_different_session_indices():
    """THE property: the default calendar is not reproducible across a midnight."""
    d = clock_drift(DAY_A, DAY_B)
    assert d["reproducible"] is False
    assert d["symmetric_difference"] > 0
    assert d["lost"] >= 1 and d["gained"] >= 1
    # sessions leave at the front and arrive at the back
    assert d["first_a"] < d["first_b"] and d["last_a"] < d["last_b"]
    assert d["bounds_a"] != d["bounds_b"]


def test_the_drift_grows_with_the_gap_between_the_clocks():
    one = clock_drift(DAY_A, DAY_B)["symmetric_difference"]
    month = clock_drift(DAY_A, "2021-04-01")["symmetric_difference"]
    assert month > one >= 2


def test_the_same_clock_twice_is_reproducible():
    d = clock_drift(DAY_A, DAY_A)
    assert d["reproducible"] is True and d["symmetric_difference"] == 0


@requires("exchange_calendars")
def test_the_faked_clock_really_reaches_exchange_calendars():
    a = sessions_under_clock(DAY_A)
    b = sessions_under_clock(DAY_B)
    assert a is not None and b is not None
    for got, day in ((a, DAY_A), (b, DAY_B)):
        start, end, sessions = got
        assert (start, end) == default_bounds(day)
        assert isinstance(sessions, pd.DatetimeIndex) and len(sessions) > 4000
    assert not a[2].equals(b[2])
    assert clock_drift(DAY_A, DAY_B)["source"] == "exchange_calendars"


@requires("exchange_calendars")
def test_faking_the_clock_leaves_the_interpreter_clean():
    before = {name: value for name, value in sys.modules.items()
              if name.split(".")[0] == "exchange_calendars"}
    sessions_under_clock("2021-05-03")
    # pd.Timestamp is restored (a leaked subclass would make now() return the fake date)
    assert pd.Timestamp("2020-01-01").__class__ is pd.Timestamp
    assert type(pd.Timestamp.now()).__name__ == "Timestamp"
    # Caller modules are preserved; only the isolated child's clock was changed.
    after = {name: value for name, value in sys.modules.items()
             if name.split(".")[0] == "exchange_calendars"}
    assert after == before


def test_truncation_is_counted_not_guessed():
    t = truncation_cost("2000-01-03", "2026-09-09", "2026-09-10")
    assert t["kept"] + t["dropped"] == t["panel_rows"]
    assert t["dropped"] > 1500
    assert t["default_start"] == "2006-09-10"


# --------------------------------------------------------- 2. session bars vs the wall clock
def test_session_minutes_excludes_the_lunch_break():
    tokyo = session_minutes(pd.Timestamp("2024-06-12"), SESSION_SHAPES["XTKS"])
    assert len(tokyo) == minutes_per_session(SESSION_SHAPES["XTKS"]) == 300
    hours = set(tokyo.hour)
    assert 11 in hours and 12 in hours          # 11:00-11:29 and 12:30-12:59 trade
    assert not any((t.hour == 11 and t.minute >= 30) or (t.hour == 12 and t.minute < 30)
                   for t in tokyo)
    nyse = session_minutes(pd.Timestamp("2024-06-12"), SESSION_SHAPES["XNYS"])
    assert len(nyse) == 390


def test_synthetic_bars_are_deterministic_for_a_seed():
    a = synthetic_minute_bars(n_sessions=5, seed=3)
    b = synthetic_minute_bars(n_sessions=5, seed=3)
    pd.testing.assert_series_equal(a, b)
    assert not a.equals(synthetic_minute_bars(n_sessions=5, seed=4))


def test_wall_clock_resampling_fabricates_most_of_its_buckets():
    r = resample_comparison(n_sessions=40)
    assert r["wall_fabricated"] > 0
    assert r["wall_nonempty"] == r["session_bars"] == 40 * 10
    assert r["wall_nonempty_pct"] < 20.0
    assert r["wall_buckets"] > 4 * r["session_bars"]


def test_trading_time_comparison_compresses_lunch_while_wall_bins_are_partial():
    r = resample_comparison(n_sessions=40)
    assert set(r["session_minutes_per_bar"]) == {60}
    assert 30 in r["wall_minutes_per_bar"]
    assert sum(r["wall_minutes_per_bar"]) == sum(r["session_minutes_per_bar"]) == 300


def test_the_midday_trough_is_an_artefact_of_the_bucket_edges():
    """Half the minutes, half the variance - on a process with a FLAT true profile."""
    r = resample_comparison(n_sessions=250)
    assert 0.35 < r["midday_trough"] < 0.65
    assert r["wall_var_spread"] > 1.8 > r["session_var_spread"]


def test_session_bars_partition_the_minutes_exactly():
    px = synthetic_minute_bars("XHKG", n_sessions=12)
    bars = session_bars(px, 30)
    assert int(bars["minutes"].sum()) == len(px)
    assert (bars["minutes"] == 30).all()
    wall = wall_clock_bars(px, "30min")
    assert int(wall["minutes"].sum()) == len(px)
    assert (wall["minutes"] == 0).any()          # the empty buckets the session grid has not


# ------------------------------------------------------------------- 3. cross-venue sessions
def test_reference_sessions_match_the_bundled_closure_table():
    for code in CLOSURES_2024:
        s = reference_sessions(code)
        assert len(s) == 262 - len(CLOSURES_2024[code])
        assert s.is_monotonic_increasing and not s.has_duplicates
        assert set(s.weekday) <= {0, 1, 2, 3, 4}
        assert not s.intersection(pd.to_datetime(list(CLOSURES_2024[code]))).size
    assert len(reference_sessions("XNYS")) == 252
    assert len(reference_sessions("XTKS")) == 245


def test_venue_overlap_is_asymmetric_in_both_directions():
    t = venue_overlap(["XNYS", "XTKS"]).iloc[0]
    assert t["only_a"] == 16 and t["only_b"] == 9
    assert t["only_a"] + t["both"] == t["n_a"]
    assert t["only_b"] + t["both"] == t["n_b"]


def test_the_union_calendar_is_not_a_session_everywhere():
    h = panel_holes()
    assert h["union"] == 261 and h["intersection"] == 222
    assert h["rows_not_a_session_everywhere"] == 39
    assert h["worst"] == "XTKS"
    assert sum(h["holes_per_venue"].values()) > 0


def test_a_cross_venue_reindex_loses_rows_in_the_invisible_direction():
    r = reindex_report("XTKS", "XNYS")
    assert r["holes"] == 16                      # visible as NaN
    assert r["source_sessions_dropped"] == 9      # not visible at all
    assert r["stale_rows_after_ffill"] > 0
    back = reindex_report("XNYS", "XTKS")
    assert back["holes"] == 9 and back["source_sessions_dropped"] == 16


@requires("exchange_calendars")
def test_the_bundled_tables_agree_with_the_library():
    assert set(verify_closures_against_library().values()) == {"agrees"}
    for code in CLOSURES_2024:
        assert reference_sessions(code).equals(library_sessions(code))


# --------------------------------------------------------------- 4. the library truth table
def test_a_public_holiday_list_is_not_a_trading_calendar():
    f = federal_holidays_vs_sessions()
    assert f["n_nyse_closures"] == 10
    assert f["federal_but_nyse_open"] == ["2024-10-14", "2024-11-11"]   # Columbus, Veterans
    assert f["nyse_closed_but_not_federal"] == ["2024-03-29"]            # Good Friday
    assert f["wrong_days"] == 3


@requires("holidays")
def test_the_financial_calendar_in_the_same_package_is_exact():
    f = federal_holidays_vs_sessions()
    assert f["financial_holidays_exact"] is True
    assert f["empty_until_first_lookup"] == 0     # holidays.US() is EMPTY until you look up


@requires("pandas_market_calendars")
def test_one_library_invents_a_monday_to_friday_year_and_the_other_refuses():
    rows = {(r["library"], r["calendar"]): r for r in forward_coverage(2027)}
    nse = rows[("pandas_market_calendars", "NSE")]
    assert nse["days"] == nse["weekdays"] and "fabricated" in nse["verdict"]
    nyse = rows[("pandas_market_calendars", "NYSE")]
    assert nyse["days"] < nyse["weekdays"]
    if ("exchange_calendars", "XBOM") in rows:
        assert rows[("exchange_calendars", "XBOM")]["verdict"].startswith("raised")


@requires("pandas_market_calendars")
@requires("exchange_calendars")
def test_the_two_calendar_libraries_agree_except_on_a_weather_day():
    rows = {r["venue"]: r for r in library_agreement()}
    assert rows["XNYS/NYSE"]["symmetric_difference"] == 0
    assert rows["XTKS/JPX"]["symmetric_difference"] == 0
    assert rows["XHKG/HKEX"]["symmetric_difference"] == 1
    assert "2024-09-06" in rows["XHKG/HKEX"]["dates"]
    ind = india_disagreement()
    assert ind["xnse_in_exchange_calendars"] is False
    assert ind["nse_in_pandas_market_calendars"] is True
    assert ind["disagreements"] == 3


def test_early_closes_are_recorded_per_venue():
    assert len(EARLY_CLOSES_2024["XNYS"]) == 3
    assert "2024-07-03" in EARLY_CLOSES_2024["XNYS"]
    assert EARLY_CLOSES_2024["XTKS"] == ()
    for code, days in EARLY_CLOSES_2024.items():
        sessions = set(reference_sessions(code))
        assert all(pd.Timestamp(d) in sessions for d in days)


# ----------------------------------------------------------------------------- housekeeping
def test_reference_sessions_rejects_what_it_cannot_serve():
    with pytest.raises(KeyError):
        reference_sessions("XNOPE")
    with pytest.raises(ValueError):
        reference_sessions("XNYS", year=2019)


def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.market_data.calendars")
    for head in ("1. exchange_calendars default bounds move with the wall clock",
                 "2. session-aware bars vs wall-clock resampling",
                 "3. two venues do not share a session index",
                 "4. calendar library coverage and disagreements"):
        assert head in out
    assert "symmetric difference 2" in out
    assert "0.4768" in out and "14264" in out
    assert "union of the five calendars: 261" in out
    assert out.strip().splitlines()[-1].startswith("The rule:")
    assert all(ord(c) < 128 for c in out)


def test_no_nan_leaks_into_the_measured_numbers():
    r = resample_comparison(n_sessions=30)
    assert not np.isnan(r["midday_trough"])
    assert all(isinstance(v, (int, float, str, list)) for v in r.values())


def _lunch_schedule():
    return pd.DataFrame({"open": [pd.Timestamp("2024-06-12 09:00", tz="Asia/Tokyo")],
                         "break_start": [pd.Timestamp("2024-06-12 11:30", tz="Asia/Tokyo")],
                         "break_end": [pd.Timestamp("2024-06-12 12:30", tz="Asia/Tokyo")],
                         "close": [pd.Timestamp("2024-06-12 15:00", tz="Asia/Tokyo")]},
                        index=["session-A"])


def test_schedule_bars_reset_after_break_and_preserve_partial_buckets():
    stamps = session_minutes(pd.Timestamp("2024-06-12"), SESSION_SHAPES["XTKS"])
    px = pd.Series(np.arange(len(stamps)), index=stamps.tz_localize("Asia/Tokyo"))
    bars = schedule_bars(px, _lunch_schedule(), 60)
    assert bars["minutes"].tolist() == [60, 60, 30, 60, 60, 30]
    assert bars.index.get_level_values("segment").tolist() == [0, 0, 0, 1, 1, 1]
    assert bars["minutes"].sum() == len(px)
    assert bars.loc[("session-A", 0)].iloc[-1]["end"].hour == 11


def test_missing_minutes_do_not_shift_buckets_and_empty_bucket_is_retained():
    idx = pd.date_range("2024-01-02 09:00", periods=90, freq="min", tz="UTC")
    px = pd.Series(np.arange(90, dtype=float), index=idx)
    schedule = pd.DataFrame({"open": [idx[0]], "close": [idx[-1] + pd.Timedelta(minutes=1)]})
    full = schedule_bars(px, schedule)
    missing = schedule_bars(px.drop(index=idx[:30].append(idx[[40]])), schedule)
    pd.testing.assert_index_equal(full.index, missing.index)
    assert missing["minutes"].tolist() == [0, 29, 30]
    assert np.isnan(missing.iloc[0]["last"])
    assert missing.iloc[1]["last"] == full.iloc[1]["last"]


def test_timezone_conversion_dst_early_close_and_overnight_labels():
    opens = [pd.Timestamp("2024-03-08 09:30", tz="America/New_York"),
             pd.Timestamp("2024-03-11 09:30", tz="America/New_York")]
    closes = [x + pd.Timedelta(minutes=n) for x, n in zip(opens, [60, 30])]
    schedule = pd.DataFrame({"open": opens, "close": closes}, index=["fri", "mon"])
    idx = pd.DatetimeIndex([opens[0], opens[1]]).tz_convert("UTC")
    result = schedule_bars(pd.Series([1., 2.], index=idx), schedule)
    assert len(result.loc["fri"]) == 2 and len(result.loc["mon"]) == 1
    starts = result.index.get_level_values("start").tz_convert("UTC")
    assert starts[0].hour == 14 and starts[-1].hour == 13
    night = pd.DataFrame({"open": [pd.Timestamp("2024-01-01 23:00", tz="UTC")],
                          "close": [pd.Timestamp("2024-01-02 01:00", tz="UTC")]},
                         index=["2024-01-02"])
    prices = pd.Series([3.], index=pd.DatetimeIndex([night.iloc[0]["open"]]))
    assert set(schedule_bars(prices, night).index.get_level_values("session")) == {"2024-01-02"}


def test_schedule_bars_reject_break_observation_overlap_and_mixed_timezone():
    schedule = _lunch_schedule()
    px = pd.Series([1.], index=pd.DatetimeIndex([schedule.iloc[0]["break_start"]]))
    with pytest.raises(ValueError, match="outside"):
        schedule_bars(px, schedule)
    px.index = px.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone"):
        schedule_bars(px, schedule)
    px.index = pd.DatetimeIndex([schedule.iloc[0]["open"]])
    overlap = pd.concat([schedule, schedule.rename(index={"session-A": "session-B"})])
    with pytest.raises(ValueError, match="nonoverlapping"):
        schedule_bars(px, overlap)
