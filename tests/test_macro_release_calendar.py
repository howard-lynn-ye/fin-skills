"""fin_skills.macro.release_calendar - when a number becomes tradeable, to the minute.

Properties under test: available_at is timezone-correct on both sides of a DST boundary;
the transcribed EIA holiday exceptions really do fall on the weekdays EIA printed; the
as-of join's look-ahead vanishes only for the release-timestamp stamping and only the
timestamp survives a change of decision clock; the latency/capture curve is monotone and
the spliced fill assumption is wrong in opposite directions on the two sides of
2020-06-03; the demo is deterministic and prints the rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.macro import release_calendar as rc


def test_the_verified_embargo_timeline_carries_the_dates_the_skill_quotes():
    tl = rc.EMBARGO_TIMELINE.set_index("date")
    assert "2018-08-01" in tl.index and tl.loc["2018-08-01", "agency"] == "USDA NASS"
    assert "2020-06-03" in tl.index and tl.loc["2020-06-03", "agency"] == "DOL"
    assert list(tl.index) == sorted(tl.index)
    assert rc.LOCKUP_END == pd.Timestamp("2020-06-03")
    # the release clocks the skill tabulates
    r = rc.RELEASES.set_index("release")
    assert r.loc["Employment Situation", "clock"] == "08:30"
    assert r.loc["WASDE", "clock"] == "12:00"
    assert r.loc["Weekly Petroleum Status Report", "clock"] == "10:30"
    assert len(rc.WASDE_2026) == 12
    assert all(8 <= int(d[-2:]) <= 12 for d in rc.WASDE_2026)


def test_available_at_is_localised_not_offset():
    winter = rc.available_at("2017-01-06")
    summer = rc.available_at("2017-07-07")
    assert str(winter) == "2017-01-06 13:30:00+00:00"      # EST, UTC-5
    assert str(summer) == "2017-07-07 12:30:00+00:00"      # EDT, UTC-4
    assert winter.tz is not None and summer.tz is not None
    noon = rc.available_at("2026-09-11", "12:00")
    assert noon.tz_convert(rc.ET).strftime("%H:%M") == "12:00"


def test_dst_audit_counts_the_releases_a_hardcoded_offset_gets_wrong():
    cal = rc.payroll_release_dates()
    da = rc.dst_audit([d.strftime("%Y-%m-%d") for d in cal["release_date"]])
    assert da["n"] == 72
    assert set(da["offsets"]) == {-5.0, -4.0}
    assert da["offsets"][-5.0] + da["offsets"][-4.0] == 72
    assert da["error_hours"] == 1.0
    assert da["n_wrong_if_hardcoded"] == min(da["offsets"].values())
    assert 0.2 < da["n_wrong_if_hardcoded"] / da["n"] < 0.5


def test_release_calendar_is_causal_and_lags_its_reference_period():
    cal = rc.payroll_release_dates()
    assert (cal["release_date"] > cal["ref_end"]).all()
    assert (cal["ref_start"] < cal["ref_end"]).all()
    lag = (cal["release_date"] - cal["ref_end"]).dt.days
    assert 1 <= lag.min() and lag.max() <= 8
    assert cal["release_date"].dt.day_name().eq("Friday").all()


def test_eia_holiday_exceptions_fall_on_the_weekdays_eia_printed():
    """shift_audit asserts the transcription internally; this pins the totals."""
    w = rc.shift_audit(rc.WPSR_SHIFTS, "Wednesday", "10:30")
    n = rc.shift_audit(rc.NGSR_SHIFTS, "Thursday", "10:30")
    assert w["n"] == 14 and w["n_day_changed"] == 14
    assert n["n"] == 9 and n["n_day_changed"] == 9
    assert w["max_hours_late"] == pytest.approx(6.5)       # Monday 2025-12-29, 17:00 ET
    assert "17:00" in w["distinct_clocks"] and len(w["distinct_clocks"]) == 4
    assert n["max_hours_late"] == pytest.approx(1.5)
    worst = w["table"].sort_values("hours_late").iloc[-1]
    assert worst["date"] == "2025-12-29" and worst["day"] == "Monday"


def test_the_trap_fires_on_reference_date_stamping_and_clears_on_the_timestamp():
    cal = rc.payroll_release_dates()
    ja = rc.join_audit(cal, "09:30")
    assert list(ja.index) == list(rc.STAMPS)
    assert ja.loc["reference period start", "share"] > 0.99
    assert ja.loc["reference period start", "mean_days"] > 15
    assert 0.05 < ja.loc["reference period end", "share"] < 0.30
    assert ja.loc["release timestamp", "days_with_lookahead"] == 0
    assert ja.loc["release timestamp", "max_hours"] == 0.0


def test_only_the_timestamp_survives_a_change_of_decision_clock():
    cal = rc.payroll_release_dates()
    at_open = rc.join_audit(cal, "09:30")
    premarket = rc.join_audit(cal, "06:00")
    # midnight stamping looks clean at 09:30 purely because the print lands at 08:30
    assert at_open.loc["release date midnight", "days_with_lookahead"] == 0
    assert premarket.loc["release date midnight", "days_with_lookahead"] > 0
    assert premarket.loc["release date midnight", "max_hours"] == pytest.approx(2.5)
    assert premarket.loc["release timestamp", "days_with_lookahead"] == 0


def test_capture_is_monotone_and_bounded_and_faster_with_the_lockup():
    lat = [10, 50, 250, 1000, 5000]
    pre = [rc.capture(x, 40.0) for x in lat]
    post = [rc.capture(x, 1500.0) for x in lat]
    assert pre == sorted(pre) and post == sorted(post)
    assert all(0.0 <= x <= 1.0 for x in pre + post)
    assert all(a > b for a, b in zip(pre, post))
    assert rc.capture(40.0, 40.0) == pytest.approx(0.5)     # one half-life
    assert rc.capture(0.0, 40.0) == 0.0
    grid = rc.latency_grid()
    assert grid.loc[250, "captured_pre_2020"] > 0.9
    assert grid.loc[250, "captured_post_2020"] < 0.2


def test_the_spliced_fill_assumption_is_wrong_in_opposite_directions():
    ss = rc.splice_study()
    again = rc.splice_study()
    assert ss == again                                      # seeded
    assert ss["n_pre"] > 100 and ss["n_post"] > 100
    assert ss["edge_true_post"] > 10 * ss["edge_true_pre"]
    assert ss["err_pre"] > 0 > ss["err_post"]               # opposite signs
    assert abs(ss["err_full"]) < 0.2 * min(abs(ss["err_pre"]), abs(ss["err_post"]))
    # a longer latency narrows the gap between the regimes
    slow = rc.splice_study(latency_ms=5000.0)
    assert (slow["capture_pre"] - slow["capture_post"]
            < ss["capture_pre"] - ss["capture_post"])


@pytest.mark.slow
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.macro.release_calendar")
    for head in ("1. The release-time table", "2. available_at",
                 "3. 'Wednesday at 10:30'", "4. The as-of join", "5. The splice"):
        assert head in out
    assert "2020-06-03" in out and "2018-08-01" in out and "2025-12-29" in out
    assert out.strip().splitlines()[-1].startswith("Rule:")
    assert all(ord(c) < 128 for c in out)
