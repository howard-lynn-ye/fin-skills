"""fin_skills.market_data.pit_fundamentals - the value that was actually KNOWN on a past date."""
from __future__ import annotations

import pandas as pd
import pytest

from fin_skills.market_data.pit_fundamentals import (DEMO_FACTS, available_at, facts_to_frame,
                                              naive_latest, pit_facts, restatement_report)

Q3 = "2022-07-01..2022-09-30"


def test_facts_to_frame_keeps_instantaneous_facts_and_fills_missing_fields():
    df = facts_to_frame(DEMO_FACTS)
    assert len(df) == len(DEMO_FACTS)
    assert "instant..2022-12-31" in set(df["period"])
    partial = facts_to_frame([{"end": "2022-12-31", "val": "5", "filed": "2023-01-01"}])
    assert partial["val"].iloc[0] == 5.0
    assert pd.isna(partial["fy"].iloc[0]) and partial["period"].iloc[0] == "instant..2022-12-31"


def test_point_in_time_returns_the_original_where_naive_returns_the_restatement():
    pit = pit_facts(DEMO_FACTS, "2023-01-15").set_index("period")
    naive = naive_latest(DEMO_FACTS).set_index("period")
    assert pit.loc[Q3, "val"] == 1_000_000_000              # the 10-Q on file that day
    assert pit.loc[Q3, "form"] == "10-Q"
    assert naive.loc[Q3, "val"] == 940_000_000               # filed a month AFTER the as-of
    assert naive.loc[Q3, "filed"] > pd.Timestamp("2023-01-15")
    assert list(pit.index) == [Q3]                            # nothing else was filed yet


def test_later_as_of_shows_the_restated_value_and_the_instant_fact():
    later = pit_facts(DEMO_FACTS, "2023-06-30").set_index("period")
    assert later.loc[Q3, "val"] == 940_000_000 and later.loc[Q3, "form"] == "10-Q/A"
    assert later.loc["instant..2022-12-31", "val"] == 48_300_000_000
    assert later.loc["2022-10-01..2022-12-31", "val"] == 1_120_000_000   # the 2024 rewrite unseen
    assert later["end"].is_monotonic_increasing
    only_k = pit_facts(DEMO_FACTS, "2024-12-31", forms=("10-K",))
    assert set(only_k["form"]) == {"10-K"}


def test_restatement_report_keys():
    default = restatement_report(DEMO_FACTS).set_index("period")
    assert set(default.index) == {Q3, "2022-10-01..2022-12-31"}
    assert default.loc[Q3, "n_vintages"] == 2 and default.loc[Q3, "form"] == "10-Q"
    assert default.loc[Q3, "pct_change"] == pytest.approx(-0.06)
    cross = restatement_report(DEMO_FACTS, by=("start", "end")).set_index("period")
    assert cross.loc[Q3, "n_vintages"] == 3                    # the 10-Q/A joins the group
    assert cross.loc[Q3, "form"] == "10-Q|10-Q/A"
    assert cross.loc["2022-10-01..2022-12-31", "original_val"] == 1_120_000_000
    assert restatement_report(DEMO_FACTS[3:4]).empty


def test_post_close_acceptance_rolls_to_the_next_session():
    a = available_at("2026-07-30T20:30:28Z")                  # 16:30 ET
    assert a.post_close and a.first_tradeable_session == pd.Timestamp("2026-07-31")
    b = available_at("2026-07-30T19:30:00Z")                  # 15:30 ET
    assert not b.post_close and b.first_tradeable_session == pd.Timestamp("2026-07-30")
    c = available_at("2026-07-31T21:05:00Z")                  # Friday 17:05 ET
    assert c.first_tradeable_session == pd.Timestamp("2026-08-03")
    assert "POST-CLOSE" in str(a) and "intraday" in str(b)


def test_tz_in_matters_the_eastern_stamp_read_as_utc_lands_in_session():
    assert available_at("2026-07-30T16:30:00", tz_in="America/New_York").post_close
    assert not available_at("2026-07-30T16:30:00", tz_in="UTC").post_close


def test_explicit_session_calendar_skips_holidays():
    sessions = pd.DatetimeIndex(["2026-07-30", "2026-08-04"])   # 07-31 and 08-03 closed
    a = available_at("2026-07-30T20:30:28Z", sessions=sessions)
    assert a.first_tradeable_session == pd.Timestamp("2026-08-04")
    with pytest.raises(ValueError, match="no session"):
        available_at("2026-08-05T20:30:28Z", sessions=sessions)


def test_demo_prints_the_gap(run_main):
    out = run_main("fin_skills.market_data.pit_fundamentals")
    assert "days AFTER the as-of date" in out
