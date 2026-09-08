"""fin_skills.core.pit_universe - a point-in-time universe LOSES names."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.core.pit_universe import (_synthetic_index, annualised,
                                          audit_universe_stability, equal_weight_backtest,
                                          rebalance_universe, universe_on)


@pytest.fixture
def members():
    return pd.DataFrame({
        "ticker": ["OLD", "GONE", "LATE", "EVER"],
        "start_date": ["2015-01-01", "2015-01-01", "2019-06-01", "2015-01-01"],
        "end_date": ["2019-03-14", "2019-03-15", pd.NaT, pd.NaT],
    })


def test_membership_intervals_are_closed_on_both_ends(members):
    assert universe_on("2019-03-15", members) == ["EVER", "GONE"]   # GONE traded that session
    assert universe_on("2019-03-14", members) == ["EVER", "GONE", "OLD"]
    assert universe_on("2019-06-01", members) == ["EVER", "LATE"]
    assert universe_on("2014-12-31", members) == []


def test_membership_table_validation(members):
    with pytest.raises(ValueError, match="current snapshot"):
        universe_on("2019-01-01", members.drop(columns="start_date"))
    with pytest.raises(ValueError, match="ticker"):
        universe_on("2019-01-01", members.drop(columns="ticker"))
    bad = members.copy()
    bad.loc[0, "end_date"] = "2010-01-01"
    with pytest.raises(ValueError, match="precedes"):
        universe_on("2019-01-01", bad)
    bad = members.copy()
    bad.loc[0, "start_date"] = "not a date"
    with pytest.raises(ValueError, match="unparseable"):
        universe_on("2019-01-01", bad)


def test_liquidity_screen_uses_only_trailing_history():
    idx = pd.bdate_range("2024-01-01", periods=60)
    adv = pd.DataFrame({"A": 1e7, "B": np.r_[np.full(30, 1e5), np.full(30, 1e9)]}, index=idx)
    m = pd.DataFrame({"ticker": ["A", "B"], "start_date": ["2020-01-01"] * 2})
    early, late = idx[29], idx[59]
    assert universe_on(early, m, liquidity=adv, min_adv=5e6) == ["A"]     # B not liquid YET
    assert universe_on(late, m, liquidity=adv, min_adv=5e6) == ["A", "B"]
    assert (adv.mean() > 5e6).all()             # the full-sample screen would admit B on day 30
    assert universe_on("2023-12-01", m, liquidity=adv, min_adv=5e6) == []  # no history at all
    reb = rebalance_universe([early, late], m, liquidity=adv, min_adv=5e6)
    assert reb == {early: ["A"], late: ["A", "B"]}


def test_rebalance_universe_matches_universe_on_per_date(members):
    dates = pd.bdate_range("2019-03-13", periods=4)
    reb = rebalance_universe(dates, members)
    assert reb == {d: universe_on(d, members) for d in dates}


def test_stability_audit_flags_a_universe_that_never_loses_a_name():
    dates = pd.bdate_range("2016-01-01", periods=8, freq="BQE")
    growing = {d: [f"S{i}" for i in range(10 + k)] for k, d in enumerate(dates)}
    a = audit_universe_stability(growing)
    assert a.never_loses_a_name and a.monotonic_growth
    assert a.verdict.startswith("CURRENT-SNAPSHOT SCREEN")
    assert a.total_removals == 0 and a.total_additions == 7
    assert any("fixed ticker list" in n for n in a.notes)
    assert "VERDICT" in a.report()


def test_stability_audit_accepts_a_universe_with_deletions():
    dates = pd.bdate_range("2016-01-01", periods=6, freq="BQE")
    names = [f"S{i}" for i in range(50)]
    uni = {d: names[k:k + 40] for k, d in enumerate(dates)}      # rotate one in, one out
    a = audit_universe_stability(uni)
    assert not a.never_loses_a_name
    assert a.verdict.startswith("PLAUSIBLY POINT-IN-TIME")
    assert a.total_removals == 5 and a.mean_turnover == pytest.approx(2 / 40)
    with pytest.raises(ValueError, match="at least two"):
        audit_universe_stability({dates[0]: names})


def test_equal_weight_backtest_and_annualised():
    idx = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"A": 100 * 1.01 ** np.arange(6), "B": 100 * 1.03 ** np.arange(6)}, index=idx)
    r = equal_weight_backtest({idx[0]: ["A", "B"]}, prices)
    assert np.allclose(r.to_numpy(), 0.02)
    assert len(r) == 5                                           # held from the day AFTER
    assert annualised(pd.Series([0.01] * 252)) == pytest.approx(1.01 ** 252 - 1)
    assert np.isnan(annualised(pd.Series([0.01])))


def test_survivor_screen_inflates_the_synthetic_index():
    prices, members = _synthetic_index()
    again, _ = _synthetic_index()
    pd.testing.assert_frame_equal(prices, again)                  # seeded
    rebals = pd.bdate_range(prices.index[0], prices.index[-1], freq="BQE")
    pit = rebalance_universe(rebals, members)
    today = sorted(members.loc[members["end_date"].isna(), "ticker"])
    snap = {pd.Timestamp(d): [t for t in today if pd.notna(prices[t].asof(pd.Timestamp(d)))]
            for d in rebals}
    a_pit, a_snap = audit_universe_stability(pit), audit_universe_stability(snap)
    assert a_pit.total_removals > 0 and a_snap.total_removals == 0
    c_pit = annualised(equal_weight_backtest(pit, prices))
    c_snap = annualised(equal_weight_backtest(snap, prices))
    assert c_snap > c_pit + 0.02                                  # the documented look-ahead


def test_demo_prints_both_traps(run_main):
    out = run_main("fin_skills.core.pit_universe")
    assert "bps/yr of pure look-ahead" in out and "winner filter" in out
