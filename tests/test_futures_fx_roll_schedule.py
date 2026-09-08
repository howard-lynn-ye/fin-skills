"""fin_skills.futures_fx.roll_schedule - roll rules by rule, and the roll rule as an uncounted trial."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.futures_fx.roll_schedule import (DEFAULT_OFFSET, RULES, _crossover,
                                                 _synthetic_market, _tradeable_roll,
                                                 compare_rules, render, roll_costs, roll_dates)


@pytest.fixture(scope="module")
def market():
    return _synthetic_market(years=3)


def lead_days(px, expiries, rolls):
    return [int(px.index.searchsorted(expiries.iloc[k], side="right") - 1
                - px.index.searchsorted(d, side="left")) for k, d in enumerate(rolls)]


def test_synthetic_market_is_seeded(market):
    px, oi, vo, exp, fnd = market
    again = _synthetic_market(years=3)
    pd.testing.assert_frame_equal(px, again[0])
    pd.testing.assert_frame_equal(oi, again[1])
    assert list(px.columns) == list(exp.index) == list(fnd.index)
    assert (fnd < exp).all()


@pytest.mark.parametrize("rule", RULES)
def test_every_rule_yields_one_tradeable_strictly_increasing_roll_per_pair(market, rule):
    px, oi, vo, exp, fnd = market
    rolls = roll_dates(px, rule, exp, open_interest=oi, volume=vo, first_notice=fnd)
    assert len(rolls) == px.shape[1] - 1
    assert all(b > a for a, b in zip(rolls, rolls[1:]))
    for k, d in enumerate(rolls):
        assert d in px.index
        assert np.isfinite(px.at[d, px.columns[k]]) and np.isfinite(px.at[d, px.columns[k + 1]])
        assert d < exp.iloc[k]


def test_rules_fire_where_the_docstring_says(market):
    px, oi, vo, exp, fnd = market
    leads = {rule: lead_days(px, exp, roll_dates(px, rule, exp, open_interest=oi, volume=vo,
                                                 first_notice=fnd)) for rule in RULES}
    assert set(leads["calendar"]) == {DEFAULT_OFFSET["calendar"]}            # pinned
    assert set(leads["first_notice"]) == {22 + DEFAULT_OFFSET["first_notice"]}  # pinned, earliest
    assert np.mean(leads["volume"]) > np.mean(leads["open_interest"])         # volume migrates first
    assert np.mean(leads["first_notice"]) > np.mean(leads["volume"])
    assert len(set(leads["open_interest"])) > 1                               # crossovers drift
    later = lead_days(px, exp, roll_dates(px, "calendar", exp, offset=10))
    assert set(later) == {10}


def test_crossover_must_stick():
    w = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=4))
    front = pd.Series([5.0, 5.0, 5.0, 5.0], index=w)
    assert _crossover(front, pd.Series([6.0, 4.0, 6.0, 6.0], index=w), w) == w[2]   # not the head-fake
    assert _crossover(front, pd.Series([6.0, 6.0, 6.0, 6.0], index=w), w) == w[0]
    assert _crossover(front, pd.Series([4.0, 4.0, 4.0, 4.0], index=w), w) is None
    assert _crossover(front, pd.Series([4.0, 4.0, 4.0, 6.0], index=w), w) == w[3]


def test_tradeable_roll_walks_back_to_a_day_both_legs_printed():
    idx = pd.bdate_range("2024-01-01", periods=5)
    c = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0], "B": [np.nan, 2.0, np.nan, np.nan, 2.0]}, index=idx)
    assert _tradeable_roll(c, "A", "B", idx[3], None, idx[4]) == idx[1]
    with pytest.raises(ValueError, match="no tradeable roll date"):
        _tradeable_roll(c, "A", "B", idx[3], idx[1], idx[4])


def test_missing_inputs_are_refused_not_defaulted(market):
    px, oi, vo, exp, fnd = market
    with pytest.raises(ValueError, match="rule must be one of"):
        roll_dates(px, "vibes", exp)
    with pytest.raises(ValueError, match="needs the open_interest frame"):
        roll_dates(px, "open_interest", exp)
    with pytest.raises(ValueError, match="needs the volume frame"):
        roll_dates(px, "volume", exp)
    with pytest.raises(ValueError, match="first_notice"):
        roll_dates(px, "first_notice", exp)
    with pytest.raises(ValueError, match="expiries missing"):
        roll_dates(px, "calendar", exp.iloc[:-1])
    with pytest.raises(ValueError, match="same index and columns"):
        roll_dates(px, "volume", exp, volume=vo.iloc[1:])
    with pytest.raises(ValueError, match="expiries are required"):
        compare_rules(px)


def test_roll_costs_floor_and_illiquidity_penalty(market):
    px, oi, vo, exp, fnd = market
    rolls = roll_dates(px, "calendar", exp)
    flat = roll_costs(px, rolls, base_bps=1.5)
    assert (flat == 3.0).all() and list(flat.index) == rolls
    with_oi = roll_costs(px, rolls, oi, base_bps=1.5)
    assert (with_oi >= 3.0).all() and with_oi.mean() > 3.0
    early = roll_costs(px, roll_dates(px, "first_notice", exp, first_notice=fnd), oi)
    assert early.mean() > with_oi.mean()                # rolling early lifts a thin book


def test_compare_rules_is_one_trial_per_rule(market):
    px, oi, vo, exp, fnd = market
    tab = compare_rules(px, RULES, exp, open_interest=oi, volume=vo, first_notice=fnd)
    assert list(tab.index) == list(RULES)
    assert tab["n_rolls"].nunique() == 1
    assert (tab["ann_net"] <= tab["ann_gross"]).all()
    assert tab["roll_cost_bps_per_roll"].min() >= 3.0
    assert tab["sharpe_net"].max() > tab["sharpe_net"].min()
    text = render(tab)
    assert "SHARPE SPREAD" in text and "RETURN SPREAD" in text


def test_demo_names_every_rule_as_a_trial(run_main):
    out = run_main("fin_skills.futures_fx.roll_schedule")
    assert "EACH RULE YOU EVALUATED IS A TRIAL" in out
    assert "led.deflated_sharpe" in out
