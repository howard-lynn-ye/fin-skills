"""fin_skills.futures_fx.continuous_contract - levels -> unadjusted, returns -> ratio.

The two-contract toy: F0 prints 100..103 through the roll, F1 prints 106..112, one roll on
the third day, so every roll-day quantity can be checked by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.futures_fx.continuous_contract import (METHODS, _synthetic_contango,
                                                       active_contract, compare_methods,
                                                       render, roll_following_days,
                                                       safe_returns, stitch, true_roll_return)


@pytest.fixture
def toy():
    idx = pd.bdate_range("2024-01-01", periods=6)
    contracts = pd.DataFrame({"F0": [100.0, 101.0, 102.0, 103.0, np.nan, np.nan],
                              "F1": [np.nan, 106.0, 107.0, 109.0, 110.0, 112.0]}, index=idx)
    return contracts, [idx[2]]


def test_contract_k_is_held_through_roll_k_inclusive(toy):
    contracts, rolls = toy
    assert active_contract(contracts, rolls).tolist() == ["F0", "F0", "F0", "F1", "F1", "F1"]
    assert roll_following_days(contracts, rolls).tolist() == [contracts.index[3]]


def test_true_roll_return_is_priced_off_one_contracts_own_two_closes(toy):
    contracts, rolls = toy
    pts = true_roll_return(contracts, rolls, in_points=True)
    assert pts.tolist() == pytest.approx([1.0, 1.0, 2.0, 1.0, 2.0])   # roll day earns F1 107->109
    ret = true_roll_return(contracts, rolls)
    assert ret.tolist() == pytest.approx([1 / 100, 1 / 101, 2 / 107, 1 / 109, 2 / 110])
    assert ret.name == "true_roll_return" and pts.name == "true_roll_pnl_points"


def test_ratio_adjusted_returns_equal_the_true_rolled_position_returns(toy):
    contracts, rolls = toy
    ratio = stitch(contracts, rolls, "ratio")
    assert ratio.attrs == {"method": "ratio", "levels_are_prices": False,
                           "returns_valid": True, "n_rolls": 1}
    assert ratio.tolist()[2:] == pytest.approx([107.0, 109.0, 110.0, 112.0])   # anchored today
    np.testing.assert_allclose(safe_returns(ratio).to_numpy(),
                               true_roll_return(contracts, rolls).to_numpy(), atol=1e-12)


def test_unadjusted_levels_are_the_prices_that_printed_and_carry_a_fake_jump(toy):
    contracts, rolls = toy
    unadj = stitch(contracts, rolls, "unadjusted")
    assert unadj.tolist() == [100.0, 101.0, 102.0, 109.0, 110.0, 112.0]
    assert unadj.attrs["levels_are_prices"] and not unadj.attrs["returns_valid"]
    fake = unadj.pct_change(fill_method=None).iloc[3]
    assert fake == pytest.approx(109 / 102 - 1)
    assert fake != pytest.approx(2 / 107)                     # nobody earned the roll gap


def test_difference_adjusted_diff_equals_true_dollar_pnl(toy):
    contracts, rolls = toy
    diff = stitch(contracts, rolls, "difference")
    np.testing.assert_allclose(diff.diff().dropna().to_numpy(),
                               true_roll_return(contracts, rolls, in_points=True).to_numpy(),
                               atol=1e-12)


def test_difference_adjusted_series_is_continuous_at_the_roll_off_the_roll_days(toy):
    # what does hold: the series is anchored at today and shifted by the cumulative gap
    contracts, rolls = toy
    diff = stitch(contracts, rolls, "difference")
    assert diff.tolist()[3:] == [109.0, 110.0, 112.0]
    assert diff.attrs["returns_valid"] is False and diff.attrs["levels_are_prices"] is False
    off_roll = diff.diff().dropna().drop(contracts.index[3])
    pts = true_roll_return(contracts, rolls, in_points=True).drop(contracts.index[3])
    np.testing.assert_allclose(off_roll.to_numpy(), pts.to_numpy(), atol=1e-12)


def test_safe_returns_refuses_the_two_series_whose_pct_change_is_meaningless(toy):
    contracts, rolls = toy
    with pytest.raises(ValueError, match="crosses zero"):
        safe_returns(stitch(contracts, rolls, "difference"))
    with pytest.raises(ValueError, match="fictitious jump"):
        safe_returns(stitch(contracts, rolls, "unadjusted"))
    with pytest.raises(ValueError, match="provenance"):
        safe_returns(pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="method must be one of"):
        stitch(contracts, rolls, "panama")


def test_validation_catches_the_silent_nan_gap_and_bad_roll_lists(toy):
    contracts, rolls = toy
    with pytest.raises(ValueError, match="need BOTH legs"):
        stitch(contracts, [contracts.index[0]], "ratio")        # F1 has no print on day 1
    with pytest.raises(ValueError, match="exactly one roll date"):
        stitch(contracts, [], "ratio")
    with pytest.raises(ValueError, match="not a row"):
        stitch(contracts, ["2030-01-01"], "ratio")
    with pytest.raises(TypeError):
        stitch(contracts.to_numpy(), rolls, "ratio")
    with pytest.raises(ValueError, match="sorted"):
        stitch(contracts.iloc[::-1], rolls, "ratio")


def test_compare_methods_scores_each_method_against_the_truth(toy):
    contracts, rolls = toy
    tab = compare_methods(contracts, rolls)
    assert list(tab.index) == list(METHODS)
    assert tab["returns_valid"].tolist() == [False, False, True]
    assert tab.loc["ratio", "bad_roll_days"] == 0
    assert tab.loc["unadjusted", "bad_roll_days"] == tab.loc["unadjusted", "n_rolls_scored"] == 1
    assert tab.loc["unadjusted", "goes_negative"] == False  # noqa: E712
    assert tab.attrs["n_days"] == 5
    assert "THE RULE" in render(tab)


def test_persistent_contango_climbs_and_the_rolled_long_bleeds():
    # Back-adjustment ADDS the gap, so contango (deferred dearer) pushes history UP, not
    # negative; spot still rises while the rolled long loses to the roll each quarter.
    contracts, rolls, spot = _synthetic_contango()
    again = _synthetic_contango()
    pd.testing.assert_frame_equal(contracts, again[0])            # seeded
    assert stitch(contracts, rolls, "difference").min() > 0        # contango does NOT go negative
    ratio = stitch(contracts, rolls, "ratio")
    assert (ratio > 0).all()
    err = (safe_returns(ratio) - true_roll_return(contracts, rolls)).abs().max()
    assert err < 1e-12
    assert spot.iloc[-1] > spot.iloc[0]                            # spot up ...
    assert (1 + true_roll_return(contracts, rolls)).prod() < 1.0    # ... the rolled long lost


def test_backwardation_drives_the_difference_series_negative():
    # The crude-oil case: deferred cheaper, gap < 0, added -> distant history goes below zero.
    contracts, rolls, spot = _synthetic_contango(contango=-0.20, seed=5)
    diff = stitch(contracts, rolls, "difference")
    assert diff.min() < 0                                          # back-adjusted goes negative
    with pytest.raises(ValueError, match="crosses zero"):
        safe_returns(diff)
    ratio = stitch(contracts, rolls, "ratio")
    assert (ratio > 0).all()
    err = (safe_returns(ratio) - true_roll_return(contracts, rolls)).abs().max()
    assert err < 1e-12
    assert (1 + true_roll_return(contracts, rolls)).prod() > 1.0    # the rolled long earned the roll


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.futures_fx.continuous_contract")
    assert "THE RULE:  levels -> unadjusted.   returns -> ratio." in out
