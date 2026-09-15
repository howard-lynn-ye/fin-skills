"""fin_skills.market_data.corporate_actions - the events one adjustment ratio cannot carry.

Each headline gap has a closed form, so the tests assert the identity rather than a
transcribed decimal: the spin-off gap IS the distributed value over the cum price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.market_data.corporate_actions import (ACTION_TAXONOMY, feed_coverage_demo,
                                                      gap_table, merger_treatments,
                                                      reverse_split_gap, rights_issue_gap,
                                                      screen_false_positives, spin_off_gap,
                                                      spin_off_panel, terp,
                                                      ticker_change_cost, ticker_recycling,
                                                      unexplained_jumps)


# ------------------------------------------------------------------------------- taxonomy
def test_the_taxonomy_identifies_position_information_not_vendor_coverage():
    d = {name: ok for name, ok, _ in ACTION_TAXONOMY}
    assert d["forward split"] and d["cash dividend"] and d["reverse split"]
    for name in ("spin-off", "rights issue", "merger - cash", "merger - stock",
                 "ticker change", "delisting"):
        assert d[name] is False, name
    assert all(need for _, _, need in ACTION_TAXONOMY)


# ------------------------------------------------------------------------------- spin-off
def test_spin_off_panel_conserves_value_at_the_ex_date():
    p = spin_off_panel()
    i = p["parent"].index.get_loc(p["ex_date"])
    p_cum = float(p["parent"].iloc[i - 1])
    p_ex = float(p["parent"].iloc[i])
    s0 = float(p["spinco"].iloc[i])
    assert p_ex == pytest.approx(p_cum * (1 - p["value_fraction"]))
    assert p_ex + p["spin_ratio"] * s0 == pytest.approx(p_cum)
    assert p["spinco"].iloc[:i].isna().all()      # SpinCo does not exist before the ex-date


def test_spin_off_gap_equals_the_value_of_the_spun_off_entity():
    """THE property: the gap IS the distribution, expressed as a fraction of the cum price."""
    g = spin_off_gap()
    assert g["gap_pp"] == pytest.approx(g["closed_form_pp"], abs=1e-9)
    # the same identity from the reported (4-dp rounded) prices
    assert g["gap_pp"] == pytest.approx(
        100 * g["distributed_per_share"] / g["parent_last_cum"], abs=1e-3)
    assert g["gap_pp"] == pytest.approx(30.0, abs=1e-9)
    assert g["series_return_exdate_pct"] == pytest.approx(-30.0, abs=1e-9)
    assert g["holder_return_exdate_pct"] == pytest.approx(0.0, abs=1e-9)


def test_the_spin_off_gap_survives_the_whole_window():
    g = spin_off_gap()
    assert g["holder_return_window_pct"] > g["series_return_window_pct"]
    assert g["gap_window_pp"] > 25.0


def test_a_bigger_spin_off_is_a_bigger_loss_in_the_series():
    for w in (0.10, 0.30, 0.55):
        g = spin_off_gap(spin_off_panel(value_fraction=w))
        assert g["series_return_exdate_pct"] == pytest.approx(-100 * w, abs=1e-9)
        assert g["gap_pp"] == pytest.approx(100 * w, abs=1e-9)


def test_spin_off_panel_is_deterministic_for_a_seed():
    a, b = spin_off_panel(seed=5), spin_off_panel(seed=5)
    pd.testing.assert_series_equal(a["parent"], b["parent"])
    assert not a["parent"].equals(spin_off_panel(seed=6)["parent"])


# --------------------------------------------------------------------------- rights issue
def test_terp_is_the_weighted_average_of_old_and_new_shares():
    assert terp(100.0, 60.0, 0.5) == pytest.approx((100 + 0.5 * 60) / 1.5)
    assert terp(100.0, 100.0, 0.5) == pytest.approx(100.0)     # no discount, no drop
    assert terp(100.0, 60.0, 0.0) == pytest.approx(100.0)      # no rights, no drop


def test_the_rights_gap_is_exactly_the_value_of_the_rights():
    r = rights_issue_gap()
    assert r["terp"] == pytest.approx(86.6667, abs=1e-4)
    assert r["series_return_exdate_pct"] == pytest.approx(-13.3333, abs=1e-4)
    assert r["gap_pp"] == pytest.approx(r["closed_form_pp"], abs=1e-9)
    assert r["gap_pp"] == pytest.approx(
        100 * r["rights_value_per_share"] / r["cum_price"], abs=1e-9)


def test_selling_and_subscribing_are_worth_the_same_and_doing_nothing_is_not():
    r = rights_issue_gap()
    assert r["holder_return_exdate_pct"] == pytest.approx(0.0, abs=1e-9)
    assert r["subscriber_return_pct"] == pytest.approx(0.0, abs=1e-9)
    # doing nothing = keeping only the ex-rights share
    assert r["series_return_exdate_pct"] < -10.0


def test_a_deeper_discount_prints_a_bigger_fake_loss():
    shallow = rights_issue_gap(subscription_price=90.0)
    deep = rights_issue_gap(subscription_price=20.0, rights_per_share=1.0)
    assert deep["series_return_exdate_pct"] < shallow["series_return_exdate_pct"]
    assert deep["gap_pp"] > shallow["gap_pp"] > 0.0


# -------------------------------------------------------------------------- reverse split
def test_a_missed_reverse_split_prints_a_gain_the_holder_never_had():
    rs = reverse_split_gap()
    assert rs["series_return_exdate_pct"] == pytest.approx(900.0, abs=1e-9)
    assert rs["holder_return_exdate_pct"] == pytest.approx(0.0, abs=1e-6)
    assert rs["gap_pp"] == pytest.approx(rs["closed_form_pp"], abs=1e-9)
    assert rs["gap_pp"] == pytest.approx(-100 * (rs["ratio"] - 1), abs=1e-9)


def test_the_fractional_share_is_cashed_out_not_carried():
    rs = reverse_split_gap(ratio=10.0, price=0.80, shares=1237)
    assert rs["shares_after"] == 123
    assert rs["cash_in_lieu"] == pytest.approx(0.7 * 8.0, abs=1e-9)
    assert (rs["shares_after"] * rs["price_after"] + rs["cash_in_lieu"]
            == pytest.approx(1237 * 0.80, abs=1e-9))


def test_the_three_gaps_carry_opposite_signs():
    t = gap_table().set_index("event")
    assert t.loc["spin-off", "gap_pp"] > 0
    assert t.loc["rights issue", "gap_pp"] > 0
    assert t.loc["reverse split", "gap_pp"] < 0
    for ev in t.index:
        assert t.loc[ev, "gap_pp"] == pytest.approx(t.loc[ev, "closed_form_pp"], abs=1e-6)
        assert t.loc[ev, "holder_pct"] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------------------ the screen
def test_unexplained_jumps_reports_and_never_adjusts():
    idx = pd.bdate_range("2024-01-02", periods=10)
    px = pd.Series(100.0, index=idx)
    px.iloc[5:] *= 2.0
    out = unexplained_jumps(px, None)
    assert len(out) == 1 and out.loc[0, "explained_by_feed"] is np.False_ or \
        not bool(out.loc[0, "explained_by_feed"])
    assert out.loc[0, "date"] == idx[5]
    explained = unexplained_jumps(px, [(idx[5], 2.0)])
    assert bool(explained.loc[0, "explained_by_feed"])
    # the input is untouched: this is a report, not a fix
    assert float(px.iloc[5]) == 200.0


def test_an_empty_feed_explains_nothing_and_a_splits_feed_explains_the_split():
    f = feed_coverage_demo()
    assert f["n_events"] == 3
    assert f["jumps_found"] == 2                     # the rights drop is under the threshold
    assert f["unexplained_empty_feed"] == 2
    assert f["unexplained_splits_only_feed"] == 1
    assert f["events_missed_by_the_screen"] == ["rights issue"]


def test_lowering_the_threshold_costs_false_positives():
    fp = screen_false_positives().set_index("threshold_pct")
    assert fp.loc[20.00, "flags"] < fp.loc[13.33, "flags"] < fp.loc[10.00, "flags"]
    assert fp.loc[13.33, "flags_per_name_per_year"] > 0.2
    assert fp["flags"].min() > 0                     # an event-free panel still flags


# --------------------------------------------------------------------------------- merger
def test_the_nan_convention_decides_the_sign_of_a_merger():
    m = merger_treatments()
    assert m["truth_pct"] > m["dropped_pct"] > m["delisted_at_zero_pct"]
    assert m["truth_minus_delisted_pp"] > 5.0
    assert m["spread_pp"] == pytest.approx(m["truth_minus_delisted_pp"], abs=1e-9)


def test_a_bigger_premium_widens_the_gap_between_the_conventions():
    small = merger_treatments(premium=0.05)
    big = merger_treatments(premium=0.60)
    assert big["truth_minus_dropped_pp"] > small["truth_minus_dropped_pp"]


def test_merger_treatments_is_deterministic():
    assert merger_treatments() == merger_treatments()


# --------------------------------------------------------------------------------- tickers
def test_a_rename_takes_a_momentum_name_out_of_the_universe_for_a_year():
    t = ticker_change_cost()
    assert t["signal_days_after_change_by_permid"] == t["sessions_after_change"] == 504
    assert t["signal_days_after_change_by_ticker"] == 252
    assert t["days_out_of_the_universe"] == 252 == t["lookback"]


def test_a_shorter_lookback_loses_less():
    short = ticker_change_cost(lookback=21)
    assert short["days_out_of_the_universe"] == 21


def test_a_recycled_ticker_invents_a_return_nobody_earned():
    rc = ticker_recycling()
    assert rc["seam_return_pct"] < -50.0
    assert abs(rc["seam_z_score"]) > 5.0
    assert rc["first_price_company_b"] < rc["last_price_company_a"]


# ---------------------------------------------------------------------------- housekeeping
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.market_data.corporate_actions")
    for head in ("0. which actions need position information beyond a price ratio",
                 "1. spin-off", "2. rights issue", "3. reverse split",
                 "4. the three numbers", "5. is the feed carrying them at all",
                 "6. a cash merger is indistinguishable from a delisting",
                 "7. a ticker change moves the key"):
        assert head in out
    assert "30.0000" in out and "13.3333" in out and "-900.0000" in out
    assert out.strip().splitlines()[-1].startswith("The rule:")
    assert all(ord(c) < 128 for c in out)


def test_an_explicit_adjustment_can_account_for_spin_off_value():
    result = spin_off_gap()
    assert result["adjusted_return_exdate_pct"] == pytest.approx(0.)
    assert result["series_return_exdate_pct"] < 0
    # Holding SpinCo is a different portfolio from reinvesting its value into Parent.
    assert result["adjusted_return_window_pct"] != result["holder_return_window_pct"]


@pytest.mark.parametrize("kwargs", [{"rights_per_share": -1}, {"cum_price": 0},
                                    {"subscription_price": float("nan")},
                                    {"subscription_price": 120}])
def test_rights_reject_invalid_or_out_of_model_terms(kwargs):
    with pytest.raises(ValueError):
        rights_issue_gap(**kwargs)


@pytest.mark.parametrize("kwargs", [{"ratio": 0}, {"shares": 0}, {"price": -2},
                                    {"shares": 1.5}])
def test_reverse_split_rejects_invalid_terms(kwargs):
    with pytest.raises(ValueError):
        reverse_split_gap(**kwargs)


def test_jump_threshold_means_absolute_simple_return_in_both_directions():
    dates = pd.bdate_range("2024-01-02", periods=3)
    close = pd.Series([100., 82., 100.], index=dates)
    flagged = unexplained_jumps(close, None, tol=0.2)
    assert flagged["date"].tolist() == [dates[2]]
