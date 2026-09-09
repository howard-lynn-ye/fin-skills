"""fin_skills.fixed_income.sofr - compounded-in-arrears RFRs, the index, and the windows.

The documented properties: the reference implementation reproduces the NY Fed's published SOFR
Index example to 0.0e+00, compounding is not averaging and day weights are not equal weights,
the day-count basis appears twice so changing it in one place is the full 365/360, the two
conventions both called a "5-day lookback" give different rates, and QuantLib's
OvernightIndexedCoupon agrees on all four windows.
"""
from __future__ import annotations

from datetime import date

import pytest

from conftest import has_module, requires
from fin_skills.fixed_income.sofr import (NYFED_INDEX_EXAMPLE, RFR_CONVENTIONS, SOFR_BASIS,
                                          SOFR_CALENDAR_HOLIDAYS_2026, SONIA_BASIS,
                                          business_days, calendar_day_weights, compound_factor,
                                          compounded_rate, mixed_basis_rate,
                                          observation_window, quantlib_cross_checks,
                                          quantlib_index_day_counts, rate_for_window,
                                          rate_from_index, shift_business_days,
                                          simple_average_rate, sofr_index_path,
                                          synthetic_sofr, unweighted_average_rate)

START, END = date(2026, 4, 1), date(2026, 7, 1)
HOLIDAYS = list(SOFR_CALENDAR_HOLIDAYS_2026)
POLICY = date(2026, 6, 18)
NOTIONAL = 100_000_000.0
FIXINGS = synthetic_sofr(date(2026, 1, 2), END, HOLIDAYS, seed=7, policy_date=POLICY)
BASE = observation_window(START, END, HOLIDAYS)
RATES = [FIXINGS[d] for d in BASE["fixing_dates"]]
WEIGHTS = BASE["weights"]


# ------------------------------------------------------------------ the NY Fed example
def test_the_reference_implementation_reproduces_the_ny_fed_published_index_exactly():
    rates = [r for _, r, _, _ in NYFED_INDEX_EXAMPLE]
    weights = [d for _, _, d, _ in NYFED_INDEX_EXAMPLE]
    published = [p for _, _, _, p in NYFED_INDEX_EXAMPLE]
    got = sofr_index_path(rates, weights, 1.0, SOFR_BASIS, dp=8)
    assert got == published                       # to the published eighth decimal, exactly
    assert weights[-1] == 3                       # a Friday fixing carries three calendar days
    assert got[0] == pytest.approx(1.0 * (1 + 0.0180 / 360), abs=1e-12)
    with pytest.raises(ValueError, match="same length"):
        compound_factor(rates, weights[:-1])


def test_a_rate_can_be_rebuilt_from_two_index_values_and_a_day_count():
    idx = sofr_index_path(RATES, WEIGHTS, 1.0, SOFR_BASIS, dp=None)
    exact = rate_from_index(1.0, idx[-1], BASE["total_days"], SOFR_BASIS)
    assert exact == pytest.approx(compounded_rate(RATES, WEIGHTS, SOFR_BASIS), abs=1e-15)
    # and from the rounded, published index it is still under a hundredth of a basis point
    rounded = sofr_index_path(RATES, WEIGHTS, 1.0, SOFR_BASIS, dp=8)
    from_rounded = rate_from_index(1.0, rounded[-1], BASE["total_days"], SOFR_BASIS)
    assert 1e4 * abs(from_rounded - exact) < 0.01
    with pytest.raises(ValueError, match="must be positive"):
        rate_from_index(0.0, 1.01, 91)


# ------------------------------------------------------------------ calendars and weights
def test_calendar_day_weights_give_a_friday_three_days_and_a_pre_holiday_more():
    days = business_days(START, END, HOLIDAYS)
    w = calendar_day_weights(days, END)
    assert sum(w) == (END - START).days == 91
    assert len(days) == 62                        # Good Friday and Memorial Day and Juneteenth
    by_date = dict(zip(days, w))
    assert by_date[date(2026, 4, 10)] == 3        # an ordinary Friday
    assert by_date[date(2026, 4, 2)] == 4         # Thursday before Good Friday + the weekend
    assert by_date[date(2026, 5, 22)] == 4        # Friday before Memorial Day
    assert by_date[date(2026, 6, 18)] == 4        # Thursday before Juneteenth
    assert date(2026, 4, 3) not in by_date        # Good Friday is not a SOFR fixing day
    with pytest.raises(ValueError, match="no fixing dates"):
        calendar_day_weights([], END)


def test_business_day_shifting_skips_the_sofr_calendar_holidays():
    assert shift_business_days(date(2026, 4, 6), -1, HOLIDAYS) == date(2026, 4, 2)
    assert shift_business_days(date(2026, 4, 2), 1, HOLIDAYS) == date(2026, 4, 6)
    # Mon 22 Jun back five business days skips Juneteenth (Fri 19 Jun) and lands on Fri 12 Jun
    assert shift_business_days(date(2026, 6, 22), -5, HOLIDAYS) == date(2026, 6, 12)
    assert shift_business_days(date(2026, 6, 22), -5) == date(2026, 6, 15)   # without it
    assert len(SOFR_CALENDAR_HOLIDAYS_2026) == 12


# ------------------------------------------------------------------ the four errors
def test_compounding_is_not_averaging_and_day_weights_are_not_equal_weights():
    comp = compounded_rate(RATES, WEIGHTS, SOFR_BASIS)
    assert comp == pytest.approx(0.04352685, abs=5e-9)
    simple = simple_average_rate(RATES, WEIGHTS)
    assert 1e4 * (simple - comp) == pytest.approx(-2.32, abs=0.01)
    unweighted_compounded = compounded_rate(RATES, [1] * len(RATES), SOFR_BASIS)
    assert 1e4 * (unweighted_compounded - comp) == pytest.approx(-1.13, abs=0.01)
    plain_mean = unweighted_average_rate(RATES)
    assert 1e4 * (plain_mean - comp) == pytest.approx(-2.72, abs=0.01)
    # every wrong method here is LOW: compounding and the extra weekend days both add
    for wrong in (simple, unweighted_compounded, plain_mean):
        assert wrong < comp
    cash = NOTIONAL * (simple - comp) * BASE["total_days"] / SOFR_BASIS
    assert cash == pytest.approx(-5873.0, abs=1.0)


def test_the_day_count_basis_appears_twice_and_changing_one_of_them_is_the_full_365_over_360():
    comp = compounded_rate(RATES, WEIGHTS, SOFR_BASIS)
    consistent = compounded_rate(RATES, WEIGHTS, SONIA_BASIS)
    mixed = mixed_basis_rate(RATES, WEIGHTS, SOFR_BASIS, SONIA_BASIS)
    assert 1e4 * (consistent - comp) == pytest.approx(-0.03, abs=0.01)
    assert 1e4 * (mixed - comp) == pytest.approx(+6.05, abs=0.01)
    # the mixed rate is exactly 365/360 times the correct one, by construction
    assert mixed / comp == pytest.approx(365.0 / 360.0, abs=1e-12)
    assert abs(mixed - comp) > 100.0 * abs(consistent - comp)


def test_the_two_conventions_both_called_a_five_day_lookback_give_different_rates():
    plain = rate_for_window(FIXINGS, BASE, SOFR_BASIS)
    lb = rate_for_window(FIXINGS, observation_window(START, END, HOLIDAYS, lookback=5),
                         SOFR_BASIS)
    shift = rate_for_window(
        FIXINGS, observation_window(START, END, HOLIDAYS, lookback=5, observation_shift=True),
        SOFR_BASIS)
    lock = rate_for_window(FIXINGS, observation_window(START, END, HOLIDAYS, lockout=5),
                           SOFR_BASIS)
    assert 1e4 * (lb - plain) == pytest.approx(-2.24, abs=0.01)
    assert 1e4 * (shift - plain) == pytest.approx(-1.94, abs=0.01)
    assert 1e4 * (lock - plain) == pytest.approx(-0.39, abs=0.01)
    assert 1e4 * (shift - lb) == pytest.approx(+0.30, abs=0.01)
    assert lb != shift                              # same name, different number


def test_the_windows_have_the_shape_they_claim():
    lb = observation_window(START, END, HOLIDAYS, lookback=5)
    shift = observation_window(START, END, HOLIDAYS, lookback=5, observation_shift=True)
    # the two lookbacks use the SAME rates and differ only in the day weights
    assert lb["fixing_dates"] == shift["fixing_dates"]
    assert lb["weights"] != shift["weights"]
    assert sum(lb["weights"]) == sum(shift["weights"]) == 91
    # a lockout repeats its last observable fixing
    lock = observation_window(START, END, HOLIDAYS, lockout=5)
    assert len(set(lock["fixing_dates"][-5:])) == 1
    assert lock["fixing_dates"][-1] == lock["fixing_dates"][-6]
    with pytest.raises(ValueError, match="must not be negative"):
        observation_window(START, END, HOLIDAYS, lookback=-5)
    with pytest.raises(ValueError, match="needs a lookback"):
        observation_window(START, END, HOLIDAYS, observation_shift=True)
    with pytest.raises(ValueError, match="longer than the observation window"):
        observation_window(START, END, HOLIDAYS, lockout=200)


def test_the_synthetic_path_is_deterministic_and_carries_the_policy_step():
    again = synthetic_sofr(date(2026, 1, 2), END, HOLIDAYS, seed=7, policy_date=POLICY)
    assert again == FIXINGS
    other = synthetic_sofr(date(2026, 1, 2), END, HOLIDAYS, seed=8, policy_date=POLICY)
    assert other != FIXINGS
    before = FIXINGS[date(2026, 6, 17)]
    after = FIXINGS[date(2026, 6, 18)]
    assert after - before == pytest.approx(0.0025, abs=0.0005)
    # SOFR is published to the nearest basis point
    assert all(abs(r * 1e4 - round(r * 1e4)) < 1e-9 for r in FIXINGS.values())


# ------------------------------------------------------------------ conventions
def test_the_day_count_split_is_two_bases_over_five_currencies():
    by_name = {n: b for n, _, _, b, _ in RFR_CONVENTIONS}
    assert by_name["SOFR"] == by_name["ESTR"] == by_name["SARON"] == 360.0
    assert by_name["SONIA"] == by_name["TONA"] == 365.0
    assert set(by_name.values()) == {360.0, 365.0}


@requires("QuantLib")
def test_quantlibs_own_index_definitions_confirm_the_day_count_split():
    dcs = quantlib_index_day_counts()
    assert dcs is not None
    for name in ("SOFR", "ESTR", "SARON"):
        assert dcs[name] == "Actual/360", name
    for name in ("SONIA", "TONA"):
        assert dcs[name].startswith("Actual/365"), name


@pytest.mark.skipif(has_module("QuantLib"), reason="QuantLib is installed")
def test_cross_checks_return_none_without_quantlib():
    assert quantlib_cross_checks(FIXINGS, START, END, HOLIDAYS) is None
    assert quantlib_index_day_counts() is None


@requires("QuantLib")
def test_quantlib_agrees_on_all_four_window_conventions():
    live = quantlib_cross_checks(FIXINGS, START, END, HOLIDAYS)
    assert live is not None
    # the hard-coded 2026 holiday list IS QuantLib's SOFR fixing calendar
    assert live["calendar_mismatches"] == []
    assert "SOFR" in live["calendar"]
    pairs = {"compounded": rate_for_window(FIXINGS, BASE, SOFR_BASIS),
             "simple": None,
             "lookback5": rate_for_window(
                 FIXINGS, observation_window(START, END, HOLIDAYS, lookback=5), SOFR_BASIS),
             "lookback5_shift": rate_for_window(
                 FIXINGS, observation_window(START, END, HOLIDAYS, lookback=5,
                                             observation_shift=True), SOFR_BASIS)}
    for key, mine in pairs.items():
        if mine is not None:
            assert live[key] == pytest.approx(mine, abs=1e-12), key
    assert live["simple"] == pytest.approx(simple_average_rate(RATES, WEIGHTS), abs=1e-12)
    assert 1e4 * (live["compounded"] - live["simple"]) == pytest.approx(2.32, abs=0.01)


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.fixed_income.sofr")
    assert "a compounded RFR is a PRODUCT of (1 + r_i x d_i / basis)" in out
    assert "until you say whether it shifts" in out
    assert "1.00034365" in out
    assert out.isascii()
