"""fin_skills.fixed_income.ex_dividend - ex-dividend windows, rebate interest, DMO formulae.

The documented properties: inside the ex-dividend window the DMO formula subtracts exactly one
coupon so accrued goes negative while the CLEAN price stays continuous, QuantLib's
exCouponPeriod is already a look-back so a negative Period silently disables the whole feature,
and the DMO price formula and QuantLib's pricer agree to 1e-13 once the payment convention is
Unadjusted.
"""
from __future__ import annotations

from datetime import date

import pytest

from conftest import has_module, requires
from fin_skills.fixed_income.ex_dividend import (GILT_EX_DIVIDEND_BUSINESS_DAYS,
                                                 WAR_LOAN_EX_DIVIDEND_BUSINESS_DAYS,
                                                 WAR_LOAN_REDEEMED, add_business_days,
                                                 add_months, dmo_accrued, ex_dividend_date,
                                                 ex_dividend_window, gilt_price_from_yield,
                                                 is_ex_dividend, quantlib_ex_coupon_signs,
                                                 quasi_coupon_dates, surrounding_quasi_period)

COUPON, YIELD = 0.04, 0.042
FIRST, MAT = date(2025, 6, 7), date(2035, 6, 7)
NEXT_CPN, EXDIV = date(2026, 12, 7), date(2026, 11, 26)
FACE = 1_000_000.0
WINDOW = [date(2026, 11, 24), date(2026, 11, 25), date(2026, 11, 26), date(2026, 11, 27),
          date(2026, 11, 30), date(2026, 12, 4), date(2026, 12, 8)]


# ------------------------------------------------------------------ the rule
def test_the_gilt_ex_dividend_date_is_seven_business_days_before_the_coupon():
    assert GILT_EX_DIVIDEND_BUSINESS_DAYS == 7
    assert ex_dividend_date(NEXT_CPN) == EXDIV
    assert (NEXT_CPN - EXDIV).days == 11               # 11 calendar days across two weekends
    assert NEXT_CPN.weekday() == 0 and EXDIV.weekday() == 3
    # the dead War Loan carve-out is kept only as a dated constant
    assert WAR_LOAN_EX_DIVIDEND_BUSINESS_DAYS == 10
    assert WAR_LOAN_REDEEMED == date(2015, 3, 9)
    with pytest.raises(ValueError, match="pass it positive"):
        ex_dividend_date(NEXT_CPN, -7)


def test_the_quasi_coupon_cycle_runs_off_the_maturity_date():
    dates = quasi_coupon_dates(MAT, date(2026, 1, 1))
    assert dates[-1] == MAT
    assert all(add_months(a, 6) == b for a, b in zip(dates, dates[1:]))
    assert NEXT_CPN in dates
    prev_q, next_q = surrounding_quasi_period(date(2026, 11, 26), MAT)
    assert (prev_q, next_q) == (date(2026, 6, 7), NEXT_CPN)
    assert (next_q - prev_q).days == 183
    with pytest.raises(ValueError, match="freq must divide"):
        quasi_coupon_dates(MAT, date(2026, 1, 1), freq=7)
    # end-of-month clamping
    assert add_months(date(2026, 8, 31), 6) == date(2027, 2, 28)


# ------------------------------------------------------------------ negative accrued
def test_inside_the_window_the_dmo_formula_subtracts_exactly_one_coupon():
    cum = dmo_accrued(COUPON, date(2026, 11, 26), MAT)
    ex = dmo_accrued(COUPON, date(2026, 11, 27), MAT)
    assert cum["in_ex_dividend"] is False and ex["in_ex_dividend"] is True
    assert cum["accrued"] == pytest.approx(1.879781, abs=5e-7)
    assert ex["accrued"] == pytest.approx(-0.109290, abs=5e-7)
    assert ex["d1"] == pytest.approx(2.0)
    # one day of accrual apart, one coupon apart: (t/s) vs (t/s - 1)
    same_day = dmo_accrued(COUPON, date(2026, 11, 27), MAT, boundary="quantlib")
    assert cum["accrued"] - same_day["accrued"] == pytest.approx(2.0 - 2.0 / 183.0, abs=1e-9)
    assert ex["accrued"] < 0.0 < cum["accrued"]
    assert FACE * (cum["accrued"] - ex["accrued"]) / 100.0 == pytest.approx(19_890.71, abs=0.01)


def test_the_clean_price_is_continuous_across_the_ex_dividend_date_and_the_dirty_one_is_not():
    cum = gilt_price_from_yield(COUPON, YIELD, date(2026, 11, 26), MAT)
    ex = gilt_price_from_yield(COUPON, YIELD, date(2026, 11, 27), MAT)
    assert cum["d1"] == pytest.approx(2.0) and ex["d1"] == 0.0
    assert ex["dirty"] - cum["dirty"] == pytest.approx(-1.986321, abs=5e-7)
    assert ex["clean"] - cum["clean"] == pytest.approx(+0.002750, abs=5e-7)
    # the dirty jump is ~700x the clean one, and it is one discounted coupon
    assert abs(ex["dirty"] - cum["dirty"]) > 500 * abs(ex["clean"] - cum["clean"])
    assert abs(ex["dirty"] - cum["dirty"]) < 2.0


def test_the_window_table_flags_exactly_the_ex_dividend_settlement_dates():
    rows = ex_dividend_window(COUPON, YIELD, WINDOW, MAT)
    flags = [r["in_ex_dividend"] for r in rows]
    assert flags == [False, False, False, True, True, True, False]
    for r in rows:
        assert (r["accrued"] < 0) is r["in_ex_dividend"]
        assert r["dirty"] == pytest.approx(r["clean"] + r["accrued"], abs=1e-12)
    cleans = [r["clean"] for r in rows]
    assert max(cleans) - min(cleans) < 0.01          # clean barely moves across the whole window


def test_the_two_boundary_conventions_differ_on_exactly_one_day():
    assert is_ex_dividend(EXDIV, EXDIV, "dmo") is False
    assert is_ex_dividend(EXDIV, EXDIV, "quantlib") is True
    for d in WINDOW:
        if d != EXDIV:
            assert is_ex_dividend(d, EXDIV, "dmo") == is_ex_dividend(d, EXDIV, "quantlib")
    with pytest.raises(ValueError, match="boundary must be"):
        is_ex_dividend(EXDIV, EXDIV, "isda")


def test_business_day_arithmetic_goes_backwards_over_weekends():
    assert add_business_days(NEXT_CPN, -1) == date(2026, 12, 4)      # Monday back to Friday
    assert add_business_days(NEXT_CPN, -7) == EXDIV
    assert add_business_days(NEXT_CPN, -7, holidays=[date(2026, 11, 26)]) == date(2026, 11, 25)


# ------------------------------------------------------------------ QuantLib
@pytest.mark.skipif(has_module("QuantLib"), reason="QuantLib is installed")
def test_cross_checks_return_none_without_quantlib():
    assert quantlib_ex_coupon_signs(COUPON, FIRST, MAT, WINDOW) is None


@requires("QuantLib")
def test_a_negative_ex_coupon_period_puts_the_ex_date_after_the_coupon_and_is_never_flagged():
    live = quantlib_ex_coupon_signs(COUPON, FIRST, MAT, WINDOW)
    assert live is not None
    wrong = live["ex_coupon_dates"]["wrong_negative"][NEXT_CPN]
    right = live["ex_coupon_dates"]["correct"][NEXT_CPN]
    assert wrong == date(2026, 12, 16) and wrong > NEXT_CPN     # nine days AFTER the coupon
    assert right == EXDIV and right < NEXT_CPN
    assert live["ex_coupon_dates"]["dmo_boundary"][NEXT_CPN] == date(2026, 11, 27)
    a = live["accrued"]
    # the wrong sign is character-for-character the no-ex-coupon bond, on every date
    assert a["wrong_negative"] == a["none"]
    # and it is exactly one coupon adrift wherever the window actually applies
    for i, d in enumerate(live["settle_dates"]):
        gap = a["wrong_negative"][i] - a["correct"][i]
        assert gap == pytest.approx(0.0 if a["correct"][i] > 0 else 2.0, abs=1e-12), d
    i = live["settle_dates"].index(date(2026, 11, 27))
    assert a["correct"][i] == pytest.approx(-0.109290, abs=5e-7)
    assert a["wrong_negative"][i] == pytest.approx(+1.890710, abs=5e-7)
    assert FACE * (a["wrong_negative"][i] - a["correct"][i]) / 100.0 == pytest.approx(20_000.0,
                                                                                      abs=1e-6)


@requires("QuantLib")
def test_quantlib_matches_the_dmo_formula_on_both_the_boundary_and_the_price():
    live = quantlib_ex_coupon_signs(COUPON, FIRST, MAT, WINDOW, y=YIELD)
    a = live["accrued"]
    for i, d in enumerate(live["settle_dates"]):
        # Period(6) reproduces the DMO cum/ex boundary exactly
        assert a["dmo_boundary"][i] == pytest.approx(dmo_accrued(COUPON, d, MAT)["accrued"],
                                                     abs=1e-12), d
        # Period(7) matches QuantLib's own boundary rule
        assert a["correct"][i] == pytest.approx(
            dmo_accrued(COUPON, d, MAT, boundary="quantlib")["accrued"], abs=1e-12), d
        # and the two pricers are the same pricer
        mine = gilt_price_from_yield(COUPON, YIELD, d, MAT, boundary="quantlib")
        assert mine["dirty"] == pytest.approx(live["dirty"][i], abs=1e-11), d
        assert mine["clean"] == pytest.approx(live["clean"][i], abs=1e-11), d
    # the usual Following payment convention bumps coupons off weekends and shifts the price
    bump = live["clean_following"][0] - live["clean"][0]
    assert bump == pytest.approx(-0.000947, abs=5e-7)
    assert bump != 0.0


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.fixed_income.ex_dividend")
    assert "inside an ex-dividend window accrued is NEGATIVE" in out
    assert "Period(7, Days), never Period(-7, Days)" in out
    assert "20,000.00" in out
    assert out.isascii()
