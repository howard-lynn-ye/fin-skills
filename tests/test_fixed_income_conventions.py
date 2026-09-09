"""fin_skills.fixed_income.conventions - day count, accrued, clean vs dirty, settlement lag.

The documented properties: ACT/ACT ICMA handed no coupon schedule guesses the frequency and
returns a whole quasi-period, the 30/360 name covers five rules that disagree on the 31st and
on February, QuantLib's nine Thirty360 constants collapse onto those five, and one business
day of settlement moves as much cash as the entire 30/360 argument.
"""
from __future__ import annotations

from datetime import date

import pytest

from conftest import has_module, requires
from fin_skills.fixed_income.conventions import (CONVENTIONS, THIRTY_FLAVOURS, accrued_interest,
                                                 act_act_icma, act_act_isda, add_business_days,
                                                 convention_table, dirty_price,
                                                 icma_frequency_guess, is_last_of_february,
                                                 quantlib_cross_checks, settlement_amount,
                                                 settlement_lag_table, thirty_360_days,
                                                 thirty_360_table, year_fraction)

PREV, SETTLE, NEXT = date(2026, 1, 15), date(2026, 4, 30), date(2026, 7, 15)
MAT = date(2036, 1, 15)
CPN, FACE = 0.05, 1_000_000.0
FEB = (date(2026, 2, 28), date(2026, 8, 31))


# ------------------------------------------------------------------ the ICMA fallback
def test_act_act_icma_without_a_schedule_guesses_quarterly_and_returns_exactly_a_quarter():
    good = act_act_icma(PREV, SETTLE, PREV, NEXT)
    bad = act_act_icma(PREV, SETTLE)
    assert good == pytest.approx(105 / (181 * 2.0), abs=1e-15)
    assert good == pytest.approx(0.29005525, abs=5e-9)
    # the trap: no error, and the number is a round quarter because it inferred 3 months
    assert bad == 0.25
    assert icma_frequency_guess(PREV, SETTLE) == 3          # from a 105-day stub
    assert icma_frequency_guess(PREV, NEXT) == 6            # the real semiannual period
    assert act_act_icma(PREV, NEXT, PREV, NEXT) == pytest.approx(0.5)


def test_the_schedule_free_fallback_costs_2002_dollars_of_accrued_per_million():
    good = accrued_interest(CPN, PREV, SETTLE, NEXT, "ACT/ACT ICMA", 2)
    bad = accrued_interest(CPN, PREV, SETTLE, NEXT, "ACT/ACT ICMA", 2, use_schedule=False)
    assert good == pytest.approx(1.45027624, abs=5e-9)
    assert bad == pytest.approx(1.25, abs=1e-12)
    assert FACE * (good - bad) / 100.0 == pytest.approx(2002.76, abs=0.01)


def test_the_icma_fallback_is_the_only_row_that_moves_by_more_than_a_hundred_dollars():
    rows = convention_table(CPN, PREV, SETTLE, NEXT, 2, FACE)
    by = {r["convention"]: r for r in rows}
    assert len(rows) == len(CONVENTIONS) + 1
    assert by["ACT/ACT ICMA"]["vs_icma_per_face"] == pytest.approx(0.0, abs=1e-9)
    assert by["ACT/ACT ISDA"]["vs_icma_per_face"] == pytest.approx(-119.20, abs=0.01)
    assert by["ACT/365F"]["vs_icma_per_face"] == pytest.approx(-119.20, abs=0.01)
    assert by["ACT/360"]["vs_icma_per_face"] == pytest.approx(+80.57, abs=0.01)
    assert by["30/360 BondBasis"]["vs_icma_per_face"] == pytest.approx(+80.57, abs=0.01)
    assert by["ACT/ACT ICMA (no schedule)"]["vs_icma_per_face"] == pytest.approx(-2002.76,
                                                                                 abs=0.01)
    others = [abs(r["vs_icma_per_face"]) for r in rows if "no schedule" not in r["convention"]]
    assert max(others) < 150.0                              # every real convention is small
    assert abs(by["ACT/ACT ICMA (no schedule)"]["vs_icma_per_face"]) > 10 * max(others)


def test_act_act_isda_splits_at_the_year_end_and_icma_does_not():
    # inside one non-leap year ISDA and ACT/365F agree exactly ...
    assert act_act_isda(PREV, SETTLE) == pytest.approx(105 / 365.0, abs=1e-15)
    # ... and across a leap boundary they do not, which is how the mismatch survives testing
    across = act_act_isda(date(2027, 11, 1), date(2028, 3, 1))
    assert across == pytest.approx(61 / 365.0 + 60 / 366.0, abs=1e-15)
    assert across != pytest.approx((date(2028, 3, 1) - date(2027, 11, 1)).days / 365.0,
                                   abs=1e-9)


# ------------------------------------------------------------------ 30/360 family
def test_the_thirty_360_flavours_disagree_only_on_the_31st_and_on_february():
    # an ordinary pair: all five agree
    a, b = date(2026, 3, 15), date(2026, 9, 15)
    assert len({thirty_360_days(a, b, f) for f in THIRTY_FLAVOURS}) == 1
    # the 31st splits US/BondBasis from the European family
    a, b = date(2026, 1, 15), date(2026, 7, 31)
    days = {f: thirty_360_days(a, b, f) for f in THIRTY_FLAVOURS}
    assert days["30/360 US"] == days["30/360 BondBasis"] == 196
    assert days["30E/360"] == days["30E/360 ISDA"] == days["30/360 Italian"] == 195
    # February splits it three ways, and US is NOT BondBasis
    days = {f: thirty_360_days(*FEB, f) for f in THIRTY_FLAVOURS}
    assert days["30/360 US"] == 180 and days["30/360 BondBasis"] == 183
    assert days["30E/360"] == 182
    assert days["30E/360 ISDA"] == days["30/360 Italian"] == 180
    assert len(set(days.values())) == 3


def test_the_february_spread_is_over_four_hundred_dollars_per_million():
    rows = thirty_360_table(*FEB, CPN, FACE)
    assert max(r["cash_vs_lowest"] for r in rows) == pytest.approx(416.67, abs=0.01)
    rows = thirty_360_table(date(2026, 1, 15), date(2026, 7, 31), CPN, FACE)
    assert max(r["cash_vs_lowest"] for r in rows) == pytest.approx(138.89, abs=0.01)
    rows = thirty_360_table(date(2028, 2, 29), date(2028, 8, 31), CPN, FACE)
    assert max(r["cash_vs_lowest"] for r in rows) == pytest.approx(277.78, abs=0.01)


def test_thirty_e_360_isda_needs_the_termination_date_to_be_evaluated():
    end = date(2027, 2, 28)
    assert is_last_of_february(end) and not is_last_of_february(date(2028, 2, 28))
    plain = thirty_360_days(date(2026, 8, 30), end, "30E/360 ISDA")
    final = thirty_360_days(date(2026, 8, 30), end, "30E/360 ISDA", termination=end)
    assert plain == 180                       # last of February -> 30
    assert final == 178                       # ... unless it is the termination date
    assert plain != final


def test_the_conventions_refuse_what_they_cannot_compute():
    with pytest.raises(ValueError, match="unknown convention"):
        year_fraction(PREV, SETTLE, "ACT/ACT")
    with pytest.raises(ValueError, match="unknown 30/360 flavour"):
        thirty_360_days(PREV, SETTLE, "30/360")
    with pytest.raises(ValueError, match="must not precede"):
        year_fraction(SETTLE, PREV, "ACT/360")
    with pytest.raises(ValueError, match="ref_start"):
        act_act_icma(PREV, SETTLE, NEXT, MAT)


# ------------------------------------------------------------------ clean / dirty / lag
def test_the_quote_is_clean_and_the_cash_is_dirty():
    acc = accrued_interest(CPN, PREV, SETTLE, NEXT, "ACT/ACT ICMA", 2)
    assert dirty_price(98.5, acc) == pytest.approx(99.950276, abs=5e-7)
    assert settlement_amount(98.5, acc, FACE) == pytest.approx(999_502.76, abs=0.01)
    assert settlement_amount(98.5, acc, FACE) - 985_000.0 == pytest.approx(14_502.76, abs=0.01)


def test_one_business_day_of_settlement_costs_as_much_as_the_whole_thirty_360_argument():
    rows = settlement_lag_table(CPN, PREV, SETTLE, NEXT, (0, 1, 2, 3), 2, FACE)
    by = {r["lag"]: r for r in rows}
    assert by[0]["settle"] == SETTLE
    assert by[1]["settle"] == date(2026, 5, 1)              # Friday
    assert by[2]["settle"] == date(2026, 5, 4)              # over the weekend
    assert by[1]["vs_t0_per_face"] == pytest.approx(138.12, abs=0.01)
    assert by[2]["vs_t0_per_face"] == pytest.approx(552.49, abs=0.01)
    assert by[3]["vs_t0_per_face"] == pytest.approx(690.61, abs=0.01)
    # one day of accrual is the same order as the 30/360 spread in the other direction
    assert by[1]["vs_t0_per_face"] == pytest.approx(138.89, abs=1.0)


def test_business_day_arithmetic_skips_weekends_and_the_holidays_it_is_given():
    fri = date(2026, 5, 1)
    assert add_business_days(fri, 1) == date(2026, 5, 4)
    assert add_business_days(fri, 1, holidays=[date(2026, 5, 4)]) == date(2026, 5, 5)
    assert add_business_days(date(2026, 5, 4), -1) == fri
    assert add_business_days(fri, 0) == fri


# ------------------------------------------------------------------ QuantLib
@pytest.mark.skipif(has_module("QuantLib"), reason="QuantLib is installed")
def test_cross_checks_return_none_without_quantlib():
    assert quantlib_cross_checks(PREV, SETTLE, NEXT, MAT, CPN) is None


@requires("QuantLib")
def test_quantlib_reproduces_every_number_and_shows_where_the_fallback_bites():
    live = quantlib_cross_checks(PREV, SETTLE, NEXT, MAT, CPN, pair=FEB)
    assert live is not None
    # nine named constants, three distinct answers, and USA != BondBasis
    assert len(live["thirty360"]) == 9
    assert live["thirty360_distinct"] == [180, 182, 183]
    assert live["thirty360"]["USA"] != live["thirty360"]["BondBasis"]
    mapping = {"USA": "30/360 US", "BondBasis": "30/360 BondBasis", "ISMA": "30/360 BondBasis",
               "NASD": "30/360 BondBasis", "European": "30E/360", "EurobondBasis": "30E/360",
               "ISDA": "30E/360 ISDA", "German": "30E/360 ISDA", "Italian": "30/360 Italian"}
    for const, flavour in mapping.items():
        assert live["thirty360"][const] == thirty_360_days(*FEB, flavour), const
    # the ICMA fallback, both ways
    assert live["isma_with_ref"] == pytest.approx(act_act_icma(PREV, SETTLE, PREV, NEXT),
                                                  abs=1e-15)
    assert live["isma_no_ref"] == 0.25
    assert live["isda_yf"] == pytest.approx(act_act_isda(PREV, SETTLE), abs=1e-15)
    assert live["act365f_yf"] == pytest.approx(105 / 365.0, abs=1e-15)
    assert live["act360_yf"] == pytest.approx(105 / 360.0, abs=1e-15)
    # a real bond agrees with the reference implementation on every counter
    for name, val in live["bond_accrued"].items():
        assert val == pytest.approx(accrued_interest(CPN, PREV, SETTLE, NEXT, name, 2),
                                    abs=1e-13), name
    # SAFE: the bond and BondFunctions supply the coupon period themselves
    assert live["bond_accrued_no_schedule"] == pytest.approx(
        live["bond_accrued"]["ACT/ACT ICMA"], abs=1e-14)
    assert live["mod_duration_no_schedule"] == pytest.approx(
        live["mod_duration_with_schedule"], abs=1e-12)
    # BITES: anything that computes a year fraction from two bare dates
    assert live["curve_t_isma"] == 0.25
    assert live["curve_t_act365"] == pytest.approx(105 / 365.0, abs=1e-15)
    bp = 1e4 * (live["curve_df_isma"] / live["curve_df_act365"] - 1.0)
    assert bp == pytest.approx(18.6, abs=0.1)
    bp2 = 1e4 * (live["compound_factor_act365"] / live["compound_factor_isma"] - 1.0)
    assert bp2 == pytest.approx(18.6, abs=0.1)


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.fixed_income.conventions")
    assert "accrued is (day count, coupon schedule, settlement date)" in out
    assert "the quote is clean while the cash is dirty" in out
    assert "2,002.76" in out
    assert out.isascii()
