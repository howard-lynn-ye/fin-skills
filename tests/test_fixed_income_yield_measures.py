"""fin_skills.fixed_income.yield_measures - discount rates, BEY, YTM, current yield, YTW.

The documented properties: a bill discount rate is not a yield and the gap widens with both
tenor and level, Treasury's investment rate branches at one half-year and this file reproduces
both worked examples printed in 31 CFR 356 Appendix B, current yield is right only at par, and
yield to worst on a callable above par is far below the yield to maturity.
"""
from __future__ import annotations

import pytest

from conftest import has_module, requires
from fin_skills.fixed_income.yield_measures import (bill_discount_from_price, bill_price_from_discount,
                                                    bill_table, bond_equivalent_yield,
                                                    bond_price_from_yield, current_yield,
                                                    money_market_yield, quantlib_cross_checks,
                                                    treasury_investment_rate, ytm,
                                                    yield_to_worst)

DISCOUNT = 0.05
DAYS = (28, 91, 182, 364)


# ------------------------------------------------------------------ bills
def test_the_discount_rate_divides_by_face_and_the_yield_divides_by_price():
    p = bill_price_from_discount(DISCOUNT, 182)
    assert p == pytest.approx(97.472222, abs=5e-7)
    assert bill_discount_from_price(p, 182) == pytest.approx(DISCOUNT, abs=1e-15)
    # the three add-on rates, in order, on the same bill
    assert money_market_yield(p, 182) == pytest.approx(0.051297, abs=5e-7)
    assert bond_equivalent_yield(p, 182) == pytest.approx(0.05200912, abs=5e-9)
    assert DISCOUNT < money_market_yield(p, 182) < bond_equivalent_yield(p, 182)
    # 365/360 is exactly the step between the last two
    assert bond_equivalent_yield(p, 182) == pytest.approx(
        money_market_yield(p, 182) * 365.0 / 360.0, abs=1e-15)


def test_the_gap_between_discount_and_yield_grows_with_tenor_and_with_level():
    rows = {r["days"]: r for r in bill_table(DISCOUNT, DAYS)}
    assert rows[28]["bp_discount_to_bey"] == pytest.approx(8.9, abs=0.05)
    assert rows[91]["bp_discount_to_bey"] == pytest.approx(13.4, abs=0.05)
    assert rows[182]["bp_discount_to_bey"] == pytest.approx(20.1, abs=0.05)
    assert rows[364]["bp_discount_to_bey"] == pytest.approx(33.9, abs=0.05)
    gaps = [rows[d]["bp_discount_to_bey"] for d in DAYS]
    assert gaps == sorted(gaps)                     # monotone in maturity
    level = [1e4 * (bond_equivalent_yield(bill_price_from_discount(d, 182), 182) - d)
             for d in (0.01, 0.03, 0.05, 0.08, 0.12)]
    assert level == sorted(level)                   # and monotone in the level
    assert level[0] == pytest.approx(1.9, abs=0.05)
    assert level[-1] == pytest.approx(95.2, abs=0.05)
    assert level[-1] / level[2] > 4.0               # 12% is over 4x the 5% error


def test_the_investment_rate_reproduces_both_worked_examples_in_31_cfr_356_appendix_b():
    # VI.B.1, cash management bill 1990-06-01 -> 1990-06-21; the regulation prints 8.076%
    short = treasury_investment_rate(99.559444, 20, 365)
    assert round(short * 100, 3) == 8.076
    assert short == pytest.approx(0.08075725, abs=5e-9)
    # VI.B.2, 52-week bill 1990-06-07 -> 1991-06-06; the regulation prints 8.237%
    long = treasury_investment_rate(92.265000, 364, 365)
    assert round(long * 100, 3) == 8.237
    assert long == pytest.approx(0.08237324, abs=5e-9)
    # and the short formula on the long bill is 16.9 bp too high
    naive = bond_equivalent_yield(92.265000, 364, 365)
    assert 1e4 * (naive - long) == pytest.approx(16.9, abs=0.05)


def test_the_investment_rate_branches_at_exactly_one_half_year():
    rows = {r["days"]: r for r in bill_table(DISCOUNT, DAYS)}
    for d in (28, 91, 182):
        assert rows[d]["uses_quadratic"] is False
        assert rows[d]["investment_rate"] == pytest.approx(rows[d]["bond_equivalent_yield"],
                                                           abs=1e-15)
    assert rows[364]["uses_quadratic"] is True
    assert rows[364]["investment_rate"] == pytest.approx(0.05270135, abs=5e-8)
    assert rows[364]["bp_bey_to_investment"] == pytest.approx(-6.9, abs=0.05)
    # 182 is inside the half-year (365/2 = 182.5) and 183 is not
    p = bill_price_from_discount(DISCOUNT, 183)
    assert treasury_investment_rate(p, 183) != pytest.approx(bond_equivalent_yield(p, 183),
                                                             abs=1e-9)


def test_the_bill_functions_refuse_impossible_inputs():
    with pytest.raises(ValueError, match="days must be positive"):
        bill_price_from_discount(DISCOUNT, 0)
    with pytest.raises(ValueError, match="price must be positive"):
        bond_equivalent_yield(0.0, 91)
    with pytest.raises(ValueError, match="price must be positive"):
        treasury_investment_rate(-1.0, 364)
    with pytest.raises(ValueError, match="r must be positive"):
        treasury_investment_rate(95.0, 0)


# ------------------------------------------------------------------ coupon bonds
def test_a_par_bond_prices_at_par_and_the_yield_round_trips():
    at_par = bond_price_from_yield(0.05, 0.05, 20)
    assert at_par["clean"] == pytest.approx(100.0, abs=1e-12)
    assert at_par["accrued"] == 0.0                 # w = 1 is a coupon date
    for y in (0.0, 0.01, 0.05, 0.12):
        p = bond_price_from_yield(0.05, y, 20)["clean"]
        assert ytm(p, 0.05, 20) == pytest.approx(y, abs=1e-12)
    # mid-period: accrued is the part of the coupon already run
    mid = bond_price_from_yield(0.05, 0.05, 20, w=0.5)
    assert mid["accrued"] == pytest.approx(1.25, abs=1e-12)
    assert mid["dirty"] == pytest.approx(mid["clean"] + mid["accrued"], abs=1e-12)
    with pytest.raises(ValueError, match="n must be at least"):
        bond_price_from_yield(0.05, 0.05, 0)
    with pytest.raises(ValueError, match="w must be"):
        bond_price_from_yield(0.05, 0.05, 20, w=1.5)


def test_current_yield_is_right_only_at_par_and_worst_on_short_bonds():
    assert current_yield(0.05, 100.0) == pytest.approx(ytm(100.0, 0.05, 20), abs=1e-12)
    disc5 = 1e4 * (ytm(85.0, 0.03, 10) - current_yield(0.03, 85.0))
    disc20 = 1e4 * (ytm(85.0, 0.03, 40) - current_yield(0.03, 85.0))
    prem5 = 1e4 * (ytm(118.0, 0.08, 10) - current_yield(0.08, 118.0))
    assert disc5 == pytest.approx(303.9, abs=0.1)
    assert disc20 == pytest.approx(57.8, abs=0.1)
    assert prem5 == pytest.approx(-278.7, abs=0.1)
    assert disc5 > disc20 > 0 > prem5               # sign flips with the discount/premium
    with pytest.raises(ValueError, match="clean price must be positive"):
        current_yield(0.05, 0.0)


def test_yield_to_worst_takes_each_call_at_its_own_redemption_price():
    out = yield_to_worst(108.0, 0.05, [(4, 102.0), (6, 101.0), (10, 100.0)], 20)
    ys = [c["yield"] for c in out["candidates"]]
    assert ys == sorted(ys)                         # earliest call is worst here
    assert out["yield_to_worst"] == pytest.approx(0.018909, abs=5e-7)
    assert out["yield_to_maturity"] == pytest.approx(0.040205, abs=5e-7)
    assert out["worst_at"] == 4
    assert out["bp_ytm_over_ytw"] == pytest.approx(213.0, abs=0.1)
    # repricing the 2y call to par instead of 102 UNDERSTATES the yield, so it is not a shortcut
    to_par = yield_to_worst(108.0, 0.05, [(4, 100.0)], 20)
    assert to_par["yield_to_worst"] < out["yield_to_worst"]
    # a bond below par is not called: the worst is maturity
    below = yield_to_worst(92.0, 0.05, [(4, 102.0), (6, 101.0)], 20)
    assert below["worst_at"] == 20
    assert below["bp_ytm_over_ytw"] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------ QuantLib
@pytest.mark.skipif(has_module("QuantLib"), reason="QuantLib is installed")
def test_cross_checks_return_none_without_quantlib():
    assert quantlib_cross_checks(DISCOUNT, DAYS, 0.05, 0.045, 10) is None


@requires("QuantLib")
def test_quantlib_confirms_the_bill_arithmetic_and_the_price_yield_pair():
    live = quantlib_cross_checks(DISCOUNT, DAYS, 0.05, 0.045, 10)
    assert live is not None
    for b in live["bills"]:
        assert b["ql_compound_factor"] == pytest.approx(b["direct"], abs=1e-15)
        assert b["bey"] == pytest.approx(bond_equivalent_yield(b["price"], b["days"]),
                                         abs=1e-15)
    mine = bond_price_from_yield(0.05, 0.045, live["periods"])["clean"]
    assert live["bond_clean"] == pytest.approx(mine, abs=1e-11)
    assert live["bond_yield"] == pytest.approx(ytm(live["bond_clean"], 0.05, live["periods"]),
                                               abs=1e-12)
    # the default Following payment convention is a different bond, not a different yield
    assert live["bond_clean_following"] - live["bond_clean"] == pytest.approx(-0.003831,
                                                                             abs=5e-7)


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.fixed_income.yield_measures")
    assert "a bill DISCOUNT RATE is not a yield" in out
    assert "on a callable, quote YTW" in out
    assert "8.076%" in out and "8.237%" in out
    assert out.isascii()
