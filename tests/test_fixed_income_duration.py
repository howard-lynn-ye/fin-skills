"""fin_skills.fixed_income.duration - Macaulay, modified, effective and spread duration.

The documented properties: Macaulay is exactly (1 + y/m) times modified, duration alone always
understates the price and misses 10.19 points at -200 bp on a 30y bond, a DV01 on the clean
price is short by ModDur x accrued, a floater's rate duration is the time to its next reset
while its spread duration equals the fixed comparator's modified duration, and QuantLib's
Duration.Modified silently returns a 28.6% smaller number on a non-compounded InterestRate.
"""
from __future__ import annotations

import pytest

from conftest import has_module, requires
from fin_skills.fixed_income.duration import (DURATION_TYPES, bond_cashflows, bond_price,
                                              convexity, duration, dv01, effective_convexity,
                                              effective_duration, frn_durations, frn_price,
                                              macaulay_duration, modified_duration,
                                              price_change, pv01, quantlib_cross_checks)

NOTIONAL = 1_000_000.0
INDEX, QM, DM, NQ = 0.05, 0.004, 0.004, 20


# ------------------------------------------------------------------ Macaulay vs modified
def test_macaulay_is_exactly_one_plus_y_over_m_times_modified():
    for cpn, y, yrs, m in ((0.05, 0.05, 10, 2), (0.05, 0.05, 10, 4), (0.04, 0.04, 30, 2),
                           (0.12, 0.12, 10, 2), (0.0, 0.05, 10, 2), (0.07, 0.03, 3, 1)):
        n = yrs * m
        mac, mod = macaulay_duration(cpn, y, n, m), modified_duration(cpn, y, n, m)
        assert mac / mod == pytest.approx(1.0 + y / m, abs=1e-14), (cpn, y, yrs, m)
        assert mac > mod > 0.0
    # the documented headline numbers
    assert macaulay_duration(0.05, 0.05, 20) == pytest.approx(7.98944567, abs=5e-9)
    assert modified_duration(0.05, 0.05, 20) == pytest.approx(7.79458114, abs=5e-9)
    assert modified_duration(0.04, 0.04, 60) == pytest.approx(17.38044334, abs=5e-9)
    # a zero-coupon bond's Macaulay duration IS its maturity - the reason the two get confused
    assert macaulay_duration(0.0, 0.05, 20) == pytest.approx(10.0, abs=1e-13)
    assert modified_duration(0.0, 0.05, 20) == pytest.approx(10.0 / 1.025, abs=1e-13)


def test_duration_dispatch_rejects_a_name_it_does_not_implement():
    assert DURATION_TYPES == ("macaulay", "modified")
    assert duration(0.05, 0.05, 20, "modified") == modified_duration(0.05, 0.05, 20)
    assert duration(0.05, 0.05, 20, "macaulay") == macaulay_duration(0.05, 0.05, 20)
    with pytest.raises(ValueError, match="kind must be"):
        duration(0.05, 0.05, 20, "effective")
    with pytest.raises(ValueError, match="n must be at least"):
        bond_cashflows(0.05, 0)


def test_a_par_bond_prices_at_par_and_the_last_cashflow_carries_the_principal():
    cf = bond_cashflows(0.05, 20)
    assert cf[0] == pytest.approx(2.5) and cf[-1] == pytest.approx(102.5)
    assert bond_price(0.05, 0.05, 20) == pytest.approx(100.0, abs=1e-12)
    assert bond_price(0.04, 0.04, 60) == pytest.approx(100.0, abs=1e-12)


# ------------------------------------------------------------------ convexity
def test_duration_alone_always_understates_the_price_and_convexity_fixes_most_of_it():
    assert convexity(0.04, 0.04, 60) == pytest.approx(420.8130, abs=5e-5)
    for dy in (0.0025, 0.005, 0.01, 0.02, -0.0025, -0.005, -0.01, -0.02):
        r = price_change(0.04, 0.04, 60, dy)
        assert r["err_duration_only"] < 0.0, dy          # one-sided, both directions
        assert abs(r["err_duration_convexity"]) < abs(r["err_duration_only"]), dy
    m100 = price_change(0.04, 0.04, 60, -0.01)
    m200 = price_change(0.04, 0.04, 60, -0.02)
    p200 = price_change(0.04, 0.04, 60, +0.02)
    assert m100["err_duration_only"] == pytest.approx(-2.30969, abs=5e-6)
    assert m100["pct_err_duration_only"] == pytest.approx(-1.930, abs=0.001)
    assert m200["err_duration_only"] == pytest.approx(-10.19415, abs=5e-6)
    assert m200["pct_err_duration_only"] == pytest.approx(-7.033, abs=0.001)
    assert p200["pct_err_duration_only"] == pytest.approx(-9.797, abs=0.001)
    assert abs(m200["err_duration_convexity"]) == pytest.approx(1.77789, abs=5e-6)
    # the residual shrinks like dy^3, so halving the shift cuts it by roughly eight
    small = price_change(0.04, 0.04, 60, -0.01)["err_duration_convexity"]
    assert abs(m200["err_duration_convexity"] / small) > 6.0


def test_effective_duration_reproduces_the_closed_form_where_the_closed_form_applies():
    for cpn, y, n in ((0.04, 0.04, 60), (0.05, 0.045, 20), (0.0, 0.05, 60)):
        eff = effective_duration(lambda yy, c=cpn, nn=n: bond_price(c, yy, nn), y)
        assert eff == pytest.approx(modified_duration(cpn, y, n), abs=1e-4)
        effc = effective_convexity(lambda yy, c=cpn, nn=n: bond_price(c, yy, nn), y)
        assert effc == pytest.approx(convexity(cpn, y, n), rel=1e-5)


# ------------------------------------------------------------------ DV01
def test_a_dv01_on_the_clean_price_is_short_by_modified_duration_times_accrued():
    clean = bond_price(0.05, 0.045, 20)
    md = modified_duration(0.05, 0.045, 20)
    on_coupon = dv01(0.05, 0.045, 20, notional=NOTIONAL, accrued=0.0)
    mid = dv01(0.05, 0.045, 20, notional=NOTIONAL, accrued=1.25)
    assert on_coupon == pytest.approx(md * clean * 1e-4 * NOTIONAL / 100.0, abs=1e-9)
    assert on_coupon == pytest.approx(817.24, abs=0.01)
    assert mid == pytest.approx(827.06, abs=0.01)
    assert on_coupon - mid == pytest.approx(-md * 1.25 * 1e-4 * NOTIONAL / 100.0, abs=1e-9)
    assert on_coupon - mid == pytest.approx(-9.82, abs=0.01)


def test_pv01_reprices_and_dv01_approximates_and_the_gap_is_the_convexity_inside_one_bp():
    d = dv01(0.05, 0.045, 20, notional=NOTIONAL)
    p = pv01(0.05, 0.045, 20, notional=NOTIONAL)
    assert d == pytest.approx(817.2379, abs=5e-5)
    assert p == pytest.approx(816.8504, abs=5e-5)
    assert 0.0 < d - p < 1.0
    # on a 30y bond the same gap is larger, because convexity is larger
    d30 = dv01(0.04, 0.04, 60, notional=NOTIONAL)
    p30 = pv01(0.04, 0.04, 60, notional=NOTIONAL)
    assert (d30 - p30) > (d - p)


# ------------------------------------------------------------------ floaters
def test_an_frn_prices_at_par_when_the_quoted_margin_equals_the_discount_margin():
    assert frn_price(INDEX, QM, DM, NQ, 4) == pytest.approx(100.0, abs=1e-12)
    # and at any index level, because the coupon moves with the discount rate
    for i in (0.0, 0.02, 0.05, 0.10):
        assert frn_price(i, QM, DM, NQ, 4) == pytest.approx(100.0, abs=1e-12)
    # a discount margin above the quoted margin prices it below par
    assert frn_price(INDEX, QM, DM + 0.005, NQ, 4) < 100.0
    with pytest.raises(ValueError, match="n must be at least"):
        frn_price(INDEX, QM, DM, 0, 4)
    with pytest.raises(ValueError, match="w must be"):
        frn_price(INDEX, QM, DM, NQ, 4, w=0.0)


def test_rate_duration_tracks_the_next_reset_while_spread_duration_runs_to_maturity():
    fixed = modified_duration(INDEX + QM, INDEX + DM, NQ, 4)
    assert fixed == pytest.approx(4.356304, abs=5e-7)
    just_after = frn_durations(INDEX, QM, DM, NQ, 4, 1.0)
    mid = frn_durations(INDEX, QM, DM, NQ, 4, 0.5)
    almost = frn_durations(INDEX, QM, DM, NQ, 4, 0.01)
    assert just_after["rate_duration"] == pytest.approx(0.246670, abs=5e-7)
    assert mid["rate_duration"] == pytest.approx(0.124162, abs=5e-7)
    assert almost["rate_duration"] == pytest.approx(0.002500, abs=5e-7)
    for f in (just_after, mid, almost):
        # rate duration is the time to the next reset, to within the discounting
        assert f["rate_duration"] == pytest.approx(f["time_to_reset"], rel=0.02)
        assert f["spread_duration"] > 10.0 * f["rate_duration"]
    # on a reset date at par the spread duration IS the fixed comparator's modified duration
    assert just_after["spread_duration"] == pytest.approx(fixed, abs=1e-6)
    assert just_after["spread_duration"] == pytest.approx(4.356304, abs=5e-7)
    phantom = (fixed - just_after["rate_duration"]) * 1e-2 * NOTIONAL
    assert phantom == pytest.approx(41_096.0, abs=1.0)


# ------------------------------------------------------------------ QuantLib
@pytest.mark.skipif(has_module("QuantLib"), reason="QuantLib is installed")
def test_cross_checks_return_none_without_quantlib():
    assert quantlib_cross_checks(0.05, 0.05, 10) is None


@requires("QuantLib")
def test_quantlib_agrees_under_compounding_and_is_silently_wrong_under_simple():
    live = quantlib_cross_checks(0.05, 0.05, 10)
    assert live is not None
    n = live["periods"]
    comp = live["by_compounding"]["Compounded"]
    assert comp["macaulay"] == pytest.approx(macaulay_duration(0.05, 0.05, n), abs=1e-12)
    assert comp["modified"] == pytest.approx(modified_duration(0.05, 0.05, n), abs=1e-12)
    assert comp["convexity"] == pytest.approx(convexity(0.05, 0.05, n), abs=1e-10)
    # three enum values, two numbers
    assert comp["simple"] == pytest.approx(comp["macaulay"], abs=1e-12)
    assert comp["simple"] != pytest.approx(comp["modified"], abs=1e-6)
    # the guard covers Macaulay only
    for label in ("Continuous", "Simple"):
        row = live["by_compounding"][label]
        assert row["macaulay"] is None
        assert "compounded rate required" in row["error"]
        assert row["modified"] is not None and row["simple"] is not None
    simple = live["by_compounding"]["Simple"]
    assert simple["modified"] == pytest.approx(5.56346580, abs=5e-9)
    assert simple["modified"] / comp["modified"] - 1.0 == pytest.approx(-0.286, abs=0.001)
    assert simple["bpv"] / comp["bpv"] - 1.0 == pytest.approx(-0.286, abs=0.001)
    # basisPointValue is a signed price CHANGE, a DV01 is quoted positive
    assert comp["bpv"] < 0.0
    assert -comp["bpv"] * NOTIONAL / 100.0 == pytest.approx(
        dv01(0.05, 0.05, n, notional=NOTIONAL), abs=0.05)


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.fixed_income.duration")
    assert "Macaulay = (1 + y/m) x Modified" in out
    assert "duration is the time to its next reset" in out
    assert "10.19 points" in out
    assert out.isascii()
