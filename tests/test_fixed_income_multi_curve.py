"""fin_skills.fixed_income.multi_curve - projection vs discount curves on a swap.

The documented properties: a swap struck at its own par rate prices at zero under EVERY discount
curve so the standard check has no power, the par rate is a discount-weighted average of the
forwards and is exactly invariant when the forwards are flat and the schedules match, the
annuity is a level and moves by the whole basis (-1.286% here), and the resulting NPV error
vanishes at the money while the PV01 error does not.
"""
from __future__ import annotations

import math

import pytest

from conftest import has_module, requires
from fin_skills.fixed_income.multi_curve import (DEFAULT_BASIS, Curve, annuity,
                                                 basis_sensitivity, discounting_comparison,
                                                 flat_curve, floating_leg_pv, par_swap_rate,
                                                 pv01, quantlib_cross_checks, reprices_at_par,
                                                 schedule, swap_npv)

TIMES = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)
ZEROS = (0.0290, 0.0295, 0.0305, 0.0315, 0.0330, 0.0340, 0.0348, 0.0355, 0.0358, 0.0360)
NOTIONAL, STRIKE, TENOR = 100_000_000.0, 0.0450, 10.0
OIS = Curve(TIMES, ZEROS)
PROJ = Curve(TIMES, ZEROS, DEFAULT_BASIS)


# ------------------------------------------------------------------ the curve
def test_the_curve_round_trips_zeros_discounts_and_forwards():
    for t, z in zip(TIMES, ZEROS):
        assert OIS.zero(t) == pytest.approx(z, abs=1e-14)
        assert OIS.discount(t) == pytest.approx(math.exp(-z * t), abs=1e-14)
    assert OIS.discount(0.0) == 1.0
    # a constant spread is a constant shift of every zero rate
    for t in TIMES:
        assert PROJ.zero(t) - OIS.zero(t) == pytest.approx(DEFAULT_BASIS, abs=1e-14)
        assert PROJ.discount(t) < OIS.discount(t)
    assert OIS.shifted(10.0).zero(5.0) == pytest.approx(OIS.zero(5.0) + 0.0010, abs=1e-14)
    # forwards are the discount-factor ratio, and they are positive on this curve
    assert OIS.forward(1.0, 2.0) == pytest.approx(OIS.discount(1.0) / OIS.discount(2.0) - 1.0,
                                                  abs=1e-14)
    assert all(OIS.forward(t, t + 0.25) > 0 for t in (0.0, 1.0, 5.0, 9.75))


def test_the_curve_refuses_input_it_cannot_interpolate():
    with pytest.raises(ValueError, match="same length"):
        Curve([1.0, 2.0], [0.03])
    with pytest.raises(ValueError, match="strictly increasing"):
        Curve([2.0, 1.0], [0.03, 0.03])
    with pytest.raises(ValueError, match="strictly increasing"):
        Curve([0.0, 1.0], [0.03, 0.03])
    with pytest.raises(ValueError, match="must not be negative"):
        OIS.discount(-1.0)
    with pytest.raises(ValueError, match="t1 must exceed"):
        OIS.forward(2.0, 1.0)
    with pytest.raises(ValueError, match="must be positive"):
        schedule(10.0, 0)


# ------------------------------------------------------------------ the blind check
def test_a_swap_struck_at_its_own_par_rate_prices_at_zero_under_every_discount_curve():
    for disc in (OIS, PROJ, OIS.shifted(200.0), OIS.shifted(-150.0)):
        assert abs(reprices_at_par(PROJ, disc, TENOR, NOTIONAL)) < 1e-6
    # and the two par rates it produces are only 0.045 bp apart
    s_ois = par_swap_rate(PROJ, OIS, TENOR)
    s_prj = par_swap_rate(PROJ, PROJ, TENOR)
    assert s_ois == pytest.approx(0.0375361656, abs=5e-11)
    assert 1e4 * (s_prj - s_ois) == pytest.approx(-0.045, abs=0.001)


def test_the_par_rate_is_exactly_invariant_when_the_forwards_are_flat_and_the_dates_match():
    flat = flat_curve(0.0330)
    flat_p = Curve(flat.times, flat.zeros, DEFAULT_BASIS)
    same = par_swap_rate(flat_p, flat, TENOR, 4, 4)
    other = par_swap_rate(flat_p, flat_p, TENOR, 4, 4)
    assert same == pytest.approx(other, abs=1e-15)          # exact cancellation
    assert same == pytest.approx(0.035775135, abs=5e-9)
    # ... and stays invariant under any discount curve at all
    for disc in (OIS, OIS.shifted(300.0), flat_curve(0.10)):
        assert par_swap_rate(flat_p, disc, TENOR, 4, 4) == pytest.approx(same, abs=1e-12)
    # the annuity is NOT invariant on the same curves
    assert annuity(flat_p, TENOR, 4) / annuity(flat, TENOR, 4) - 1.0 == pytest.approx(
        -0.012582, abs=5e-7)
    # a semiannual-vs-quarterly schedule leaves a small residual, not zero
    mismatch = 1e4 * (par_swap_rate(flat_p, flat_p, TENOR, 2, 4)
                      - par_swap_rate(flat_p, flat, TENOR, 2, 4))
    assert mismatch == pytest.approx(0.118, abs=0.001)
    assert 0.0 < abs(mismatch) < 1.0


# ------------------------------------------------------------------ what moved
def test_the_annuity_moves_by_the_whole_basis_and_scales_every_risk_number():
    c = discounting_comparison(OIS, PROJ, TENOR, NOTIONAL, STRIKE)
    assert c["bp_par_rate"] == pytest.approx(-0.045, abs=0.001)
    assert c["pct_annuity"] == pytest.approx(-1.286, abs=0.001)
    assert c["pct_pv01"] == pytest.approx(c["pct_annuity"], abs=1e-9)   # PV01 is A x 1bp x N
    assert c["ois"]["annuity"] == pytest.approx(8.42485066, abs=5e-9)
    assert c["ois"]["pv01"] == pytest.approx(84_248.51, abs=0.01)
    assert c["pv01_error"] == pytest.approx(-1083.02, abs=0.01)
    assert c["npv_error"] == pytest.approx(77_083.24, abs=0.01)
    # the par rate is three orders of magnitude smaller a signal than the annuity
    assert abs(c["bp_par_rate"]) < 0.1
    assert abs(c["pct_annuity"]) > 1.0


def test_the_npv_error_vanishes_at_the_money_and_the_pv01_error_does_not():
    rows = basis_sensitivity(OIS, (5, 10, 26.161, 50, 100), TENOR, NOTIONAL, STRIKE)
    ann = [r["pct_annuity"] for r in rows]
    pv = [r["pv01_error"] for r in rows]
    assert ann == sorted(ann, reverse=True)          # monotone in the basis
    assert pv == sorted(pv, reverse=True)
    assert rows[2]["pct_annuity"] == pytest.approx(-1.286, abs=0.001)
    assert rows[-1]["pct_annuity"] == pytest.approx(-4.796, abs=0.001)
    # at a 100 bp basis the par rate crosses the strike and the NPV error collapses
    assert rows[-1]["par_rate"] == pytest.approx(STRIKE, abs=5e-4)
    assert abs(rows[-1]["npv_error"]) < abs(rows[0]["npv_error"])
    assert abs(rows[-1]["pv01_error"]) > abs(rows[0]["pv01_error"])
    # the NPV error is exactly (S - K) x dA x N
    c = discounting_comparison(OIS, PROJ, TENOR, NOTIONAL, STRIKE)
    d_annuity = c["projection"]["annuity"] - c["ois"]["annuity"]
    assert c["npv_error"] == pytest.approx(
        (c["projection"]["par_rate"] - STRIKE) * c["projection"]["annuity"] * NOTIONAL
        - (c["ois"]["par_rate"] - STRIKE) * c["ois"]["annuity"] * NOTIONAL, abs=1e-6)
    assert d_annuity < 0.0


def test_the_pieces_are_consistent_with_each_other():
    a = annuity(OIS, TENOR, 2)
    assert pv01(NOTIONAL, OIS, TENOR, 2) == pytest.approx(a * 1e-4 * NOTIONAL, abs=1e-9)
    s = par_swap_rate(PROJ, OIS, TENOR)
    assert floating_leg_pv(PROJ, OIS, TENOR, 4) == pytest.approx(s * a, abs=1e-12)
    # paying fixed at a rate above par is a negative NPV, and the sign flips with the direction
    npv = swap_npv(STRIKE, NOTIONAL, PROJ, OIS, TENOR)
    assert npv < 0.0 and STRIKE > s
    assert swap_npv(STRIKE, NOTIONAL, PROJ, OIS, TENOR, pay_fixed=False) == pytest.approx(-npv)
    assert swap_npv(s, NOTIONAL, PROJ, OIS, TENOR) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------ QuantLib
@pytest.mark.skipif(has_module("QuantLib"), reason="QuantLib is installed")
def test_cross_checks_return_none_without_quantlib():
    assert quantlib_cross_checks(list(zip(TIMES, ZEROS)), DEFAULT_BASIS) is None


@requires("QuantLib")
def test_quantlib_reproduces_the_shape_and_shows_what_a_live_handle_does():
    live = quantlib_cross_checks(list(zip(TIMES, ZEROS)), DEFAULT_BASIS, int(TENOR), NOTIONAL,
                                 STRIKE)
    assert live is not None
    # the same shape as the reference implementation, on real schedules and day counts
    assert live["bp_par_rate"] == pytest.approx(-0.054, abs=0.005)
    assert live["pct_annuity"] == pytest.approx(-1.290, abs=0.005)
    assert live["npv_error"] == pytest.approx(76_434.38, abs=1.0)
    assert abs(live["bp_par_rate"]) < 0.1 < abs(live["pct_annuity"])
    # a relinkable handle is a live reference: an already-built swap moves under it
    assert live["npv_after_relink"] != live["npv_before_relink"]
    assert live["npv_after_relink"] - live["npv_before_relink"] == pytest.approx(29_372.83,
                                                                                abs=1.0)
    # and a stale evaluation date gives exactly zero, not an error
    assert live["npv_after_maturity"] == 0.0


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.fixed_income.multi_curve")
    assert "the ANNUITY is a level and moves by" in out
    assert "Check the annuity, not the par rate." in out
    assert "BOTH reprice at par" in out
    assert out.isascii()
