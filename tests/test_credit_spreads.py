"""fin_skills.credit.spreads - G, I, Z, ASW, DM and OAS, and the curve each is measured against.

The documented properties: the G-spread is measured against the government PAR yield and
"YTM minus the government ZERO rate" is a different number that collapses onto it only on a
FLAT curve; the par/par ASW diverges from the Z-spread in proportion to the bond's distance
from par; the discount margin equals the quoted margin exactly at 100 and nowhere else; the
lattice reprices the zero curve to machine precision; and OAS equals the Z-spread on a bullet
bond, so the gap on a CALLABLE bond is the option cost and it vanishes at zero volatility.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires
from fin_skills.credit.spreads import (asset_swap_spread, bond_price, calibrate_tree,
                                       discount_factors, discount_margin, dm_table,
                                       g_spread_trap, govt_zeros, lattice_bond_price,
                                       lattice_oas, oas_repricing_error, oas_vs_zspread,
                                       option_cost_by_vol, par_yields, price_from_yield,
                                       quantlib_cross_check, slope_sensitivity, spread_table,
                                       swap_zeros, yield_to_maturity, z_spread)


# ------------------------------------------------------------------ the measured headline
def test_the_documented_three_spreads_reproduce_exactly():
    t = g_spread_trap()
    assert t["price"] == pytest.approx(85.700750, abs=5e-7)
    assert t["ytm_pct"] == pytest.approx(6.434845, abs=5e-7)
    assert t["g_spread_bp"] == pytest.approx(153.22, abs=0.005)
    assert t["naive_bp"] == pytest.approx(143.48, abs=0.005)
    assert t["z_spread_bp"] == pytest.approx(150.00, abs=1e-6)
    # the naive number is SMALLER on an upward-sloping curve - the sign is systematic
    assert t["naive_error_bp"] < 0


def test_every_measure_in_the_table_is_a_different_number():
    s = spread_table()
    vals = [s.g_spread_bp, s.naive_bp, s.i_spread_bp, s.z_spread_bp, s.z_over_swap_bp, s.asw_bp]
    assert len(vals) == len({round(v, 2) for v in vals}), "six measures, six distinct numbers"
    assert max(vals) - min(vals) == pytest.approx(153.22 - 110.71, abs=0.02)


# ------------------------------------------------------------------ the G-spread trap
def test_the_par_yield_sits_below_the_zero_rate_on_an_upward_sloping_curve():
    gz = govt_zeros(5)
    par = par_yields(gz)
    assert np.all(np.diff(gz) > 0)
    assert par[0] == pytest.approx(gz[0], abs=1e-15)      # 1y par IS the 1y zero
    assert np.all(par[1:] < gz[1:])                        # and every longer par is below


def test_the_trap_vanishes_on_a_flat_curve_and_scales_with_the_slope():
    rows = slope_sensitivity()
    flat = rows[0]
    assert flat["slope_bp_per_year"] == 0.0
    assert flat["error_bp"] == pytest.approx(0.0, abs=1e-9), "flat curve: the two must agree"
    errs = [r["error_bp"] for r in rows]
    assert all(b < a for a, b in zip(errs, errs[1:])), "error grows monotonically with slope"
    assert rows[2]["error_bp"] == pytest.approx(-9.74, abs=0.005)


def test_a_par_bond_reprices_to_100_on_the_curve_that_produced_its_par_yield():
    for zeros in (govt_zeros(), swap_zeros()):
        par = par_yields(zeros)
        for m in range(1, len(zeros) + 1):
            assert bond_price(par[m - 1] * 100.0, m, zeros) == pytest.approx(100.0, abs=1e-11)


def test_the_z_spread_and_the_ytm_round_trip():
    gz = govt_zeros()
    p = bond_price(3.0, 5, gz, 0.015)
    assert z_spread(p, 3.0, 5, gz) == pytest.approx(0.015, abs=1e-12)
    y = yield_to_maturity(p, 3.0, 5)
    assert price_from_yield(3.0, 5, y) == pytest.approx(p, abs=1e-10)


def test_the_curve_arithmetic_refuses_impossible_input():
    with pytest.raises(ValueError):
        discount_factors([-1.5])
    with pytest.raises(ValueError):
        bond_price(3.0, 99, govt_zeros())
    with pytest.raises(ValueError):
        yield_to_maturity(0.0, 3.0, 5)


# ------------------------------------------------------------------ asset swap
def test_the_asw_and_the_z_spread_coincide_at_par_and_separate_below_it():
    sz = swap_zeros()
    par5 = float(par_yields(sz)[4]) * 100.0
    at_par = asset_swap_spread(100.0, par5, 5, sz) * 1e4
    assert at_par == pytest.approx(0.0, abs=1e-9), "a par bond on its own curve has zero ASW"
    s = spread_table()
    assert s.asw_bp == pytest.approx(110.71, abs=0.005)
    assert s.z_over_swap_bp - s.asw_bp == pytest.approx(13.82, abs=0.01)
    assert s.asw_bp < s.z_over_swap_bp, "a discount bond's ASW is below its Z-spread"


# ------------------------------------------------------------------ discount margin
def test_the_discount_margin_equals_the_quoted_margin_only_at_par():
    rows = {r["price"]: r for r in dm_table()}
    assert rows[100.0]["dm_bp"] == pytest.approx(120.0, abs=1e-8)
    assert rows[102.0]["dm_bp"] == pytest.approx(74.83, abs=0.005)
    assert rows[98.50]["dm_bp"] == pytest.approx(154.56, abs=0.005)
    assert rows[95.0]["dm_bp"] == pytest.approx(237.57, abs=0.005)
    # the straight-line shortcut always UNDERSTATES the DM on a discount bond
    for p in (98.5, 95.0):
        assert rows[p]["approx_error_bp"] < 0
    assert rows[95.0]["approx_error_bp"] == pytest.approx(-17.57, abs=0.005)


def test_the_discount_margin_inverts_the_frn_price():
    dm = discount_margin(97.25, 0.04, 0.0120, 5.0)
    from fin_skills.credit.spreads import frn_price
    assert frn_price(0.04, 0.0120, dm, 5.0) == pytest.approx(97.25, abs=1e-10)


# ------------------------------------------------------------------ the lattice and the OAS
def test_the_lattice_reprices_the_zero_curve_it_was_calibrated_to():
    assert oas_repricing_error() < 1e-12
    gz = govt_zeros(10)
    target = discount_factors(gz)
    for sigma in (0.0, 0.10, 0.30):
        rates = calibrate_tree(gz, sigma)
        assert len(rates) == 10
        for i, r in enumerate(rates):
            assert len(r) == i + 1 and np.all(r > 0.0)
            assert np.all(np.diff(r) > 0.0) if sigma > 0 else np.all(np.diff(r) == 0.0)
        got = [lattice_bond_price(rates, 0.0, m, 0.0, redemption=1.0) for m in range(1, 11)]
        assert np.max(np.abs(np.array(got) - target)) < 1e-12


def test_oas_equals_the_z_spread_on_a_bullet_bond():
    o = oas_vs_zspread()
    # the identity that proves the engine; the residual is the lattice's own convexity
    assert o["bullet_oas_bp"] == pytest.approx(80.0, abs=1e-6)
    assert o["bullet_gap_bp"] == pytest.approx(-0.1455, abs=0.001)
    assert abs(o["bullet_gap_bp"]) < 0.25


def test_the_gap_on_a_callable_bond_is_the_option_cost():
    o = oas_vs_zspread()
    assert o["callable_price"] == pytest.approx(99.345962, abs=5e-6)
    assert o["bullet_price"] == pytest.approx(102.219687, abs=5e-6)
    assert o["z_spread_bp"] == pytest.approx(120.04, abs=0.005)
    assert o["oas_bp"] == pytest.approx(80.00, abs=1e-6)
    assert o["option_cost_bp"] == pytest.approx(40.04, abs=0.005)
    assert o["callable_price"] < o["bullet_price"], "the call can only hurt the holder"


def test_the_option_cost_is_zero_at_zero_volatility_and_grows_with_it():
    rows = option_cost_by_vol()
    assert rows[0]["sigma"] == 0.0
    assert rows[0]["option_cost_bp"] == pytest.approx(0.0049, abs=0.002)
    costs = [r["option_cost_bp"] for r in rows]
    assert all(b > a for a, b in zip(costs, costs[1:])), "more vol, more option cost"
    assert rows[-1]["option_cost_bp"] - rows[1]["option_cost_bp"] == pytest.approx(
        62.32 - 17.46, abs=0.02), "10% vs 30% vol is ~45 bp of 'spread' on one bond"
    # the OAS is invariant to the vol used to GENERATE the price, by construction
    assert all(r["oas_bp"] == pytest.approx(80.0, abs=1e-6) for r in rows)


def test_a_callable_never_prices_above_its_bullet_and_the_call_binds_from_the_first_call_date():
    gz = govt_zeros(10)
    rates = calibrate_tree(gz, 0.20)
    bullet = lattice_bond_price(rates, 6.5, 10, 0.008)
    for first in (5, 3, 1):
        callable_ = lattice_bond_price(rates, 6.5, 10, 0.008, call_price=100.0, first_call=first)
        assert callable_ <= bullet + 1e-12
    early = lattice_bond_price(rates, 6.5, 10, 0.008, call_price=100.0, first_call=1)
    late = lattice_bond_price(rates, 6.5, 10, 0.008, call_price=100.0, first_call=5)
    assert early < late, "a longer call window is worth more to the issuer"
    with pytest.raises(ValueError):
        lattice_bond_price(rates, 6.5, 30, 0.0)


def test_the_lattice_oas_inverts_the_lattice_price():
    gz = govt_zeros(10)
    rates = calibrate_tree(gz, 0.20)
    for s in (0.0, 0.005, 0.02):
        p = lattice_bond_price(rates, 6.5, 10, s, call_price=100.0, first_call=5)
        assert lattice_oas(p, rates, 6.5, 10, 100.0, 5) == pytest.approx(s, abs=1e-11)


# ------------------------------------------------------------------ determinism and the demo
def test_the_run_is_deterministic():
    a, b = spread_table(), spread_table()
    assert a == b
    assert [r["option_cost_bp"] for r in option_cost_by_vol()] == \
           [r["option_cost_bp"] for r in option_cost_by_vol()]


def test_the_demo_prints_the_rule_and_stays_ascii(run_main):
    out = run_main("fin_skills.credit.spreads")
    assert all(ord(c) < 128 for c in out), "a non-ASCII char dies on a stock Windows console"
    assert "THE RULE:" in out
    assert "the number AND the curve it was measured against" in out
    for token in ("153.22", "143.48", "150.00", "110.71", "120.04", "80.00", "40.04"):
        assert token in out


@requires("QuantLib")
def test_quantlib_agrees_on_the_z_spread_and_the_yield():
    q = quantlib_cross_check()
    assert q is not None
    assert abs(q["ql_z_spread_bp"] - q["mine_z_spread_bp"]) < 1e-6
    assert abs(q["ql_ytm_pct"] - q["mine_ytm_pct"]) < 1e-9
    assert q["ql_z_spread_bp"] == pytest.approx(150.0, abs=1e-6)


def test_the_cross_check_returns_none_rather_than_raising_when_quantlib_is_absent(monkeypatch):
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "QuantLib":
            raise ImportError("blocked")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert quantlib_cross_check() is None
