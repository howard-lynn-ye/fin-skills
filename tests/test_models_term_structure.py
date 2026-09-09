"""fin_skills.models.term_structure - bootstrap, conventions, Nelson-Siegel, short-rate bonds.

The documented properties: a bootstrapped curve reprices its own par bonds exactly, a discount
factor is a different zero rate under every convention, a free Nelson-Siegel lambda destabilises
the betas it is meant to shape, the affine A and B are individually right and collapse to the
deterministic limit, and an Euler CIR step turns one path in five into NaN.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from conftest import has_module, requires
from fin_skills.models.term_structure import (DIEBOLD_LI_LAMBDA_PER_MONTH,
                                              DIEBOLD_LI_LAMBDA_PER_YEAR,
                                              DIEBOLD_LI_MATURITIES_MONTHS, affine_ab,
                                              bootstrap_from_par, cir_bond, convention_table,
                                              curvature_loading_peak, deterministic_bond,
                                              diebold_li_lambda_facts, discount_factor,
                                              fit_nelson_siegel, fit_svensson,
                                              hull_white_bond, lambda_stability, ns_loadings,
                                              nelson_siegel, par_bond_price,
                                              quantlib_cross_checks, simulate_cir, svensson,
                                              svensson_collinearity, synthetic_yields,
                                              vasicek_bond, year_fraction, zero_rate,
                                              zero_curve_from_discounts)

PAR = (0.03, 0.035, 0.04, 0.044, 0.047)
D0, D1 = date(2026, 9, 8), date(2031, 9, 8)
VAS = dict(a=0.1, b=0.05, sigma=0.01)
CIR = dict(kappa=0.3, theta=0.05, sigma=0.1)


# ------------------------------------------------------------------ bootstrap
def test_a_bootstrapped_curve_reprices_every_par_bond_at_exactly_100():
    dfs = bootstrap_from_par(PAR)
    assert len(dfs) == len(PAR)
    for n, c in enumerate(PAR, 1):
        assert par_bond_price(c, dfs, n) == pytest.approx(100.0, abs=1e-10)
    assert (np.diff(dfs) < 0).all()                      # discount factors fall with maturity
    assert dfs[0] == pytest.approx(1.0 / 1.03)
    assert dfs[-1] == pytest.approx(0.79203794, abs=5e-9)


def test_the_discount_factor_round_trip_is_exact_in_every_compounding():
    dfs = bootstrap_from_par(PAR)
    times = np.arange(1.0, len(PAR) + 1.0)
    for comp in ("continuous", "annual", "semiannual", "simple"):
        zs = zero_curve_from_discounts(dfs, times, comp)
        back = np.array([discount_factor(z, t, comp) for z, t in zip(zs, times)])
        assert np.max(np.abs(back - dfs)) < 1e-15, comp
    assert zero_rate(dfs[0], 1.0, "annual") == pytest.approx(0.03, abs=1e-15)


def test_bootstrap_refuses_par_rates_it_cannot_strip():
    with pytest.raises(ValueError, match="below -100"):
        bootstrap_from_par([-1.5])
    with pytest.raises(ValueError, match="not positive"):
        bootstrap_from_par([0.03, 3.0, 3.0, 3.0, 3.0, 3.0])
    with pytest.raises(ValueError, match="must be positive"):
        zero_rate(0.0, 1.0)
    with pytest.raises(ValueError, match="compounding"):
        zero_rate(0.9, 1.0, "quarterly")
    with pytest.raises(ValueError, match="compounding"):
        discount_factor(0.05, 1.0, "quarterly")


# ------------------------------------------------------------------ conventions
def test_day_count_conventions_give_three_different_year_fractions():
    assert (D1 - D0).days == 1826                        # 5 calendar years, one leap day
    assert year_fraction(D0, D1, "ACT/365") == pytest.approx(1826 / 365)
    assert year_fraction(D0, D1, "ACT/360") == pytest.approx(1826 / 360)
    assert year_fraction(D0, D1, "30/360") == pytest.approx(5.0)
    # 30/360 bond basis: the 31st collapses to the 30th on both legs
    assert year_fraction(date(2026, 1, 31), date(2026, 7, 31), "30/360") == pytest.approx(0.5)
    with pytest.raises(ValueError, match="convention"):
        year_fraction(D0, D1, "ACT/ACT")


def test_one_discount_factor_is_twelve_different_zero_rates():
    rows = convention_table(0.79203794, D0, D1)
    by = {(r["day_count"], r["compounding"]): r for r in rows}
    assert len(rows) == 12
    assert by[("30/360", "annual")]["bp_vs_base"] == pytest.approx(0.0, abs=1e-9)
    # the four documented decompositions, to the basis point
    assert by[("30/360", "continuous")]["bp_vs_base"] == pytest.approx(-11.04, abs=0.01)
    assert by[("ACT/365", "annual")]["bp_vs_base"] == pytest.approx(-0.27, abs=0.01)
    assert by[("ACT/360", "annual")]["bp_vs_base"] == pytest.approx(-6.95, abs=0.01)
    assert by[("ACT/360", "continuous")]["bp_vs_base"] == pytest.approx(-17.68, abs=0.01)
    # simple compounding is the biggest single error and it is the easy one to write
    assert by[("30/360", "simple")]["bp_vs_base"] == pytest.approx(+47.80, abs=0.01)
    spread = max(r["zero"] for r in rows) - min(r["zero"] for r in rows)
    assert spread * 1e4 > 60.0                           # over 60 bp between the extremes


# ------------------------------------------------------------------ Nelson-Siegel
def test_the_curvature_loading_peak_fixes_the_diebold_li_lambda_and_its_unit():
    x_star = curvature_loading_peak()
    assert x_star == pytest.approx(1.793282, abs=1e-5)
    facts = diebold_li_lambda_facts()
    # the published 0.0609 peaks at 29.45 months, not exactly 30
    assert facts["peak_months_at_paper_lambda"] == pytest.approx(29.45, abs=0.01)
    assert facts["lambda_exact_for_30_months"] == pytest.approx(0.0598, abs=1e-4)
    # months -> years is a factor of 12, and getting it wrong moves the hump by 12x
    assert DIEBOLD_LI_LAMBDA_PER_YEAR == pytest.approx(12.0 * DIEBOLD_LI_LAMBDA_PER_MONTH)
    assert x_star / DIEBOLD_LI_LAMBDA_PER_YEAR == pytest.approx(
        facts["peak_months_at_paper_lambda"] / 12.0)


def test_nelson_siegel_loadings_have_the_documented_limits():
    lam = DIEBOLD_LI_LAMBDA_PER_MONTH
    short = ns_loadings(np.array([1e-8]), lam)[0]
    long = ns_loadings(np.array([1e9]), lam)[0]
    assert short == pytest.approx([1.0, 1.0, 0.0], abs=1e-6)   # slope loads fully at the front
    assert long == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)    # only the level survives
    b = (6.0, -2.0, -1.5)
    assert nelson_siegel(np.array([1e-8]), *b, lam)[0] == pytest.approx(b[0] + b[1], abs=1e-5)
    assert nelson_siegel(np.array([1e7]), *b, lam)[0] == pytest.approx(b[0], abs=1e-5)
    # Svensson with beta3 = 0 is exactly Nelson-Siegel
    t = np.array(DIEBOLD_LI_MATURITIES_MONTHS, dtype=float)
    assert np.allclose(svensson(t, *b, 0.0, lam, 0.5), nelson_siegel(t, *b, lam))


def test_a_fixed_lambda_fit_recovers_the_generating_betas():
    t, clean, noisy = synthetic_yields(seed=0, noise_bp=5.0)
    exact = fit_nelson_siegel(t, clean, DIEBOLD_LI_LAMBDA_PER_MONTH)
    assert exact["betas"] == pytest.approx((6.0, -2.0, -1.5), abs=1e-9)
    assert exact["rmse"] < 1e-12 and exact["fixed"] is True
    fitted = fit_nelson_siegel(t, noisy, DIEBOLD_LI_LAMBDA_PER_MONTH)
    assert fitted["rmse"] * 100 < 6.0                    # under 6 bp on 5 bp of noise


def test_freeing_lambda_buys_almost_no_fit_and_costs_an_order_of_magnitude_in_stability():
    out = lambda_stability(n_boot=60, seed=0)
    assert out["rmse_free"] <= out["rmse_fixed"]                       # it must fit at least as well
    assert (out["rmse_fixed"] - out["rmse_free"]) * 100 < 1.0          # by under one basis point
    assert out["beta2_std_free"] / out["beta2_std_fixed"] > 5.0        # and 5x+ less stable
    assert out["beta1_std_free"] / out["beta1_std_fixed"] > 5.0
    assert out["lam_std"] > 0.005                                      # lambda itself wanders
    assert out["lam_min"] <= 0.011                                     # down onto the search bound
    assert out["base_fixed"]["lam"] == DIEBOLD_LI_LAMBDA_PER_MONTH


def test_svensson_loadings_become_collinear_as_the_two_decays_converge():
    t = np.array(DIEBOLD_LI_MATURITIES_MONTHS, dtype=float)
    conds = dict(svensson_collinearity(t, DIEBOLD_LI_LAMBDA_PER_MONTH))
    assert conds[2.0] < conds[1.2] < conds[1.05] < conds[1.01]
    assert conds[1.01] / conds[2.0] > 20.0
    fit = fit_svensson(t, synthetic_yields(seed=0)[2])
    assert len(fit["params"]) == 6 and fit["rmse"] * 100 < 6.0


# ------------------------------------------------------------------ affine bonds
def test_the_affine_bond_prices_are_between_zero_and_one_and_fall_with_maturity():
    for tau in (1.0, 4.0, 10.0):
        for p in (vasicek_bond(0.04, tau, **VAS), cir_bond(0.04, tau, **CIR)):
            assert 0.0 < p < 1.0
    assert vasicek_bond(0.04, 10.0, **VAS) < vasicek_bond(0.04, 4.0, **VAS)
    assert cir_bond(0.04, 10.0, **CIR) < cir_bond(0.04, 4.0, **CIR)
    assert vasicek_bond(0.04, 0.0, **VAS) == pytest.approx(1.0)
    assert cir_bond(0.04, 0.0, **CIR) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="a must be positive"):
        vasicek_bond(0.04, 4.0, 0.0, 0.05, 0.01)
    with pytest.raises(ValueError, match="model must be"):
        affine_ab("ho-lee", 1.0)


def test_b_is_minus_the_log_price_derivative_and_a_is_the_rest():
    h = 1e-6
    for model, price, kw in (("vasicek", vasicek_bond, VAS), ("cir", cir_bond, CIR)):
        A, B = affine_ab(model, 4.0, **kw)
        num = -(math.log(price(0.04 + h, 4.0, **kw))
                - math.log(price(0.04 - h, 4.0, **kw))) / (2 * h)
        assert B == pytest.approx(num, abs=1e-8)
        assert A == pytest.approx(price(0.04, 4.0, **kw) * math.exp(B * 0.04), abs=1e-14)
        # the price is affine in r in the exponent: the same B at any r
        assert -(math.log(price(0.09 + h, 4.0, **kw))
                 - math.log(price(0.09 - h, 4.0, **kw))) / (2 * h) == pytest.approx(B, abs=1e-8)


def test_both_models_collapse_to_the_deterministic_integral_and_one_stops_being_computable():
    det_v = deterministic_bond(0.04, 4.0, VAS["a"], VAS["b"])
    det_c = deterministic_bond(0.04, 4.0, CIR["kappa"], CIR["theta"])
    assert vasicek_bond(0.04, 4.0, VAS["a"], VAS["b"], 1e-9) == pytest.approx(det_v, abs=1e-14)
    assert cir_bond(0.04, 4.0, CIR["kappa"], CIR["theta"], 1e-4) == pytest.approx(det_c, abs=1e-8)
    # and the CIR exponent 2*kappa*theta/sigma^2 destroys it well before sigma reaches zero
    assert cir_bond(0.04, 4.0, CIR["kappa"], CIR["theta"], 1e-9) > 100.0
    # a bond price over 100 is the tell; Vasicek never does this
    assert 0.0 < vasicek_bond(0.04, 4.0, VAS["a"], VAS["b"], 1e-12) < 1.0


def test_hull_white_fits_the_initial_curve_it_is_given():
    flat = 0.05
    p = hull_white_bond(1.0, 5.0, 0.04, 0.1, 0.01,
                        lambda T: math.exp(-flat * T), lambda t: flat)
    assert 0.0 < p < 1.0
    # with sigma = 0 and r_t equal to the forward, it reproduces the forward discount exactly
    p0 = hull_white_bond(1.0, 5.0, flat, 0.1, 0.0,
                         lambda T: math.exp(-flat * T), lambda t: flat)
    assert p0 == pytest.approx(math.exp(-flat * 4.0), abs=1e-12)


# ------------------------------------------------------------------ CIR simulation
def test_plain_euler_kills_paths_and_full_truncation_does_not():
    kw = dict(r0=0.03, kappa=0.5, theta=0.03, T=5.0, steps=60, paths=4000, seed=0)
    exact = cir_bond(0.03, 5.0, 0.5, 0.03, 0.15)
    eu = simulate_cir(sigma=0.15, scheme="euler", **kw)
    ft = simulate_cir(sigma=0.15, scheme="full_truncation", **kw)
    assert 2 * 0.5 * 0.03 > 0.15 ** 2                    # Feller HOLDS and paths still die
    assert eu["nan_paths"] / eu["paths"] > 0.10
    assert ft["nan_paths"] == 0
    # dropping the dead paths drops the LOW-rate ones: survivorship bias, many se wide
    assert eu["mc_price"] < exact
    assert abs(eu["mc_price"] - exact) > 10.0 * eu["mc_se"]
    assert abs(ft["mc_price"] - exact) < 3.0 * ft["mc_se"]
    # full truncation lets MORE steps go negative and is far more accurate: the NaN is the bug
    assert ft["negative_steps"] > eu["negative_steps"]
    with pytest.raises(ValueError, match="scheme"):
        simulate_cir(sigma=0.15, scheme="milstein", **kw)


def test_violating_feller_makes_the_euler_failure_worse():
    kw = dict(r0=0.03, kappa=0.5, theta=0.03, T=5.0, steps=60, paths=4000, seed=0)
    holds = simulate_cir(sigma=0.15, scheme="euler", **kw)
    violated = simulate_cir(sigma=0.20, scheme="euler", **kw)
    assert 2 * 0.5 * 0.03 < 0.20 ** 2                    # Feller violated
    assert violated["nan_paths"] > 2 * holds["nan_paths"]
    ft = simulate_cir(sigma=0.20, scheme="full_truncation", **kw)
    assert ft["nan_paths"] == 0
    assert abs(ft["mc_price"] - cir_bond(0.03, 5.0, 0.5, 0.03, 0.20)) < 3.0 * ft["mc_se"]


def test_a_seeded_simulation_is_deterministic():
    kw = dict(r0=0.03, kappa=0.5, theta=0.03, sigma=0.15, T=5.0, steps=30, paths=500,
              scheme="full_truncation")
    assert simulate_cir(seed=1, **kw) == simulate_cir(seed=1, **kw)
    assert simulate_cir(seed=2, **kw) != simulate_cir(seed=1, **kw)


# ------------------------------------------------------------------ QuantLib
@pytest.mark.skipif(has_module("QuantLib"), reason="QuantLib is installed")
def test_cross_checks_return_none_without_quantlib():
    assert quantlib_cross_checks(PAR, bootstrap_from_par(PAR), D0, D1,
                                 (0.05, -0.02, 0.01, 0.5),
                                 (0.05, -0.02, 0.01, 0.02, 0.5, 0.1)) is None


@requires("QuantLib")
def test_quantlib_confirms_the_closed_forms_the_day_counts_and_its_own_argument_order():
    ns = (0.05, -0.02, 0.01, 0.5)
    sv = (0.05, -0.02, 0.01, 0.02, 0.5, 0.1)
    dfs = bootstrap_from_par(PAR)
    live = quantlib_cross_checks(PAR, dfs, D0, D1, ns, sv)
    assert live is not None
    assert live["vasicek"] == pytest.approx(vasicek_bond(0.04, 4.0, **VAS), abs=1e-12)
    assert live["cir"] == pytest.approx(cir_bond(0.04, 4.0, **CIR), abs=1e-12)
    assert live["hull_white"] == pytest.approx(
        hull_white_bond(1.0, 5.0, 0.04, 0.1, 0.01,
                        lambda T: math.exp(-0.05 * T), lambda t: 0.05), abs=1e-10)
    # Vasicek is (r0, speed, level, sigma); CoxIngersollRoss is (r0, level, speed, sigma)
    assert live["vasicek_swapped"] != live["vasicek"]
    assert abs(live["cir_swapped"] - live["cir"]) > 0.05          # over 5 price points, no error
    assert 0.0 < live["cir_swapped"] < 1.0                        # and perfectly plausible
    # the parameterizations agree, so lambda means the same thing in both
    assert live["ns_disc_5y"] == pytest.approx(
        math.exp(-nelson_siegel(np.array([5.0]), *ns)[0] * 5.0), abs=1e-12)
    assert live["sv_disc_5y"] == pytest.approx(
        math.exp(-svensson(np.array([5.0]), *sv)[0] * 5.0), abs=1e-12)
    assert live["five_pct_annual_as_continuous"] == pytest.approx(math.log(1.05), abs=1e-9)
    for conv, key in (("ACT/365", "yf_act365"), ("ACT/360", "yf_act360"), ("30/360", "yf_30360")):
        assert live[key] == pytest.approx(year_fraction(D0, D1, conv), abs=1e-12)
    for price in live["par_bond_prices"]:
        assert price == pytest.approx(100.0, abs=1e-8)


@pytest.mark.slow      # ~14 s: 200 bootstrap refits plus four 20,000-path CIR simulations
def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.models.term_structure")
    assert "a zero rate is (day count, compounding, instrument)" in out
    assert "without truncation" in out
    assert out.isascii()
