"""fin_skills.models.vol_surface - invert, fit, check for arbitrage, interpolate.

The properties asserted here are the ones the SKILL.md documents: the solver round-trips to
machine precision and refuses a sub-intrinsic price, an absolute price tolerance would return
the first guess instead, Durrleman's g agrees with Breeden-Litzenberger point for point, the
published Vogt slice fails the butterfly test while passing every box condition, and linear
interpolation in implied vol manufactures a calendar-spread arbitrage that total variance
cannot.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from _helpers import bs_call, bs_put
from conftest import has_module, requires
from fin_skills.models.vol_surface import (INTERP_METHODS, SVI_PARAM_ORDER, VOGT,
                                           black_normalised, breeden_litzenberger_density,
                                           bsm_price, bsm_vega, butterfly_check,
                                           calendar_check, check_svi_params, durrleman_g,
                                           fit_svi, heston_call, heston_smile, implied_vol,
                                           interpolate_maturity, quantlib_cross_check,
                                           svi_derivatives, svi_rho_profile,
                                           svi_total_variance, total_variance_from_normalised,
                                           vol_from_rounded_price, vollib_cross_check)

S, R, Q = 100.0, 0.03, 0.01
HESTON = dict(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)
STRIKES = np.array([70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130], dtype=float)
CASES = [(sig, mny, T) for sig in (0.10, 0.30, 0.80)
         for mny in (0.7, 1.0, 1.3) for T in (0.05, 1.0, 3.0)]


def _case(sig: float, mny: float, T: float):
    K = 100.0 * mny
    flag = "c" if mny >= 1.0 else "p"
    return bsm_price(S, K, T, R, Q, sig, flag), K, T, flag


# ------------------------------------------------------------------ the solver
def test_bsm_matches_the_independent_reference():
    assert bsm_price(S, 95.0, 1.0, R, Q, 0.2, "c") == pytest.approx(
        bs_call(S, 95.0, R, Q, 0.2, 1.0), abs=1e-12)
    assert bsm_price(S, 95.0, 1.0, R, Q, 0.2, "p") == pytest.approx(
        bs_put(S, 95.0, R, Q, 0.2, 1.0), abs=1e-12)
    assert bsm_price(S, 95.0, 0.0, R, Q, 0.2, "c") == 5.0


def test_round_trip_is_exact_across_the_whole_grid():
    worst = 0.0
    for sig, mny, T in CASES:
        price, K, Tc, flag = _case(sig, mny, T)
        worst = max(worst, abs(implied_vol(price, S, K, Tc, R, Q, flag) - sig))
    assert worst < 1e-12, worst


def test_a_relative_tolerance_is_what_makes_the_deep_wing_recoverable():
    # the SKILL.md's headline: this price is 7.9e-59, so ANY absolute 1e-12 stop is already met
    price = bsm_price(S, 70.0, 0.05, R, Q, 0.10, "p")
    assert price < 1e-50
    first_guess = min(max(math.sqrt(2.0 * math.pi / 0.05) * price / S, 1e-3), 5.0)
    assert first_guess == pytest.approx(1e-3)
    assert abs(bsm_price(S, 70.0, 0.05, R, Q, first_guess, "p") - price) < 1e-12
    assert implied_vol(price, S, 70.0, 0.05, R, Q, "p") == pytest.approx(0.10, abs=1e-12)


def test_the_solver_refuses_prices_no_volatility_can_produce():
    T = 1.0
    intrinsic = S - 100.0 * math.exp(-R * T)
    with pytest.raises(ValueError, match="discounted intrinsic"):
        implied_vol(intrinsic, S, 100.0, T, R, 0.0, "c")
    with pytest.raises(ValueError, match="discounted intrinsic"):
        implied_vol(intrinsic - 1e-9, S, 100.0, T, R, 0.0, "c")
    with pytest.raises(ValueError, match="maximum"):
        implied_vol(S + 1.0, S, 100.0, T, R, 0.0, "c")
    with pytest.raises(ValueError, match="T must be positive"):
        implied_vol(5.0, S, 100.0, 0.0, R, 0.0, "c")
    with pytest.raises(ValueError, match="flag"):
        implied_vol(5.0, S, 100.0, T, R, 0.0, "x")
    # just above intrinsic the vol collapses - a stale ITM quote is not slightly low
    assert implied_vol(intrinsic + 1e-6, S, 100.0, T, R, 0.0, "c") < 0.02


def test_a_penny_tick_destroys_the_short_dated_wing():
    resolution = {}
    for K in (100.0, 110.0, 115.0, 120.0):
        px = bsm_price(S, K, 0.05, R, Q, 0.20, "c")
        resolution[K] = 0.005 / bsm_vega(S, K, 0.05, R, Q, 0.20)
        got = vol_from_rounded_price(px, S, K, 0.05, R, Q, "c")
        if K <= 110.0:
            assert got is not None and abs(got[0] - 0.20) < 0.01
        else:
            assert got is None, f"K={K} should round to zero and have no implied vol"
    assert resolution[100.0] < 0.001                     # under a tenth of a vol point
    assert resolution[115.0] > 0.05                      # over five vol points
    assert resolution[120.0] > resolution[115.0] > resolution[110.0] > resolution[100.0]
    # a long-dated option at the same strike is fine: vega, not moneyness, is what matters
    px = bsm_price(S, 120.0, 1.0, R, Q, 0.20, "c")
    got = vol_from_rounded_price(px, S, 120.0, 1.0, R, Q, "c")
    assert got is not None and abs(got[0] - 0.20) < 5e-4


# ------------------------------------------------------------------ SVI
def test_svi_fits_a_heston_smile_to_well_under_a_tenth_of_a_vol_point():
    for T in (0.25, 1.0, 2.0):
        k, vols, w = heston_smile(S, T, R, Q, STRIKES, HESTON)
        fit = fit_svi(k, w)
        check_svi_params(*fit["params"])
        fitted_vol = np.sqrt(svi_total_variance(k, *fit["params"]) / T)
        rmse = float(np.sqrt(np.mean((fitted_vol - vols) ** 2)))
        assert rmse < 4e-4, (T, rmse)                    # < 0.04 vol points
        ok, gmin, _ = butterfly_check(fit["params"])
        assert ok and gmin > 0.0
    with pytest.raises(ValueError, match="at least five"):
        fit_svi([0.0, 0.1], [0.04, 0.05])


def test_the_svi_objective_is_flat_in_rho_so_rho_is_not_identified():
    k, vols, w = heston_smile(S, 1.0, R, Q, STRIKES, HESTON)
    fit = fit_svi(k, w)
    assert fit["params"][2] == pytest.approx(-0.999, abs=1e-6)   # ran to the bound
    prof = svi_rho_profile(k, w)
    rmses = [rm for _, rm in prof]
    assert max(rmses) / min(rmses) < 2.0                 # flat: no rho is meaningfully better
    for _, rm in prof:                                   # and every one of them is a good fit
        assert rm / (2.0 * float(np.mean(vols)) * 1.0) < 2e-4


def test_svi_derivatives_agree_with_finite_differences():
    p = (0.02, 0.10, -0.30, 0.05, 0.20)
    k = np.linspace(-0.5, 0.5, 21)
    w, w1, w2 = svi_derivatives(k, p)
    h = 1e-5
    fd1 = (svi_total_variance(k + h, *p) - svi_total_variance(k - h, *p)) / (2 * h)
    fd2 = (svi_total_variance(k + h, *p) - 2 * svi_total_variance(k, *p)
           + svi_total_variance(k - h, *p)) / h ** 2
    assert np.allclose(w, svi_total_variance(k, *p))
    assert np.allclose(w1, fd1, atol=1e-8)
    assert np.allclose(w2, fd2, atol=1e-4)
    assert SVI_PARAM_ORDER == ("a", "b", "rho", "m", "sigma")


def test_the_box_conditions_reject_what_they_cover_and_nothing_else():
    check_svi_params(0.02, 0.10, -0.30, 0.05, 0.20)      # a benign slice passes
    with pytest.raises(ValueError, match="non negative"):
        check_svi_params(0.02, -0.1, -0.3, 0.05, 0.20)
    with pytest.raises(ValueError, match="rho"):
        check_svi_params(0.02, 0.10, 1.0, 0.05, 0.20)
    with pytest.raises(ValueError, match="sigma"):
        check_svi_params(0.02, 0.10, -0.3, 0.05, -0.20)
    with pytest.raises(ValueError, match="minimum variance"):
        check_svi_params(-1.0, 0.10, -0.3, 0.05, 0.20)
    with pytest.raises(ValueError, match=r"b \(1"):
        check_svi_params(0.02, 3.5, -0.3, 0.05, 0.20)


# ------------------------------------------------------------------ static arbitrage
def test_the_vogt_slice_passes_every_box_condition_and_fails_the_butterfly_test():
    check_svi_params(*VOGT)                              # raises nothing - that is the point
    ok, gmin, kmin = butterfly_check(VOGT, np.linspace(-1.5, 1.5, 30001))
    assert not ok
    assert gmin == pytest.approx(-0.0329, abs=5e-4)
    assert kmin == pytest.approx(0.879, abs=1e-3)


def test_durrleman_g_agrees_with_breeden_litzenberger_point_for_point():
    k = np.linspace(-1.2, 1.6, 2801)
    dens = breeden_litzenberger_density(k, VOGT)
    g = durrleman_g(k, VOGT)
    assert np.array_equal(dens < 0.0, g < 0.0), "the two disagree about where the density is negative"
    assert int((dens < 0.0).sum()) == 614
    # and on a benign slice both are positive everywhere
    benign = (0.02, 0.10, -0.30, 0.05, 0.20)
    assert (durrleman_g(k, benign) > 0).all()
    assert (breeden_litzenberger_density(np.linspace(-0.8, 0.8, 401), benign) > 0).all()


def test_calendar_check_catches_a_slice_that_shrinks_in_total_variance():
    k = np.linspace(-0.4, 0.3, 141)
    rising = [(0.5, (0.010, 0.10, -0.3, 0.0, 0.20)), (1.0, (0.030, 0.10, -0.3, 0.0, 0.20))]
    ok, worst, _, pair = calendar_check(k, rising)
    assert ok and worst > 0.0 and pair == (0.5, 1.0)
    falling = [(0.5, (0.030, 0.10, -0.3, 0.0, 0.20)), (1.0, (0.010, 0.10, -0.3, 0.0, 0.20))]
    ok, worst, _, _ = calendar_check(k, falling)
    assert not ok and worst < 0.0
    with pytest.raises(ValueError, match="distinct"):
        calendar_check(k, [(1.0, rising[0][1]), (1.0, rising[1][1])])


def test_a_fitted_heston_surface_passes_both_checks():
    fits = {}
    for T in (0.25, 0.5, 1.0, 2.0):
        k, _, w = heston_smile(S, T, R, Q, STRIKES, HESTON)
        fits[T] = fit_svi(k, w)["params"]
        assert butterfly_check(fits[T])[0]
    ok, worst, _, _ = calendar_check(np.linspace(-0.4, 0.3, 141), list(fits.items()))
    assert ok and worst > 0.0


# ------------------------------------------------------------------ interpolation
def test_black_normalised_round_trips_through_its_inverse():
    for k in (-0.5, 0.0, 0.4):
        for w in (0.01, 0.09, 1.0):
            p = black_normalised(k, w)
            assert total_variance_from_normalised(p, k) == pytest.approx(w, rel=1e-8)
    assert black_normalised(-0.5, 0.0) == pytest.approx(1.0 - math.exp(-0.5))
    assert black_normalised(0.5, 0.0) == 0.0


def test_linear_in_vol_manufactures_calendar_arbitrage_that_total_variance_cannot():
    event = dict(v0=0.36, kappa=8.0, theta=0.04, sigma=0.5, rho=-0.5)
    Ta, Tb = 1.0 / 52.0, 0.5
    grid = np.linspace(Ta, Tb, 60)
    truth = np.array([implied_vol(heston_call(S, S, T, R, Q, **event), S, S, T, R, Q, "c")
                      for T in grid])
    w_truth = truth * truth * grid
    assert np.min(np.diff(w_truth)) > 0.0                # the TRUE surface is arbitrage-free
    lam = (grid - Ta) / (Tb - Ta)
    w_vol = (truth[0] + lam * (truth[-1] - truth[0])) ** 2 * grid
    w_tv = w_truth[0] + lam * (w_truth[-1] - w_truth[0])
    assert np.min(np.diff(w_vol)) < 0.0, "vol interpolation should break monotonicity here"
    assert np.min(np.diff(w_tv)) > 0.0, "total variance is a convex combination: monotone"
    # and the violation is worth money: the far option is cheaper than the near one
    i = int(np.argmax(w_vol))
    assert grid[i] < Tb
    assert black_normalised(0.0, w_vol[i]) > black_normalised(0.0, w_truth[-1])


def test_interpolate_maturity_is_a_convex_combination_and_validates_its_inputs():
    k = np.array([-0.2, 0.0, 0.2])
    w1, w2 = np.array([0.02, 0.018, 0.021]), np.array([0.05, 0.045, 0.052])
    mid = interpolate_maturity(0.75, 0.5, w1, 1.0, w2, k, "total_variance")
    assert np.allclose(mid, 0.5 * (w1 + w2))
    for method in INTERP_METHODS:
        out = interpolate_maturity(0.75, 0.5, w1, 1.0, w2, k, method)
        assert np.all(out > w1) and np.all(out < w2)
    with pytest.raises(ValueError, match="method"):
        interpolate_maturity(0.75, 0.5, w1, 1.0, w2, k, "cubic")
    with pytest.raises(ValueError, match="strictly between"):
        interpolate_maturity(1.5, 0.5, w1, 1.0, w2, k)


# ------------------------------------------------------------------ optional libraries
@pytest.mark.skipif(has_module("vollib"), reason="vollib is installed")
def test_vollib_cross_check_is_none_without_vollib():
    assert vollib_cross_check([]) is None


@requires("vollib")
def test_vollib_agrees_to_machine_precision_and_refuses_the_same_prices():
    cases = []
    for sig, mny, T in CASES:
        price, K, Tc, flag = _case(sig, mny, T)
        cases.append((price, S, K, Tc, R, Q, flag))
    out = vollib_cross_check(cases)
    assert out is not None
    assert out["max_abs_diff"] < 1e-12
    for label in ("below intrinsic", "at intrinsic"):
        mine, theirs = out["boundary"][label]
        assert mine == "raises ValueError"
        assert "BelowIntrinsic" in theirs
    for label in ("intrinsic + 1e-6", "intrinsic + 1e-3"):
        mine, theirs = out["boundary"][label]
        assert mine == theirs                            # same six decimals


@pytest.mark.skipif(has_module("QuantLib"), reason="QuantLib is installed")
def test_quantlib_cross_check_is_none_without_quantlib():
    assert quantlib_cross_check(S, R, Q, (1.0,), STRIKES, HESTON, {}) is None


@requires("QuantLib")
def test_quantlib_confirms_the_surface_and_accepts_an_arbitrageable_slice():
    fits = {}
    for T in (0.25, 0.5, 1.0, 2.0):
        k, _, w = heston_smile(S, T, R, Q, STRIKES, HESTON)
        fits[T] = fit_svi(k, w)["params"]
    live = quantlib_cross_check(S, R, Q, (0.25, 0.5, 1.0, 2.0), STRIKES, HESTON, fits)
    assert live is not None
    assert live["heston_max_abs_diff"] < 1e-10
    assert live["svi_max_abs_diff"] < 1e-12              # the QuantLib parameter order is right
    # the box conditions do not test for butterfly arbitrage
    assert live["vogt_accepted_by_quantlib"] is True and live["vogt_error"] == ""
    # the swapped parameter order is caught only when rho happens to be negative
    right_neg, wrong_neg, err_neg = live["svi_order_swap"]["rho<0"]
    assert "must be positive" in err_neg and math.isnan(wrong_neg)
    right_pos, wrong_pos, err_pos = live["svi_order_swap"]["rho>0"]
    assert err_pos == "" and abs(wrong_pos - right_pos) > 0.03      # over 3 vol points, silently


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.models.vol_surface")
    assert "interpolate" in out and "in total variance at fixed log-moneyness" in out
    assert "never in vol" in out
    assert out.isascii()
