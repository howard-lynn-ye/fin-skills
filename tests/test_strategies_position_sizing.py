"""fin_skills.strategies.position_sizing - Kelly, its drawdown, and the estimated mean."""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.strategies.position_sizing import (double_before_halve,
                                                   expected_growth_estimated, growth_binary,
                                                   growth_rate, growth_ratio, kelly_binary,
                                                   kelly_discrete, kelly_fraction,
                                                   kelly_max_growth_binary, max_drawdown_paths,
                                                   optimal_c_estimated, ruin_prob, sharpe_ratio,
                                                   simulate_estimated_kelly, simulate_ruin,
                                                   vol_target_fraction)

MU, SIG = 0.08, 0.20


# --------------------------------------------------------------- discrete Kelly, both ways
@pytest.mark.parametrize("p,b", [(0.60, 1.0), (0.55, 1.0), (0.52, 1.0), (0.40, 2.0),
                                 (0.25, 5.0), (0.70, 0.5)])
def test_closed_form_matches_the_numeric_argmax(p, b):
    """f* = (bp - q)/b against a numerical maximiser of E[ln(1 + f X)]."""
    numeric = kelly_discrete(np.array([b, -1.0]), np.array([p, 1.0 - p]))
    assert kelly_binary(p, b) == pytest.approx(numeric, abs=1e-5)


def test_even_money_kelly_is_the_edge_and_a_losing_bet_is_zero():
    assert kelly_binary(0.6) == pytest.approx(0.2)          # f* = 2p - 1
    assert kelly_binary(0.5) == 0.0
    assert kelly_binary(0.4) == 0.0                          # floored, not negative
    assert kelly_discrete(np.array([1.0, -1.0]), np.array([0.5, 0.5])) == 0.0
    assert kelly_discrete(np.array([1.0, -1.0]), np.array([0.4, 0.6])) == 0.0


def test_the_kelly_fraction_maximises_the_growth_it_claims_to():
    """The defining property: g(f*) >= g(f) for every other f."""
    f = kelly_binary(0.6, 1.0)
    grid = np.linspace(0.001, 0.9, 400)
    assert growth_binary(f, 0.6) >= growth_binary(grid, 0.6).max() - 1e-9
    assert growth_binary(1.0, 0.6) == -np.inf                # betting everything on a loss
    assert growth_binary(0.0, 0.6) == pytest.approx(0.0)


def test_kellys_own_max_growth_is_in_bits_and_matches_the_nats_version():
    """Kelly (1956) p. 920: G_max = 1 + p log2 p + q log2 q, base 2 because it is a bit rate."""
    for p in (0.55, 0.6, 0.75):
        bits = kelly_max_growth_binary(p)
        nats = growth_binary(kelly_binary(p), p)
        assert bits * np.log(2.0) == pytest.approx(nats)
    assert kelly_max_growth_binary(0.5) == pytest.approx(0.0)
    assert kelly_max_growth_binary(0.99) < 1.0
    with pytest.raises(ValueError):
        kelly_max_growth_binary(0.0)


def test_a_three_outcome_bet_has_no_closed_form_but_a_numeric_optimum():
    pay, pr = np.array([2.0, 0.0, -1.0]), np.array([0.35, 0.30, 0.35])
    f = kelly_discrete(pay, pr)
    assert 0.0 < f < 1.0
    grid = np.linspace(0.001, 0.99, 500)
    best = max(float(pr @ np.log1p(g * pay)) for g in grid)
    assert float(pr @ np.log1p(f * pay)) >= best - 1e-8
    with pytest.raises(ValueError, match="sum to 1"):
        kelly_discrete(pay, np.array([0.5, 0.5, 0.5]))
    with pytest.raises(ValueError, match="1-D arrays"):
        kelly_discrete(np.array([1.0]), np.array([1.0]))


# ------------------------------------------------------------------------ continuous Kelly
def test_continuous_kelly_and_its_growth_rate():
    f = kelly_fraction(MU, SIG)
    assert f == pytest.approx(MU / SIG ** 2) == pytest.approx(2.0)
    assert growth_rate(f, MU, SIG) == pytest.approx(MU ** 2 / (2 * SIG ** 2))
    assert growth_rate(f, MU, SIG) == pytest.approx(sharpe_ratio(MU, SIG) ** 2 / 2)
    grid = np.linspace(0.0, 6.0, 2001)
    assert growth_rate(f, MU, SIG) >= growth_rate(grid, MU, SIG).max() - 1e-12
    with pytest.raises(ValueError, match="sigma"):
        kelly_fraction(MU, 0.0)


def test_double_kelly_gives_exactly_zero_growth_for_every_mu_and_sigma():
    """Thorp p. 409: g(c f*)/g(f*) = c(2 - c), so c = 2 is zero."""
    for mu, sig in ((0.08, 0.20), (0.50, 2.00), (0.01, 0.05)):
        assert growth_rate(2 * kelly_fraction(mu, sig), mu, sig) == pytest.approx(0.0, abs=1e-14)
    assert growth_ratio(2.0) == pytest.approx(0.0)


def test_the_growth_ratio_is_c_times_two_minus_c_and_half_kelly_keeps_three_quarters():
    fstar, gstar = kelly_fraction(MU, SIG), growth_rate(kelly_fraction(MU, SIG), MU, SIG)
    for c in (0.25, 0.5, 0.75, 1.0, 1.5):
        assert growth_rate(c * fstar, MU, SIG) / gstar == pytest.approx(float(growth_ratio(c)))
    assert float(growth_ratio(0.5)) == pytest.approx(0.75)
    # growth is concave in c, dispersion is linear - the half-Kelly argument
    sdev = [SIG * c * fstar for c in (0.25, 0.5, 1.0)]
    assert sdev[1] / sdev[2] == pytest.approx(0.5)


def test_risk_free_rate_shifts_the_fraction_and_the_growth():
    assert kelly_fraction(0.10, 0.20, r=0.02) == pytest.approx(0.08 / 0.04)
    assert growth_rate(2.0, 0.10, 0.20, r=0.02) == pytest.approx(
        0.02 + 2.0 * 0.08 - 0.04 * 4.0 / 2)


# ------------------------------------------------------------------- the drawdown formula
def test_ruin_prob_reproduces_thorps_stated_numbers():
    """Thorp p. 415: 'the chance of ever losing half the starting capital is 1/2 for f = f*
    but only 1/8 for f = f*/2'."""
    assert ruin_prob(0.5, 1.0) == pytest.approx(0.5)
    assert ruin_prob(0.5, 0.5) == pytest.approx(0.125)
    assert ruin_prob(0.25, 1.0) == pytest.approx(0.25)
    assert ruin_prob(0.5, 0.25) == pytest.approx(0.5 ** 7)
    with pytest.raises(ValueError, match="x must be"):
        ruin_prob(1.0, 1.0)
    with pytest.raises(ValueError, match="c must be"):
        ruin_prob(0.5, 2.0)


def test_double_before_halve_reproduces_two_thirds_and_eight_ninths():
    """Thorp eq. (7.11) at x = 1/2, y = 2, quoted in the paper as 2/3 and 8/9."""
    assert double_before_halve(1.0) == pytest.approx(2.0 / 3.0)
    assert double_before_halve(0.5) == pytest.approx(8.0 / 9.0)
    assert double_before_halve(1.9) < double_before_halve(1.0)


@pytest.mark.slow
@pytest.mark.parametrize("c,x", [(1.0, 0.5), (0.5, 0.5)])
def test_eq_7_13_holds_under_simulation(c, x):
    """The formula, checked against paths with a Brownian-bridge crossing correction."""
    r = simulate_ruin(c, MU, SIG, x, years=200.0, n_paths=8000, seed=2)
    assert r["p_bridge"] == pytest.approx(r["p_formula"], abs=0.02)
    assert r["p_naive"] < r["p_bridge"]          # discrete sampling always understates
    assert r["f"] == pytest.approx(c * kelly_fraction(MU, SIG))


@pytest.mark.slow
def test_the_peak_relative_drawdown_is_a_different_and_much_larger_quantity():
    """Section 3's headline misquote: x^(2/c-1) is about the INITIAL stake."""
    full = simulate_ruin(1.0, MU, SIG, 0.5, years=200.0, n_paths=4000, seed=3)
    half = simulate_ruin(0.5, MU, SIG, 0.5, years=200.0, n_paths=4000, seed=3)
    assert full["p_dd_from_peak"] > 0.95 and half["p_dd_from_peak"] > 0.9
    assert full["p_dd_from_peak"] > 1.5 * full["p_formula"]
    assert half["p_dd_from_peak"] > 5.0 * half["p_formula"]
    assert full["median_max_dd"] > half["median_max_dd"]     # but full Kelly is still worse


def test_simulate_ruin_validates_its_inputs():
    with pytest.raises(ValueError):
        simulate_ruin(2.5, MU, SIG, 0.5, n_paths=10)
    with pytest.raises(ValueError):
        simulate_ruin(1.0, MU, SIG, 1.5, n_paths=10)


def test_max_drawdown_paths_is_from_the_running_peak():
    logw = np.array([[0.0], [np.log(2.0)], [np.log(1.0)], [np.log(1.5)]])
    assert max_drawdown_paths(logw)[0] == pytest.approx(0.5)      # 2 -> 1 is a 50% drawdown
    assert max_drawdown_paths(np.array([[0.0], [1.0], [2.0]]))[0] == pytest.approx(0.0)


# ------------------------------------------------- TRAP 1: Kelly on an estimated mean
def test_the_estimation_penalty_is_exactly_c_squared_over_2N():
    """E[g] = g(f*)*c(2-c) - c^2/(2N), and the penalty depends on nothing but N."""
    gstar = growth_rate(kelly_fraction(MU, SIG), MU, SIG)
    for c in (0.25, 0.5, 1.0):
        for n in (5.0, 10.0, 40.0):
            assert expected_growth_estimated(c, MU, SIG, n) == pytest.approx(
                gstar * float(growth_ratio(c)) - c ** 2 / (2 * n))
    # the penalty term itself is invariant to mu and sigma
    pen = (expected_growth_estimated(1.0, MU, SIG, 10.0)
           - growth_rate(kelly_fraction(MU, SIG), MU, SIG))
    pen2 = (expected_growth_estimated(1.0, 0.30, 0.60, 10.0)
            - growth_rate(kelly_fraction(0.30, 0.60), 0.30, 0.60))
    assert pen == pytest.approx(pen2) == pytest.approx(-1.0 / 20.0)
    with pytest.raises(ValueError, match="n_years"):
        expected_growth_estimated(1.0, MU, SIG, 0.0)


def test_optimal_c_is_t_squared_over_one_plus_t_squared():
    """A t-stat of 1 on the mean estimate gives EXACTLY half Kelly."""
    # choose N so that t = SR*sqrt(N) is exactly 1, 2, 3
    sr = sharpe_ratio(MU, SIG)
    for t in (1.0, 2.0, 3.0):
        n = (t / sr) ** 2
        assert optimal_c_estimated(MU, SIG, n) == pytest.approx(t ** 2 / (1 + t ** 2))
    assert optimal_c_estimated(MU, SIG, (1.0 / sr) ** 2) == pytest.approx(0.5)
    assert optimal_c_estimated(MU, SIG, 1e9) == pytest.approx(1.0, abs=1e-6)


def test_the_optimal_c_really_is_the_argmax_of_the_expected_growth():
    for n in (5.0, 10.0, 30.0):
        c_opt = optimal_c_estimated(MU, SIG, n)
        grid = np.linspace(0.01, 1.99, 3000)
        best = max(expected_growth_estimated(float(c), MU, SIG, n) for c in grid)
        assert expected_growth_estimated(c_opt, MU, SIG, n) >= best - 1e-9
        assert expected_growth_estimated(c_opt, MU, SIG, n) > expected_growth_estimated(
            1.0, MU, SIG, n)


@pytest.mark.slow
def test_half_kelly_beats_full_kelly_on_a_ten_year_estimate():
    """The trap, measured: full Kelly on an estimated mean is an over-bet on every axis."""
    full = simulate_estimated_kelly(1.0, MU, SIG, est_years=10.0, n_paths=8000, seed=4)
    half = simulate_estimated_kelly(0.5, MU, SIG, est_years=10.0, n_paths=8000, seed=4)
    assert half["growth_mean"] > full["growth_mean"]           # more growth
    assert half["p_growth_negative"] < 0.5 * full["p_growth_negative"]
    assert half["median_max_dd"] < full["median_max_dd"]
    # and a long enough sample restores full Kelly's advantage
    long_full = simulate_estimated_kelly(1.0, MU, SIG, est_years=200.0, n_paths=8000, seed=4)
    long_half = simulate_estimated_kelly(0.5, MU, SIG, est_years=200.0, n_paths=8000, seed=4)
    assert long_full["growth_mean"] > long_half["growth_mean"]


def test_simulate_estimated_kelly_matches_its_own_closed_form_direction():
    r = simulate_estimated_kelly(0.5, MU, SIG, est_years=30.0, n_paths=8000, seed=5)
    # the simulation floors f at 0, so it sits at or above the unconstrained theory
    assert r["growth_mean"] >= r["theory_growth"] - 0.01
    assert r["f_mean"] == pytest.approx(0.5 * kelly_fraction(MU, SIG), rel=0.1)
    with pytest.raises(ValueError):
        simulate_estimated_kelly(1.0, MU, SIG, est_years=0.0)


# ---------------------------------------------- TRAP 2: a Sharpe ratio is not a fraction
def test_using_the_sharpe_ratio_as_the_fraction_is_wrong_by_one_over_sigma():
    sr = 0.5
    for sig in (0.10, 0.20, 0.50, 1.00, 2.00):
        mu = sr * sig
        assert kelly_fraction(mu, sig) == pytest.approx(sr / sig)
    # right only at sigma = 1, and exactly double Kelly (zero growth) at sigma = 2
    assert kelly_fraction(sr * 1.0, 1.0) == pytest.approx(sr)
    assert growth_rate(sr, sr * 2.0, 2.0) == pytest.approx(0.0, abs=1e-15)
    # a large under-bet at equity volatility
    kept = growth_rate(sr, sr * 0.10, 0.10) / growth_rate(kelly_fraction(sr * 0.10, 0.10),
                                                          sr * 0.10, 0.10)
    assert 0.15 < kept < 0.25


def test_vol_targeting_is_kelly_exactly_when_the_target_equals_the_sharpe_ratio():
    sig, sr = 0.20, 0.5
    assert vol_target_fraction(sr, sig) == pytest.approx(kelly_fraction(sr * sig, sig))
    assert vol_target_fraction(0.10, sig) != pytest.approx(kelly_fraction(sr * sig, sig))
    assert vol_target_fraction(0.10, 0.40) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        vol_target_fraction(0.1, 0.0)
    with pytest.raises(ValueError):
        vol_target_fraction(-0.1, 0.2)


@pytest.mark.slow
def test_demo_reproduces_thorps_numbers_and_prints_the_rule(run_main):
    out = run_main("fin_skills.strategies.position_sizing")
    assert "TRAP 1" in out and "TRAP 2" in out
    assert "0.6667" in out and "0.8889" in out         # Thorp's 2/3 and 8/9
    assert "t^2/(1 + t^2)" in out
    assert "Rule: f* = SR/sigma, not SR" in out
    assert out.isascii()
