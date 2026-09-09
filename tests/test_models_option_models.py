"""fin_skills.models.option_models - the four ways a textbook option model returns a wrong number.

Each test asserts the PROPERTY the SKILL.md documents, not the printed digits: the closed form
matches an independent reference, the tree oscillates with the parity of the step count, the
1993 Heston formulation breaks past a maturity the safe one handles, and a Monte Carlo standard
error is blind to discretisation bias.
"""
from __future__ import annotations

import math

import pytest

from _helpers import bs_call, bs_put
from conftest import has_module, requires
from fin_skills.models.option_models import (FORMULATIONS, bsm_price, crr_convergence,
                                             crr_price, crr_smoothed,
                                             heston_cf_discontinuities,
                                             heston_first_nonfinite_phi, heston_kappa_scan,
                                             heston_price, heston_trap_scan, mc_european,
                                             payoff_std, quantlib_cross_checks,
                                             sabr_atm_vol, sabr_implied_vol)

S = K = 100.0
T, R, Q, SIGMA = 1.0, 0.05, 0.02, 0.20
HESTON = dict(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)
R_H, Q_H = 0.03, 0.0
SABR = dict(F=100.0, T=1.0, alpha=0.25, beta=0.6, nu=0.4, rho=-0.25)
STRIKES = (70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0)


# ------------------------------------------------------------------ Black-Scholes-Merton
def test_bsm_matches_an_independent_closed_form_and_satisfies_parity():
    call = bsm_price(S, K, T, R, Q, SIGMA, "c")
    put = bsm_price(S, K, T, R, Q, SIGMA, "p")
    assert call == pytest.approx(bs_call(S, K, R, Q, SIGMA, T), abs=1e-10)
    assert put == pytest.approx(bs_put(S, K, R, Q, SIGMA, T), abs=1e-10)
    forward = S * math.exp(-Q * T) - K * math.exp(-R * T)
    assert call - put == pytest.approx(forward, abs=1e-12)


def test_bsm_edge_cases_and_flag_validation():
    assert bsm_price(120.0, K, 0.0, R, Q, SIGMA, "c") == 20.0        # expired -> intrinsic
    assert bsm_price(80.0, K, -1.0, R, Q, SIGMA, "c") == 0.0
    assert bsm_price(80.0, K, 0.0, R, Q, SIGMA, "p") == 20.0
    with pytest.raises(ValueError, match="flag"):
        bsm_price(S, K, T, R, Q, SIGMA, "x")
    with pytest.raises(ValueError, match="sigma"):
        bsm_price(S, K, T, R, Q, 0.0, "c")


# ------------------------------------------------------------------ CRR tree
def test_crr_converges_to_black_scholes_but_oscillates_with_the_step_parity():
    ref = bsm_price(S, K, T, R, Q, SIGMA, "p")
    rows = {n: (p, e) for n, p, e in crr_convergence(S, K, T, R, Q, SIGMA, "p")}
    for n in (50, 100, 200, 400, 800):
        assert rows[n][1] < 0.0, f"even n={n} should sit below the limit"
        assert rows[n + 1][1] > 0.0, f"odd n={n + 1} should sit above the limit"
    # the documented headline: 64x the work buys well under 64x the accuracy
    ratio = abs(rows[100][1] / rows[800][1])
    assert 5.0 < ratio < 12.0, ratio
    assert rows[800][0] == pytest.approx(ref, abs=5e-3)


def test_richardson_needs_both_legs_on_the_same_parity():
    ref = bsm_price(S, K, T, R, Q, SIGMA, "p")
    raw_even = abs(crr_price(S, K, T, R, Q, SIGMA, 100, "p") - ref)
    raw_odd = abs(crr_price(S, K, T, R, Q, SIGMA, 101, "p") - ref)
    avg = abs(crr_smoothed(S, K, T, R, Q, SIGMA, 100, "p", method="average") - ref)
    rich = abs(crr_smoothed(S, K, T, R, Q, SIGMA, 100, "p", method="richardson") - ref)
    rich_odd = abs(crr_smoothed(S, K, T, R, Q, SIGMA, 101, "p", method="richardson") - ref)
    assert avg < raw_even / 10.0                      # averaging the parities: >=10x better
    assert rich < avg                                 # Richardson from an even n: better again
    assert rich_odd > raw_odd                         # the trap: from an odd n it is WORSE
    with pytest.raises(ValueError, match="method"):
        crr_smoothed(S, K, T, R, Q, SIGMA, 100, "p", method="romberg")


def test_american_put_is_worth_more_than_european_and_a_call_without_dividends_is_not():
    eu = crr_price(S, K, T, R, Q, SIGMA, 800, "p", american=False)
    am = crr_price(S, K, T, R, Q, SIGMA, 800, "p", american=True)
    assert am > eu and 0.1 < am - eu < 1.0
    # q = 0: an American call on a non-dividend payer is never exercised early (Merton 1973)
    eu_c = crr_price(S, K, T, R, 0.0, SIGMA, 400, "c", american=False)
    am_c = crr_price(S, K, T, R, 0.0, SIGMA, 400, "c", american=True)
    assert am_c == pytest.approx(eu_c, abs=1e-10)


def test_the_two_crr_probabilities_differ_at_order_dt_and_both_converge():
    ref = bsm_price(S, K, T, R, Q, SIGMA, "p")
    coarse = abs(crr_price(S, K, T, R, Q, SIGMA, 100, "p")
                 - crr_price(S, K, T, R, Q, SIGMA, 100, "p", prob="quantlib"))
    fine = abs(crr_price(S, K, T, R, Q, SIGMA, 800, "p")
               - crr_price(S, K, T, R, Q, SIGMA, 800, "p", prob="quantlib"))
    assert coarse > 5e-5 and fine < coarse / 4.0      # an O(dt) gap, not a bug
    assert crr_price(S, K, T, R, Q, SIGMA, 2000, "p", prob="quantlib") == pytest.approx(ref, abs=2e-3)
    with pytest.raises(ValueError, match="prob"):
        crr_price(S, K, T, R, Q, SIGMA, 100, "p", prob="jarrow-rudd")
    with pytest.raises(ValueError, match="steps"):
        crr_price(S, K, T, R, Q, SIGMA, 0, "p")


def test_a_tree_too_coarse_for_the_drift_refuses_rather_than_returning_a_price():
    with pytest.raises(ValueError, match="outside"):
        crr_price(S, K, 10.0, 0.5, 0.0, 0.05, 2, "c")


# ------------------------------------------------------------------ Heston
def test_heston_collapses_to_black_scholes_as_the_vol_of_vol_vanishes():
    v0 = 0.04
    ref = bsm_price(S, K, T, R_H, Q_H, math.sqrt(v0), "c")
    prev = None
    for vov in (1e-2, 1e-3, 1e-4):
        err = abs(heston_price(S, K, T, R_H, Q_H, v0, 1.5, v0, vov, -0.7, "c") - ref)
        if prev is not None:
            assert err < prev
        prev = err
    assert prev < 1e-4


def test_heston_put_call_parity_and_edge_cases():
    kw = dict(S=100.0, K=90.0, T=2.0, r=0.03, q=0.01, flag="c", **HESTON)
    call = heston_price(**kw)
    put = heston_price(**{**kw, "flag": "p"})
    forward = 100.0 * math.exp(-0.01 * 2.0) - 90.0 * math.exp(-0.03 * 2.0)
    assert call - put == pytest.approx(forward, abs=1e-10)
    assert heston_price(100.0, 90.0, 0.0, 0.03, 0.0, flag="c", **HESTON) == 10.0
    assert heston_price(100.0, 90.0, 0.0, 0.03, 0.0, flag="p", **HESTON) == 0.0
    with pytest.raises(ValueError, match="formulation"):
        heston_price(S, K, T, R_H, Q_H, flag="c", formulation="carr-madan", **HESTON)
    with pytest.raises(ValueError, match="j must be"):
        from fin_skills.models.option_models import heston_log_cf_terms
        heston_log_cf_terms(None, 1.0, 0.0, 0.0, 1.5, 0.04, 0.3, -0.7, 3)
    assert FORMULATIONS == ("albrecher", "heston1993")


def test_the_1993_formulation_breaks_and_the_albrecher_form_does_not():
    scan = {row["T"]: row for row in heston_trap_scan(S, K, R_H, Q_H, **HESTON)}
    for short in (0.5, 1.0):
        assert abs(scan[short]["error"]) < 1e-6, "both forms agree at short maturity"
    for long in (2.0, 3.0, 5.0, 10.0, 15.0):
        assert abs(scan[long]["error"]) > 1e-6, f"the 1993 form should be wrong at T={long}"
    assert abs(scan[15.0]["error"]) / scan[15.0]["albrecher"] > 0.30      # 39% of the price
    for blown in (20.0, 30.0):
        assert math.isnan(scan[blown]["heston1993"])
        assert math.isfinite(scan[blown]["albrecher"]) and scan[blown]["albrecher"] > 0.0


def test_the_branch_jumps_are_the_fingerprint_and_the_nan_is_an_overflow():
    for Tm in (1.0, 10.0, 30.0):
        assert heston_cf_discontinuities(Tm, R_H, Q_H, HESTON["kappa"], HESTON["theta"],
                                         HESTON["sigma"], HESTON["rho"], "albrecher") == 0
    jumps = [heston_cf_discontinuities(Tm, R_H, Q_H, HESTON["kappa"], HESTON["theta"],
                                       HESTON["sigma"], HESTON["rho"], "heston1993")
             for Tm in (1.0, 10.0, 30.0)]
    assert jumps[0] == 0 and jumps[1] > 0 and jumps[2] > jumps[1]
    kw = (R_H, Q_H, HESTON["kappa"], HESTON["theta"], HESTON["sigma"], HESTON["rho"])
    phi = heston_first_nonfinite_phi(20.0, *kw, "heston1993")
    assert phi is not None and 100.0 < phi < 200.0
    assert heston_first_nonfinite_phi(20.0, *kw, "albrecher") is None


def test_raising_kappa_theta_breaks_the_1993_form_at_a_fixed_two_year_maturity():
    rows = {row["kappa"]: row for row in heston_kappa_scan(S, K, 2.0, R_H, Q_H, HESTON["v0"],
                                                           HESTON["theta"], HESTON["sigma"],
                                                           HESTON["rho"])}
    for slow in (0.25, 0.5, 1.0):
        assert abs(rows[slow]["error"]) < 1e-6 and rows[slow]["jumps"] == 0
    for fast in (1.5, 3.0, 6.0, 12.0):
        assert abs(rows[fast]["error"]) > 1e-6 and rows[fast]["jumps"] > 0
    assert rows[12.0]["jumps"] > rows[1.5]["jumps"]              # more crossings, not fewer
    assert rows[3.0]["a_over_sigma2"] == pytest.approx(3.0 * HESTON["theta"] / HESTON["sigma"] ** 2)


# ------------------------------------------------------------------ SABR
def test_sabr_atm_branch_matches_the_closed_form_special_case():
    general = sabr_implied_vol(SABR["F"], SABR["F"], SABR["T"], SABR["alpha"], SABR["beta"],
                               SABR["nu"], SABR["rho"])
    special = sabr_atm_vol(SABR["F"], SABR["T"], SABR["alpha"], SABR["beta"], SABR["nu"],
                           SABR["rho"])
    assert general == pytest.approx(special, abs=1e-12)
    # alpha is NOT the ATM vol unless beta == 1: it scales as alpha / F^(1-beta)
    assert general == pytest.approx(0.0401, abs=5e-4)
    assert abs(general - SABR["alpha"]) > 0.2
    unit_beta = sabr_implied_vol(100.0, 100.0, 1e-9, 0.25, 1.0, 1e-9, 0.0)
    assert unit_beta == pytest.approx(0.25, abs=1e-6)            # beta=1 -> alpha IS the ATM vol


def test_sabr_smile_is_a_smile_and_the_parameters_are_validated():
    vols = [sabr_implied_vol(Ks, SABR["F"], SABR["T"], SABR["alpha"], SABR["beta"],
                             SABR["nu"], SABR["rho"]) for Ks in STRIKES]
    assert min(vols) == vols[3]                                  # minimum near the money
    assert vols[0] > vols[3] and vols[-1] > vols[3]              # both wings above it
    for bad in (dict(alpha=-0.1), dict(beta=1.5), dict(rho=1.0), dict(nu=-0.1)):
        with pytest.raises(ValueError, match="out of range"):
            sabr_implied_vol(100.0, 100.0, 1.0, **{**{k: SABR[k] for k in
                                                     ("alpha", "beta", "nu", "rho")}, **bad})
    with pytest.raises(ValueError, match="positive"):
        sabr_implied_vol(0.0, 100.0, 1.0, 0.25, 0.6, 0.4, -0.25)


# ------------------------------------------------------------------ Monte Carlo
def test_a_seeded_run_is_deterministic_and_the_exact_scheme_is_unbiased():
    a = mc_european(S, K, T, R, Q, SIGMA, "c", 20_000, 1, True, "log", seed=3)
    b = mc_european(S, K, T, R, Q, SIGMA, "c", 20_000, 1, True, "log", seed=3)
    assert a == b
    assert mc_european(S, K, T, R, Q, SIGMA, "c", 20_000, 1, True, "log", seed=4) != a
    exact = bsm_price(S, K, T, R, Q, SIGMA, "c")
    price, se = mc_european(S, K, T, R, Q, SIGMA, "c", 200_000, 1, True, "log", seed=5)
    assert abs(price - exact) < 3.0 * se                         # within three standard errors
    for steps in (1, 12):                                        # the log scheme is exact at any dt
        p, _ = mc_european(S, K, T, R, Q, SIGMA, "c", 200_000, steps, True, "log", seed=5)
        assert abs(p - exact) < 3.0 * se


def test_paths_shrink_the_error_bar_and_steps_do_not():
    ses = [mc_european(S, K, T, R, Q, SIGMA, "c", n, 1, False, "log", seed=1)[1]
           for n in (10_000, 40_000, 160_000)]
    assert ses[0] / ses[1] == pytest.approx(2.0, abs=0.05)
    assert ses[1] / ses[2] == pytest.approx(2.0, abs=0.05)
    flat = [mc_european(S, K, T, R, Q, SIGMA, "c", 40_000, k, False, "log", seed=1)[1]
            for k in (1, 12, 52)]
    assert max(flat) / min(flat) < 1.01                          # steps move it by under 1%
    anti = mc_european(S, K, T, R, Q, SIGMA, "c", 40_000, 1, True, "log", seed=1)[1]
    assert anti < flat[0] and 1.2 < flat[0] / anti < 1.6


def test_the_euler_bias_is_many_standard_errors_wide_and_steps_are_what_close_it():
    exact = bsm_price(S, K, T, R, Q, SIGMA, "c")
    one = mc_european(S, K, T, R, Q, SIGMA, "c", 400_000, 1, True, "euler", seed=2)
    many = mc_european(S, K, T, R, Q, SIGMA, "c", 400_000, 16, True, "euler", seed=2)
    assert abs(one[0] - exact) > 5.0 * one[1]                    # a tiny error bar, badly placed
    assert abs(many[0] - exact) < abs(one[0] - exact) / 5.0      # steps, not paths, fix it
    assert many[1] > one[1] * 0.9                                # and the error bar barely moved
    with pytest.raises(ValueError, match="scheme"):
        mc_european(S, K, T, R, Q, SIGMA, "c", 1000, 1, True, "milstein", seed=0)


def test_payoff_std_inverts_an_error_estimate_back_into_a_sample_count():
    n = 40_000
    _, se = mc_european(S, K, T, R, Q, SIGMA, "c", n, 1, False, "log", seed=7)
    sd = payoff_std(S, K, T, R, Q, SIGMA, "c", n_paths=n, antithetic=False, seed=7)
    assert (sd / se) ** 2 == pytest.approx(n, rel=1e-9)
    # one antithetic PAIR is less variable than one raw payoff - that is the whole variance
    # reduction, and it is why the pair is the unit the error must be computed over
    assert payoff_std(S, K, T, R, Q, SIGMA, "c", antithetic=True) < \
        payoff_std(S, K, T, R, Q, SIGMA, "c", antithetic=False)


# ------------------------------------------------------------------ QuantLib cross-checks
@pytest.mark.skipif(has_module("QuantLib"), reason="QuantLib is installed")
def test_cross_checks_return_none_without_quantlib():
    assert quantlib_cross_checks(S, K, T, R, Q, SIGMA, HESTON,
                                 {**SABR, "strikes": STRIKES}) is None


@requires("QuantLib")
def test_quantlib_confirms_the_albrecher_form_the_sabr_vols_and_its_own_crr_probability():
    live = quantlib_cross_checks(S, K, T, R, Q, SIGMA, HESTON, {**SABR, "strikes": STRIKES},
                                 r_heston=R_H, q_heston=Q_H)
    assert live is not None
    for row in live["heston"]:
        mine = heston_price(S, K, row["T"], R_H, Q_H, flag="c", **HESTON)
        assert mine == pytest.approx(row["Gatheral"], abs=1e-8)
        assert row["default_ctor"] == pytest.approx(row["Gatheral"], abs=1e-8)
        assert row["BranchCorrection"] == pytest.approx(row["Gatheral"], abs=1e-8)
        if row["T"] > 3.0:                       # the unrepaired 1993 form does NOT agree
            old = heston_price(S, K, row["T"], R_H, Q_H, flag="c",
                               formulation="heston1993", **HESTON)
            assert math.isnan(old) or abs(old - row["Gatheral"]) > 1e-3

    for Ks in STRIKES:
        mine = sabr_implied_vol(Ks, SABR["F"], SABR["T"], SABR["alpha"], SABR["beta"],
                                SABR["nu"], SABR["rho"])
        assert mine == pytest.approx(live["sabr"][Ks], abs=1e-14)

    for n, ql_price in live["crr_european_put"].items():
        assert crr_price(S, K, T, R, Q, SIGMA, n, "p", prob="quantlib") == \
            pytest.approx(ql_price, abs=1e-10)
        assert abs(crr_price(S, K, T, R, Q, SIGMA, n, "p") - ql_price) > 1e-5

    tree_am = crr_price(S, K, T, R, Q, SIGMA, 800, "p", american=True)
    assert tree_am == pytest.approx(live["american_put_qdfp"], abs=5e-3)
    assert live["bs_put"] == pytest.approx(bsm_price(S, K, T, R, Q, SIGMA, "p"), abs=1e-8)

    # QuantLib's own error estimate reproduces the two Monte Carlo facts
    assert live["mc"][(False, 52)][1] == pytest.approx(live["mc"][(False, 1)][1], rel=0.02)
    assert live["mc"][(True, 1)][1] < live["mc"][(False, 1)][1]


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.models.option_models")
    assert "price Heston with the Albrecher/Gatheral form" in out
    assert "steps fix one" in out
    assert out.isascii()
