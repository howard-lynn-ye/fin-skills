"""fin_skills.models.var_cvar - VaR and ES estimators, their backtests, and the scaling guard.

The properties asserted here are the ones the SKILL.md states: the parametric-normal estimator
understates a fat tail and the EVT one does not; Cornish-Fisher stops being a quantile function
at excess kurtosis 8 exactly; Kupiec matches an independent closed form and is zero at its null;
the Christoffersen test is blind when pi01 == pi11; VaR is not subadditive and ES is; and the
sqrt(h) guard refuses to scale unless the assumption is asserted.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from fin_skills.models.var_cvar import (GARCH, LEVEL, backtest_size_under_null,
                                        christoffersen_independence, conditional_coverage,
                                        cornish_fisher_is_monotone, cornish_fisher_z,
                                        discrete_var_es, gpd_es_numeric, horizon_var_from_state,
                                        kupiec_pof, kupiec_via_binomial, losses, rolling_var,
                                        scale_var_sqrt_time, simulate_garch_t, t_es_unit,
                                        t_quantile_unit, var_es_cornish_fisher, var_es_evt,
                                        var_es_historical, var_es_normal)


@pytest.fixture(scope="module")
def fat():
    """A long iid Student-t(5) loss sample at 1 %/day: fat tails, no clustering."""
    rng = np.random.default_rng(0)
    nu = 5.0
    return -(rng.standard_t(nu, 200_000) * math.sqrt((nu - 2) / nu) * 0.01)


# --------------------------------------------------------------------------- the estimators
def test_the_normal_estimator_understates_a_fat_tail_and_evt_does_not(fat):
    nu = GARCH["nu"]
    truth_var = 0.01 * t_quantile_unit(LEVEL, nu)
    truth_es = 0.01 * t_es_unit(LEVEL, nu)
    h_var, h_es = var_es_historical(fat, LEVEL)
    n_var, n_es = var_es_normal(fat, LEVEL)
    e_var, e_es, info = var_es_evt(fat, LEVEL)
    assert h_var == pytest.approx(truth_var, rel=0.03)      # the empirical quantile is unbiased
    assert n_var < truth_var * 0.95                          # the normal quantile is too small
    assert n_es < truth_es * 0.85                            # and its ES is much too small
    assert e_var == pytest.approx(truth_var, rel=0.05)
    assert info["xi"] > 0.0                                  # a heavy tail, correctly detected
    assert info["n_exceedances"] == pytest.approx(0.10 * len(fat), rel=0.02)


def test_the_student_t_closed_forms_match_a_large_simulation():
    nu = 5.0
    rng = np.random.default_rng(3)
    x = rng.standard_t(nu, 2_000_000) * math.sqrt((nu - 2) / nu)
    assert x.std() == pytest.approx(1.0, rel=0.02)           # the rescaling gives unit variance
    q = t_quantile_unit(0.99, nu)
    assert np.quantile(-x, 0.99) == pytest.approx(q, rel=0.02)
    tail = -x[-x >= q]
    assert tail.mean() == pytest.approx(t_es_unit(0.99, nu), rel=0.03)
    assert q > stats.norm.ppf(0.99)                          # fatter than normal at 99 %


def test_cornish_fisher_stops_being_a_quantile_function_at_excess_kurtosis_eight():
    # closed form at skew 0: derivative 1 + (3z^2 - 3) K/24, minimised at z = 0 as 1 - K/8
    assert cornish_fisher_is_monotone(0.0, 7.9)
    assert not cornish_fisher_is_monotone(0.0, 8.2)
    assert cornish_fisher_z(0.0, 0.0, 0.0) == pytest.approx(0.0)
    assert cornish_fisher_z(2.0, 0.0, 0.0) == pytest.approx(2.0)   # no adjustment at S=K=0
    # past the break the failure is in the MIDDLE: a higher confidence level returns a smaller
    # "quantile". At K = 30 the expansion falls over z in about [-0.86, +0.86].
    z_lo, z_hi = stats.norm.ppf(0.25), stats.norm.ppf(0.75)
    assert cornish_fisher_z(z_hi, 0.0, 30.0) < cornish_fisher_z(z_lo, 0.0, 30.0)


def test_cornish_fisher_reports_its_own_non_monotonicity():
    r, _ = simulate_garch_t(2500, seed=0)
    _, _, info = var_es_cornish_fisher(losses(r), LEVEL)
    assert info["exkurt"] > 8.0                     # the GARCH-t sample is past the break
    assert info["monotone"] is False                # and the estimator says so
    # the region where it decreases, in confidence-level terms: about 31 % to 75 %
    z = np.linspace(-3.5, 3.5, 7001)
    falling = np.where(np.diff(cornish_fisher_z(z, info["skew"], info["exkurt"])) <= 0)[0]
    assert stats.norm.cdf(z[falling[0]]) == pytest.approx(0.31, abs=0.02)
    assert stats.norm.cdf(z[falling[-1] + 1]) == pytest.approx(0.75, abs=0.02)


def test_evt_closed_form_es_matches_numerical_integration_of_the_fitted_tail(fat):
    var, es, info = var_es_evt(fat, LEVEL)
    numeric = gpd_es_numeric(var, info["xi"], info["beta"], info["threshold"])
    assert es == pytest.approx(numeric, rel=1e-6)
    assert es > var                                  # ES is beyond VaR by construction
    with pytest.raises(ValueError, match="beyond the threshold"):
        var_es_evt(fat, 0.85, threshold_quantile=0.90)


def test_normal_es_is_the_closed_form():
    rng = np.random.default_rng(4)
    loss = rng.normal(0.0, 1.0, 400_000)
    var, es = var_es_normal(loss, 0.99)
    z = stats.norm.ppf(0.99)
    assert var == pytest.approx(z, abs=0.02)
    assert es == pytest.approx(stats.norm.pdf(z) / 0.01, abs=0.03)


# --------------------------------------------------------------------------- the backtests
def test_kupiec_matches_an_independent_binomial_computation_and_is_zero_at_its_null():
    for x, n, p in ((28, 2000, 0.01), (5, 500, 0.01), (60, 1000, 0.05)):
        lr, pv = kupiec_pof(x, n, p)
        assert lr == pytest.approx(kupiec_via_binomial(x, n, p), abs=1e-9)
        assert pv == pytest.approx(stats.chi2.sf(lr, 1))
    assert kupiec_pof(20, 2000, 0.01)[0] == pytest.approx(0.0, abs=1e-9)
    assert kupiec_pof(20, 2000, 0.01)[1] == pytest.approx(1.0)
    assert kupiec_pof(0, 500, 0.01)[0] > 0.0          # zero exceptions is also a failure


def test_christoffersen_is_blind_when_pi01_equals_pi11_and_fires_on_regularity():
    lr_blk, p_blk, c = christoffersen_independence(np.tile([1, 1, 0, 0], 100))
    assert c["pi01"] == pytest.approx(c["pi11"], abs=0.01)      # 0.4975 vs 0.500 at the edges
    assert lr_blk == pytest.approx(0.0, abs=1e-2) and p_blk > 0.9   # perfectly predictable, passes
    lr_reg, p_reg, _ = christoffersen_independence(np.tile([1, 0, 0, 0, 0], 100))
    assert lr_reg > 20.0 and p_reg < 0.001          # regularity is rejected, not only clustering


def test_conditional_coverage_is_the_sum_of_the_two_statistics():
    rng = np.random.default_rng(6)
    hits = (rng.random(2000) < 0.01).astype(int)
    res = conditional_coverage(hits, 0.01)
    lr_uc, _ = kupiec_pof(int(hits.sum()), len(hits), 0.01)
    lr_ind, _, _ = christoffersen_independence(hits)
    assert res["lr_uc"] == pytest.approx(lr_uc)
    assert res["lr_ind"] == pytest.approx(lr_ind)
    assert res["lr_cc"] == pytest.approx(lr_uc + lr_ind)
    assert res["p_cc"] == pytest.approx(stats.chi2.sf(res["lr_cc"], 2))
    assert res["expected"] == pytest.approx(20.0)


def test_the_independence_test_is_undersized_at_the_99_percent_level():
    size = backtest_size_under_null(p=0.01, n_obs=500, n_seq=2000, alpha=0.05, seed=0)
    assert size["ind"] < 0.03                # nominal 5 %, measured near 1.4 %
    assert size["uc"] > 0.05                 # Kupiec goes the other way
    assert size["cc"] < 0.05


# --------------------------------------------------------------------------- the rolling test
def test_filtered_historical_simulation_beats_the_unconditional_window_on_a_garch_path():
    r, sig = simulate_garch_t(2500, seed=0)
    loss = losses(r)
    test_loss = loss[500:]
    out = {}
    for m in ("normal", "ewma_fhs"):
        v = rolling_var(loss, 500, LEVEL, m, sigma=sig, nu=GARCH["nu"])
        out[m] = conditional_coverage((test_loss > v).astype(int), 1.0 - LEVEL)
    assert out["normal"]["p_uc"] < 0.05           # the normal quantile fails the count
    assert out["ewma_fhs"]["p_uc"] > 0.20         # filtered historical simulation does not
    assert abs(out["ewma_fhs"]["n_exceptions"] - 20) < abs(out["normal"]["n_exceptions"] - 20)
    with pytest.raises(ValueError, match="method must be"):
        rolling_var(loss, 500, LEVEL, "kernel")


def test_simulate_garch_t_is_deterministic_and_has_the_stated_unconditional_vol():
    r1, s1 = simulate_garch_t(3000, seed=0)
    r2, s2 = simulate_garch_t(3000, seed=0)
    assert np.array_equal(r1, r2) and np.array_equal(s1, s2)
    assert not np.array_equal(r1, simulate_garch_t(3000, seed=1)[0])
    long, _ = simulate_garch_t(200_000, seed=42)
    assert long.std() == pytest.approx(GARCH["uncond_vol"], rel=0.20)
    assert stats.kurtosis(long) > 3.0              # clustering plus t innovations: very fat


# --------------------------------------------------------------------------- the sqrt(h) guard
def test_sqrt_time_scaling_is_refused_unless_the_assumption_is_asserted():
    with pytest.raises(ValueError, match="iid normal zero-mean"):
        scale_var_sqrt_time(0.02, 10)
    assert scale_var_sqrt_time(0.02, 10, iid_normal_zero_mean=True) == pytest.approx(
        0.02 * math.sqrt(10))
    assert scale_var_sqrt_time(0.02, 1, iid_normal_zero_mean=True) == pytest.approx(0.02)


def test_sqrt_time_is_exact_for_iid_normal_and_wrong_in_both_directions_otherwise():
    rng = np.random.default_rng(5)
    h = 10
    x = rng.normal(0.0, 0.01, 400_000)
    v1 = float(np.quantile(-x, LEVEL))
    vh = float(np.quantile(-x.reshape(-1, h).sum(1), LEVEL))
    assert v1 * math.sqrt(h) == pytest.approx(vh, rel=0.03)          # the assumption holds

    quiet = horizon_var_from_state((0.5 * GARCH["uncond_vol"]) ** 2, h, LEVEL, n_paths=60_000)
    loud = horizon_var_from_state((2.0 * GARCH["uncond_vol"]) ** 2, h, LEVEL, n_paths=60_000)
    q1 = 0.5 * GARCH["uncond_vol"] * t_quantile_unit(LEVEL, GARCH["nu"])
    l1 = 2.0 * GARCH["uncond_vol"] * t_quantile_unit(LEVEL, GARCH["nu"])
    assert q1 * math.sqrt(h) < quiet          # from a quiet day sqrt(h) UNDERSTATES
    assert l1 * math.sqrt(h) > loud           # from a stressed day it OVERSTATES
    assert loud > quiet


# --------------------------------------------------------------------------- coherence
def test_var_is_not_subadditive_and_es_is():
    lvl = 0.975
    va, ea = discrete_var_es([0.0, 1.0], [0.98, 0.02], lvl)
    vs, es = discrete_var_es([0.0, 1.0, 2.0], [0.98 ** 2, 2 * 0.98 * 0.02, 0.02 ** 2], lvl)
    assert va == 0.0 and vs == 1.0
    assert vs > 2 * va                        # VaR(A+B) > VaR(A) + VaR(B): not subadditive
    assert ea == pytest.approx(0.8) and es == pytest.approx(1.016)
    assert es <= 2 * ea                       # ES is
    assert es >= vs                           # and ES dominates VaR at the same level


def test_discrete_var_es_matches_the_normal_closed_form_on_a_fine_grid():
    x = np.linspace(-6.0, 6.0, 40_001)
    p = stats.norm.pdf(x)
    p = p / p.sum()
    var, es = discrete_var_es(x, p, 0.99)
    z = stats.norm.ppf(0.99)
    assert var == pytest.approx(z, abs=0.01)
    assert es == pytest.approx(stats.norm.pdf(z) / 0.01, abs=0.01)


# --------------------------------------------------------------------------- the demo
def test_demo_prints_the_rule_and_the_measured_traps(run_main):
    out = run_main("fin_skills.models.var_cvar")
    assert "THE RULE:" in out and "name the estimator, the horizon and the scaling" in out
    assert "scale_var_sqrt_time(var, 10) -> ValueError" in out
    assert "VaR IS NOT SUBADDITIVE" in out and "SIZE OF THE TESTS UNDER THE NULL" in out
    assert out.isascii()
