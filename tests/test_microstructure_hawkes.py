"""fin_skills.microstructure.hawkes - self-exciting arrivals, fitted and tested.

Each test asserts the PROPERTY the SKILL.md documents: two unrelated simulators land on the
same closed-form stationary rate, the O(k) recursive log-likelihood equals the O(k^2)
definition, MLE recovers the branching ratio more sharply than either of its parts, the random
time change detects a Poisson misspecification that the residual MEAN cannot, and a Poisson
confidence interval on a clustered count under-covers by the documented factor.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fin_skills.microstructure.hawkes import (TRUE, branching_ratio, compensator_residuals,
                                              count_dispersion, fit_mle, fit_poisson,
                                              goodness_of_fit, intensity, ks_exp1,
                                              log_likelihood, rate_study, recovery_study,
                                              simulate_cluster, simulate_thinning,
                                              stationary_intensity)


# ------------------------------------------------------------------- the closed forms
def test_the_closed_forms_and_their_refusals():
    assert branching_ratio(1.4, 2.0) == pytest.approx(0.7)
    assert stationary_intensity(0.5, 1.4, 2.0) == pytest.approx(0.5 / 0.3)
    assert stationary_intensity(0.5, 0.0, 2.0) == 0.5           # no excitation -> Poisson
    with pytest.raises(ValueError, match="not stationary"):
        stationary_intensity(0.5, 2.0, 2.0)
    with pytest.raises(ValueError, match="positive"):
        stationary_intensity(0.0, 1.0, 2.0)
    with pytest.raises(ValueError, match="beta must be positive"):
        branching_ratio(1.0, 0.0)


def test_the_intensity_uses_strictly_past_points_and_decays():
    ts = [1.0, 2.0]
    mu, alpha, beta = 0.5, 1.4, 2.0
    assert intensity(0.5, ts, mu, alpha, beta) == pytest.approx(mu)     # nothing yet
    assert intensity(1.0, ts, mu, alpha, beta) == pytest.approx(mu)     # strictly before t
    right_after = intensity(1.0 + 1e-12, ts, mu, alpha, beta)
    assert right_after == pytest.approx(mu + alpha, abs=1e-8)           # the jump is alpha
    later = intensity(2.0, ts, mu, alpha, beta)
    assert later == pytest.approx(mu + alpha * math.exp(-beta * 1.0))
    assert intensity(50.0, ts, mu, alpha, beta) == pytest.approx(mu, abs=1e-12)


# ---------------------------------------------------------------------- simulation
def test_two_unrelated_simulators_agree_with_the_stationary_rate():
    target = stationary_intensity(**TRUE)
    for sim in (simulate_thinning, simulate_cluster):
        d = rate_study(sim, T=1500.0, n_runs=12, seed=101, **TRUE)
        assert abs(d["rate"] - target) < 3.0 * d["se"]
    a = simulate_thinning(T=800.0, seed=4, **TRUE)
    b = simulate_cluster(T=800.0, seed=4, **TRUE)
    assert a.size > 0 and b.size > 0
    assert np.all(np.diff(a) > 0) and np.all(np.diff(b) >= 0)    # sorted, inside [0, T]
    assert a.max() <= 800.0 and b.max() <= 800.0


def test_a_seeded_path_is_deterministic_and_the_arguments_are_validated():
    assert np.array_equal(simulate_thinning(T=200.0, seed=3, **TRUE),
                          simulate_thinning(T=200.0, seed=3, **TRUE))
    assert not np.array_equal(simulate_thinning(T=200.0, seed=3, **TRUE),
                              simulate_thinning(T=200.0, seed=4, **TRUE))
    for sim in (simulate_thinning, simulate_cluster):
        with pytest.raises(ValueError, match="T must be positive"):
            sim(0.5, 1.4, 2.0, 0.0)
        with pytest.raises(ValueError, match="positive"):
            sim(0.5, 1.4, 0.0, 10.0)
    with pytest.raises(ValueError, match="never dies out"):
        simulate_cluster(0.5, 2.5, 2.0, 10.0)


def test_zero_excitation_is_an_ordinary_poisson_process():
    ts = simulate_thinning(1.5, 0.0, 2.0, 3000.0, seed=9)
    assert ts.size / 3000.0 == pytest.approx(1.5, rel=0.06)
    gaps = np.diff(ts)
    assert gaps.mean() == pytest.approx(1 / 1.5, rel=0.06)
    assert gaps.std() / gaps.mean() == pytest.approx(1.0, abs=0.06)   # Exp CV is 1


# ---------------------------------------------------------------------- likelihood
def test_the_recursion_equals_the_definition():
    ts = simulate_thinning(T=300.0, seed=2, **TRUE)
    for pars in (TRUE, {"mu": 0.3, "alpha": 0.9, "beta": 3.0}, {"mu": 2.0, "alpha": 0.1,
                                                                "beta": 0.5}):
        a = log_likelihood(ts, T=300.0, recursive=True, **pars)
        b = log_likelihood(ts, T=300.0, recursive=False, **pars)
        assert a == pytest.approx(b, rel=1e-12)


def test_the_likelihood_nests_the_poisson_case_and_validates_its_input():
    ts = simulate_thinning(T=300.0, seed=2, **TRUE)
    rate = ts.size / 300.0
    nested = log_likelihood(ts, mu=rate, alpha=0.0, beta=1.0, T=300.0)
    assert nested == pytest.approx(fit_poisson(ts, 300.0)["loglik"], rel=1e-12)
    # the horizon matters: a longer quiet tail is less likely, never more
    assert log_likelihood(ts, T=600.0, **TRUE) < log_likelihood(ts, T=300.0, **TRUE)
    with pytest.raises(ValueError, match="sorted"):
        log_likelihood([3.0, 1.0], **TRUE)
    with pytest.raises(ValueError, match="before the last event"):
        log_likelihood(ts, T=1.0, **TRUE)


def test_mle_recovers_the_branching_ratio_more_sharply_than_alpha_or_beta():
    rec = recovery_study(n_paths=12, T=1500.0, seed=77)
    for k in ("mu", "alpha", "beta", "n"):
        assert abs(rec[k]["bias"]) < 0.08                        # unbiased to a few per cent
    assert rec["n"]["cv"] < rec["alpha"]["cv"]                   # the documented headline
    assert rec["n"]["cv"] < rec["beta"]["cv"]
    with pytest.raises(ValueError, match="at least 3"):
        fit_mle([1.0, 2.0])


def test_the_fit_beats_a_poisson_fit_on_its_own_likelihood():
    ts = simulate_thinning(T=1500.0, seed=13, **TRUE)
    h, p = fit_mle(ts, 1500.0), fit_poisson(ts, 1500.0)
    assert h["success"] and h["loglik"] > p["loglik"] + 100.0
    assert h["n"] == pytest.approx(0.7, abs=0.08)


# ------------------------------------------------------------- the random time change
def test_the_residual_variance_sees_the_misspecification_that_the_mean_cannot():
    ts = simulate_thinning(T=2000.0, seed=21, **TRUE)
    h = goodness_of_fit(ts, fit_mle(ts, 2000.0))
    p = goodness_of_fit(ts, fit_poisson(ts, 2000.0))
    # both residual series have mean ~1: under a Poisson fit that is arithmetic, not evidence
    assert h["mean"] == pytest.approx(1.0, abs=0.05)
    assert p["mean"] == pytest.approx(1.0, abs=0.05)
    # Exp(1) has variance 1; only the misspecified fit misses it, and only KS rejects
    assert h["var"] == pytest.approx(1.0, abs=0.15)
    assert p["var"] > 3.0
    assert h["p"] > 0.01 and p["p"] < 1e-10
    assert p["ks"] > 10.0 * h["ks"]


def test_the_compensator_increments_are_exp1_under_the_true_parameters():
    ts = simulate_thinning(T=4000.0, seed=31, **TRUE)
    r = compensator_residuals(ts, **TRUE)
    assert r.size == ts.size
    assert np.all(r > 0)
    assert r.mean() == pytest.approx(1.0, abs=0.05)
    ks, p = ks_exp1(r)
    assert p > 0.01
    assert compensator_residuals([], **TRUE).size == 0
    with pytest.raises(ValueError, match="sorted"):
        compensator_residuals([2.0, 1.0], **TRUE)


# ------------------------------------------------------ what assuming Poisson costs you
def test_a_poisson_interval_under_covers_a_clustered_count_by_the_documented_factor():
    plain = count_dispersion(1.6, 0.0, 2.0, T=300.0, n_runs=600, seed=41)
    assert plain["fano"] == pytest.approx(1.0, abs=0.2)          # n = 0: a real Poisson
    assert plain["coverage"] == pytest.approx(0.95, abs=0.04)
    clustered = count_dispersion(0.48, 1.4, 2.0, T=300.0, n_runs=600, seed=41)
    assert clustered["mean_count"] == pytest.approx(plain["mean_count"], rel=0.06)
    assert clustered["fano_theory"] == pytest.approx(1.0 / 0.3 ** 2)
    assert abs(clustered["fano"] - clustered["fano_theory"]) < 4.0 * clustered["fano_se"]
    assert clustered["se_ratio"] == pytest.approx(1 / 0.3)
    assert clustered["coverage"] < 0.6                            # nominal 95%, measured ~46%


@pytest.mark.slow
def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.microstructure.hawkes")
    assert "Rule: fit the branching ratio" in out
    assert "1/(1 - n)" in out
    assert out.isascii()
