"""fin_skills.models.risk_model - covariance estimators, and the two ways one lies to a solver.

The properties asserted here are the ones the SKILL.md states: a sample covariance from T < N
observations is singular and the guard refuses it while clearing the shrunk matrix from the same
sample; the minimum-variance portfolio realizes more variance than it predicts, and structure
narrows the gap; the EWMA recursion is the RiskMetrics one and its effective sample size
reproduces Table 5.7 of the Technical Document; and the shrinkage estimators match sklearn and
PyPortfolioOpt to machine precision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import requires
from fin_skills.models.risk_model import (LAMBDA_DAILY, LAMBDA_MONTHLY, PYPFOPT_PROBE,
                                          SKLEARN_PROBE, barra_cov, bias_ratio, check_invertible,
                                          condition_report, effective_days, ewma_cov,
                                          ewma_variance_path, gmv_weights, half_life, ledoit_wolf,
                                          marchenko_pastur_edge, n_factors_above_mp,
                                          pca_factor_cov, portfolio_variance, probe, sample_cov,
                                          simulate_factor_returns, span_to_lambda)


@pytest.fixture(scope="module")
def sim():
    return simulate_factor_returns(30, 400, seed=1)


# --------------------------------------------------------------------------- the N > T trap
def test_sample_covariance_is_singular_when_t_is_short_and_shrinkage_fixes_it():
    short = simulate_factor_returns(40, 25, seed=1)
    S = sample_cov(short["R"])
    lw, intensity = ledoit_wolf(short["R"], "identity", ddof=1)
    assert condition_report(S)["rank"] == 24                    # at most T - 1
    assert condition_report(lw)["rank"] == 40                   # full rank from the same data
    assert 0.0 < intensity <= 1.0

    with pytest.raises(ValueError, match="singular"):
        check_invertible(S, n_obs=25)
    rep = check_invertible(lw, n_obs=25)                        # the guard clears the fix
    assert rep["rank"] == 40 and rep["short_sample"] is True and rep["cond"] < 1e8

    # and the singular matrix is what produces the absurd portfolio
    assert np.abs(gmv_weights(S)).sum() > 2 * np.abs(gmv_weights(lw)).sum()


def test_check_invertible_refuses_an_ill_conditioned_but_full_rank_matrix():
    Sigma = np.diag([1.0, 1e-12])
    assert condition_report(Sigma)["rank"] == 2
    with pytest.raises(ValueError, match="condition number"):
        check_invertible(Sigma, max_cond=1e8)
    assert check_invertible(Sigma, max_cond=1e20)["cond"] > 1e11


def test_gmv_weights_sum_to_one_and_match_the_closed_form(sim):
    S = sample_cov(sim["R"])
    w = gmv_weights(S)
    assert w.sum() == pytest.approx(1.0)
    x = np.linalg.solve(S, np.ones(len(S)))
    assert w == pytest.approx(x / x.sum())
    # the minimum-variance portfolio is the minimum: perturb it and variance rises
    d = np.zeros(len(w))
    d[0], d[1] = 0.05, -0.05
    assert portfolio_variance(w + d, S) > portfolio_variance(w, S)


def test_the_optimizer_underpredicts_its_own_variance_and_structure_narrows_the_gap():
    ratios = {}
    for name in ("sample", "lw_cc", "pca"):
        vals = []
        for s in range(6):
            sm = simulate_factor_returns(40, 400, seed=200 + s)
            R_in = sm["R"][:200]
            Sig = {"sample": lambda: sample_cov(R_in),
                   "lw_cc": lambda: ledoit_wolf(R_in, "constant_correlation", ddof=1)[0],
                   "pca": lambda: pca_factor_cov(R_in, 3)[0]}[name]()
            vals.append(bias_ratio(gmv_weights(Sig), Sig, sm["Sigma_true"]))
        ratios[name] = float(np.median(vals))
    assert ratios["sample"] > 1.2                       # realizes more than it promised
    assert ratios["lw_cc"] < ratios["sample"]
    assert ratios["pca"] < ratios["sample"]


# --------------------------------------------------------------------------- RiskMetrics EWMA
def test_effective_days_reproduces_table_5_7_of_the_technical_document():
    # RiskMetrics Technical Document 4th ed., Eq. [5.26] K = ln(tolerance)/ln(lambda);
    # Table 5.7 prints 74 days for lambda 0.94 and 151 for 0.97 at the 1 % tolerance level.
    assert round(effective_days(LAMBDA_DAILY, 0.01)) == 74
    assert round(effective_days(LAMBDA_MONTHLY, 0.01)) == 151
    assert effective_days(0.94, 0.01) == pytest.approx(np.log(0.01) / np.log(0.94))
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="lambda"):
            effective_days(bad)


def test_half_life_and_span_conversions():
    assert half_life(0.94) == pytest.approx(11.20, abs=0.01)
    assert half_life(0.97) == pytest.approx(22.76, abs=0.01)
    assert 0.94 ** half_life(0.94) == pytest.approx(0.5)
    assert span_to_lambda(180) == pytest.approx(1.0 - 2.0 / 181.0)
    assert span_to_lambda(180) > 0.98          # pypfopt's default decays far slower than 0.94


def test_the_ewma_recursion_is_the_riskmetrics_one_and_matches_adjust_false():
    x = simulate_factor_returns(1, 500, 1, seed=7)["R"][:, 0]
    s2 = ewma_variance_path(x, 0.94)
    assert s2[0] == pytest.approx(x[0] ** 2)
    # Eq. [5.23]: sigma2[t+1|t] = 0.94 sigma2[t|t-1] + 0.06 r[t]^2
    assert s2[1] == pytest.approx(0.94 * s2[0] + 0.06 * x[0] ** 2)
    adj_false = pd.Series(x ** 2).ewm(alpha=1 - 0.94, adjust=False).mean().to_numpy()
    assert np.max(np.abs(s2[1:] - adj_false)) < 1e-15
    # pandas' DEFAULT differs over the warm-up, which is what a rolling backtest keeps re-entering
    adj_true = pd.Series(x ** 2).ewm(alpha=1 - 0.94, adjust=True).mean().to_numpy()
    assert abs(adj_true[9] / adj_false[9] - 1) > 0.2
    assert abs(adj_true[-1] / adj_false[-1] - 1) < 1e-10


def test_ewma_cov_does_not_demean_unless_asked():
    rng = np.random.default_rng(2)
    X = rng.normal(0.05, 0.01, (400, 3))          # a large, deliberate mean
    raw = ewma_cov(X, 0.94)
    centred = ewma_cov(X, 0.94, demean=True)
    assert raw[0, 0] > 5 * centred[0, 0]          # RiskMetrics centres on zero, not the mean
    assert np.allclose(raw, raw.T) and np.allclose(centred, centred.T)


def test_ewma_carries_far_less_information_than_the_equally_weighted_estimate():
    sm = simulate_factor_returns(50, 250, seed=11)
    assert np.linalg.cond(ewma_cov(sm["R"], LAMBDA_DAILY)) > 10 * np.linalg.cond(
        sample_cov(sm["R"]))
    assert effective_days(LAMBDA_DAILY) / 50 < 2.0      # T_eff/N of 1.5: not a sample to invert


# --------------------------------------------------------------------------- factor structure
def test_pca_factor_cov_reproduces_the_diagonal_and_has_the_requested_rank(sim):
    Sigma, B, vals = pca_factor_cov(sim["R"], 3)
    S = sample_cov(sim["R"])
    assert np.diag(Sigma) == pytest.approx(np.diag(S))        # the residual makes it exact
    assert B.shape == (30, 3)
    assert np.linalg.matrix_rank(B @ B.T) == 3
    assert (np.diff(vals) <= 1e-9).all()                      # eigenvalues descending
    assert np.linalg.eigvalsh(Sigma).min() > 0                # PSD once the diagonal is added


def test_marchenko_pastur_edge_and_the_factor_count(sim):
    assert marchenko_pastur_edge(100, 100) == pytest.approx(4.0)
    assert marchenko_pastur_edge(1, 1_000_000) == pytest.approx(1.0, abs=1e-2)
    k, vals, edge = n_factors_above_mp(sim["R"])
    assert 1 <= k <= 3 and vals[0] > edge > 1.0
    # a shorter sample raises the edge and the count cannot rise
    k_short, _, edge_short = n_factors_above_mp(sim["R"][:100])
    assert edge_short > edge and k_short <= k


def test_barra_cov_recovers_the_factor_returns_when_the_exposures_are_true(sim):
    out = barra_cov(sim["R"], sim["B"])
    assert out["Sigma"].shape == (30, 30)
    assert out["factor_returns"].shape == (400, 3)
    assert np.allclose(out["Sigma"], out["Sigma"].T)
    # the cross-sectional regression is a projection: residuals are orthogonal to the exposures
    assert np.abs(out["specific"] @ sim["B"]).max() < 1e-8
    assert np.diag(out["D"]).min() > 0


# --------------------------------------------------------------------------- library agreement
def _probe_value(code: str, timeout: float = 240.0) -> str:
    rc, out = probe(code, timeout=timeout)
    assert rc == 0, f"probe crashed with returncode {rc}"
    return out


@requires("sklearn")
def test_matches_sklearn_ledoit_wolf_to_machine_precision():
    out = _probe_value(SKLEARN_PROBE)
    assert "max |cov diff|" in out, out
    diff = float(out.split("max |cov diff|")[1].split(";")[0])
    assert diff < 1e-15


@requires("pypfopt")
def test_matches_pypfopt_constant_correlation_shrinkage():
    out = _probe_value(PYPFOPT_PROBE)
    assert "max |cov diff|" in out, out
    diff = float(out.split("max |cov diff|")[1].split(";")[0])
    assert diff < 1e-15


@requires("sklearn")
@requires("osqp")
def test_the_import_order_that_kills_the_interpreter_is_still_what_the_skill_says():
    """SKILL.md section 5 claims sklearn-then-osqp crashes and the reverse does not.

    A property test, not a correctness test: it asserts the claim is still true here. If this
    ever fails because BOTH orders survive, the trap has been fixed upstream and the SKILL.md
    section should be re-dated, not the test deleted.
    """
    good, _ = probe("import osqp; import sklearn.covariance; print('ok')")
    assert good == 0
    bad, _ = probe("import sklearn.covariance; import osqp; print('ok')")
    assert bad != 0, "sklearn -> osqp no longer crashes; re-verify SKILL.md section 5"


# --------------------------------------------------------------------------- the demo
@pytest.mark.slow
def test_demo_prints_the_rule_and_the_measured_traps(run_main):
    out = run_main("fin_skills.models.risk_model")
    assert "THE RULE:" in out and "N/T decides the estimator" in out
    assert "check_invertible(sample, n_obs=60): ValueError" in out
    assert "IMPORT ORDER" in out and "RiskMetrics Eq. [5.26]" in out
    assert out.isascii()
