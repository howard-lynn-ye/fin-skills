"""fin_skills.models.optimizers - mean-variance, Black-Litterman, risk parity, minimum CVaR.

The properties asserted here are the ones the SKILL.md states: the two algebraic forms of the
Black-Litterman posterior agree and reproduce PyPortfolioOpt's numbers; a zero prior is refused;
risk parity really equalises risk contributions; the minimum-CVaR LP's objective IS the sample
CVaR of its own weights; a 1-bp perturbation of mu moves an unconstrained mean-variance book by
more than the guard allows and a mu-free rule by nothing; and 1/N beats the sample-based rule out
of sample on a universe where 1/N is NOT optimal.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires
from fin_skills.models.optimizers import (BL_CORR, BL_DELTA, BL_P, BL_Q, BL_TAU, BL_VOLS,
                                          assert_stable_weights, black_litterman,
                                          implied_returns, ledoit_wolf_identity,
                                          max_population_sharpe, mean_variance,
                                          mean_variance_closed_form, min_cvar_lp, min_variance,
                                          oos_backtest, perturbation_turnover, population_sharpe,
                                          pypfopt_bl_check, risk_contributions, risk_parity,
                                          sample_cvar, sharpe, simulate_universe, tangency)

BL_NAMES = ["A", "B", "C", "D", "E"]
BL_CAPS = np.array([40.0, 30.0, 15.0, 5.0, 10.0])


@pytest.fixture(scope="module")
def bl_inputs():
    Sigma = np.outer(BL_VOLS, BL_VOLS) * BL_CORR
    return Sigma, BL_CAPS / BL_CAPS.sum()


@pytest.fixture(scope="module")
def uni():
    return simulate_universe(n_assets=15, n_months=600, seed=0)


# --------------------------------------------------------------------------- mean-variance
def test_the_closed_form_and_the_solver_agree_when_no_constraint_binds(uni):
    mu, Sigma = uni["mu"], uni["Sigma"]
    closed = mean_variance_closed_form(mu, Sigma, risk_aversion=3.0)
    assert closed.sum() == pytest.approx(1.0)
    loose = mean_variance(mu, Sigma, 3.0, long_only=False,
                          bounds=[(-10.0, 10.0)] * len(mu))
    # SLSQP's ftol is on an objective of order 1e-4 here, so the weights land within ~1e-4 of
    # the closed form and no closer. That is the solver's resolution, not a bug.
    assert loose == pytest.approx(closed, abs=2e-4)
    # long-only really binds: some closed-form weight is negative and the solver has none
    assert closed.min() < 0.0
    lo = mean_variance(mu, Sigma, 3.0, long_only=True)
    assert lo.min() >= -1e-9 and lo.sum() == pytest.approx(1.0)


def test_min_variance_is_the_minimum(uni):
    Sigma = uni["Sigma"]
    w = min_variance(Sigma)
    assert w.sum() == pytest.approx(1.0)
    for i in range(5):
        d = np.zeros(len(w))
        d[i], d[(i + 1) % len(w)] = 0.03, -0.03
        assert (w + d) @ Sigma @ (w + d) > w @ Sigma @ w
    lo = min_variance(Sigma, long_only=True)
    assert lo.min() >= -1e-9
    assert lo @ Sigma @ lo >= w @ Sigma @ w - 1e-12      # a constraint cannot lower the minimum


def test_tangency_keeps_the_budget_positive_and_maximises_the_population_sharpe(uni):
    mu, Sigma = uni["mu"], uni["Sigma"]
    w = tangency(mu, Sigma)
    assert w.sum() == pytest.approx(1.0)
    ceiling = max_population_sharpe(mu, Sigma)
    assert population_sharpe(w, mu, Sigma) == pytest.approx(ceiling, rel=1e-9)
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = w + rng.normal(0, 0.05, len(w))
        assert population_sharpe(v, mu, Sigma) <= ceiling + 1e-9
    # dividing by |1'Sigma^-1 mu| preserves the direction: the budget goes to -1 (visible)
    # rather than every weight flipping sign (invisible)
    flipped = tangency(-mu, Sigma)
    assert flipped.sum() == pytest.approx(-1.0)
    assert flipped == pytest.approx(-w)


# --------------------------------------------------------------------------- Black-Litterman
def test_the_two_algebraic_forms_of_the_posterior_agree(bl_inputs):
    Sigma, w_mkt = bl_inputs
    bl = black_litterman(Sigma, BL_P, BL_Q, w_mkt=w_mkt, tau=BL_TAU, risk_aversion=BL_DELTA)
    assert bl["mean_form_gap"] < 1e-12
    assert bl["cov_form_gap"] < 1e-12
    # the default Omega is He-Litterman's diag(diag(tau P Sigma P'))
    assert bl["omega"] == pytest.approx(np.diag(np.diag(BL_TAU * BL_P @ Sigma @ BL_P.T)))
    # posterior variance is Sigma + M: parameter uncertainty is ADDED, never subtracted
    assert (np.diag(bl["Sigma_bl"]) > np.diag(Sigma)).all()
    # the views name C and B/D; A and E move too, through the covariance
    assert abs(bl["mu_bl"][0] - bl["pi"][0]) > 0
    assert abs(bl["mu_bl"][4] - bl["pi"][4]) > 0


def test_a_zero_prior_is_refused(bl_inputs):
    Sigma, _ = bl_inputs
    with pytest.raises(ValueError, match="a zero prior is not a prior"):
        black_litterman(Sigma, BL_P, BL_Q)
    # ... and an explicit prior is accepted
    out = black_litterman(Sigma, BL_P, BL_Q, pi=np.zeros(5))
    assert out["pi"] == pytest.approx(np.zeros(5))


def test_implied_returns_is_reverse_optimization(bl_inputs):
    Sigma, w_mkt = bl_inputs
    pi = implied_returns(Sigma, w_mkt, BL_DELTA)
    assert pi == pytest.approx(BL_DELTA * Sigma @ w_mkt)
    # feeding pi back into an unconstrained optimizer at the same delta returns the market
    assert np.linalg.solve(BL_DELTA * Sigma, pi) == pytest.approx(w_mkt)
    # caps that do not sum to one are normalized
    assert implied_returns(Sigma, BL_CAPS, BL_DELTA) == pytest.approx(pi)


@requires("pypfopt")
def test_reproduces_pypfopt_black_litterman(bl_inputs):
    Sigma, w_mkt = bl_inputs
    chk = pypfopt_bl_check(Sigma, w_mkt, BL_P, BL_Q, BL_TAU, BL_DELTA, BL_NAMES)
    assert chk is not None
    assert chk["max_abs_pi_diff"] < 1e-15
    assert chk["max_abs_mu_diff"] < 1e-15
    assert chk["max_abs_cov_diff"] < 1e-15
    assert chk["max_abs_omega_diff"] < 1e-15


@requires("pypfopt")
def test_pypfopt_still_warns_and_uses_zeros_when_pi_is_none(bl_inputs):
    """SKILL.md section 3 says pi=None is a WARNING and a zero prior, not an error."""
    import warnings

    import pandas as pd
    from pypfopt.black_litterman import BlackLittermanModel

    Sigma, _ = bl_inputs
    S = pd.DataFrame(Sigma, index=BL_NAMES, columns=BL_NAMES)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bl = BlackLittermanModel(S, pi=None, P=BL_P, Q=BL_Q, tau=BL_TAU)
    assert any("no prior" in str(w.message) for w in caught)
    assert np.asarray(bl.pi).ravel() == pytest.approx(np.zeros(5))


# --------------------------------------------------------------------------- risk parity
def test_risk_parity_equalises_the_risk_contributions(uni):
    Sigma = np.cov(uni["R"][:240].T)
    n = len(Sigma)
    w = risk_parity(Sigma)
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()
    share = risk_contributions(w, Sigma)
    share = share / share.sum()
    assert share == pytest.approx(np.full(n, 1.0 / n), abs=1e-10)
    # risk contributions add up to the portfolio volatility, by Euler's theorem
    assert risk_contributions(w, Sigma).sum() == pytest.approx(np.sqrt(w @ Sigma @ w))
    # equal WEIGHTS are not equal risk
    eq = np.full(n, 1.0 / n)
    eq_share = risk_contributions(eq, Sigma)
    eq_share = eq_share / eq_share.sum()
    assert eq_share.max() / eq_share.min() > 2.0


def test_risk_parity_respects_an_unequal_budget(uni):
    Sigma = np.cov(uni["R"][:240].T)
    n = len(Sigma)
    b = np.arange(1, n + 1, dtype=float)
    b = b / b.sum()
    w = risk_parity(Sigma, budget=b)
    share = risk_contributions(w, Sigma)
    assert share / share.sum() == pytest.approx(b, abs=1e-9)


# --------------------------------------------------------------------------- minimum CVaR
def test_the_lp_objective_is_the_sample_cvar_of_its_own_weights(uni):
    R = uni["R"][:400, :8]
    lp = min_cvar_lp(R, level=0.95)
    assert lp["w"].sum() == pytest.approx(1.0) and lp["w"].min() >= -1e-9
    assert lp["cvar"] == pytest.approx(lp["cvar_check"], abs=1e-12)
    # alpha is the VaR at the optimum, and CVaR >= VaR
    assert lp["cvar"] >= lp["var"]
    # no long-only portfolio on this sample has a lower CVaR
    rng = np.random.default_rng(1)
    for _ in range(30):
        w = rng.dirichlet(np.ones(8))
        assert sample_cvar(-(R @ w), 0.95) >= lp["cvar"] - 1e-9


def test_sample_cvar_splits_the_atom_and_a_return_floor_binds(uni):
    # 100 equiprobable losses, level 0.95 -> exactly the mean of the worst 5
    loss = np.arange(100.0)
    assert sample_cvar(loss, 0.95) == pytest.approx(np.mean([95, 96, 97, 98, 99]))
    R = uni["R"][:400, :8]
    free = min_cvar_lp(R, level=0.95)
    target = float(R.mean(0).max())                       # only the best asset can reach it
    tight = min_cvar_lp(R, level=0.95, min_return=target)
    assert R.mean(0) @ tight["w"] >= target - 1e-9
    assert tight["cvar"] >= free["cvar"] - 1e-9           # a constraint cannot help


# --------------------------------------------------------------------------- stability
def test_a_one_bp_perturbation_moves_the_unconstrained_book_and_not_a_mu_free_one(uni):
    est = uni["R"][:120]
    mu_hat, S_hat = est.mean(0), np.cov(est.T)
    tan = perturbation_turnover(mu_hat, S_hat, lambda m, S: tangency(m, S))
    free = perturbation_turnover(mu_hat, S_hat, lambda m, S: min_variance(S, long_only=True))
    lo = perturbation_turnover(mu_hat, S_hat, lambda m, S: mean_variance(m, S, 3.0))
    assert tan["mean_turnover"] > 0.02
    assert free["mean_turnover"] == pytest.approx(0.0, abs=1e-12)     # mu-free: no sensitivity
    assert lo["mean_turnover"] < tan["mean_turnover"]                 # constraints damp it
    assert tan["gross_leverage"] > lo["gross_leverage"]

    with pytest.raises(ValueError, match="estimation error, not a view"):
        assert_stable_weights(mu_hat, S_hat, lambda m, S: tangency(m, S), max_turnover=0.01)
    rep = assert_stable_weights(mu_hat, S_hat, lambda m, S: min_variance(S, long_only=True))
    assert rep["mean_turnover"] == pytest.approx(0.0, abs=1e-12)


def test_ledoit_wolf_identity_shrinks_the_condition_number(uni):
    est = uni["R"][:120]
    S, s = ledoit_wolf_identity(est)
    assert 0.0 < s < 1.0
    assert np.linalg.cond(S) < np.linalg.cond(np.cov(est.T))
    assert np.allclose(S, S.T)


# --------------------------------------------------------------------------- out of sample
def test_the_universe_makes_one_over_n_suboptimal_and_the_sample_rule_still_loses():
    ceilings, naive, mv, eq = [], [], [], []
    for s in (0, 1, 2):
        u = simulate_universe(seed=s)
        n = len(u["mu"])
        w_eq = np.full(n, 1.0 / n)
        ceilings.append(max_population_sharpe(u["mu"], u["Sigma"]))
        naive.append(population_sharpe(w_eq, u["mu"], u["Sigma"]))
        oos = oos_backtest(u["R"], 120,
                           {"mv": lambda est, _: tangency(est.mean(0), np.cov(est.T)),
                            "eq": lambda est, _: np.full(est.shape[1], 1.0 / est.shape[1])},
                           n_oos=360)
        mv.append(sharpe(oos["mv"]))
        eq.append(sharpe(oos["eq"]))
    # the DGP is NOT rigged: the true tangency portfolio is materially better than 1/N
    assert np.mean(ceilings) > 1.5 * np.mean(naive)
    # and yet the sample-based rule loses to 1/N out of sample
    assert np.mean(mv) < np.mean(eq)


def test_oos_backtest_holds_the_estimated_weights_over_the_next_period(uni):
    R = uni["R"]
    out = oos_backtest(R, 120, {"eq": lambda est, _: np.full(est.shape[1], 1.0 / est.shape[1])},
                       n_oos=50)
    assert len(out) == 50
    assert out["eq"].to_numpy() == pytest.approx(R[-50:].mean(axis=1))
    with pytest.raises(ValueError, match="not enough history"):
        oos_backtest(R, 120, {"eq": lambda est, _: np.full(est.shape[1], 1.0 / est.shape[1])},
                     n_oos=len(R))


def test_simulate_universe_is_deterministic():
    a, b = simulate_universe(seed=0), simulate_universe(seed=0)
    assert np.array_equal(a["R"], b["R"]) and np.array_equal(a["mu"], b["mu"])
    assert not np.array_equal(a["R"], simulate_universe(seed=1)["R"])
    assert a["mu"] == pytest.approx(a["alpha"] + a["beta"] * 0.005)
    # the realized mean tracks mu, but every asset shares one factor draw, so the whole
    # cross-section shifts together by beta * (fbar - factor_mean): a common, not an
    # independent, error. Correlation is the property; a tight per-asset bound is not.
    assert np.corrcoef(a["R"].mean(0), a["mu"])[0, 1] > 0.7


# --------------------------------------------------------------------------- the demo
@pytest.mark.slow
def test_demo_prints_the_rule_and_the_measured_traps(run_main):
    out = run_main("fin_skills.models.optimizers")
    assert "THE RULE:" in out and "an optimizer maximizes the error in its inputs" in out
    assert "assert_stable_weights(tangency) -> ValueError" in out
    assert "POPULATION Sharpe at the TRUE moments" in out
    assert "a zero prior is not a prior" in out
    assert out.isascii()
