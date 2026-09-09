"""fin_skills.microstructure.copulas - same correlation, four different tails.

Each test asserts the PROPERTY the SKILL.md documents: the tau/parameter maps invert each
other, the deterministic bivariate normal CDF matches scipy while the bivariate t one is what
scipy cannot be trusted for, every analytic log-density matches a finite difference of its own
CDF, the finite-u ratio C(u,u)/u converges to the closed-form tail dependence, both estimators
recover the parameter, and a Gaussian copula fitted at the same Kendall's tau understates the
joint tail by a factor that grows as you go deeper.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from fin_skills.microstructure.copulas import (DF, FAMILIES, RHO, bvn_cdf, bvt_cdf,
                                               copula_cdf, copula_logpdf, fit_by_tau, fit_mle,
                                               joint_tail_study, kendall_tau, portfolio_tail,
                                               sample, sample_elliptical,
                                               scipy_mvt_reproducibility, tail_dependence,
                                               tail_ratio, tau_to_param)

TAU = 1.0 / 3.0
PARAMS = {"gaussian": (RHO, None), "t": (RHO, DF), "clayton": 1.0, "gumbel": 1.5}


def _pars(fam):
    return PARAMS[fam] if isinstance(PARAMS[fam], tuple) else (PARAMS[fam], None)


# --------------------------------------------------------------- tau and its inversion
def test_every_family_is_set_to_the_same_kendall_tau_and_the_maps_invert():
    for fam in FAMILIES:
        par, _ = _pars(fam)
        assert kendall_tau(fam, par) == pytest.approx(TAU, abs=1e-12)
        assert tau_to_param(fam, TAU) == pytest.approx(par, abs=1e-12)
        assert kendall_tau(fam, tau_to_param(fam, 0.6)) == pytest.approx(0.6, abs=1e-12)
    # the t copula's tau does not contain df - the whole point of the skill
    assert kendall_tau("t", RHO, df=2.0) == kendall_tau("t", RHO, df=50.0)


def test_the_parameter_maps_refuse_out_of_range_input():
    for fam, bad in (("gaussian", 1.0), ("t", -1.0), ("clayton", 0.0), ("gumbel", 0.5)):
        with pytest.raises(ValueError):
            kendall_tau(fam, bad)
    with pytest.raises(ValueError, match="family must be"):
        kendall_tau("frank", 1.0)
    with pytest.raises(ValueError, match="family must be"):
        tau_to_param("frank", 0.3)
    with pytest.raises(ValueError, match="Clayton needs"):
        tau_to_param("clayton", 0.0)


def test_the_tail_dependence_closed_forms():
    assert tail_dependence("gaussian", 0.99) == (0.0, 0.0)       # exactly zero at any rho < 1
    lo, hi = tail_dependence("t", RHO, DF)
    assert lo == hi                                              # radially symmetric
    assert lo == pytest.approx(2 * stats.t.cdf(-math.sqrt(5 * 0.5 / 1.5), 5.0))
    assert tail_dependence("clayton", 1.0) == (0.5, 0.0)         # 2^(-1/1)
    assert tail_dependence("gumbel", 1.5)[1] == pytest.approx(2.0 - 2.0 ** (1 / 1.5))
    # more degrees of freedom -> less tail dependence, and it vanishes only in the limit
    assert tail_dependence("t", RHO, 100.0)[0] < tail_dependence("t", RHO, 4.0)[0]
    assert tail_dependence("t", RHO, 1000.0)[0] > 0.0
    with pytest.raises(ValueError, match="needs df"):
        tail_dependence("t", RHO)


# --------------------------------------------------- the deterministic bivariate CDFs
def test_the_plackett_bivariate_normal_matches_scipy_and_its_own_limits():
    for a, b, rho in ((0.0, 0.0, 0.5), (-2.0, -2.0, 0.5), (1.0, -0.5, -0.4)):
        want = float(stats.multivariate_normal.cdf([a, b], mean=[0, 0],
                                                   cov=[[1, rho], [rho, 1]]))
        assert bvn_cdf(a, b, rho) == pytest.approx(want, rel=1e-9, abs=1e-15)
    assert bvn_cdf(0.0, 0.0, 0.5) == pytest.approx(0.25 + math.asin(0.5) / (2 * math.pi))
    assert bvn_cdf(1.0, 2.0, 1e-12) == pytest.approx(stats.norm.cdf(1) * stats.norm.cdf(2),
                                                     abs=1e-12)
    with pytest.raises(ValueError, match="rho must be"):
        bvn_cdf(0.0, 0.0, 1.0)


def test_the_bivariate_t_is_deterministic_where_scipys_is_not():
    x = float(stats.t.ppf(1e-4, DF))
    exact = bvt_cdf(x, x, RHO, DF)
    assert bvt_cdf(x, x, RHO, DF) == exact                       # bit-identical on a rerun
    assert exact > 0.0
    # it collapses to the normal one as df grows
    assert bvt_cdf(0.0, 0.0, RHO, 1e7) == pytest.approx(bvn_cdf(0.0, 0.0, RHO), abs=1e-6)
    rep = scipy_mvt_reproducibility(n_calls=5)
    assert len(set(rep["mvt_calls"])) > 1                        # scipy's is randomised
    assert len(set(rep["mvn_calls"])) == 1                       # scipy's normal one is not
    assert rep["mvn_maxabs"] / rep["mvn_exact"] < 1e-12
    with pytest.raises(ValueError, match="df must be positive"):
        bvt_cdf(0.0, 0.0, RHO, 0.0)


def test_every_copula_cdf_obeys_the_frechet_bounds_and_the_uniform_margins():
    for fam in FAMILIES:
        par, d = _pars(fam)
        for (u, v) in ((0.3, 0.7), (0.5, 0.5), (0.9, 0.2)):
            c = copula_cdf(fam, u, v, par, d)
            assert max(u + v - 1.0, 0.0) - 1e-9 <= c <= min(u, v) + 1e-9
        assert copula_cdf(fam, 1.0, 0.4, par, d) == pytest.approx(0.4, abs=1e-6)
        assert copula_cdf(fam, 0.4, 1.0, par, d) == pytest.approx(0.4, abs=1e-6)
    with pytest.raises(ValueError, match="must be in"):
        copula_cdf("gaussian", 1.5, 0.5, RHO)


def test_each_log_density_matches_a_finite_difference_of_its_own_cdf():
    h = 1e-4
    for fam in FAMILIES:
        par, d = _pars(fam)
        for (u, v) in ((0.3, 0.7), (0.9, 0.85)):
            num = (copula_cdf(fam, u + h, v + h, par, d) - copula_cdf(fam, u + h, v - h, par, d)
                   - copula_cdf(fam, u - h, v + h, par, d)
                   + copula_cdf(fam, u - h, v - h, par, d)) / (4 * h * h)
            ana = float(np.exp(copula_logpdf(fam, u, v, par, d)))
            assert ana == pytest.approx(num, rel=1e-6)


# ---------------------------------------------------------- the limit, evaluated exactly
def test_the_finite_u_ratio_converges_to_the_closed_form_tail_dependence():
    for fam in ("t", "clayton"):
        par, d = _pars(fam)
        lam = tail_dependence(fam, par, d)[0]
        vals = [tail_ratio(fam, u, par, d)[0] for u in (1e-2, 1e-4, 1e-6)]
        assert abs(vals[-1] - lam) < abs(vals[0] - lam)          # converging
        assert vals[-1] == pytest.approx(lam, abs=5e-4)
    par, _ = _pars("gumbel")
    up = [tail_ratio("gumbel", u, par)[1] for u in (1e-2, 1e-4, 1e-6)]
    assert up[-1] == pytest.approx(tail_dependence("gumbel", par)[1], abs=5e-5)
    # the Gaussian ratio decays to zero, but SLOWLY - the documented headline
    g = [tail_ratio("gaussian", u, RHO)[0] for u in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)]
    assert all(x > y for x, y in zip(g, g[1:]))
    assert g[0] > 0.12                                           # still 0.13 at u = 1%
    assert g[-1] < 0.01
    assert tail_ratio("clayton", 1e-3, 1.0)[1] < 1e-2            # Clayton has no upper tail
    with pytest.raises(ValueError, match="u must be"):
        tail_ratio("gaussian", 0.6, RHO)


# ----------------------------------------------------------------- sampling and fitting
def test_the_samplers_reproduce_their_own_kendall_tau_and_are_deterministic():
    for fam in FAMILIES:
        par, d = _pars(fam)
        uv = sample(fam, 60_000, par, d, seed=5)
        assert uv.shape == (60_000, 2)
        assert uv.min() > 0.0 and uv.max() < 1.0
        emp = float(stats.kendalltau(uv[:, 0], uv[:, 1]).statistic)
        assert emp == pytest.approx(TAU, abs=0.01)
        assert np.array_equal(uv, sample(fam, 60_000, par, d, seed=5))
        assert not np.array_equal(uv, sample(fam, 60_000, par, d, seed=6))
    # the independence cases the constructions must reproduce exactly
    ind = sample("gumbel", 40_000, 1.0, seed=1)                  # theta = 1 -> independence
    assert abs(float(stats.kendalltau(ind[:, 0], ind[:, 1]).statistic)) < 0.02
    with pytest.raises(ValueError, match="n must be"):
        sample("gaussian", 0, RHO)


def test_both_estimators_recover_the_parameter():
    for fam in FAMILIES:
        par, d = _pars(fam)
        uv = sample(fam, 20_000, par, d, seed=17)
        assert fit_by_tau(fam, uv[:, 0], uv[:, 1]) == pytest.approx(par, rel=0.06)
        assert fit_mle(fam, uv[:, 0], uv[:, 1], d) == pytest.approx(par, rel=0.06)


def test_sample_elliptical_has_the_requested_equicorrelation():
    for df in (None, DF):
        u = sample_elliptical(6, 60_000, 0.5, df, seed=8)
        z = stats.norm.ppf(u)
        c = np.corrcoef(z.T)
        off = c[~np.eye(6, dtype=bool)]
        assert off.mean() == pytest.approx(0.5, abs=0.03)
    with pytest.raises(ValueError, match="d >= 2"):
        sample_elliptical(1, 10, 0.5)
    with pytest.raises(ValueError, match="0 <= rho"):
        sample_elliptical(3, 10, 1.0)


# ------------------------------------------------------------------------------ the trap
def test_a_gaussian_copula_at_the_same_tau_understates_the_joint_tail_more_and_more():
    rows = joint_tail_study(n=200_000, seed=23)
    assert rows[0]["rho_hat"] == pytest.approx(RHO, abs=0.02)     # same correlation
    for r in rows:
        assert abs(r["empirical"] - r["t_model"]) < 4.0 * r["se"] + 2e-4
        assert r["gaussian"] < r["t_model"]                       # always understated
        assert r["gaussian"] > r["indep"]                         # but not independent either
    ratios = [r["ratio"] for r in rows]
    assert all(x < y for x, y in zip(ratios, ratios[1:]))          # worse the deeper you go
    assert ratios[-1] > 3.0


def test_the_gaussian_copula_overstates_the_diversification_benefit_at_every_level():
    pt = portfolio_tail(n=120_000, d=10, seed=29)
    for r in pt["rows"]:
        assert r["gaussian"] == pytest.approx(r["gaussian_exact"], rel=0.02)
        assert r["t"] > r["gaussian"]                              # fatter portfolio tail
        assert r["t_es"] > r["gaussian_es"]
        assert r["div_gaussian"] > r["div_t"]                      # the memo number
    deep = pt["rows"][-1]
    assert deep["ratio"] > pt["rows"][0]["ratio"]                  # worse deeper in the tail
    assert deep["es_ratio"] > deep["ratio"]                        # and worse for ES than VaR
    assert pt["sd_exact"] == pytest.approx(math.sqrt((1 + 9 * RHO) / 10))


@pytest.mark.slow
def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.microstructure.copulas")
    assert "Rule: correlation fixes the middle of the joint distribution" in out
    assert "0.253170" in out                                       # the t tail dependence
    assert out.isascii()
