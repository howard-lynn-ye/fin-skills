"""fin_skills.microstructure.monte_carlo - converge to the RIGHT number.

Each test asserts the PROPERTY the SKILL.md documents: the standard error falls as 1/sqrt(N)
and each variance-reduction method beats plain by the documented order of magnitude; the
Longstaff-Schwartz eight-path example reproduces the paper's two printed regressions and its
.1144; a Bermudan tree, the paper's finite-difference column and LSM meet; footnote 9's
max-value rule is biased UP on the same paths; scipy's Sobol scrambles by default and its
unscrambled first point is -inf under norm.ppf; and a discretely monitored barrier price is
tens of standard errors from the continuous closed form no matter how many paths you throw
at it, while the Broadie-Glasserman-Kou shift removes it.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from fin_skills.microstructure.monte_carlo import (BGK_BETA, LS_AMERICAN, LS_COEF_T1,
                                                   LS_COEF_T2, LS_EUROPEAN, LS_PATHS,
                                                   LS_TABLE1, ZETA_HALF, asian_mc,
                                                   asian_qmc, asian_reference, barrier_mc,
                                                   bermudan_tree_put, bgk_corrected, bs_call,
                                                   bs_put, crr_american_put, design_matrix,
                                                   down_and_out_call, gbm_paths,
                                                   geometric_asian_call, ls_paper_example,
                                                   lsm_american_put, lsm_price, mc_stats,
                                                   option_models_cross_check, qmc_vs_mc,
                                                   sobol_first_point, sobol_normals,
                                                   variance_reduction_table)

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.25)


# ------------------------------------------------------------------------ the closed forms
def test_black_scholes_and_the_geometric_asian_behave():
    call, put = bs_call(**BASE), bs_put(**BASE)
    fwd = BASE["S"] * math.exp(-BASE["q"]) - BASE["K"] * math.exp(-BASE["r"])
    assert call - put == pytest.approx(fwd, abs=1e-12)
    assert bs_call(120.0, 100.0, 0.0, 0.05, 0.0, 0.25) == 20.0
    # a geometric average has less variance than the terminal value, so it is worth less
    geo = geometric_asian_call(m=12, **BASE)
    assert 0.0 < geo < call
    # one monitoring date IS the terminal value: the Asian collapses to the vanilla
    assert geometric_asian_call(m=1, **BASE) == pytest.approx(call, rel=1e-10)
    with pytest.raises(ValueError, match="m must be"):
        geometric_asian_call(m=0, **BASE)
    with pytest.raises(ValueError, match="at least 2"):
        mc_stats([1.0])


# ------------------------------------------------------------------- rate and reductions
def test_the_error_bar_falls_as_one_over_root_n():
    ses = [asian_mc(n=n, method="plain", seed=1)["se"] for n in (10_000, 40_000, 160_000)]
    assert ses[0] / ses[1] == pytest.approx(2.0, abs=0.08)
    assert ses[1] / ses[2] == pytest.approx(2.0, abs=0.08)


def test_each_variance_reduction_delivers_its_documented_order_of_magnitude():
    rows = {r["method"]: r for r in variance_reduction_table(60_000)}
    assert rows["plain"]["vrf"] == pytest.approx(1.0)
    assert 1.4 < rows["antithetic"]["vrf"] < 3.0
    assert rows["antithetic"]["units"] == 30_000            # PAIRS, not draws
    assert rows["control"]["vrf"] > 200.0                   # ~859x at 100,000
    assert 2.0 < rows["stratified"]["vrf"] < 8.0
    ref, ref_se = asian_reference(n_blocks=3, block=100_000, seed=4)
    for meth in ("plain", "antithetic", "control", "stratified"):
        r = rows[meth]
        assert abs(r["price"] - ref) < 4.0 * (r["se"] + ref_se)
    with pytest.raises(ValueError, match="method must be"):
        asian_mc(method="importance")


def test_a_seeded_run_is_deterministic():
    a = asian_mc(n=5_000, method="control", seed=3)
    assert a == asian_mc(n=5_000, method="control", seed=3)
    assert a["price"] != asian_mc(n=5_000, method="control", seed=4)["price"]


# ---------------------------------------------------------------- Longstaff-Schwartz
def test_the_eight_path_example_reproduces_both_printed_regressions_and_the_price():
    ex = ls_paper_example()
    for got, want in ((ex["coef_t2"], LS_COEF_T2), (ex["coef_t1"], LS_COEF_T1)):
        for g, w in zip(got, want):
            assert g == pytest.approx(w, abs=1e-3)          # the paper prints 3 decimals
    assert ex["price"] == pytest.approx(LS_AMERICAN, abs=5e-5)
    assert ex["european"] == pytest.approx(LS_EUROPEAN, abs=5e-5)
    assert ex["price"] > 2.0 * ex["european"] - 0.01        # "roughly twice"


def test_lsm_and_a_bermudan_tree_land_on_the_papers_finite_difference_column():
    for (S0, sg, T0), (fd, _eu, _sim, _se) in LS_TABLE1.items():
        tree = bermudan_tree_put(S0, 40.0, float(T0), 0.06, sg)
        assert tree == pytest.approx(fd, abs=2e-3)
        mine = lsm_american_put(S0, 40.0, float(T0), 0.06, sg, 40_000, 50, basis="laguerre")
        assert mine == pytest.approx(fd, abs=0.03)


def test_this_files_tree_matches_the_one_in_option_pricing_models():
    xc = option_models_cross_check()
    assert xc is not None, "fin_skills is importable inside the test suite by construction"
    for steps in (400, 800):
        assert xc[steps]["mine"] == pytest.approx(xc[steps]["theirs"], abs=1e-10)
    assert xc[800]["mine"] == pytest.approx(xc["documented_american_800"], abs=1e-6)


def test_a_continuously_exercisable_tree_is_worth_more_than_the_bermudan_one():
    for (S0, sg, T0) in ((36, 0.20, 1.0), (44, 0.20, 1.0)):
        berm = bermudan_tree_put(S0, 40.0, T0, 0.06, sg)
        cont = crr_american_put(S0, 40.0, T0, 0.06, 0.0, sg, 2000)
        assert cont > berm                                  # more exercise dates, more value
        assert 0.002 < cont - berm < 0.02                   # and the gap dwarfs the LSM s.e.
        eu = bs_put(S0, 40.0, T0, 0.06, 0.0, sg)
        assert berm > eu                                    # ... and both beat European
    with pytest.raises(ValueError, match="multiple of ex_every"):
        crr_american_put(36, 40.0, 1.0, 0.06, 0.0, 0.2, steps=99, ex_every=10)
    with pytest.raises(ValueError, match="outside"):
        crr_american_put(36, 40.0, 10.0, 0.5, 0.0, 0.05, steps=2)


def test_footnote_nine_the_max_value_rule_is_biased_up_on_the_very_same_paths():
    for (S0, sg) in ((36, 0.20), (40, 0.20), (44, 0.20)):
        kw = dict(n_paths=40_000, per_year=50, basis="laguerre")
        good = lsm_american_put(S0, 40.0, 1.0, 0.06, sg, **kw)
        bad = lsm_american_put(S0, 40.0, 1.0, 0.06, sg, rule="maxvalue", **kw)
        assert bad > good                                   # same paths, same seed, same fits
        assert bad > bermudan_tree_put(S0, 40.0, 1.0, 0.06, sg)
    with pytest.raises(ValueError, match="rule must be"):
        lsm_price(LS_PATHS, 1.10, 0.06, 1.0, rule="lookahead")


def test_regressing_on_all_paths_and_the_underflowing_basis_both_lose_real_money():
    tree = bermudan_tree_put(36, 40.0, 1.0, 0.06, 0.20)
    kw = dict(n_paths=60_000, per_year=50)
    paper = lsm_american_put(36, 40.0, 1.0, 0.06, 0.20, basis="laguerre", **kw)
    allp = lsm_american_put(36, 40.0, 1.0, 0.06, 0.20, basis="laguerre", itm_only=False, **kw)
    raw = lsm_american_put(36, 40.0, 1.0, 0.06, 0.20, basis="laguerre_raw", **kw)
    assert abs(paper - tree) < 0.01                         # the paper's recipe is right
    assert allp < tree - 0.02 and raw < tree - 0.02         # both lose 5-7 cents, silently
    assert abs(allp - tree) > 5.0 * abs(paper - tree)
    # the raw basis underflows: exp(-S/2) at S = 40 is 2e-9, so the columns vanish
    A = design_matrix(np.array([36.0, 40.0, 44.0]), "laguerre_raw")
    assert np.abs(A[:, 1:]).max() < 1e-4                     # against a constant column of 1
    assert np.linalg.cond(A.T @ A) > 1e12                    # numerically an intercept only
    assert np.abs(design_matrix(np.array([36.0, 40.0]), "laguerre", 40.0)[:, 1:]).max() > 0.3
    with pytest.raises(ValueError, match="basis must be"):
        design_matrix(np.array([1.0]), "chebyshev")


def test_the_out_of_sample_check_agrees_with_the_in_sample_one():
    kw = dict(n_paths=50_000, per_year=50, basis="laguerre")
    ins = lsm_american_put(36, 40.0, 1.0, 0.06, 0.20, **kw)
    oos = lsm_american_put(36, 40.0, 1.0, 0.06, 0.20, out_of_sample=True, **kw)
    assert abs(oos - ins) < 0.02                            # "virtually identical" (Table 2)
    # the low bias is an EXPECTATION property of the out-of-sample estimator, not a per-sample
    # guarantee: at 50,000 paths the Monte Carlo noise is several times the bias, so the test
    # asserts the bracket rather than a strict inequality
    berm = bermudan_tree_put(36, 40.0, 1.0, 0.06, 0.20)
    assert bs_put(36, 40.0, 1.0, 0.06, 0.0, 0.20) < oos < berm + 0.03
    with pytest.raises(ValueError, match="at least one exercise date"):
        lsm_price(np.ones((4, 1)), 1.0, 0.05, 1.0)


def test_gbm_paths_are_antithetic_and_start_at_the_spot():
    p = gbm_paths(100.0, 1.0, 0.05, 0.2, 1000, 10, seed=2)
    assert p.shape == (1000, 11)
    assert np.all(p[:, 0] == 100.0)
    logs = np.log(p[:, 1:] / 100.0)
    assert logs[:500].sum() + logs[500:].sum() == pytest.approx(
        2 * (0.05 - 0.5 * 0.04) * 0.1 * np.arange(1, 11).sum() * 500, rel=1e-9)


# --------------------------------------------------------------------- quasi-Monte Carlo
def test_sobol_scrambles_by_default_and_the_unscrambled_first_point_is_minus_infinity():
    sf = sobol_first_point()
    assert np.all(sf["unscrambled_first"] == 0.0)
    assert np.all(np.isneginf(sf["unscrambled_ppf"]))
    assert np.all((sf["scrambled_first"] > 0.0) & (sf["scrambled_first"] < 1.0))
    assert any("power of 2" in m for m in sf["non_power_of_two_warning"])
    z = sobol_normals(3, 8, scramble=True, seed=1)
    assert z.shape == (256, 3) and np.all(np.isfinite(z))
    assert abs(float(z.mean())) < 0.05                       # a balanced sample of N(0,1)
    assert float(z.std()) == pytest.approx(1.0, abs=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # random_base2 must not warn
        sobol_normals(2, 6, seed=3)


def test_qmc_beats_plain_mc_at_the_same_budget_and_converges_faster():
    rows = qmc_vs_mc(powers=(10, 12, 14), n_reps=6, seed=13)
    for r in rows:
        assert r["qmc_rmse"] < r["mc_rmse"] / 3.0
    assert rows[0]["mc_slope"] == pytest.approx(-0.5, abs=0.15)
    assert rows[0]["qmc_slope"] < rows[0]["mc_slope"]        # steeper, i.e. faster
    # randomised QMC: the error bar comes from independent SCRAMBLES, not from the points
    price, se = asian_qmc(12, n_scrambles=8, seed=13)
    ref, ref_se = asian_reference(n_blocks=3, block=100_000, seed=17)
    assert abs(price - ref) < 4.0 * (se + ref_se)
    assert se > 0.0


# --------------------------------------------------------- the bias the se cannot see
def test_the_continuous_barrier_closed_form_is_bracketed_correctly():
    exact = down_and_out_call(**BASE, H=90.0)
    vanilla = bs_call(**BASE)
    assert 0.0 < exact < vanilla
    # as the barrier vanishes the knock-out risk does too
    assert down_and_out_call(**BASE, H=1.0) == pytest.approx(vanilla, abs=1e-6)
    assert down_and_out_call(**BASE, H=99.0) < exact         # a nearer barrier is worth less
    for bad in (0.0, 100.0, 120.0):
        with pytest.raises(ValueError, match="H < min"):
            down_and_out_call(**BASE, H=bad)


def test_discrete_monitoring_is_a_bias_that_paths_cannot_touch():
    exact = down_and_out_call(**BASE, H=90.0)
    coarse = barrier_mc(m=52, n=150_000, seed=5)
    fine = barrier_mc(m=52, n=600_000, seed=5)
    assert coarse["price"] > exact and fine["price"] > exact          # discrete knocks out less
    assert fine["se"] == pytest.approx(coarse["se"] / 2.0, rel=0.2)   # 4x paths, half the bar
    assert (fine["price"] - exact) / fine["se"] > 20.0               # ... and MORE sigmas
    assert (fine["price"] - exact) / fine["se"] > (coarse["price"] - exact) / coarse["se"]
    # monitoring dates are what close it, at the 1/sqrt(m) rate
    errs = [barrier_mc(m=m, n=150_000, seed=5)["price"] - exact for m in (12, 52, 252)]
    assert errs[0] > errs[1] > errs[2] > 0.0
    assert errs[0] / errs[2] == pytest.approx(math.sqrt(252 / 12), rel=0.3)
    with pytest.raises(ValueError, match="m >= 1"):
        barrier_mc(m=0)


def test_the_bgk_continuity_correction_removes_the_bias_and_its_constant_is_right():
    assert ZETA_HALF == pytest.approx(-1.4603545088095866, abs=1e-12)
    assert BGK_BETA == pytest.approx(0.5826, abs=1e-4)
    assert BGK_BETA == pytest.approx(-ZETA_HALF / math.sqrt(2 * math.pi), rel=1e-15)
    exact = down_and_out_call(**BASE, H=90.0)
    for m in (12, 52, 252):
        d = barrier_mc(m=m, n=150_000, seed=5)
        tgt = bgk_corrected(m=m)
        assert tgt > exact                                   # the shift moves the barrier away
        assert abs(d["price"] - tgt) < 2.5 * d["se"]         # within a couple of sigma
        assert abs(d["price"] - tgt) < abs(d["price"] - exact) / 3.0
    # the shifted barrier converges back to H, so the correction converges to the closed form
    far = [abs(bgk_corrected(m=mm) - exact) for mm in (10**4, 10**6, 10**8)]
    assert all(x > y for x, y in zip(far, far[1:])) and far[-1] < 1e-3


@pytest.mark.slow
def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.microstructure.monte_carlo")
    assert "Rule: a standard error measures how much the ESTIMATOR wobbles" in out
    assert "paper: -1.070 +2.983 X -1.813 X^2" in out
    assert out.isascii()
