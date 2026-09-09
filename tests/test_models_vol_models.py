"""fin_skills.models.vol_models - percent units, range-estimator assumptions, HAR causality.

The properties the SKILL.md claims, each asserted rather than printed:

  * the numpy GARCH(1,1) recursion is arch's recursion (checked against a plain loop, and
    against arch itself when it is installed);
  * the estimator recovers a known (omega, alpha, beta) on a seeded series;
  * arch's DataScaleWarning fires on decimals and `res.scale` stays 1 - the default warns and
    does NOT rescale, which is the whole trap;
  * Rogers-Satchell is the drift-independent estimator, and three of the five break when the
    day gaps;
  * the HAR design matrix is causal.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from conftest import requires
from fin_skills.models import vol_models as vm


# ------------------------------------------------------------------------------ GARCH ------
@pytest.fixture(scope="module")
def garch_series():
    return vm.simulate_garch11(vm.N_GARCH, seed=vm.SEED, **vm.TRUE_GARCH)


def test_simulation_is_seeded_and_matches_its_unconditional_variance(garch_series):
    r, s2 = garch_series
    r2, s2b = vm.simulate_garch11(vm.N_GARCH, seed=vm.SEED, **vm.TRUE_GARCH)
    assert np.array_equal(r, r2) and np.array_equal(s2, s2b)
    assert not np.array_equal(r, vm.simulate_garch11(vm.N_GARCH, seed=vm.SEED + 1,
                                                     **vm.TRUE_GARCH)[0])
    t = vm.TRUE_GARCH
    uncond = t["omega"] / (1.0 - t["alpha"] - t["beta"])
    assert s2.mean() == pytest.approx(uncond, rel=0.10)     # 1.0 in percent^2 = 1 %/day
    with pytest.raises(ValueError):
        vm.simulate_garch11(100, mu=0.0, omega=0.02, alpha=0.5, beta=0.6)   # not stationary


def test_the_vectorised_recursion_is_the_loop(garch_series):
    r, _ = garch_series
    e = r - r.mean()
    omega, alpha, beta = 0.02, 0.08, 0.90
    bc = vm.arch_backcast(e)
    fast = vm.garch11_variance(e, omega, alpha, beta, bc)
    slow = np.empty_like(fast)
    slow[0] = omega + alpha * bc + beta * bc
    for t in range(1, len(e)):
        slow[t] = omega + alpha * e[t - 1] ** 2 + beta * slow[t - 1]
    assert np.abs(fast - slow).max() < 1e-12


def test_backcast_is_arch_s_0_94_weighted_mean_of_the_first_75_squares(garch_series):
    r, _ = garch_series
    e = r - r.mean()
    w = 0.94 ** np.arange(75)
    assert vm.arch_backcast(e) == pytest.approx(float((e[:75] ** 2 * (w / w.sum())).sum()))
    short = e[:10]
    assert vm.arch_backcast(short) == pytest.approx(vm.arch_backcast(short[:10]))


def test_the_estimator_recovers_the_true_variance_parameters(garch_series):
    r, s2 = garch_series
    fit = vm.fit_garch11(r)
    p, t = fit["params"], vm.TRUE_GARCH
    assert fit["converged"]
    assert abs(fit["persistence"] - (t["alpha"] + t["beta"])) < 0.02
    assert abs(p["alpha"] - t["alpha"]) < 0.02 and abs(p["beta"] - t["beta"]) < 0.02
    assert np.corrcoef(fit["sigma2"], s2)[0, 1] > 0.99
    # ... and does NOT recover the mean: it is inside two standard errors and no better
    se_mu = r.std(ddof=1) / math.sqrt(len(r))
    assert abs(p["mu"] - t["mu"]) < 2.0 * se_mu
    rel_mu = abs(p["mu"] / t["mu"] - 1.0)
    rel_persist = abs(fit["persistence"] / (t["alpha"] + t["beta"]) - 1.0)
    assert rel_mu > 0.5 and rel_persist < 0.01 and rel_mu > 50 * rel_persist
    with pytest.raises(ValueError):
        vm.fit_garch11(r[:10])


def test_the_fit_is_deterministic(garch_series):
    r, _ = garch_series
    a, b = vm.fit_garch11(r[:1200]), vm.fit_garch11(r[:1200])
    assert a["llf"] == b["llf"] and a["params"] == b["params"]


def test_multi_step_forecasts_converge_to_the_unconditional_variance():
    omega, alpha, beta = 0.02, 0.08, 0.90
    f = vm.garch11_forecast_variance(4.0, 4.0, omega, alpha, beta, horizon=2000)
    uncond = omega / (1.0 - alpha - beta)
    assert f[0] > uncond and f[-1] == pytest.approx(uncond, rel=1e-6)
    assert np.all(np.diff(f) <= 0)                      # monotone decay from above
    assert np.all(np.diff(f[:200]) < 0)                 # strictly, until it hits the fixed point
    half_life = math.log(0.5) / math.log(alpha + beta)  # 34.3 days at persistence 0.98
    assert f[int(half_life)] - uncond == pytest.approx(0.5 * (f[0] - uncond), rel=0.05)


def test_annualise_vol_is_where_the_100x_error_lives():
    daily_var_pct2 = 1.0                                # 1 %/day, fitted on returns x 100
    assert vm.annualise_vol(daily_var_pct2, scale=100.0) == pytest.approx(math.sqrt(252) / 100)
    wrong = vm.annualise_vol(daily_var_pct2, scale=1.0)
    assert wrong / vm.annualise_vol(daily_var_pct2, scale=100.0) == pytest.approx(100.0)


@requires("arch")
def test_the_numpy_mle_reproduces_arch(garch_series):
    r, _ = garch_series
    own = vm.fit_garch11(r)
    chk = vm.arch_crosscheck(r, own)
    assert chk is not None and chk["percent"]["converged"]
    assert max(abs(v) for v in chk["param_diff"].values()) < 1e-4
    assert abs(own["llf"] - chk["percent"]["llf"]) < 1e-5
    # the recursion itself, at arch's own parameters, to machine precision
    assert chk["max_abs_cond_var_diff_at_arch_params"] < 1e-10
    assert chk["llf_at_arch_params_ours"] == pytest.approx(chk["percent"]["llf"], abs=1e-6)


@requires("arch")
def test_decimals_warn_and_are_not_rescaled_but_rescale_true_multiplies_by_100(garch_series):
    from arch import arch_model
    from arch.utility.exceptions import DataScaleWarning

    r, _ = garch_series
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = arch_model(r / 100.0).fit(disp="off", show_warning=False)
    assert any(issubclass(c.category, DataScaleWarning) for c in caught)
    assert res.scale == 1.0                       # rescale=None WARNS and carries on

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        keep = arch_model(r / 100.0, rescale=True).fit(disp="off", show_warning=False)
        quiet = arch_model(r / 100.0, rescale=False).fit(disp="off", show_warning=False)
    assert keep.scale == 100.0
    assert quiet.scale == 1.0
    # rescale=True reproduces the percent fit; the default does not
    pct = vm.fit_garch11(r)["params"]
    assert float(keep.params["alpha[1]"]) == pytest.approx(pct["alpha"], abs=1e-4)
    assert abs(float(res.params["alpha[1]"]) - pct["alpha"]) > 1e-3


@requires("arch")
def test_the_default_decimal_fit_reports_its_starting_values(garch_series):
    r, _ = garch_series
    chk = vm.arch_crosscheck(r, vm.fit_garch11(r))
    moved = chk["decimal_default"]["moved_from_start"]
    assert max(abs(moved[k]) for k in ("omega", "alpha", "beta")) < 1e-6
    assert abs(chk["percent"]["moved_from_start"]["alpha"]) > 1e-3   # the percent fit did move
    assert chk["decimal_llf_deficit"] < -0.1        # and it is a WORSE optimum, flagged converged
    assert chk["decimal_default"]["converged"]


# ------------------------------------------------------------------------- range estimators -
@pytest.fixture(scope="module")
def ohlc_clean():
    return vm.simulate_ohlc(4200, 390, vm.ANNUAL_VOL, 0.0, 0.0, seed=vm.SEED)


def test_simulated_ohlc_is_internally_consistent(ohlc_clean):
    d = ohlc_clean
    assert np.all(d["H"] >= np.maximum(d["O"], d["C"]))
    assert np.all(d["L"] <= np.minimum(d["O"], d["C"]))
    assert np.allclose(d["C"][:-1], d["C_prev"][1:])
    r = np.log(d["C"] / d["C_prev"])
    assert r.var(ddof=1) == pytest.approx(d["true_var"], rel=0.05)
    with pytest.raises(ValueError):
        vm.simulate_ohlc(10, 10, 0.2, overnight_share=1.0)
    with pytest.raises(ValueError):
        vm.ohlc_components([1.0], [0.5], [0.4], [1.0], [1.0])       # high below the open


def test_every_estimator_is_near_unbiased_on_a_continuous_driftless_path(ohlc_clean):
    comp = vm.ohlc_components(**{k: ohlc_clean[k] for k in ("O", "H", "L", "C", "C_prev")})
    tab = vm.efficiency_table(vm.all_estimators(comp, vm.WINDOW), ohlc_clean["true_var"])
    assert set(tab) == set(vm.EST_NAMES)
    for name in vm.EST_NAMES:
        assert 0.85 < tab[name]["bias_ratio"] < 1.15, name
    for name in vm.EST_NAMES[1:]:
        assert tab[name]["efficiency"] > 3.0, name          # 5-8x in the SKILL.md table
        assert tab[name]["rmse_ratio"] < 0.7, name
    assert tab["close-to-close"]["efficiency"] == pytest.approx(1.0)


def test_the_discretisation_bias_shrinks_as_the_path_is_sampled_more_finely():
    coarse, fine = (vm.simulate_ohlc(600, steps, vm.ANNUAL_VOL, seed=vm.SEED)
                    for steps in (26, 3900))
    out = []
    for d in (coarse, fine):
        comp = vm.ohlc_components(**{k: d[k] for k in ("O", "H", "L", "C", "C_prev")})
        est = vm.all_estimators(comp, vm.WINDOW)
        out.append({k: v.mean() / d["true_var"] for k, v in est.items()})
    for name in ("Parkinson", "Garman-Klass", "Rogers-Satchell", "Yang-Zhang"):
        assert out[0][name] < 0.85, name                    # 26 steps/day: badly low
        assert out[1][name] > out[0][name] + 0.1, name      # and it walks back towards 1.0


def test_rogers_satchell_is_the_drift_independent_estimator():
    bias = {}
    for label, drift in (("flat", 0.0), ("trend", 2.0)):
        d = vm.simulate_ohlc(4200, 390, vm.ANNUAL_VOL, drift, 0.0, seed=vm.SEED)
        comp = vm.ohlc_components(**{k: d[k] for k in ("O", "H", "L", "C", "C_prev")})
        bias[label] = {k: v.mean() / d["true_var"]
                       for k, v in vm.all_estimators(comp, vm.WINDOW).items()}
    moved = {k: bias["trend"][k] - bias["flat"][k] for k in vm.EST_NAMES}
    assert abs(moved["Rogers-Satchell"]) < 0.02
    assert moved["Parkinson"] > 0.10                        # a trend inflates the range
    assert bias["trend"]["Parkinson"] > 1.0                 # ... past the true variance
    assert abs(moved["close-to-close"]) < 0.02              # ddof=1 removes the drift


def test_an_overnight_gap_breaks_three_of_the_five_and_yang_zhang_survives():
    d = vm.simulate_ohlc(4200, 390, vm.ANNUAL_VOL, 0.0, 0.30, seed=vm.SEED)
    comp = vm.ohlc_components(**{k: d[k] for k in ("O", "H", "L", "C", "C_prev")})
    bias = {k: v.mean() / d["true_var"] for k, v in vm.all_estimators(comp, vm.WINDOW).items()}
    for name in ("Parkinson", "Garman-Klass", "Rogers-Satchell"):
        assert bias[name] < 0.75, name                       # ~0.64: only the daytime variance
    assert bias["Yang-Zhang"] > 0.90
    assert bias["close-to-close"] > 0.90


def test_yang_zhang_k_matches_its_published_form():
    for n in (2, 5, 21, 100):
        assert vm.yang_zhang_k(n) == pytest.approx(0.34 / (1.34 + (n + 1) / (n - 1)))
    assert 0 < vm.yang_zhang_k(21) < 0.34
    with pytest.raises(ValueError):
        vm.yang_zhang_k(1)


# -------------------------------------------------------------------------------- HAR-RV ---
@pytest.fixture(scope="module")
def rv_series():
    return vm.simulate_realized_variance(1200, vm.HAR_STEPS, seed=vm.SEED)[0]


def test_the_har_design_matrix_is_causal(rv_series):
    X, y = vm.har_design(rv_series)
    L = max(vm.HAR_LAGS)
    assert X.shape == (len(rv_series) - L, len(vm.HAR_LAGS) + 1)
    assert np.array_equal(y, rv_series[L:])
    assert np.all(X[:, 0] == 1.0)
    # column k is the mean of the k values BEFORE the target - shifting the target must not
    # change the regressors
    shocked = rv_series.copy()
    shocked[L] *= 50.0                       # blow up the FIRST target
    X2, _ = vm.har_design(shocked)
    assert np.array_equal(X[0], X2[0])
    assert not np.array_equal(X[1], X2[1])   # ... it does enter the NEXT row's regressors
    for j, k in enumerate(vm.HAR_LAGS, start=1):
        assert X[0, j] == pytest.approx(rv_series[L - k:L].mean())
    with pytest.raises(ValueError):
        vm.har_design(rv_series[:5])


def test_har_beats_the_naive_and_trailing_mean_forecasts_out_of_sample(rv_series):
    oos = vm.har_out_of_sample(rv_series, 800)
    assert oos["n_test"] == len(rv_series) - 800
    assert oos["HAR"]["mse"] < oos["naive (yesterday)"]["mse"] < oos["22-day mean"]["mse"]
    assert oos["HAR"]["qlike"] < oos["naive (yesterday)"]["qlike"]
    assert oos["coef"][1:].sum() == pytest.approx(1.0, abs=0.2)   # near a unit root in variance


def test_har_forecast_is_the_design_row_dotted_with_the_coefficients(rv_series):
    coef = vm.har_fit(rv_series)
    X, _ = vm.har_design(rv_series)
    t = len(rv_series) - 1
    assert vm.har_forecast(coef, rv_series[:t]) == pytest.approx(float(X[-1] @ coef))


def test_qlike_is_zero_at_a_perfect_forecast(rv_series):
    assert vm.qlike(rv_series, rv_series) == pytest.approx(0.0, abs=1e-12)
    assert vm.qlike(rv_series * 1.5, rv_series) > 0
    assert vm.qlike(rv_series * 0.5, rv_series) > 0


@requires("arch")
def test_the_har_design_matches_arch_harx(rv_series):
    hc = vm.arch_har_crosscheck(rv_series)
    assert hc is not None and hc["max_abs_diff"] < 1e-10
    assert hc["names"] == ["Const", "y[0:1]", "y[0:5]", "y[0:22]"]


# ---------------------------------------------------------------------------------- demo ---
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.models.vol_models")
    for head in ("=== 1. GARCH(1,1) by maximum likelihood",
                 "=== 2. The same returns in decimals",
                 "=== 3. Range-based realized variance",
                 "=== 4. HAR-RV (Corsi 2009)"):
        assert head in out
    assert out.isascii()
    rule = [ln for ln in out.splitlines() if ln.startswith("Rule: ")]
    assert len(rule) == 1 and "res.scale" in rule[0] and "Yang-Zhang" in rule[0]
    assert out.strip().splitlines()[-1].startswith("total runtime")
