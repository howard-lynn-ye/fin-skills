"""fin_skills.models.kalman_models - predicted, filtered and smoothed states are not the same.

The properties the SKILL.md claims:

  * the numpy filter satisfies the recursions it says it does, and matches statsmodels;
  * changing the LAST observation moves earlier SMOOTHED states and no filtered or predicted
    one - the operational definition of "it saw the future";
  * a hedge on the smoothed or same-day-filtered beta drops BELOW the residual variance of the
    true beta, and the exact decomposition says which term did it;
  * the profile likelihood in q has a q -> 0 optimum that a bounded scalar search falls into;
  * statsmodels' RecursiveLS reports a CONSTANT "smoothed" coefficient (the full-sample OLS).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import requires
from fin_skills.models import kalman_models as km

SHORT = 400


@pytest.fixture(scope="module")
def data():
    return km.simulate(km.N, km.SEED, "break")


@pytest.fixture(scope="module")
def fit(data):
    return km.fit_local_level(data["y"], data["x"])


def test_simulation_is_seeded_and_the_three_shapes_differ():
    a = km.simulate(SHORT, km.SEED, "break")
    b = km.simulate(SHORT, km.SEED, "break")
    for k in ("y", "x", "beta", "eps"):
        assert np.array_equal(a[k], b[k])
    assert not np.array_equal(a["y"], km.simulate(SHORT, km.SEED + 1, "break")["y"])
    # y is exactly alpha + beta x + eps, so eps is the floor the hedge cannot beat
    assert np.allclose(a["y"], km.ALPHA + a["beta"] * a["x"] + a["eps"])
    drift = km.simulate(SHORT, km.SEED, "drift")["beta"]
    switch = km.simulate(SHORT, km.SEED, "switch", switch_every=50)["beta"]
    assert abs(np.corrcoef(drift, np.arange(SHORT))[0, 1]) > 0.9      # monotone
    assert np.sum(np.abs(np.diff(switch)) > 0.3) == SHORT // 50 - 1   # one jump per flip
    with pytest.raises(ValueError):
        km.simulate(SHORT, km.SEED, "wobble")
    with pytest.raises(ValueError):
        km.simulate(SHORT, km.SEED, "switch", switch_every=0)


def test_the_filter_satisfies_its_own_recursions(data):
    y, x = data["y"][:SHORT], data["x"][:SHORT]
    s = km.local_level_system(x, 1e-4, 2.5e-5)
    a0, P0star = km._prior(2, 1)
    kf = km.kalman_filter(y, s["Z"], s["T"], s["Q"], s["H"], a0, P0star * 2.5e-5)
    # T is the identity here, so a_{t+1|t} == a_{t|t}
    assert np.allclose(kf["predicted"][1:], kf["filtered"][:-1])
    # v_t is the one-step forecast error and F_t its variance
    v = y - np.einsum("tk,tk->t", s["Z"], kf["predicted"])
    assert np.allclose(kf["v"], v)
    F = np.einsum("tk,tkj,tj->t", s["Z"], kf["predicted_cov"], s["Z"]) + s["H"]
    assert np.allclose(kf["F"], F)
    assert kf["llf"] == pytest.approx(
        float(-0.5 * np.sum(np.log(2 * np.pi) + np.log(kf["F"]) + kf["v"] ** 2 / kf["F"])))
    with pytest.raises(ValueError):
        km.kalman_filter(y[:-1], s["Z"], s["T"], s["Q"], s["H"], a0, P0star)


def test_the_smoother_ends_at_the_filter_and_reads_the_future(data):
    y, x = data["y"][:SHORT], data["x"][:SHORT]
    s = km.local_level_system(x, 1e-4, 2.5e-5)
    a0, P0 = km._prior(2, 1)
    P0 = P0 * 2.5e-5

    def run(yy):
        kf = km.kalman_filter(yy, s["Z"], s["T"], s["Q"], s["H"], a0, P0)
        sm, _ = km.rts_smoother(s["T"], kf)
        return kf, sm

    kf, sm = run(y)
    assert np.allclose(sm[-1], kf["filtered"][-1])          # nothing after N to condition on
    shocked = y.copy()
    shocked[-1] += 0.20                                     # a 40-sigma print on the LAST day
    kf2, sm2 = run(shocked)
    assert np.array_equal(kf2["filtered"][:-1], kf["filtered"][:-1])
    assert np.array_equal(kf2["predicted"][:-1], kf["predicted"][:-1])
    moved = np.abs(sm2[:-1, 1] - sm[:-1, 1])
    assert moved.max() > 1e-3
    assert int(np.argmax(moved > 1e-9)) < SHORT - 50        # an estimate far in the PAST moved


def test_concentrating_h_out_gives_the_same_optimum_as_carrying_it():
    d = km.simulate(SHORT, km.SEED, "break")
    a0, P0star = km._prior(2, 1)
    ratio = 8.0
    s1 = km.local_level_system(d["x"], ratio, 1.0)
    kf1 = km.kalman_filter(d["y"], s1["Z"], s1["T"], s1["Q"], 1.0, a0, P0star)
    h, llf = km.concentrated_llf(kf1)
    assert h == pytest.approx(float(np.mean(kf1["v"] ** 2 / kf1["F"])))
    # the same model with H = h explicitly, and Q and P0 scaled by h, has that log-likelihood
    s2 = km.local_level_system(d["x"], ratio * h, h)
    kf2 = km.kalman_filter(d["y"], s2["Z"], s2["T"], s2["Q"], s2["H"], a0, P0star * h)
    assert kf2["llf"] == pytest.approx(llf, rel=1e-10)


def test_the_local_level_fit_recovers_the_observation_variance_and_is_deterministic():
    d = km.simulate(km.N, km.SEED, "break")
    f = km.fit_local_level(d["y"], d["x"])
    assert f["converged"] and f["search"] == "grid"
    assert f["h"] == pytest.approx(km.SIG_EPS ** 2, rel=0.10)
    assert f["q_beta"] > 0
    again = km.fit_local_level(d["y"], d["x"])
    assert again["q_beta"] == f["q_beta"] and again["llf"] == f["llf"]
    with pytest.raises(ValueError):
        km.fit_local_level(d["y"], d["x"], search="brent")


@pytest.mark.slow
def test_the_bounded_scalar_search_falls_into_the_q_to_zero_optimum():
    d = km.simulate(km.N, km.SEED, "switch", switch_every=km.SWITCH_EVERY)
    naive = km.fit_local_level(d["y"], d["x"], search="bounded")
    good = km.fit_local_level(d["y"], d["x"], search="grid")
    assert naive["converged"] and good["converged"]          # both report success
    assert good["profile_llf"] - naive["profile_llf"] > 50   # 74.7 nats on this series
    assert naive["q_beta"] < 1e-9 < good["q_beta"]
    flat = km.beta_series(naive)["predicted"][km.ROLL_WINDOW:]
    live = km.beta_series(good)["predicted"][km.ROLL_WINDOW:]
    assert flat.std(ddof=1) < 0.05 < live.std(ddof=1)        # the bad fit stopped tracking


def test_beta_series_names_and_alignment(fit):
    b = km.beta_series(fit)
    assert set(b) == {"predicted", "filtered", "smoothed"}
    assert all(v.shape == (km.N,) for v in b.values())
    assert np.allclose(b["predicted"][1:], b["filtered"][:-1])   # random-walk state, T = I


def test_the_smoothed_beta_anticipates_the_break_and_the_causal_ones_lag(data, fit):
    b = km.beta_series(fit)
    win = (km.BREAK_AT - km.BREAK_WINDOW, km.BREAK_AT + km.BREAK_WINDOW)
    ks = {n: km.best_alignment(b[n], data["beta"], window=win)
          for n in ("smoothed", "filtered", "predicted")}
    assert ks["smoothed"] > ks["filtered"] and ks["smoothed"] > ks["predicted"]
    assert ks["filtered"] < 0 and ks["predicted"] < 0
    cross = {n: km.break_crossing(b[n], km.BREAK_AT, km.BETA_BEFORE, km.BETA_AFTER)
             for n in b}
    assert cross["smoothed"] <= 0 < cross["predicted"]
    moved = {n: km.share_moved_before(b[n], km.BREAK_AT, km.BETA_BEFORE, km.BETA_AFTER) for n in b}
    assert moved["smoothed"] > 0.4 > moved["predicted"]
    with pytest.raises(ValueError):
        km.lead_lag_profile(b["smoothed"], data["beta"], window=(0, 50))


def test_break_crossing_returns_none_when_the_estimate_never_crosses():
    flat = np.ones(km.N)
    assert km.break_crossing(flat, km.BREAK_AT, km.BETA_BEFORE, km.BETA_AFTER) is None
    step = np.where(np.arange(km.N) < km.BREAK_AT, 1.0, 0.5)
    assert km.break_crossing(step, km.BREAK_AT, km.BETA_BEFORE, km.BETA_AFTER) == 0


def test_the_look_ahead_hedges_drop_below_the_true_beta_floor(data, fit):
    lad = km.ladder(data, fit)
    floor = lad[km.ORACLE]["resid_var"]
    assert lad[km.SMOOTHED]["resid_var"] < floor
    assert lad[km.FILTERED]["resid_var"] < floor
    assert lad[km.PREDICTED]["resid_var"] > floor
    assert lad[km.ROLLING]["resid_var"] > floor
    assert lad[km.ORACLE]["beta_rmse"] == 0.0
    assert lad[km.PREDICTED]["beta_rmse"] < lad[km.ROLLING]["beta_rmse"]
    assert lad["full-sample OLS (constant)"]["beta_rmse"] > lad[km.ROLLING]["beta_rmse"]


def test_the_variance_decomposition_is_exact_and_only_peekers_absorb_noise(data, fit):
    b = km.beta_series(fit)
    sl = slice(km.ROLL_WINDOW, None)
    lad = km.ladder(data, fit)
    floor = lad[km.ORACLE]["resid_var"]
    rolling = km.rolling_ols_beta(data["y"], data["x"], km.ROLL_WINDOW)
    for key, used in ((km.SMOOTHED, b["smoothed"]), (km.FILTERED, b["filtered"]),
                      (km.PREDICTED, b["predicted"]), (km.ROLLING, rolling)):
        na = km.noise_absorption(data["x"][sl], data["eps"][sl], data["beta"][sl], used[sl])
        assert na["total"] == pytest.approx(lad[key]["resid_var"] / floor - 1.0, abs=2e-3)
        assert na["mishedge_term"] > 0
    peek = [km.noise_absorption(data["x"][sl], data["eps"][sl], data["beta"][sl], u[sl])
            for u in (b["smoothed"], b["filtered"])]
    causal = [km.noise_absorption(data["x"][sl], data["eps"][sl], data["beta"][sl], u[sl])
              for u in (b["predicted"], rolling)]
    assert all(p["t_stat"] < -4.0 for p in peek)
    assert all(abs(c["t_stat"]) < 3.0 for c in causal)


def test_rolling_ols_is_causal_and_warms_up():
    d = km.simulate(SHORT, km.SEED, "break")
    out = km.rolling_ols_beta(d["y"], d["x"], 100)
    assert np.isnan(out[:100]).all() and np.isfinite(out[100:]).all()
    shocked = d["y"].copy()
    shocked[150] += 1.0
    out2 = km.rolling_ols_beta(shocked, d["x"], 100)
    assert out2[150] == out[150]                    # the shocked day is not in its own window
    assert out2[151] != out[151]


def test_the_local_linear_trend_wins_on_a_drifting_beta():
    d = km.simulate(km.N, km.SEED + 1, "drift")
    ll = km.fit_local_level(d["y"], d["x"])
    llt = km.fit_local_linear_trend(d["y"], d["x"])
    sl = slice(km.ROLL_WINDOW, None)
    err = [float(np.sqrt(np.mean((km.beta_series(f)["predicted"][sl] - d["beta"][sl]) ** 2)))
           for f in (ll, llt)]
    assert err[1] < err[0]
    assert llt["q_slope"] >= 0 and llt["converged"]


@requires("statsmodels")
def test_the_numpy_recursions_reproduce_statsmodels(data, fit):
    chk = km.statsmodels_check(data["y"], fit["system"], fit["a0"], fit["P0"])
    assert chk is not None
    for key in ("predicted", "filtered", "smoothed"):
        assert chk[key] < 1e-10, (key, chk[key])
    assert chk["llf"] < 1e-8


@requires("statsmodels")
def test_recursive_ls_smoothed_coefficients_are_the_full_sample_ols(data):
    rls = km.recursive_ls_check(data["y"], data["x"])
    assert rls is not None
    assert rls["smoothed_ptp"] < 1e-10          # constant across every date
    assert rls["smoothed_minus_ols"] < 1e-10    # ... and equal to the full-sample slope
    assert rls["filtered_ptp"] > 0.1            # the filtered series really does move
    assert rls["n_predicted"] == rls["nobs"] + 1


@pytest.mark.slow
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.models.kalman_models")
    for head in ("=== 1. Local-level fit", "=== 2. Which estimate knows the future?",
                 "=== 3. A beta hedge on each estimate", "=== 4. Local level vs local linear trend",
                 "=== 5. The profile likelihood in q is not unimodal",
                 "=== 5b. When the look-ahead actually pays"):
        assert head in out
    assert out.isascii()
    rule = [ln for ln in out.splitlines() if ln.startswith("Rule: ")]
    assert len(rule) == 1 and "PREDICTED" in rule[0]
    assert out.strip().splitlines()[-1].startswith("total runtime")
    assert math.isfinite(float(out.strip().splitlines()[-1].split()[-1].rstrip("s")))
