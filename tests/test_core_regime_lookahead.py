"""fin_skills.core.regime_lookahead - smoothed uses the future, filtered uses today, predicted trades.

The numpy Hamilton filter / Kim smoother are tested at the true parameters (no estimation).
Everything that fits a Markov-switching model needs statsmodels and is skipped without it;
the full demo (~40-75 s) and the strategy ladder (~7 s of walk-forward refits) are slow.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires
from fin_skills.core import regime_lookahead as rl

LO, HI = rl.TEST_START, rl.N


@pytest.fixture(scope="module")
def dgp():
    return rl.simulate(rl.N, rl.SEED, rl.MAIN_RATIO)


@pytest.fixture(scope="module")
def probs(dgp):
    r, _ = dgp
    return rl.true_param_probs(r, rl.MAIN_RATIO)


def test_simulation_is_seeded_and_the_turbulent_regime_is_the_volatile_one(dgp):
    r, s = dgp
    r2, s2 = rl.simulate(rl.N, rl.SEED, rl.MAIN_RATIO)
    assert np.array_equal(r, r2) and np.array_equal(s, s2)
    assert not np.array_equal(r, rl.simulate(rl.N, rl.SEED + 1, rl.MAIN_RATIO)[0])
    assert set(np.unique(s)) == {0, 1}
    assert r[s == 1].std() > 1.5 * r[s == 0].std()
    assert 0.1 < s.mean() < 0.4


def test_all_three_probability_series_are_probabilities(probs):
    for name in ("smoothed", "filtered", "predicted"):
        p = probs[name]
        assert p.shape == (rl.N,) and (p >= 0).all() and (p <= 1).all()
    # predicted[0] is the ergodic distribution of the chain
    pi_calm = (1 - rl.P_STAY[1]) / (2 - rl.P_STAY[0] - rl.P_STAY[1])
    assert probs["predicted"][0] == pytest.approx(pi_calm)


def test_smoothed_sees_the_future_filtered_and_predicted_do_not(dgp, probs):
    r, _ = dgp
    shocked = r.copy()
    shocked[-1] = 0.10                                  # a 10-sigma print on the LAST day
    p2 = rl.true_param_probs(shocked, rl.MAIN_RATIO)
    assert np.array_equal(p2["filtered"][:-1], probs["filtered"][:-1])
    assert np.array_equal(p2["predicted"], probs["predicted"])
    moved = np.abs(p2["smoothed"][:-1] - probs["smoothed"][:-1])
    assert moved.max() > 0.1
    assert int(np.argmax(moved > 1e-6)) < rl.N - 1     # an EARLIER value changed


def test_predicted_is_the_transition_of_yesterdays_filtered_and_smoothed_ends_at_filtered(dgp):
    r, _ = dgp
    P = np.array([[rl.P_STAY[0], 1 - rl.P_STAY[1]], [1 - rl.P_STAY[0], rl.P_STAY[1]]])
    mu = np.asarray(rl.MU)
    sig2 = np.array([rl.SIG_CALM, rl.SIG_CALM * rl.MAIN_RATIO]) ** 2
    pi0 = np.array([0.5, 0.5])
    pred, filt = rl.hamilton_filter(r, P, mu, sig2, pi0)
    assert np.allclose(pred[1:], filt[:-1] @ P.T)
    assert np.allclose(pred.sum(axis=1), 1.0) and np.allclose(filt.sum(axis=1), 1.0)
    sm = rl.kim_smoother(P, pred, filt)
    assert np.allclose(sm[-1], filt[-1]) and np.allclose(sm.sum(axis=1), 1.0)


def test_detection_scores_the_oracle_and_a_lagged_oracle_as_documented(dgp):
    r, s = dgp
    oracle = (s == 0).astype(float)
    d = rl.detection(s, oracle, LO, HI)
    assert d["median_delay"] == 0 and d["share_anticipated"] == 0
    assert d["missed"] == 0 and d["false_alarms"] == 0
    assert d["n_switches"] == len(rl.switch_points(s, LO, HI))
    lag = np.r_[1.0, (s[:-1] == 0)]
    assert rl.detection(s, lag, LO, HI)["median_delay"] == 1
    assert rl.label_accuracy(s, oracle, LO, HI) == 1.0
    assert rl.label_accuracy(s, lag, LO, HI) < 1.0


def test_smoothed_anticipates_switches_that_predicted_can_only_follow(dgp, probs):
    _, s = dgp
    sm = rl.detection(s, probs["smoothed"], LO, HI)
    fi = rl.detection(s, probs["filtered"], LO, HI)
    pr = rl.detection(s, probs["predicted"], LO, HI)
    assert sm["share_anticipated"] > pr["share_anticipated"]
    assert sm["median_delay"] < fi["median_delay"] <= pr["median_delay"]
    acc = [rl.label_accuracy(s, probs[k], LO, HI) for k in ("smoothed", "filtered", "predicted")]
    assert acc == sorted(acc, reverse=True)


def test_strategy_stats_conventions(dgp):
    r, _ = dgp
    st = rl.strategy_stats(r[LO:HI], np.ones(HI - LO))
    assert st["in_mkt"] == 1.0 and st["switches"] == 0 and st["maxdd"] <= 0
    assert st["sharpe"] == pytest.approx(r[LO:HI].mean() / r[LO:HI].std(ddof=1) * np.sqrt(252))
    flat = rl.strategy_stats(r[LO:HI], np.zeros(HI - LO))
    assert np.isnan(flat["sharpe"]) and flat["cagr"] == 0.0


@requires("statsmodels")
def test_numpy_recursions_reproduce_statsmodels_at_the_fitted_parameters(dgp):
    r, _ = dgp
    res, llfs = rl.fit_ms(r[:800], starts=((0, 1),))
    assert len(llfs) == 1 and np.isfinite(res.llf)
    chk = rl.verify_reference(res)
    assert max(chk.values()) < 1e-8
    calm = rl.calm_index(res)
    names = list(res.model.param_names)
    s2 = [res.params[names.index(f"sigma2[{k}]")] for k in range(2)]
    assert s2[calm] == min(s2)
    series = rl.p_calm_series(res, calm)
    assert set(series) == {"smoothed", "filtered", "predicted"}
    assert all(0 <= v.min() and v.max() <= 1 and v.shape == (800,) for v in series.values())


@requires("statsmodels")
def test_walk_forward_uses_only_parameters_fitted_before_the_block(dgp):
    r, _ = dgp
    out, n_fits = rl.walk_forward_predicted(r[:1300], start=800, step=250)
    assert n_fits == 2
    assert np.isnan(out[:800]).all()
    tail = out[800:]
    assert np.isfinite(tail).all() and (tail >= 0).all() and (tail <= 1).all()


@requires("statsmodels")
@pytest.mark.slow
def test_ladder_orders_the_signals_by_how_much_future_they_use(dgp):
    r, s = dgp
    tab, info = rl.ladder(r, s, LO, HI)
    assert "walk-forward predicted" in tab.index and info["n_fits"] == 7
    assert max(info["ref_check"].values()) < 1e-8
    assert tab.loc["smoothed", "anticipated"] > tab.loc["predicted", "anticipated"]
    assert tab.loc["oracle (true regime today)", "sharpe"] > tab.loc["buy & hold", "sharpe"]


@pytest.mark.slow
def test_demo_runs_every_section_and_reports_its_runtime(run_main):
    out = run_main("fin_skills.core.regime_lookahead")
    assert "=== 1. Strategy ladder" in out and "=== 3. Sweep" in out
    if rl.HAVE_SM:
        assert "=== 4. Things the fit does that nothing warns you about ===" in out
    assert out.strip().splitlines()[-1].startswith("total runtime")
