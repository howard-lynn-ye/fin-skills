"""fin_skills.ml.meta_labeling - the primary picks the side, the secondary decides whether to act.

The properties the SKILL.md claims:

  * the generator is seeded and its two states really do differ in volatility and in the sign of
    the momentum effect;
  * features are causal and the bets do not overlap, so the forward return uses no bar the
    features saw;
  * the numpy IRLS logistic reproduces scikit-learn's LogisticRegression, and standardisation
    uses training statistics only;
  * meta-labeling raises precision and lowers recall out of sample, and improves Sharpe after
    costs by more as costs rise;
  * training the secondary on the primary's own fitted rows raises its coverage, lowers its test
    precision, and destroys most of the Sharpe gain - while raising F1.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires
from fin_skills.ml import meta_labeling as ml


@pytest.fixture(scope="module")
def bets():
    log_ret, state = ml.regime_series()
    feat = ml.build_features(log_ret)
    t0, y_fwd, x_prim, x_meta = ml.make_bets(feat, log_ret)
    return log_ret, state, feat, t0, y_fwd, x_prim, x_meta


@pytest.fixture(scope="module")
def honest(bets):
    _, _, _, _, y_fwd, x_prim, x_meta = bets
    return ml.run_meta(x_prim, x_meta, y_fwd, leak=False)


@pytest.fixture(scope="module")
def leaked(bets):
    _, _, _, _, y_fwd, x_prim, x_meta = bets
    return ml.run_meta(x_prim, x_meta, y_fwd, leak=True)


def test_the_generator_is_seeded_and_the_two_states_differ(bets):
    log_ret, state, feat, *_ = bets
    r2, s2 = ml.regime_series()
    assert np.array_equal(log_ret, r2) and np.array_equal(state, s2)
    assert not np.array_equal(log_ret, ml.regime_series(seed=ml.SEED + 1)[0])
    assert 0.3 < float(np.mean(state == 1)) < 0.7
    v = feat["vol"]
    assert np.nanmean(v[state == 0]) > 1.2 * np.nanmean(v[state == 1])
    # momentum continues in state 1 and reverses in state 0
    m = feat["mom"]
    fwd = np.r_[log_ret[1:], np.nan]
    for s, sign in ((1, 1.0), (0, -1.0)):
        sel = (state == s) & np.isfinite(m) & np.isfinite(fwd)
        assert sign * np.corrcoef(m[sel], fwd[sel])[0, 1] > 0.02


def test_rolling_is_causal_and_features_do_not_see_the_future(bets):
    log_ret, _, feat, t0, y_fwd, *_ = bets
    r = ml.rolling(np.arange(10.0), 3, np.sum)
    assert np.isnan(r[:2]).all() and r[2] == 3.0 and r[9] == 24.0
    shocked = log_ret.copy()
    shocked[12_000] += 1.0
    f2 = ml.build_features(shocked)
    assert np.array_equal(f2["vol"][:12_000], feat["vol"][:12_000], equal_nan=True)
    # the forward return starts strictly after t0
    cum = np.concatenate([[0.0], np.cumsum(log_ret)])
    assert np.allclose(y_fwd, cum[t0 + ml.HOLD + 1] - cum[t0 + 1])
    assert (np.diff(t0) == ml.HOLD).all()          # non-overlapping bets


def test_the_logistic_fit_is_deterministic_and_standardises_on_training_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3)) * np.array([1.0, 50.0, 0.01])
    y = (X[:, 0] + 0.02 * X[:, 1] > 0).astype(float)
    m = ml.fit_logistic(X, y)
    assert np.array_equal(m["beta"], ml.fit_logistic(X, y)["beta"])
    assert np.allclose(m["mu"], X.mean(axis=0)) and np.allclose(m["sd"], X.std(axis=0))
    p = ml.predict_proba(m, X)
    assert ((p > 0.5) == (y > 0.5)).mean() > 0.9
    # scale invariance: standardising inside the fit makes the columns' units irrelevant
    p2 = ml.predict_proba(ml.fit_logistic(X * np.array([1.0, 100.0, 1.0]), y),
                          X * np.array([1.0, 100.0, 1.0]))
    assert np.max(np.abs(p - p2)) < 1e-6
    # a constant column does not divide by zero
    ml.fit_logistic(np.column_stack([X, np.ones(500)]), y)


@requires("sklearn")
def test_the_numpy_logistic_is_sklearns(bets):
    _, _, _, _, y_fwd, _, x_meta = bets
    d = ml.sklearn_logistic_check(x_meta[:2000], (y_fwd[:2000] > 0).astype(float))
    assert d is not None and d < 1e-6


def test_meta_labeling_raises_precision_and_lowers_recall(honest):
    p, m = honest["primary"], honest["meta"]
    assert p["recall"] == 1.0 and p["coverage"] == 1.0
    assert m["precision"] > p["precision"] + 0.03      # +0.079 in the demo
    assert m["recall"] < 0.9                           # 0.748
    assert 0.3 < m["coverage"] < 0.9                   # 66.6 %
    assert m["f1"] < p["f1"]                           # F1 goes DOWN, and that is fine
    assert honest["p_c"].min() >= 0.0 and honest["p_c"].max() <= 1.0
    assert np.array_equal(honest["acted"], honest["p_c"] > 0.5)


def test_the_sharpe_gain_widens_with_cost(honest):
    gains = []
    for c in (0.0, 0.005):
        s_p = ml.sharpe_after_cost(honest["side_c"], np.ones_like(honest["acted"]),
                                   honest["y_c"], c)
        s_m = ml.sharpe_after_cost(honest["side_c"], honest["acted"], honest["y_c"], c)
        assert s_m > s_p
        gains.append(s_m - s_p)
    assert gains[1] > gains[0]                         # +0.156 -> +0.382 in the demo
    assert np.isnan(ml.sharpe_after_cost(np.ones(5), np.zeros(5, dtype=bool), np.ones(5), 0.0))


def test_training_the_secondary_on_the_primarys_own_rows_disables_it(honest, leaked):
    assert leaked["primary_in_sample_rate"] > leaked["primary_honest_rate"] + 0.02
    assert leaked["meta_train_base_rate"] > honest["meta_train_base_rate"] + 0.02
    assert leaked["meta"]["coverage"] > honest["meta"]["coverage"] + 0.15   # 92.2 % vs 66.6 %
    assert leaked["meta"]["precision"] < honest["meta"]["precision"] - 0.03
    assert leaked["meta"]["f1"] > honest["meta"]["f1"]      # the leaked one WINS on F1
    s_h = ml.sharpe_after_cost(honest["side_c"], honest["acted"], honest["y_c"], 0.005)
    s_l = ml.sharpe_after_cost(leaked["side_c"], leaked["acted"], leaked["y_c"], 0.005)
    s_p = ml.sharpe_after_cost(honest["side_c"], np.ones_like(honest["acted"]),
                               honest["y_c"], 0.005)
    assert s_l < s_h
    assert (s_l - s_p) < 0.5 * (s_h - s_p)                 # 22 % of the gain in the demo
    # the primary itself is identical in both runs - only the meta training set changed
    assert np.array_equal(honest["side_c"], leaked["side_c"])


def test_prf_is_the_textbook_definition():
    good = np.array([True, True, False, False])
    acted = np.array([True, False, True, False])
    m = ml.prf(good, acted)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5) and m["n_acted"] == 2 and m["coverage"] == 0.5
    none = ml.prf(good, np.zeros(4, dtype=bool))
    assert np.isnan(none["precision"]) and none["recall"] == 0.0
    all_in = ml.prf(good, np.ones(4, dtype=bool))
    assert all_in["recall"] == 1.0 and all_in["precision"] == pytest.approx(0.5)


def test_raising_the_threshold_trades_coverage_for_precision(honest):
    prev_cov, prev_prec = 1.1, -1.0
    for thr in (0.40, 0.50, 0.60, 0.65):
        m = ml.prf(honest["good_c"], honest["p_c"] > thr)
        assert m["coverage"] < prev_cov
        assert m["precision"] > prev_prec
        prev_cov, prev_prec = m["coverage"], m["precision"]
    assert prev_prec > 0.80                                # 0.852 at 0.65 in the demo


def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.ml.meta_labeling")
    for head in ("=== 1. The data", "=== 2. Primary alone vs primary + meta",
                 "=== 3. The trap", "=== 4. Sharpe after costs",
                 "=== 5. The meta probability as a size"):
        assert head in out
    assert out.isascii()
    rule = [ln for ln in out.splitlines() if ln.startswith("Rule: ")]
    assert len(rule) == 1 and "OUT of sample" in rule[0] and "secondary" in rule[0]
    assert out.strip().splitlines()[-1].startswith("total runtime")
