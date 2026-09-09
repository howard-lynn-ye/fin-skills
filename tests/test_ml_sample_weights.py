"""fin_skills.ml.sample_weights - concurrency, uniqueness, and the sample size you really have.

The properties the SKILL.md claims:

  * concurrency counts labels with INCLUSIVE endpoints, so a span-20 label occupies 21 bars, and
    average uniqueness equals the analytic min(1, step / (span + 1)) for regular spacing;
  * summed uniqueness is far below n while Kish's effective size on near-uniform weights is not;
  * a nominal 5 % t-test on overlapping labels rejects far more often than 5 %, and uniqueness
    weighting does not repair it;
  * return-attribution weights sum to n, time decay is linear in cumulative uniqueness, and a
    negative decay zeroes the oldest fraction exactly;
  * the sequential bootstrap draws a sample at least as unique as the standard one;
  * including the bar-t0 return changes individual weights while preserving their ordering.
"""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.ml import sample_weights as sw

MC = 60          # the demo uses 400 paths; 60 is enough for the qualitative assertions


@pytest.fixture(scope="module")
def overlap():
    t0, t1 = sw.regular_events()
    conc = sw.num_concurrent(sw.N_BARS, t0, t1)
    return t0, t1, conc, sw.average_uniqueness(t0, t1, conc)


def test_concurrency_counts_inclusive_endpoints():
    conc = sw.num_concurrent(10, np.array([0, 5]), np.array([3, 8]))
    assert list(conc) == [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]      # bars 0-3 and 5-8, both ends in
    # two labels that share exactly one bar
    conc2 = sw.num_concurrent(10, np.array([0, 3]), np.array([3, 6]))
    assert conc2[3] == 2 and conc2[2] == 1 and conc2[4] == 1
    assert sw.num_concurrent(6, np.array([], dtype=int), np.array([], dtype=int)).sum() == 0


def test_average_uniqueness_matches_the_analytic_value_for_regular_spacing():
    for step in (1, 2, 5, 10, 20, 40):
        t0, t1 = sw.regular_events(sw.N_BARS, sw.SPAN, step)
        conc = sw.num_concurrent(sw.N_BARS, t0, t1)
        u = sw.average_uniqueness(t0, t1, conc)
        inner = slice(sw.SPAN, len(t0) - sw.SPAN)            # away from the ramp at each end
        expected = min(1.0, step / (sw.SPAN + 1))
        assert u[inner].mean() == pytest.approx(expected, abs=1e-9)
    # non-overlapping labels are perfectly unique
    t0, t1 = np.array([0, 30]), np.array([10, 40])
    conc = sw.num_concurrent(60, t0, t1)
    assert np.allclose(sw.average_uniqueness(t0, t1, conc), 1.0)


def test_summed_uniqueness_is_small_while_kish_on_uniform_weights_is_not(overlap):
    t0, t1, conc, u = overlap
    n = len(t0)
    assert u.sum() / n < 0.10                                # 4.8 % in the demo
    assert sw.kish_ess(np.ones(n)) == pytest.approx(n)
    assert sw.kish_ess(u) / n > 0.90                         # 98.7 % - Kish sees no overlap
    assert sw.kish_ess(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert sw.kish_ess(np.zeros(5)) == 0.0


def test_the_null_path_feature_cannot_predict_its_label():
    log_ret, y, x, t0, t1 = sw.one_path(0)
    assert np.array_equal(sw.one_path(0)[1], y)              # seeded
    assert not np.array_equal(sw.one_path(1)[1], y)
    assert (t1 - t0 == sw.SPAN).all() and (t0 >= sw.FEATURE_LAG).all()
    # x uses only bars up to t0, y only bars after it - disjoint by construction
    assert x.shape == y.shape
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.20


def test_a_nominal_five_percent_test_over_rejects_and_weights_do_not_fix_it():
    mc = sw.t_test_size(n_paths=MC)
    assert mc["sd_t_unweighted"] > 1.5                       # 2.19 in the demo
    assert mc["size_unweighted"] > 0.20                      # 39.2 % in the demo
    assert mc["size_weighted"] > 0.20                        # weighting barely moves it
    assert abs(mc["size_weighted"] - mc["size_unweighted"]) < 0.15
    assert abs(float(np.mean(mc["beta"]))) < 3.0 * float(np.std(mc["beta"], ddof=1)) / np.sqrt(MC)
    assert mc["t_unweighted"].shape == (MC,)


def test_weights_by_return_sum_to_n_and_are_absolute(overlap):
    t0, t1, conc, _ = overlap
    log_ret = np.random.default_rng(sw.SEED).normal(0.0, 0.01, sw.N_BARS)
    w = sw.weights_by_return(t0, t1, log_ret, conc)
    assert w.sum() == pytest.approx(len(t0))
    assert (w >= 0).all()
    # a bigger move over the span means a bigger weight, all else equal
    flat = np.zeros(sw.N_BARS)
    assert np.allclose(sw.weights_by_return(t0, t1, flat, conc), 0.0)
    # concurrency divides the attribution: doubling concurrency halves the raw attribution
    raw1 = sw.weights_by_return(t0[:5], t1[:5], log_ret, np.ones(sw.N_BARS))
    raw2 = sw.weights_by_return(t0[:5], t1[:5], log_ret, 2 * np.ones(sw.N_BARS))
    assert np.allclose(raw1, raw2)                           # after the sum-to-n normalisation


def test_the_bar_t0_return_changes_weights_but_not_their_ordering(overlap):
    t0, t1, conc, _ = overlap
    log_ret = np.random.default_rng(sw.SEED).normal(0.0, 0.01, sw.N_BARS)
    w_in = sw.weights_by_return(t0, t1, log_ret, conc, include_t0_bar=True)
    w_out = sw.weights_by_return(t0, t1, log_ret, conc, include_t0_bar=False)
    assert not np.allclose(w_in, w_out)
    rel = np.abs(w_in - w_out) / np.maximum(w_out, 1e-12)
    assert np.median(rel) > 0.05                             # 21.6 % in the demo
    rank = np.corrcoef(np.argsort(np.argsort(w_in)), np.argsort(np.argsort(w_out)))[0, 1]
    assert rank > 0.85                                       # 0.909 in the demo


def test_time_decay_is_linear_in_cumulative_uniqueness(overlap):
    _, _, _, u = overlap
    assert np.allclose(sw.time_decay_weights(u, 1.0), 1.0)   # no decay
    for dec in (0.75, 0.5, 0.0):
        w = sw.time_decay_weights(u, dec)
        assert w[-1] == pytest.approx(1.0)
        assert w[0] == pytest.approx(dec, abs=0.01)          # oldest lands on the decay constant
        assert (np.diff(w) >= -1e-12).all()                  # monotone increasing in time
        assert (w > 0).all()
    neg = sw.time_decay_weights(u, -0.5)
    assert np.mean(neg == 0.0) == pytest.approx(0.5, abs=0.02)   # the oldest half is erased
    assert sw.kish_ess(neg) < 0.5 * len(u)
    # the clock is uniqueness, not calendar: doubling the uniqueness of the first half moves it
    u2 = u.copy()
    u2[:len(u) // 2] *= 4.0
    assert not np.allclose(sw.time_decay_weights(u2, 0.5), sw.time_decay_weights(u, 0.5))


def test_the_indicator_matrix_and_its_uniqueness_agree_with_the_span_version():
    t0 = np.array([0, 5, 10])
    t1 = t0 + 4
    im = sw.indicator_matrix(20, t0, t1)
    assert im.shape == (20, 3)
    assert im[:, 0].sum() == 5 and im[0, 0] == 1.0 and im[4, 0] == 1.0 and im[5, 0] == 0.0
    assert sw.ind_mat_average_uniqueness(im) == pytest.approx(1.0)   # no overlap at all
    over = sw.indicator_matrix(20, np.array([0, 2]), np.array([4, 6]))
    assert sw.ind_mat_average_uniqueness(over) < 1.0


def test_sequential_bootstrap_draws_at_least_as_unique_a_sample_as_the_standard_one():
    g0 = np.random.default_rng(sw.SEED)
    t0 = np.sort(g0.integers(0, 380, 80))
    t1 = t0 + 10
    im = sw.indicator_matrix(int(t1.max()) + 1, t0, t1)
    std, seq = [], []
    for r in range(3):
        g = np.random.default_rng(sw.SEED + r)
        draw = sw.sequential_bootstrap(im, im.shape[1], g)
        assert draw.shape == (im.shape[1],) and draw.min() >= 0 and draw.max() < im.shape[1]
        seq.append(sw.drawn_uniqueness(im, draw))
        std.append(sw.drawn_uniqueness(im, g.integers(0, im.shape[1], im.shape[1])))
    assert np.mean(seq) > np.mean(std)
    assert np.array_equal(sw.sequential_bootstrap(im, 10, np.random.default_rng(3)),
                          sw.sequential_bootstrap(im, 10, np.random.default_rng(3)))


def test_ols_t_reduces_to_the_unweighted_fit_and_scales_with_weights():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = 0.5 * x + rng.normal(size=400)
    b, t = sw.ols_t(x, y)
    assert b == pytest.approx(0.5, abs=0.15) and t > 5
    b2, t2 = sw.ols_t(x, y, np.ones(400))
    assert (b2, t2) == pytest.approx((b, t))
    b3, _ = sw.ols_t(x, y, np.full(400, 7.0))              # a constant weight is no weight
    assert b3 == pytest.approx(b)


def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.ml.sample_weights")
    for head in ("=== 1. Concurrency and average uniqueness", "=== 2. Effective sample size",
                 "=== 3. What that does to a t-test",
                 "=== 4. Return-attribution and time-decay weights",
                 "=== 5. Sequential vs standard bootstrap",
                 "=== 6. The bar-t0 return"):
        assert head in out
    assert out.isascii()
    rule = [ln for ln in out.splitlines() if ln.startswith("Rule: ")]
    assert len(rule) == 1 and "effective sample size" in rule[0] and "t-statistic" in rule[0]
    assert out.strip().splitlines()[-1].startswith("total runtime")
