"""fin_skills.ml.structural_breaks - the CUSUM event filter and the supremum ADF.

The properties the SKILL.md claims:

  * the CUSUM filter accumulates one-signed runs, resets only the side that fired, tests the
    negative side first, and accepts a per-bar threshold;
  * raising the threshold reduces the sampling rate, removes clustering, and increases the
    detection lag - all monotonically;
  * matched on sampling rate, CUSUM clusters far less than a naive |x| > h filter;
  * the fast SADF from cumulative cross-products equals a naive least-squares refit;
  * a full-sample ADF misses a planted bubble that SADF finds, SADF's flags start late and end
    long after the bubble, and its critical value must be simulated;
  * a per-bar exceedance rate is much smaller than the per-path rate the value was built from.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires
from fin_skills.ml import structural_breaks as sb


@pytest.fixture(scope="module")
def returns():
    return sb.drift_shift_series()


@pytest.fixture(scope="module")
def bubble():
    p = sb.bubble_series()
    return p, sb.sadf(p), sb.sadf_index(p.shape[0])


def test_cusum_fires_on_runs_and_resets_only_the_side_that_fired():
    # four +0.4 steps: the third crosses 1.0 and fires, then the accumulator restarts
    x = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    assert list(sb.cusum_filter(x, 1.0)) == [2, 5]
    # the same total move arriving as alternating steps never fires
    alt = np.array([0.4, -0.4] * 8)
    assert sb.cusum_filter(alt, 1.0).size == 0
    # the negative branch is tested first: a bar that would trip both is a downside event
    both = np.array([2.0, -5.0])
    assert list(sb.cusum_filter(both, 1.0)) == [0, 1]
    # only the firing side resets - the opposite accumulator keeps its state
    seq = np.array([1.2, -0.6, -0.7])
    assert list(sb.cusum_filter(seq, 1.0)) == [0, 2]
    # a per-bar threshold is accepted and used bar by bar
    x2 = np.array([0.9, 0.9])
    assert list(sb.cusum_filter(x2, np.array([2.0, 1.0]))) == [1]
    assert sb.cusum_filter(np.array([np.nan, 5.0]), 1.0).tolist() == [1]


def test_the_generator_is_seeded_and_plants_the_drift(returns):
    assert np.array_equal(returns, sb.drift_shift_series())
    assert not np.array_equal(returns, sb.drift_shift_series(seed=sb.SEED + 1))
    inside = returns[sb.SHIFT_AT:sb.SHIFT_AT + sb.SHIFT_LEN]
    outside = np.concatenate([returns[:sb.SHIFT_AT], returns[sb.SHIFT_AT + sb.SHIFT_LEN:]])
    assert inside.mean() - outside.mean() > 0.3 * sb.SIGMA
    assert abs(returns.std() - sb.SIGMA) < 0.1 * sb.SIGMA


def test_raising_the_threshold_samples_less_clusters_less_and_lags_more(returns):
    rates, clus, lags = [], [], []
    for h in sb.CUSUM_H:
        ev = sb.cusum_filter(returns, h * sb.SIGMA)
        rates.append(ev.shape[0] / returns.shape[0])
        clus.append(sb.clustering(ev))
        lags.append(sb.detection_lag(ev, sb.SHIFT_AT))
    assert rates == sorted(rates, reverse=True)
    assert clus == sorted(clus, reverse=True)
    assert rates[0] > 0.5 and rates[-1] < 0.10          # 65.8 % -> 5.7 % in the demo
    assert clus[-1] < 0.10                               # 1.8 %
    assert lags[-1] > lags[0]                            # 17 bars vs 0
    assert all(l >= 0 for l in lags)


def test_cusum_clusters_far_less_than_a_naive_threshold_filter(returns):
    naive = sb.threshold_filter(returns, 2.0 * sb.SIGMA)
    ev = sb.cusum_filter(returns, 5.0 * sb.SIGMA)
    assert abs(naive.shape[0] - ev.shape[0]) < 0.5 * ev.shape[0]     # comparable event counts
    assert sb.clustering(naive) > 5.0 * sb.clustering(ev)
    assert sb.clustering(np.array([1])) == 0.0
    assert sb.clustering(np.array([0, 1, 2])) == 1.0
    assert sb.clustering(np.array([0, 50, 100])) == 0.0
    assert sb.detection_lag(np.array([1, 2]), 99) == -1


def test_the_adf_design_is_what_it_says():
    p = np.arange(20.0) ** 1.1
    X, y = sb.adf_design(p, lags=2, trend="c")
    assert X.shape == (17, 4) and y.shape == (17,)
    assert np.allclose(X[:, 0], p[2:-1])                 # y_{t-1} first
    assert np.allclose(X[:, 1], 1.0)                     # then the constant
    assert np.allclose(y, np.diff(p)[2:])
    assert sb.adf_design(p, 1, "nc")[0].shape[1] == 2
    assert sb.adf_design(p, 1, "ct")[0].shape[1] == 4
    with pytest.raises(ValueError):
        sb.adf_design(p, 1, "cc")
    with pytest.raises(ValueError):
        sb.adf_design(p[:4], 1)


def test_the_fast_sadf_equals_a_naive_refit():
    p = sb.random_walk(200, sb.SEED)
    s = sb.sadf(p, lags=1, min_length=40)
    X, y = sb.adf_design(p, 1, "c")
    assert s.shape[0] == X.shape[0] == 198       # 200 bars, one diff, one lag
    for end in (60, 120, 197):
        naive = max(sb.adf_t_window(X, y, st, end) for st in range(0, end - 40 + 2))
        assert s[end] == pytest.approx(naive, abs=1e-9)
    assert np.isnan(s[:39]).all() and np.isfinite(s[39:]).all()
    assert np.array_equal(sb.sadf_index(200), np.arange(2, 200))


def test_sadf_finds_a_planted_bubble_that_a_full_sample_adf_misses(bubble):
    p, s, idx = bubble
    X, y = sb.adf_design(p)
    full = sb.adf_t_window(X, y, 0, X.shape[0] - 1)
    assert full > -2.0                                   # +1.278: nowhere near rejecting
    assert np.nanmax(s) > 5.0                            # 15.58
    peak = int(idx[int(np.nanargmax(s))])
    assert sb.BUBBLE_FROM <= peak <= sb.BUBBLE_TO + 20   # the peak is at the bubble's end


@requires("statsmodels")
def test_the_full_sample_adf_matches_statsmodels(bubble):
    p, _, _ = bubble
    sm = sb.statsmodels_adf(p)
    X, y = sb.adf_design(p)
    assert sm is not None
    assert sb.adf_t_window(X, y, 0, X.shape[0] - 1) == pytest.approx(sm[0], abs=1e-9)
    assert sm[0] > sm[1]                                 # no rejection of the unit root


def test_the_simulated_null_is_a_null_and_the_flags_overshoot(bubble):
    p, s, idx = bubble
    null = sb.sadf_null_critical(n=200, paths=12, min_length=40)
    c90, c95, c99 = null["crit"]
    assert c90 < c95 < c99
    assert null["maxima"].shape == (12,)
    assert np.mean(null["maxima"] > c95) == pytest.approx(1 / 12, abs=0.1)
    runs = sb.flagged_runs(s, idx, c95)
    assert runs
    assert runs[0][0] > sb.BUBBLE_FROM                   # detection lag: flags start late
    assert runs[-1][1] > sb.BUBBLE_TO + 50               # and they persist well past the end
    caught = sum(max(0, min(b, sb.BUBBLE_TO - 1) - max(a, sb.BUBBLE_FROM) + 1) for a, b in runs)
    assert caught > 0.5 * (sb.BUBBLE_TO - sb.BUBBLE_FROM)
    assert sb.flagged_runs(np.array([0.0, 5.0, 0.0, 5.0]), np.arange(4), 1.0) == [(1, 1), (3, 3)]
    assert sb.flagged_runs(np.array([5.0, 5.0]), np.arange(2), 1.0) == [(0, 1)]
    assert sb.flagged_runs(np.array([np.nan, 0.0]), np.arange(2), 1.0) == []


def test_the_per_bar_exceedance_rate_is_far_below_the_per_path_rate():
    null = sb.sadf_null_critical(n=200, paths=12, min_length=40)
    c95 = null["crit"][1]
    per_path = float(np.mean(null["maxima"] > c95))
    per_bar = float(np.mean(np.nan_to_num(null["curves"], nan=-np.inf) > c95))
    assert per_bar < 0.25 * per_path
    assert null["curves"].shape[0] == 12


def test_bubble_series_is_seeded_and_explosive_only_inside_the_window():
    p = sb.bubble_series()
    assert np.array_equal(p, sb.bubble_series())
    assert not np.array_equal(p, sb.bubble_series(seed=sb.SEED + 1))
    inside = np.abs(p[sb.BUBBLE_FROM:sb.BUBBLE_TO])
    before = np.abs(p[:sb.BUBBLE_FROM])
    assert inside.max() > 5.0 * max(before.max(), 1e-9)
    rw = sb.random_walk(100, 3)
    assert np.array_equal(rw, sb.random_walk(100, 3)) and rw.shape == (100,)


@pytest.mark.slow
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.ml.structural_breaks")
    for head in ("=== 1. The CUSUM filter as an event sampler", "=== 2. Detection lag",
                 "=== 3. SADF on a planted bubble", "=== 4. What a single full-sample ADF sees",
                 "=== 5. False positives on a null series"):
        assert head in out
    assert out.isascii()
    rule = [ln for ln in out.splitlines() if ln.startswith("Rule: ")]
    assert len(rule) == 1 and "CUSUM" in rule[0] and "SADF" in rule[0]
    assert out.strip().splitlines()[-1].startswith("total runtime")
