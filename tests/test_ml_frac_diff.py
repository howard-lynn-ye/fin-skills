"""fin_skills.ml.frac_diff - the weight recursion, the fixed window, and the memory d=1 deletes.

The properties the SKILL.md claims:

  * the recursion w_k = -w_{k-1}(d-k+1)/k IS the closed form (-1)^k C(d,k), and d=1 is exactly
    the first difference while d=0 is the identity;
  * the fixed-width truncation reproduces mlfinpy 0.1.2's `get_weights_ffd` (transcribed in the
    module) up to its oldest-first column orientation;
  * the filter is causal - a shock at t+1 cannot move the output at t - and equals the naive
    loop over the window;
  * the ADF t-statistic is statsmodels' adfuller statistic;
  * on the seeded near-unit-root price the level does NOT reject, some minimum d does, and the
    correlation with the level is still high there;
  * over a panel of paths the information coefficient survives fractional differencing and is
    destroyed by first differences;
  * the weight threshold changes the window width and therefore the feature.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import requires
from fin_skills.ml import frac_diff as fd

PANEL_PATHS = 12          # the demo uses 50; 12 is enough for the qualitative assertions


@pytest.fixture(scope="module")
def price():
    return fd.near_unit_root_price()


def test_the_recursion_is_the_closed_form_binomial():
    for d in (0.0, 0.05, 0.15, 0.4, 0.75, 0.999):
        w = fd.expanding_weights(d, 500)
        cf = fd.weights_closed_form(d, 500)
        assert np.max(np.abs(w - cf)) < 1e-12
        assert w[0] == 1.0
    assert np.array_equal(fd.expanding_weights(1.0, 5), np.array([1.0, -1.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(np.abs(fd.expanding_weights(0.0, 5)), np.array([1.0, 0, 0, 0, 0]))
    with pytest.raises(ValueError):
        fd.expanding_weights(0.4, 0)


def test_ffd_truncation_matches_the_transcribed_mlfinpy_function():
    for d in (0.05, 0.2, 0.35, 0.5, 0.75, 1.0):
        for thresh in (1e-2, 1e-3, 1e-4):
            w = fd.ffd_weights(d, thresh)
            ref = fd.mlfinpy_get_weights_ffd(d, thresh, fd.MAX_WIDTH)
            assert ref.shape == (len(w), 1)                 # oldest first, as a column
            assert np.max(np.abs(w - ref.ravel()[::-1])) == 0.0
            assert abs(w[-1]) >= thresh or len(w) == 1      # nothing under the threshold kept
            # the NEXT weight the recursion would produce is the one that broke the loop
            nxt = -w[-1] * (d - len(w) + 1.0) / len(w)
            assert abs(nxt) < thresh


def test_the_window_shrinks_and_sum_of_weights_goes_to_zero_with_d():
    widths = [len(fd.ffd_weights(d)) - 1 for d in (0.35, 0.5, 0.75, 1.0)]
    assert widths == sorted(widths, reverse=True)
    sums = [fd.ffd_weights(d).sum() for d in (0.05, 0.2, 0.35, 0.5, 0.75)]
    assert all(a > b > 0 for a, b in zip(sums, sums[1:]))
    assert fd.ffd_weights(1.0).sum() == 0.0                 # d=1 keeps no level at all


def test_frac_diff_ffd_is_the_naive_loop_and_is_causal(price):
    x = price[:400]
    d, thresh = 0.35, 1e-2
    w = fd.ffd_weights(d, thresh)
    width = len(w) - 1
    y = fd.frac_diff_ffd(x, d, thresh)
    assert np.isnan(y[:width]).all() and np.isfinite(y[width:]).all()
    naive = np.array([float(np.dot(w, x[t::-1][:len(w)])) for t in range(width, len(x))])
    assert np.max(np.abs(y[width:] - naive)) < 1e-10
    # a shock in the future cannot move the past
    shocked = x.copy()
    shocked[300] += 5.0
    y2 = fd.frac_diff_ffd(shocked, d, thresh)
    assert np.array_equal(y2[:300], y[:300], equal_nan=True)
    assert y2[300] != y[300]
    # too few observations for one window -> all NaN, no exception
    assert np.isnan(fd.frac_diff_ffd(x[:width], d, thresh)).all()


def test_d_equals_one_is_the_first_difference(price):
    y = fd.frac_diff_ffd(price, 1.0, 1e-2)
    assert np.max(np.abs(y[1:] - np.diff(price))) < 1e-12
    y0 = fd.frac_diff_ffd(price, 0.0, 1e-2)
    assert np.max(np.abs(y0 - price)) < 1e-12


def test_adf_tstat_rejects_a_bad_trend_and_needs_data(price):
    with pytest.raises(ValueError):
        fd.adf_tstat(price, trend="ct")
    with pytest.raises(ValueError):
        fd.adf_tstat(price[:8])
    assert fd.adf_tstat(price, trend="n") != fd.adf_tstat(price, trend="c")


@requires("statsmodels")
def test_adf_tstat_is_statsmodels_adfuller(price):
    for trend in ("c", "n"):
        sm = fd.statsmodels_adf(price, fd.ADF_LAGS, trend)
        assert sm is not None
        assert fd.adf_tstat(price, fd.ADF_LAGS, trend) == pytest.approx(sm[0], abs=1e-9)
    assert fd.statsmodels_adf(price)[1] == pytest.approx(fd.ADF_CRIT_5, abs=0.02)


def test_the_generator_is_seeded_and_near_unit_root(price):
    assert np.array_equal(price, fd.near_unit_root_price())
    assert not np.array_equal(price, fd.near_unit_root_price(seed=fd.SEED + 1))
    assert price.shape == (fd.T_OBS,)
    phi_hat = float(np.corrcoef(price[:-1], price[1:])[0, 1])
    assert phi_hat > 0.99
    assert fd.adf_tstat(price) > fd.ADF_CRIT_5              # the level does NOT reject


def test_the_scan_trades_stationarity_against_memory(price):
    rows = fd.scan_one(price)
    by_d = {r["d"]: r for r in rows}
    assert by_d[0.0]["corr_level"] == pytest.approx(1.0, abs=1e-9)
    assert by_d[0.0]["adf"] > fd.ADF_CRIT_5
    assert by_d[1.0]["adf"] < -20.0
    assert abs(by_d[1.0]["corr_level"]) < 0.10             # -0.034 in the demo
    # the ADF statistic falls monotonically in d, which is what the bisection assumes
    adfs = [r["adf"] for r in rows]
    assert adfs == sorted(adfs, reverse=True)
    corrs = [r["corr_level"] for r in rows]
    assert corrs == sorted(corrs, reverse=True)


def test_min_d_is_the_smallest_that_passes(price):
    best = fd.min_d_passing_adf(price)
    assert 0.0 < best["d"] < 1.0
    assert best["adf"] < fd.ADF_CRIT_5
    assert fd.adf_tstat(fd.frac_diff_ffd(price, best["d"] - 0.02)) > fd.ADF_CRIT_5
    assert best["corr_level"] > 0.9                        # 0.967 in the demo
    assert best["width"] == len(fd.ffd_weights(best["d"])) - 1
    # a deterministic quadratic ramp is still non-stationary after d = 1, so there is no answer
    ramp = fd.min_d_passing_adf(np.arange(500.0) ** 2)
    assert math.isnan(ramp["d"]) and ramp["width"] == 0
    # a constant series has no correlation to report, and says so instead of dividing by zero
    assert math.isnan(fd.corr(np.zeros(50), np.arange(50.0)))
    assert math.isnan(fd.corr(np.array([1.0]), np.array([2.0])))


def test_first_differences_destroy_the_information_coefficient():
    panel = fd.scan_panel(n_paths=PANEL_PATHS)
    by_d = {r["d"]: r for r in panel}
    level, mid, diff = by_d[0.0], by_d[0.35], by_d[1.0]
    assert level["reject_rate"] <= 0.35                    # 0.06 over 50 paths
    assert mid["reject_rate"] == 1.0
    assert mid["corr_mean"] > 0.85                         # 0.924 over 50 paths
    assert abs(diff["corr_mean"]) < 0.10
    # the level's IC survives fractional differencing and does not survive d = 1
    assert abs(mid["ic_mean"]) > 0.8 * abs(level["ic_mean"])
    assert abs(diff["ic_mean"]) < 0.25 * abs(level["ic_mean"])
    assert abs(diff["ic_mean"]) < 3.0 * diff["ic_se"]      # indistinguishable from zero
    assert abs(mid["ic_mean"]) > 5.0 * mid["ic_se"]
    assert panel == fd.scan_panel(n_paths=PANEL_PATHS)     # seeded end to end


def test_the_threshold_changes_the_window_and_therefore_the_feature(price):
    widths = [len(fd.ffd_weights(0.35, t)) - 1 for t in (1e-2, 1e-3, 1e-4, 1e-5)]
    assert widths == sorted(widths)
    assert widths[0] < 20 < 300 < widths[-1]
    loose = fd.min_d_passing_adf(price, thresh=1e-2)
    tight = fd.min_d_passing_adf(price, thresh=1e-4)
    assert tight["width"] > 10 * loose["width"]
    assert tight["corr_level"] < loose["corr_level"]       # -0.004 vs 0.967 in the demo


def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.ml.frac_diff")
    for head in ("=== 1. The weight recursion", "=== 2. Fixed-width window",
                 "=== 3. Seeded log price", "=== 4. The d-scan",
                 "=== 5. The same scan over", "=== 6. The weight threshold"):
        assert head in out
    assert out.isascii()
    rule = [ln for ln in out.splitlines() if ln.startswith("Rule: ")]
    assert len(rule) == 1 and "smallest" in rule[0] and "d = 1" in rule[0]
    assert out.strip().splitlines()[-1].startswith("total runtime")
