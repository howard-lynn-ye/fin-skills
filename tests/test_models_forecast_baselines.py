"""fin_skills.models.forecast_baselines - the naive forecast, MASE, and Diebold-Mariano.

The properties the SKILL.md claims:

  * R^2 of a price on its own lag is ~0.99 and of a return on its own lag is ~0, on the SAME
    seeded series - and the "good" price forecast's strongest correlation is at lag 1;
  * MASE of the in-sample naive forecast is exactly 1, which is what makes the number readable;
  * rolling-origin evaluation is causal: an origin's forecast does not move when the future
    changes;
  * the Diebold-Mariano statistic equals scipy's one-sample t-test times sqrt(T/(T-1)) at
    h = 1, is antisymmetric, and has the right size under the null;
  * nothing beats the naive forecast on a random walk, and several things beat it on a series
    with real structure - the contrast is the whole point;
  * statsmodels' ARIMA(0,1,0) IS the naive forecast, and an integrated ARIMA has no drift term
    unless you pass trend='t'.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from conftest import requires
from fin_skills.models import forecast_baselines as fb


@pytest.fixture(scope="module")
def price_data():
    return fb.simulate_price()


@pytest.fixture(scope="module")
def seasonal():
    return fb.simulate_seasonal()


def test_the_generators_are_seeded_and_consistent(price_data, seasonal):
    d2 = fb.simulate_price()
    assert np.array_equal(price_data["price"], d2["price"])
    assert not np.array_equal(price_data["price"], fb.simulate_price(seed=fb.SEED + 1)["price"])
    assert np.allclose(price_data["price"], 100.0 * np.exp(np.cumsum(price_data["ret"])))
    assert price_data["ret"][0] == 0.0
    assert np.array_equal(seasonal, fb.simulate_seasonal())
    # the seasonal DGP is a line plus a sine of period SEASON_M plus noise
    t = np.arange(len(seasonal), dtype=float)
    detrended = seasonal - np.polyval(np.polyfit(t, seasonal, 1), t)
    assert np.corrcoef(detrended, np.sin(2 * np.pi * t / fb.SEASON_M))[0, 1] > 0.9
    assert np.corrcoef(detrended, np.sin(2 * np.pi * t / (fb.SEASON_M + 5)))[0, 1] < 0.5


def test_the_random_walk_illusion(price_data):
    r2_price = fb.r2_of_lag(price_data["price"])
    r2_ret = fb.r2_of_lag(price_data["ret"][1:])
    assert r2_price > 0.95                      # 0.987 on this seed
    assert r2_ret < 0.01                        # 0.000133
    assert r2_price > 100 * r2_ret
    with pytest.raises(ValueError):
        fb.r2_of_lag(price_data["price"], lag=0)
    with pytest.raises(ValueError):
        fb.r2_of_lag(price_data["price"][:3])


def test_the_lag_one_signature_of_a_price_forecast_that_is_really_a_lag(price_data):
    p = price_data["price"]
    lo = fb.PRICE_TRAIN
    naive = p[lo - 1:len(p) - 1]
    sig = fb.lag_signature(naive, p[lo:])
    assert max(sig, key=sig.get) == 1           # the forecast IS the previous actual
    assert sig[1] == pytest.approx(1.0, abs=1e-12)
    assert sig[0] < sig[1]
    # the honest score of "tomorrow = today" against itself is exactly zero
    assert fb.out_of_sample_r2(p[lo:], naive, naive) == pytest.approx(0.0)
    # ... and it looks superb against a constant-mean benchmark
    mean_bm = np.full(len(p) - lo, p[:lo].mean())
    assert fb.out_of_sample_r2(p[lo:], naive, mean_bm) > 0.95
    assert math.isnan(fb.out_of_sample_r2(np.ones(5), np.ones(5), np.ones(5)))


def test_the_four_baselines_are_what_they_say():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert list(fb.f_naive(y, 3)) == [6.0, 6.0, 6.0]
    assert list(fb.f_drift(y, 2)) == [7.0, 8.0]             # slope (6-1)/5 = 1
    assert list(fb.f_mean(y, 2)) == [3.5, 3.5]
    assert list(fb.f_seasonal_naive(y, 3, 3)) == [4.0, 5.0, 6.0]
    assert list(fb.f_seasonal_naive(y, 4, 3)) == [4.0, 5.0, 6.0, 4.0]   # wraps a whole cycle
    with pytest.raises(ValueError):
        fb.f_seasonal_naive(y, 1, 10)
    with pytest.raises(ValueError):
        fb.f_drift(y[:1], 1)


def test_mase_of_the_in_sample_naive_forecast_is_exactly_one(price_data):
    tr = price_data["price"][:fb.PRICE_TRAIN]
    sc = fb.scaling_factor(tr, 1)
    assert sc > 0
    assert fb.mase(tr[1:], tr[:-1], sc) == pytest.approx(1.0, abs=1e-12)
    # the seasonal scaling factor uses the seasonal naive on the training data
    sc12 = fb.scaling_factor(tr, 12)
    assert fb.mase(tr[12:], tr[:-12], sc12) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        fb.mase(tr[1:], tr[:-1], 0.0)
    with pytest.raises(ValueError):
        fb.scaling_factor(tr[:5], 12)
    s = fb.score(tr[1:], tr[:-1], sc)
    assert set(s) == {"mae", "rmse", "mase"} and s["rmse"] >= s["mae"]


def test_rolling_origin_is_causal_and_shaped_right(price_data):
    y = price_data["price"][:400]
    fc, act = fb.rolling_origin(y, lambda tr, h: fb.f_naive(tr, h), start=300)
    assert fc.shape == act.shape == (100,)
    assert np.array_equal(act, y[300:])
    assert np.array_equal(fc, y[299:-1])            # the naive forecast IS the lagged series
    shocked = y.copy()
    shocked[350] += 25.0
    fc2, _ = fb.rolling_origin(shocked, lambda tr, h: fb.f_naive(tr, h), start=300)
    assert np.array_equal(fc2[:51], fc[:51])        # origins 300..350 never saw y[350]
    assert fc2[51] != fc[51]                        # origin 351 does
    with pytest.raises(ValueError):
        fb.rolling_origin(y, lambda tr, h: fb.f_naive(tr, h), start=1)
    with pytest.raises(ValueError):
        fb.rolling_origin(y, lambda tr, h: fb.f_naive(tr, 2), start=300)   # wrong length


def test_horizon_two_scores_the_second_step(price_data):
    y = price_data["price"][:400]
    fc, act = fb.rolling_origin(y, lambda tr, h: fb.f_naive(tr, h), start=300, horizon=2)
    assert np.array_equal(act, y[301:])
    assert np.array_equal(fc, y[299:-2])


# ------------------------------------------------------------------------ Diebold-Mariano ---
def test_dm_equals_a_one_sample_t_test_at_h_one(price_data):
    rng = np.random.default_rng(3)
    e1, e2 = rng.standard_normal(500), rng.standard_normal(500) * 1.2
    chk = fb.dm_matches_scipy_t_test(e1, e2)
    assert chk["abs_error"] < 1e-12
    assert chk["ratio"] == pytest.approx(math.sqrt(500 / 499.0))
    d = e1 ** 2 - e2 ** 2
    assert chk["scipy_t"] == pytest.approx(float(stats.ttest_1samp(d, 0.0).statistic))


def test_dm_is_antisymmetric_and_validates_its_input():
    rng = np.random.default_rng(1)
    e1 = rng.standard_normal(300)
    e2 = e1 + rng.standard_normal(300) * 0.4
    a, b = fb.diebold_mariano(e1, e2), fb.diebold_mariano(e2, e1)
    assert a["stat"] == pytest.approx(-b["stat"])
    assert a["p_value"] == pytest.approx(b["p_value"])
    assert a["df"] == 299
    assert fb.diebold_mariano(e1, e1)["dbar"] == 0.0
    assert fb.diebold_mariano(e1, e2, loss="absolute")["stat"] != a["stat"]
    for bad in (dict(h=0), dict(loss="huber")):
        with pytest.raises(ValueError):
            fb.diebold_mariano(e1, e2, **bad)
    with pytest.raises(ValueError):
        fb.diebold_mariano(e1, e2[:-1])
    # a longer horizon widens the HAC variance, so the statistic shrinks in magnitude
    assert abs(fb.diebold_mariano(e1, e2, h=5)["stat"]) != abs(a["stat"])


def test_dm_has_the_right_size_under_the_null():
    sz = fb.dm_size_under_the_null(sims=400, T=200, seed=fb.SEED)
    assert abs(sz["size"] - 0.05) < 3 * sz["se"] + 0.01
    assert sz["sims"] == 400 and sz["T"] == 200


# ------------------------------------------------------------------------- the two DGPs -----
def test_nothing_beats_the_naive_forecast_on_a_random_walk(price_data):
    y = price_data["price"]
    sc = fb.scaling_factor(y[:fb.PRICE_TRAIN], 1)
    errs, mases = {}, {}
    for name, fn in fb.BASELINES.items():
        fc, act = fb.rolling_origin(y, lambda tr, h, _f=fn: _f(tr, h, 5), fb.PRICE_TRAIN)
        errs[name] = act - fc
        mases[name] = fb.mase(act, fc, sc)
    assert mases["naive"] < mases["seasonal naive"]
    assert mases["naive"] < mases["mean"] / 10          # the mean is hopeless on a random walk
    assert mases["drift"] == pytest.approx(mases["naive"], rel=0.01)
    for name in ("seasonal naive", "mean"):
        assert fb.diebold_mariano(errs[name], errs["naive"])["p_value"] < 0.01
    assert fb.diebold_mariano(errs["drift"], errs["naive"])["p_value"] > 0.05


def test_real_structure_is_beatable_and_the_dm_test_says_so(seasonal):
    sc = fb.scaling_factor(seasonal[:fb.SEASON_TRAIN], 1)
    fc_n, act = fb.rolling_origin(seasonal, lambda tr, h: fb.f_naive(tr, h), fb.SEASON_TRAIN)
    fc_s, _ = fb.rolling_origin(seasonal,
                                lambda tr, h: fb.f_seasonal_naive(tr, h, fb.SEASON_M),
                                fb.SEASON_TRAIN)
    assert fb.mase(act, fc_s, sc) < fb.mase(act, fc_n, sc)
    dm = fb.diebold_mariano(act - fc_s, act - fc_n)
    assert dm["stat"] < 0 and dm["p_value"] < 0.05      # significantly better, not just better


@requires("statsmodels")
@pytest.mark.slow
def test_ets_beats_the_naive_forecast_on_the_seasonal_series(seasonal):
    sc = fb.scaling_factor(seasonal[:fb.SEASON_TRAIN], 1)
    et = fb.ets_walk_forward(seasonal, fb.SEASON_TRAIN, seasonal_periods=fb.SEASON_M,
                             seasonal="add")
    assert et is not None and et["n_fits"] == len(seasonal) - fb.SEASON_TRAIN
    fc_n, act = fb.rolling_origin(seasonal, lambda tr, h: fb.f_naive(tr, h), fb.SEASON_TRAIN)
    assert np.array_equal(et["actual"], act)
    assert fb.mase(act, et["forecast"], sc) < 0.75 * fb.mase(act, fc_n, sc)


# ----------------------------------------------------------------------- statsmodels facts --
@requires("statsmodels")
def test_arima_0_1_0_is_the_naive_forecast_and_an_integrated_arima_has_no_drift():
    td = fb.arima_trend_defaults()
    assert td is not None
    assert td["d=0 on the level"]["trend"] == "c"
    assert td["d=1, default trend"]["trend"] == "n"
    assert "const" not in td["d=1, default trend"]["params"]
    assert abs(td["d=1, default trend"]["slope"]) < 0.001          # a flat forecast
    assert td["d=1, trend='t'"]["slope"] == pytest.approx(td["sample_drift"], rel=0.10)
    assert td["naive_identity"]["max_abs"] < 1e-9                  # 0.0 exactly on this data


@requires("statsmodels")
def test_the_arima_walk_forward_never_sees_the_future(price_data):
    y = price_data["price"][:500]
    out = fb.arima_walk_forward(y, 400, order=(1, 1, 1), refit_every=50)
    assert out is not None and out["n_fits"] == 2 and out["trend"] == "n"
    assert out["forecast"].shape == out["actual"].shape == (100,)
    assert np.array_equal(out["actual"], y[400:])
    shocked = y.copy()
    shocked[450] += 30.0
    out2 = fb.arima_walk_forward(shocked, 400, order=(1, 1, 1), refit_every=50)
    assert np.allclose(out2["forecast"][:50], out["forecast"][:50])
    assert not np.allclose(out2["forecast"][50:], out["forecast"][50:])


@pytest.mark.slow
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.models.forecast_baselines")
    for head in ("=== 1. The same series, two framings",
                 "=== 2. MASE (Hyndman-Koehler 2006)",
                 "=== 3. Rolling-origin walk-forward",
                 "=== 4. Is the Diebold-Mariano statistic right?",
                 "=== 5. statsmodels ARIMA"):
        assert head in out
    assert out.isascii()
    rule = [ln for ln in out.splitlines() if ln.startswith("Rule: ")]
    assert len(rule) == 1 and "MASE" in rule[0] and "R^2 on a price level" in rule[0]
    assert out.strip().splitlines()[-1].startswith("total runtime")
