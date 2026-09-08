"""fin_skills.core.event_study - market-model event study with BMP standardisation.

The documented result: statistics whose denominator comes from the estimation window
(Patell, classic Corrado) over-reject when the event itself moves volatility; the
cross-sectional ones (BMP, generalised sign, cross-sectional Corrado) are correctly sized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.core.event_study import (_car_prediction_sd, _market_model, _run,
                                         _synthetic_events, bmp_test, corrado_rank_test,
                                         corrado_rank_test_classic, event_study,
                                         generalized_sign_test, naive_cross_sectional_test,
                                         patell_test, resolve_event_date)


# ---------------------------------------------------------------------------- timing
def test_resolve_event_date_rolls_post_close_events_to_the_next_session():
    sessions = pd.bdate_range("2024-07-29", periods=10)
    assert resolve_event_date("2024-07-30 16:30", sessions) == pd.Timestamp("2024-07-31")
    assert resolve_event_date("2024-07-30 15:30", sessions) == pd.Timestamp("2024-07-30")
    assert resolve_event_date("2024-07-30", sessions) == pd.Timestamp("2024-07-30")  # date-only
    assert resolve_event_date("2024-08-02 17:00", sessions) == pd.Timestamp("2024-08-05")  # Fri -> Mon
    assert resolve_event_date("2024-09-01", sessions) is None


# ---------------------------------------------------------------------------- market model
def test_market_model_recovers_an_exact_line_and_forecast_sd_exceeds_residual_sd():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 200)
    fit = _market_model(0.001 + 1.5 * x, x)
    assert fit.alpha == pytest.approx(0.001) and fit.beta == pytest.approx(1.5)
    assert fit.sigma == pytest.approx(0.0, abs=1e-12)
    noisy = _market_model(0.001 + 1.5 * x + rng.normal(0, 0.02, 200), x)
    x_win = rng.normal(0, 0.01, 3)
    # parameter uncertainty makes the out-of-sample CAR sd larger than sigma*sqrt(L2)
    assert _car_prediction_sd(noisy, x_win) > noisy.sigma * np.sqrt(3)
    with pytest.raises(ValueError, match="zero variance"):
        _market_model(x, np.zeros(200))


# ---------------------------------------------------------------------------- tests
def test_bmp_is_scale_free_where_patell_scales_with_the_error():
    scar = np.array([0.5, 1.2, -0.3, 0.8, 1.5])
    t1, p1 = bmp_test(scar)
    t3, p3 = bmp_test(3 * scar)
    assert t3 == pytest.approx(t1) and p3 == pytest.approx(p1)
    z1, _ = patell_test(scar, 150)
    z3, _ = patell_test(3 * scar, 150)
    assert z3 == pytest.approx(3 * z1)


def test_degenerate_inputs_return_nan_not_exceptions():
    assert all(np.isnan(v) for v in bmp_test(np.array([1.0])))
    assert all(np.isnan(v) for v in naive_cross_sectional_test(np.array([2.0, 2.0])))
    assert all(np.isnan(v) for v in patell_test(np.array([1.0, 2.0]), n_est=4))


def test_generalised_sign_test_uses_the_estimation_window_base_rate():
    car = np.array([0.01, 0.02, 0.03, 0.04, -0.01])
    z_half, _ = generalized_sign_test(car, 0.5)
    z_skew, _ = generalized_sign_test(car, 0.8)
    assert z_half > 0 and z_skew < z_half           # a skewed base rate lowers the statistic
    assert generalized_sign_test(np.array([1.0, -1.0]), 0.5)[0] == 0.0   # raw == 0 -> z 0


def test_rank_tests_run_and_return_probabilities():
    rng = np.random.default_rng(1)
    est = rng.normal(size=(30, 100))
    win = rng.normal(size=(30, 3))
    for fn in (corrado_rank_test, corrado_rank_test_classic):
        stat, p = fn(est, win)
        assert np.isfinite(stat) and 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------- the study
@pytest.fixture(scope="module")
def planted():
    return _run(seed=7, true_ar=0.04)


def test_recovers_a_planted_abnormal_return(planted):
    assert planted.n_events == 120 and planted.attrition == {}
    assert planted.car.mean() == pytest.approx(0.04, abs=0.01)
    assert planted.tests.loc["BMP (standardised)", "p_value"] < 1e-6
    assert planted.window == (-1, 1) and planted.est_window == 150 and planted.gap == 15
    assert list(planted.aar.index) == list(range(-4, 7))
    pd.testing.assert_series_equal(planted.caar, planted.aar.cumsum())


def test_bmp_has_more_power_than_the_raw_cross_sectional_test(planted):
    t_bmp = planted.tests.loc["BMP (standardised)", "statistic"]
    t_raw = planted.tests.loc["naive cross-sectional (raw CAR)", "statistic"]
    assert t_bmp > t_raw


def test_seeded_runs_are_deterministic(planted):
    again = _run(seed=7, true_ar=0.04)
    pd.testing.assert_frame_equal(planted.ar, again.ar)
    pd.testing.assert_frame_equal(planted.tests, again.tests)


def test_estimation_window_denominators_over_reject_under_the_null():
    # 60 null replications: Patell fires far above 5%, BMP stays near it
    reps = 60
    reject = {"Patell standardised-residual": 0, "BMP (standardised)": 0,
              "Corrado rank (classic SE)": 0, "Corrado rank (cross-sectional SE)": 0}
    for s in range(reps):
        tests = _run(seed=20_000 + s, true_ar=0.0).tests
        for name in reject:
            reject[name] += float(tests.loc[name, "p_value"]) < 0.05
    rates = {k: v / reps for k, v in reject.items()}
    assert rates["Patell standardised-residual"] > 0.25
    assert rates["BMP (standardised)"] <= 0.15
    assert rates["Corrado rank (classic SE)"] > rates["Corrado rank (cross-sectional SE)"]


def test_events_with_missing_data_are_dropped_and_counted():
    rng = np.random.default_rng(7)
    rets, mkt, events = _synthetic_events(rng, true_ar=0.0)
    ev = events[:5] + [("ZZZ", events[0][1]), (events[0][0], pd.Timestamp("2020-01-01"))]
    res = event_study(rets, mkt, ev, window=(-1, 1), est_window=150, gap=15,
                      pre_days=4, post_days=6, min_nonzero=20)
    assert res.n_events == 5 and res.n_events_in == 7
    assert res.attrition == {"ticker not in the returns panel": 1,
                             "event date is not a session in the panel": 1}
    report = res.report()
    assert "ATTRITION" in report and "BMP (standardised)" in report
    with pytest.raises(ValueError, match="no events survived"):
        event_study(rets, mkt, [("ZZZ", events[0][1])], est_window=150, gap=15)
    with pytest.raises(ValueError, match="window"):
        event_study(rets, mkt, events[:3], window=(1, -1))


@pytest.mark.slow
def test_demo_runs_300_null_replications_and_reports_the_rates(run_main):
    out = run_main("fin_skills.core.event_study")
    assert "OVER-REJECTS - do not use" in out
    assert "the raw cross-sectional t-test is correctly sized" in out
