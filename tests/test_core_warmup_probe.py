"""fin_skills.core.warmup_probe - how much history a recursive indicator needs before it settles."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.core.warmup_probe import NOT_CONVERGED, error_curve, warmup_bars, warmup_report


def sma(x, n):
    return pd.Series(x, dtype=float).rolling(n, min_periods=n).mean().to_numpy()


def ema(x, n):
    alpha = 2.0 / (n + 1.0)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi_wilder(x, n=14):
    out = np.full(len(x), np.nan)
    if len(x) <= n:
        return out
    d = np.diff(x)
    gain, loss = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    avg_g, avg_l = float(np.mean(gain[:n])), float(np.mean(loss[:n]))
    for i in range(n, len(x)):
        if i > n:
            avg_g = (avg_g * (n - 1) + gain[i - 1]) / n
            avg_l = (avg_l * (n - 1) + loss[i - 1]) / n
        out[i] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return out


@pytest.fixture(scope="module")
def closes():
    rng = np.random.default_rng(7)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 1500)))


def test_sma_converges_within_its_window(closes):
    assert warmup_bars(lambda a: sma(a, 20), closes, tol=1e-12, max_probe=700) <= 20


def test_ema_needs_10_to_15_times_its_period(closes):
    bars = warmup_bars(lambda a: ema(a, 20), closes, tol=1e-12, max_probe=700)
    assert 10 * 20 <= bars <= 15 * 20
    assert warmup_bars(lambda a: ema(a, 20), closes, tol=1e-6, max_probe=700) < bars


def test_wilder_smoothing_needs_more_periods_than_an_ema(closes):
    rsi = warmup_bars(lambda a: rsi_wilder(a, 14), closes, tol=1e-12, max_probe=700)
    e20 = warmup_bars(lambda a: ema(a, 20), closes, tol=1e-12, max_probe=700)
    assert rsi / 14 > e20 / 20


def test_not_converged_when_the_probe_is_too_short(closes):
    assert warmup_bars(lambda a: ema(a, 50), closes, tol=1e-12, max_probe=30) == NOT_CONVERGED
    assert NOT_CONVERGED == -1


def test_error_curve_is_relative_and_nan_counts_as_infinite_error(closes):
    probes, errs = error_curve(lambda a: sma(a, 20), closes, max_probe=40)
    assert probes.tolist() == list(range(41)) and len(errs) == 41
    assert np.isinf(errs[:19]).all() and errs[19:].max() < 1e-12
    _, e_ema = error_curve(lambda a: ema(a, 20), closes, max_probe=40)
    assert e_ema[0] > 0 and e_ema[-1] < e_ema[0]


def test_input_validation(closes):
    with pytest.raises(ValueError, match="positive"):
        warmup_bars(lambda a: ema(a, 20), closes, tol=0.0)
    with pytest.raises(ValueError, match="NaN"):
        error_curve(lambda a: ema(a, 20), np.r_[closes[:100], np.nan])
    with pytest.raises(ValueError, match="length-preserving"):
        error_curve(lambda a: a[1:], closes)
    with pytest.raises(ValueError, match="at least max_probe"):
        error_curve(lambda a: ema(a, 20), closes[:50], max_probe=45)
    with pytest.raises(ValueError, match="1-D"):
        error_curve(lambda a: a, np.zeros((10, 2)))


def test_report_table_shape(closes):
    fns = {"EMA(20)": lambda a: ema(a, 20), "SMA(20)": lambda a: sma(a, 20)}
    tab = warmup_report(fns, closes, periods={"EMA(20)": 20, "SMA(20)": 20}, max_probe=400)
    assert list(tab["indicator"]) == ["EMA(20)", "SMA(20)"]
    assert {"bars@1e-06", "bars@1e-09", "bars@1e-12", "x_period@1e-12", "err@0bars"} <= set(tab.columns)
    ema_row = tab.set_index("indicator").loc["EMA(20)"]
    assert ema_row["bars@1e-06"] <= ema_row["bars@1e-09"] <= ema_row["bars@1e-12"]
    assert ema_row["x_period@1e-12"] == pytest.approx(ema_row["bars@1e-12"] / 20, abs=0.05)


def test_demo_tests_the_folklore(run_main):
    out = run_main("fin_skills.core.warmup_probe")
    assert "THE FOLKLORE, TESTED" in out and "Fetch warmup + lookback" in out
