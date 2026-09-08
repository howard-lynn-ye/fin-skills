"""fin_skills.core.assert_causal - perturb rows >= k, nothing before k may move."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.core.assert_causal import assert_causal, scan_indicators, warmup_bars


@pytest.fixture
def ohlc() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 500
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    return pd.DataFrame({"open": close.shift(1).bfill(), "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 1e6})


def test_flags_a_signal_that_uses_bar_t_plus_1(ohlc):
    with pytest.raises(AssertionError, match="LOOK-AHEAD in shift_minus_1"):
        assert_causal(lambda d: d.close.shift(-1), ohlc, k=200, name="shift_minus_1")


def test_exactly_one_cell_moves_for_a_one_bar_leak(ohlc):
    # row k-1 = shift(-1) of row k is the only pre-k cell that depends on rows >= k
    with pytest.raises(AssertionError, match="1 cells before index 200"):
        assert_causal(lambda d: d.close.shift(-1), ohlc, k=200)


def test_flags_centered_window_and_full_sample_zscore(ohlc):
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_causal(lambda d: d.close.rolling(20, center=True).mean(), ohlc, k=200)
    with pytest.raises(AssertionError, match="200 cells before index 200"):
        assert_causal(lambda d: (d.close - d.close.mean()) / d.close.std(), ohlc, k=200)


def test_clears_a_trailing_rolling_mean_and_an_expanding_zscore(ohlc):
    assert assert_causal(lambda d: d.close.rolling(20).mean(), ohlc, k=200) is None
    zs = lambda d: (d.close - d.close.expanding().mean()) / d.close.expanding().std()
    assert assert_causal(zs, ohlc, k=200) is None


def test_dataframe_valued_signals_are_supported(ohlc):
    assert assert_causal(lambda d: d[["open", "close"]].rolling(5).mean(), ohlc, k=100) is None
    with pytest.raises(AssertionError):
        assert_causal(lambda d: d[["open", "close"]].shift(-2), ohlc, k=100)


def test_tolerance_decides_what_counts_as_a_move(ohlc):
    # a future dependence of ~1e-10 passes the default 1e-9 tolerance and fails a tighter one
    tiny_leak = lambda d: d.close.rolling(20).mean() + 1e-12 * d.close.iloc[-1]
    assert assert_causal(tiny_leak, ohlc, k=200) is None
    with pytest.raises(AssertionError):
        assert_causal(tiny_leak, ohlc, k=200, tol=1e-12)


def test_scan_indicators_reports_pass_fail_and_broken(ohlc):
    fns = {
        "sma": lambda d: d.close.rolling(20).mean(),
        "lead": lambda d: d.close.shift(-1),
        "broken": lambda d: d["no_such_column"],
    }
    tab = scan_indicators(fns, ohlc).set_index("indicator")
    assert tab.loc["sma", "causal"] is True or tab.loc["sma", "causal"] == True  # noqa: E712
    assert tab.loc["sma", "error"] == ""
    assert tab.loc["lead", "causal"] == False  # noqa: E712
    assert "LOOK-AHEAD" in tab.loc["lead", "error"]
    assert tab.loc["broken", "causal"] is None
    assert tab.loc["broken", "error"].startswith("KeyError")


def test_scan_defaults_k_to_half_the_sample(ohlc):
    tab = scan_indicators({"lead": lambda d: d.close.shift(-1)}, ohlc)
    assert f"before index {len(ohlc) // 2}" in tab.loc[0, "error"]


def test_warmup_bars_first_prefix_for_causal_none_for_lookahead(ohlc):
    # a causal function computed on a prefix agrees with the full run, so the first probe
    # (50 bars) already qualifies; a centred window never agrees on its last 10 values
    assert warmup_bars(lambda d: d.close.rolling(20).mean(), ohlc) == 50
    assert warmup_bars(lambda d: d.close.rolling(20, center=True).mean(), ohlc) is None
