"""Causality, numerical definitions, risk policy and the public strategy/tool workflow."""
import json

import numpy as np
import pandas as pd
import pytest

from fin_skills.algorithms import market_state, recommend_strategy, run, research
from fin_skills.algorithms.technical import SIGNAL_DEFAULTS
from fin_skills.tools import call_tool, series_to_payload


def history(values=None):
    if values is None:
        values = 100 * np.exp(np.arange(100) * .003)
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=len(values), tz="UTC"))


@pytest.mark.parametrize("method", list(SIGNAL_DEFAULTS))
def test_signals_prefix_invariant_and_strictly_lagged(method):
    x = history(100 * np.exp(np.random.default_rng(4).normal(0, .02, 100).cumsum()))
    result = run(method, {"prices": x})
    changed = x.copy()
    changed.iloc[70:] *= 3
    pd.testing.assert_series_equal(result.iloc[:71], run(method, {"prices": changed}).iloc[:71])
    pd.testing.assert_series_equal(result.iloc[:75], run(method, {"prices": x.iloc[:75]}))
    assert result.iloc[:10].isna().all()
    assert result.attrs["lag_bars"] == 1
    assert np.isfinite(result.iloc[-1])


def test_breakout_reference_and_reversion_on_constant_prices():
    result = run("donchian_breakout", {"prices": np.arange(1., 31)}, lookback=5)
    assert result.iloc[:6].isna().all()
    assert (result.iloc[6:] == 1).all()
    for method in ("bollinger_reversion", "rsi_reversion", "vol_target_momentum"):
        result = run(method, {"prices": np.full(70, 100.)})
        assert result.iloc[-1] == 0


def test_rsi_is_explicit_simple_rolling_reversion():
    # Strictly rising prices give RSI 100 and a short reversion target after warmup.
    result = run("rsi_reversion", {"prices": np.arange(1., 41)})
    assert result.iloc[-1] == -1
    with pytest.raises(ValueError, match="thresholds"):
        run("rsi_reversion", {"prices": np.arange(1., 41)}, lower=60)


def test_bollinger_entry_and_center_exit():
    x = np.r_[np.full(24, 100.), 70., 100., 100.]
    result = run("bollinger_reversion", {"prices": x})
    assert result.iloc[25] == 1
    assert result.iloc[26] == 0


def test_vol_target_matches_manual_formula():
    x = history(100 * np.exp(np.random.default_rng(8).normal(.003, .02, 100).cumsum()))
    result = run("vol_target_momentum", {"prices": x})
    vol = x.pct_change().iloc[-21:-1].std() * np.sqrt(252)
    expected = np.sign(x.iloc[-2] / x.iloc[-22] - 1) * min(1, .10 / vol)
    assert result.iloc[-1] == pytest.approx(expected)


def test_cross_sectional_negative_assets_leave_cash_and_research_supports_it():
    returns = pd.DataFrame(np.tile([.01, -.01, -.02], (85, 1)), columns=list("ABC"))
    weights = run("cross_sectional_momentum", {"asset_returns": returns}, top_k=2)
    assert weights.to_dict() == {"A": .5, "B": 0., "C": 0.}
    assert weights.attrs["cash_weight"] == .5
    result = research("portfolio", {"asset_returns": returns}, initial_train=65,
                      horizon=5, holdout=5, candidates=["cross_sectional_momentum"],
                      parameters={"cross_sectional_momentum": {"top_k": 2}})
    assert result.status == "completed"
    assert result.selected == "cross_sectional_momentum"
    assert result.validation["holdout"]["algorithm"] == result.selected


def test_market_state_prefix_invariance_and_unknown_warmup():
    x = history()
    result = market_state(x)
    assert (result.regime.iloc[:60] == "unknown").all()
    assert result.regime.iloc[-1] == "trend_up"
    pd.testing.assert_frame_equal(result.iloc[:75], market_state(x.iloc[:75]))
    assert market_state(history(np.full(100, 100.))).regime.iloc[-1] == "range"
    assert market_state(history(np.linspace(100, 40, 100))).regime.iloc[-1] == "stress"


def test_regime_high_volatility_is_prioritized_over_trend():
    x = history(100 * np.exp(np.tile([.1, -.08], 50).cumsum()))
    assert market_state(x).regime.iloc[-1] == "high_volatility"


def test_relative_volatility_shock_detected_below_absolute_threshold():
    ret = np.r_[np.tile([.001, -.001], 35), np.tile([.005, -.005], 10)]
    x = history(100 * np.cumprod(1 + ret))
    state = market_state(x).iloc[-1]
    assert state.annualized_vol < .35
    assert state.vol_ratio >= 1.5
    assert state.regime == "high_volatility"


def test_asof_excludes_future_prices_and_news_and_default_is_long_only():
    x = history()
    as_of = x.index[80].isoformat()
    future = {"id": "halt", "source": "rss", "kind": "news", "title": "Trading halt",
              "url": "https://example.com/halt", "observed_at": x.index[81].isoformat(),
              "published_at": x.index[79].isoformat()}
    report = recommend_strategy(x, as_of=as_of, news=[future])
    assert report["excluded_future_price_rows"] == 19
    assert report["news"]["excluded"]["future"] == 1
    assert report["selected"] == "vol_target_momentum"
    assert report["target_exposure"] >= 0
    cutoff = recommend_strategy(x.iloc[:81], as_of=as_of)
    assert report["target_exposure"] == cutoff["target_exposure"]
    json.dumps(report, allow_nan=False)


def test_news_risk_caps_exposure_and_stale_or_short_history_abstains():
    x = history(100 * np.exp(np.arange(100) * .003 + np.sin(np.arange(100)) * .005))
    now = x.index[-1].isoformat()
    event = {"id": "a", "source": "rss", "kind": "news", "title": "Trading halt",
             "url": "https://example.com/a", "observed_at": now, "published_at": now}
    report = recommend_strategy(x, as_of=now, news=[event])
    assert report["exposure_cap"] == .25
    assert recommend_strategy(x.iloc[:10], as_of=x.index[9].isoformat())["selected"] == "cash"
    assert recommend_strategy(x, as_of="2027-01-01T00:00:00Z")["selected"] == "cash"
    with pytest.raises(ValueError, match="timezone"):
        recommend_strategy(x, as_of="2026-04-10")


def test_short_constraint_and_stress_cash():
    x = history(100 * np.exp(-np.arange(100) * .001))
    assert recommend_strategy(x, as_of=x.index[-1])["target_exposure"] == 0
    assert recommend_strategy(x, as_of=x.index[-1], allow_short=True)["target_exposure"] < 0
    x = history(np.linspace(100, 40, 100))
    assert recommend_strategy(x, as_of=x.index[-1])["selected"] == "cash"


def test_new_signals_are_usable_in_temporal_research_and_json_tool():
    x = history(100 * np.exp(np.arange(100) * .003 + np.sin(np.arange(100)) * .005))
    study = research("signal", {"prices": x}, initial_train=65, horizon=5, holdout=5,
                     candidates=["donchian_breakout", "macd"], cost_bps=5)
    assert study.status == "completed"
    assert study.validation["rejected"] == {}
    assert study.validation["holdout"]["algorithm"] == study.selected
    result = call_tool("recommend_trading_strategy", {"prices": series_to_payload(x),
                       "as_of": x.index[-1].isoformat()})
    assert result["selected"] == "vol_target_momentum"
