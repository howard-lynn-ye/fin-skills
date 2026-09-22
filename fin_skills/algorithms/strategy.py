"""Causal market-state descriptors and transparent research strategy policies."""
import numpy as np
import pandas as pd

from .runtime import _array, integer, number, params
from .technical import decisions

REGIME_DEFAULTS = {"lookback": 20, "baseline": 60, "periods_per_year": 252,
                   "trend_efficiency": 0.35, "high_vol": 0.35,
                   "vol_ratio": 1.5, "drawdown_limit": 0.15}


def regime_settings(given):
    p = params(given, REGIME_DEFAULTS)
    for key in ("lookback", "baseline", "periods_per_year"):
        integer(p[key], key, minimum=2)
    if p["baseline"] < p["lookback"]:
        raise ValueError("baseline must be at least lookback")
    for key in ("trend_efficiency", "high_vol", "vol_ratio", "drawdown_limit"):
        if number(p[key], key) <= 0:
            raise ValueError(f"{key} must be positive")
    if p["trend_efficiency"] > 1 or p["drawdown_limit"] >= 1:
        raise ValueError("invalid efficiency or drawdown threshold")
    return p


def market_state(prices, **parameters):
    """Descriptors at close t use rows <= t; any resulting trade belongs after t.

    Labels are threshold policy, not learned probabilities or stationary-process tests.
    Caller declares bar frequency via periods_per_year; no frequency inference.
    """
    p = regime_settings(parameters)
    x = pd.Series(_array(prices, "prices", 1))
    if (x <= 0).any():
        raise ValueError("prices must be positive")
    n, b = p["lookback"], p["baseline"]
    ret = x.pct_change(fill_method=None)
    vol = ret.rolling(n).std() * np.sqrt(p["periods_per_year"])
    base_vol = ret.rolling(b).std() * np.sqrt(p["periods_per_year"])
    distance = x.diff().abs().rolling(n).sum()
    efficiency = ((x - x.shift(n)).abs() / distance.replace(0, np.nan)).where(distance != 0, 0.0)
    trend = x / x.shift(n) - 1
    drawdown = x / x.rolling(b + 1).max() - 1
    ratio = (vol / base_vol.replace(0, np.nan)).where(base_vol != 0, 0.0)
    ready = drawdown.notna() & vol.notna()
    labels = pd.Series("unknown", index=x.index)
    labels.loc[ready] = "range"
    labels.loc[ready & (efficiency >= p["trend_efficiency"]) & (trend > 0)] = "trend_up"
    labels.loc[ready & (efficiency >= p["trend_efficiency"]) & (trend < 0)] = "trend_down"
    labels.loc[ready & ((vol >= p["high_vol"]) | (ratio >= p["vol_ratio"]))] = "high_volatility"
    labels.loc[ready & (drawdown <= -p["drawdown_limit"])] = "stress"
    out = pd.DataFrame({"regime": labels, "annualized_vol": vol, "vol_ratio": ratio,
                        "trend_return": trend, "efficiency": efficiency, "drawdown": drawdown})
    out.attrs.update(parameters=p, timing="close-time descriptor; trade only on a later bar")
    return out


def regime_adapter(data, given):
    return market_state(data["prices"], **given)


def recommend_strategy(prices, *, as_of, news=None, allow_short=False, target_vol=0.10,
                       max_exposure=1.0, max_price_age_days=7.0, regime_parameters=None,
                       news_keywords=(), risk_terms=None):
    """Return next-bar research targets, explanations and point-in-time news evidence.

    Requires timezone-aware close timestamps. Future price rows are excluded explicitly.
    No model is fit to future rows, no live orders are submitted, no efficacy is asserted.
    """
    from fin_skills.collect.news import news_digest
    if not isinstance(prices, pd.Series) or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must be a Series with timezone-aware DatetimeIndex")
    if prices.index.tz is None:
        raise ValueError("price timestamps require an explicit timezone")
    if type(allow_short) is not bool:
        raise TypeError("allow_short must be boolean")
    stamp = pd.Timestamp(as_of)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("as_of requires an explicit timezone")
    prices = _array(prices, "prices", 1)
    known = prices.loc[prices.index <= stamp]
    if known.empty:
        raise ValueError("no prices available at as_of")
    for key, val in (("target_vol", target_vol), ("max_exposure", max_exposure),
                     ("max_price_age_days", max_price_age_days)):
        if number(val, key) <= 0:
            raise ValueError(f"{key} must be positive")
    p = regime_settings(regime_parameters or {})
    states = market_state(known, **p)
    last = states.iloc[-1]
    digest = news_digest(news or [], as_of=stamp.isoformat(), keywords=news_keywords,
                         risk_terms=risk_terms)
    warnings = ["Rule-based research candidates; profitability has not been validated.",
                "Range describes low directional efficiency, not proven mean reversion."]
    age = (stamp - known.index[-1]).total_seconds() / 86400
    blocked = last["regime"] == "unknown" or age > max_price_age_days
    if age > max_price_age_days:
        warnings.append("Price history is stale; no active strategy is proposed.")
    if digest["status"] != "available":
        warnings.append("No eligible fresh news; event risk is unknown, not absent.")
    regime = last["regime"]
    choices = {
        "trend_up": ["vol_target_momentum", "donchian_breakout", "macd"],
        "trend_down": ["vol_target_momentum", "donchian_breakout", "macd"],
        "range": ["bollinger_reversion", "rsi_reversion"],
        "high_volatility": ["vol_target_momentum"],
        "stress": [], "unknown": [],
    }[regime]
    vol = float(last["annualized_vol"])
    cap = min(max_exposure, target_vol / vol) if np.isfinite(vol) and vol > 0 else 0.0
    if regime == "high_volatility":
        cap = min(cap, 0.25)
    if digest["risk_matches"]:
        cap = min(cap, 0.25)
        warnings.append("Configured event-risk keywords matched; exposure cap reduced to 0.25.")
    if blocked or regime == "stress":
        cap, choices = 0.0, []
    candidates = []
    for method in choices:
        parameters = ({"target_vol": target_vol, "max_exposure": max_exposure,
                       "periods_per_year": p["periods_per_year"]}
                      if method == "vol_target_momentum" else {})
        value = float(decisions(method, known, parameters).iloc[-1])
        if not np.isfinite(value):
            continue
        target = float(np.clip(value, -cap if allow_short else 0.0, cap))
        candidates.append({"algorithm": method, "parameters": parameters,
                           "target_exposure": target,
                           "reason": f"policy maps {regime} to {method}; risk cap {cap:.4f}",
                           "requires_validation": True})
    return {"as_of": stamp.isoformat(), "price_as_of": known.index[-1].isoformat(),
            "excluded_future_price_rows": len(prices) - len(known),
            "regime": regime, "indicators": {k: (float(v) if np.isfinite(v) else None)
                for k, v in last.items() if k != "regime"}, "regime_parameters": p,
            "status": "insufficient_or_stale" if blocked else "ready",
            "candidates": candidates, "selected": candidates[0]["algorithm"] if candidates else "cash",
            "target_exposure": candidates[0]["target_exposure"] if candidates else 0.0,
            "exposure_cap": cap, "allow_short": allow_short, "news": digest,
            "timing": "target for a bar strictly after price_as_of and as_of",
            "warnings": warnings}
