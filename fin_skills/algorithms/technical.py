"""Dependency-light strategy baselines. All historical positions are lagged one bar."""
import numpy as np
import pandas as pd

from .runtime import integer, number, params


SIGNAL_DEFAULTS = {
    "donchian_breakout": {"lookback": 20},
    "bollinger_reversion": {"lookback": 20, "entry_z": 2.0},
    "rsi_reversion": {"lookback": 14, "lower": 30.0, "upper": 70.0},
    "macd": {"fast": 12, "slow": 26, "signal_span": 9},
    "vol_target_momentum": {"lookback": 20, "target_vol": 0.10,
                            "periods_per_year": 252, "max_exposure": 1.0},
}


def settings(method, given):
    p = params(given, SIGNAL_DEFAULTS[method])
    for key in ("lookback", "fast", "slow", "signal_span", "periods_per_year"):
        if key in p:
            p[key] = integer(p[key], key, minimum=2)
    if method == "macd" and p["fast"] >= p["slow"]:
        raise ValueError("fast must be less than slow")
    if method == "rsi_reversion":
        lo, hi = (number(p[k], k) for k in ("lower", "upper"))
        if not 0 < lo < 50 < hi < 100:
            raise ValueError("RSI thresholds must satisfy 0 < lower < 50 < upper < 100")
    for key in ("entry_z", "target_vol", "max_exposure"):
        if key in p and number(p[key], key) <= 0:
            raise ValueError(f"{key} must be positive")
    return p


def warmup(method, p):
    return p["slow"] + p["signal_span"] if method == "macd" else p["lookback"] + 1


def signal_check(method):
    def check(data, given):
        p = settings(method, given)
        if len(data["prices"]) < warmup(method, p) + 1:
            raise ValueError("insufficient history for a warmed, lagged signal")
    return check


def decisions(method, prices, given):
    """Close-time desired exposures, for the NEXT bar; internal unlagged primitive."""
    p = settings(method, given)
    x = pd.Series(prices, dtype=float)
    n = p.get("lookback")
    if method == "donchian_breakout":
        upper, lower = x.rolling(n).max().shift(1), x.rolling(n).min().shift(1)
        target = pd.Series(np.nan, index=x.index)
        target.loc[x > upper] = 1.0
        target.loc[x < lower] = -1.0
        target = target.ffill().where(upper.notna())
        return target.where(target.notna(), 0.0).where(upper.notna())
    if method in ("bollinger_reversion", "rsi_reversion"):
        if method == "bollinger_reversion":
            mean, std = x.rolling(n).mean(), x.rolling(n).std()
            indicator = ((x - mean) / std.replace(0, np.nan)).where(std != 0, 0.0)
            lower, upper, center = -p["entry_z"], p["entry_z"], 0.0
        else:
            delta = x.diff()
            # Explicit simple rolling RSI, not Wilder's recursive smoothing.
            gain, loss = delta.clip(lower=0).rolling(n).mean(), (-delta.clip(upper=0)).rolling(n).mean()
            indicator = 100 * gain / (gain + loss).replace(0, np.nan)
            indicator = indicator.where((gain + loss) != 0, 50.0)
            lower, upper, center = p["lower"], p["upper"], 50.0
        values, position = [], 0.0
        for value in indicator:
            if pd.isna(value):
                values.append(np.nan)
                continue
            if value < lower:
                position = 1.0
            elif value > upper:
                position = -1.0
            elif (position > 0 and value >= center) or (position < 0 and value <= center):
                position = 0.0
            values.append(position)
        return pd.Series(values, index=x.index)
    if method == "macd":
        line = (x.ewm(span=p["fast"], adjust=False, min_periods=p["fast"]).mean()
                - x.ewm(span=p["slow"], adjust=False, min_periods=p["slow"]).mean())
        baseline = line.ewm(span=p["signal_span"], adjust=False,
                            min_periods=p["signal_span"]).mean()
        return np.sign(line - baseline)
    ret = x.pct_change(fill_method=None)
    vol = ret.rolling(n).std() * np.sqrt(p["periods_per_year"])
    scale = (p["target_vol"] / vol.replace(0, np.nan)).clip(upper=p["max_exposure"])
    scale = scale.where(vol != 0, 0.0)
    return np.sign(x / x.shift(n) - 1) * scale


def signal(method):
    def run(data, given):
        signal_check(method)(data, given)
        result = decisions(method, data["prices"], given).shift(1)
        result.attrs["lag_bars"] = 1
        return result
    return run


def cross_sectional_check(data, given):
    p = params(given, {"lookback": 60, "top_k": 3})
    integer(p["lookback"], "lookback", minimum=2)
    integer(p["top_k"], "top_k", maximum=np.shape(data["asset_returns"])[1])
    if len(data["asset_returns"]) < p["lookback"]:
        raise ValueError("history must cover lookback")
    if (np.asarray(data["asset_returns"]) <= -1).any():
        raise ValueError("momentum requires returns strictly greater than -1")


def cross_sectional_momentum(data, given):
    cross_sectional_check(data, given)
    p = params(given, {"lookback": 60, "top_k": 3})
    frame = pd.DataFrame(data["asset_returns"])
    cumulative = np.log1p(frame.iloc[-p["lookback"]:]).sum()
    # Stable ties follow input column order; positive absolute momentum filter leaves cash.
    chosen = cumulative.sort_values(ascending=False, kind="stable").iloc[:p["top_k"]]
    weights = pd.Series(0.0, index=frame.columns)
    weights.loc[chosen[chosen > 0].index] = 1.0 / p["top_k"]
    weights.attrs["cash_weight"] = float(1 - weights.sum())
    weights.attrs["apply_to"] = "subsequent returns only"
    return weights
