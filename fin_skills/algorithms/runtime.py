"""Bounded input contracts and lazy adapters over the existing algorithm implementations."""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np
import pandas as pd

MAX_POINTS = 1_000_000


def integer(value, name, maximum=100_000, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return int(value)


def number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside the supported range")
    return float(value)


def params(given, defaults):
    unknown = set(given) - set(defaults)
    if unknown:
        raise TypeError(f"unexpected parameters: {sorted(unknown)}; accepted: {sorted(defaults)}")
    return dict(defaults, **given)


def _array(value, name, ndim):
    raw = np.asarray(value)
    if raw.size > MAX_POINTS:
        raise ValueError(f"{name} exceeds {MAX_POINTS} values")
    if raw.ndim != ndim or raw.size == 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty numeric {ndim}-D array")
    a = raw.astype(float)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains missing or nonfinite values; clean them explicitly")
    if isinstance(value, (pd.Series, pd.DataFrame)):
        if (value.index.has_duplicates or value.index.hasnans or
                not value.index.is_monotonic_increasing):
            raise ValueError(f"{name} index must be unique, nonmissing and increasing")
        if isinstance(value, pd.DataFrame) and (value.columns.has_duplicates or value.columns.hasnans):
            raise ValueError(f"{name} columns must be unique and nonmissing")
    return value.astype(float).copy() if isinstance(value, (pd.Series, pd.DataFrame)) else a


def validate_data(registry, task, data):
    if not isinstance(data, Mapping) or not data:
        raise TypeError("data must be a nonempty mapping")
    known = {key for a in registry.algorithms(task) for key in a.inputs}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown data fields for {task}: {sorted(unknown)}")
    out = dict(data)
    rows = []
    for key in ("series", "returns", "prices", "y", "asset_returns", "X", "X_predict",
                "volume_forecast"):
        if key not in out:
            continue
        ndim = 2 if key in ("asset_returns", "X", "X_predict") else 1
        out[key] = _array(out[key], key, ndim)
        if key not in ("X_predict", "volume_forecast"):
            rows.append(len(out[key]))
        if key in ("returns", "asset_returns"):
            a = np.asarray(out[key])
            if (a < -1).any() or np.max(np.abs(a.mean(axis=0))) > 0.10:
                raise ValueError(f"{key} requires decimal simple returns; routing policy rejects "
                                 "values below -1 or absolute per-period mean above 0.10")
        if key == "prices" and (np.asarray(out[key]) <= 0).any():
            raise ValueError("prices must be positive")
    if len(set(rows)) > 1:
        raise ValueError("training arrays must have the same number of rows")
    historical = [out[k] for k in ("series", "returns", "prices", "X", "y", "asset_returns")
                  if k in out and isinstance(out[k], (pd.Series, pd.DataFrame))]
    if historical and any(not historical[0].index.equals(v.index) for v in historical[1:]):
        raise ValueError("training indexes must match exactly; align them explicitly")
    if "X" in out and "X_predict" in out:
        if np.shape(out["X"])[1] != np.shape(out["X_predict"])[1]:
            raise ValueError("X_predict must have the same feature count as X")
        if isinstance(out["X"], pd.DataFrame) and isinstance(out["X_predict"], pd.DataFrame):
            if not out["X"].columns.equals(out["X_predict"].columns):
                raise ValueError("X_predict columns must match X in order")
    for key in ("n_bins", "seasonal_period"):
        if key in out:
            out[key] = integer(out[key], key)
    if "shares" in out:
        out["shares"] = number(out["shares"], "shares", minimum=0)
    return out, min(rows) if rows else None


def portfolio(method, backend=None):
    def run(data, given):
        p = params(given, {"linkage": None} if method == "hrp" else {})
        returns = pd.DataFrame(data["asset_returns"])
        if method == "equal_weight":
            weights = np.full(returns.shape[1], 1 / returns.shape[1])
        else:
            variance = returns.var(ddof=1)
            if (variance <= 1e-16).any():
                raise ValueError("allocation requires nonconstant assets")
            if backend:
                from fin_skills.bridges.optimizers import OptimizerInput, optimize
                return optimize(OptimizerInput(returns, linkage=p.get("linkage")),
                                backend=backend, model="HRP" if method == "hrp" else "min_vol")
            if method == "inverse_volatility":
                weights = 1 / np.sqrt(variance.to_numpy())
                weights /= weights.sum()
            elif method == "min_variance":
                from fin_skills.models.optimizers import min_variance
                weights = min_variance(returns.cov().to_numpy(), long_only=True)
            else:
                from fin_skills.bridges.optimizers import require_linkage
                from fin_skills.libraries.weight_traps import hrp_weights
                linkage = require_linkage("HRP", p["linkage"])
                if returns.shape[1] < 2:
                    raise ValueError("HRP requires at least two assets")
                weights = hrp_weights(returns, linkage_method=linkage)
        return pd.Series(weights, index=returns.columns, name="weight")
    return run


def forecast(method):
    def run(data, given):
        p = params(given, {"horizon": 1})
        h = integer(p["horizon"], "horizon", maximum=10_000)
        series = np.asarray(data["series"])
        if method == "arima":
            from statsmodels.tsa.arima.model import ARIMA
            return np.asarray(ARIMA(series, order=(1, 1, 0)).fit().forecast(h))
        from fin_skills.models import forecast_baselines as f
        if method == "seasonal_naive":
            return f.f_seasonal_naive(series, h, data["seasonal_period"])
        return getattr(f, "f_" + method)(series, h)
    return run


def volatility(method):
    def run(data, given):
        defaults = {"periods_per_year": 1}
        if method == "ewma":
            defaults["decay"] = 0.94
        p = params(given, defaults)
        periods = integer(p["periods_per_year"], "periods_per_year")
        returns = np.asarray(data["returns"])
        if method == "ewma":
            decay = number(p["decay"], "decay", minimum=0, maximum=1)
            if not 0 < decay < 1:
                raise ValueError("decay must be strictly between 0 and 1")
            # Zero-mean recursive variance. Initialization uses the first observation only.
            variance = returns[0] ** 2
            for r in returns[1:]:
                variance = decay * variance + (1 - decay) * r ** 2
        else:
            variance = np.var(returns, ddof=1)
        return {"volatility": float(np.sqrt(variance * periods)),
                "periods_per_year": periods, "method": method}
    return run


def risk(method):
    def run(data, given):
        p = params(given, {"confidence": 0.95})
        level = number(p["confidence"], "confidence", minimum=0, maximum=1)
        if not 0 < level < 1:
            raise ValueError("confidence must be strictly between 0 and 1")
        from fin_skills.models.var_cvar import var_es_historical, var_es_normal
        fn = var_es_historical if method == "historical" else var_es_normal
        var, es = fn(-np.asarray(data["returns"]), level)
        return {"var": var, "expected_shortfall": es, "confidence": level,
                "units": "per-period decimal loss", "observations": len(data["returns"])}
    return run


def signal(method):
    def run(data, given):
        from fin_skills.strategies.trend_models import ma_crossover_signal, tsmom_signal
        if method == "momentum":
            p = params(given, {"lookback": 20})
            window = integer(p["lookback"], "lookback")
            source = pd.Series(data["returns"])
            if (source <= -1).any():
                raise ValueError("momentum requires returns strictly greater than -1")
            result = tsmom_signal(source, window)
        else:
            p = params(given, {"fast": 5, "slow": 20})
            fast, window = (integer(p[k], k) for k in ("fast", "slow"))
            source = pd.Series(data["prices"])
            result = ma_crossover_signal(source, fast, window)
        if len(source) <= window:
            raise ValueError("need more observations than the signal window for a lagged signal")
        # Position at t uses information strictly before t; warmup remains visibly unavailable.
        result = result.shift(1)
        result.attrs["lag_bars"] = 1
        return result
    return run


def execution(method):
    def run(data, given):
        params(given, {})
        from fin_skills.strategies.execution_algos import twap_schedule, vwap_schedule
        if method == "twap":
            return twap_schedule(data["shares"], data["n_bins"])
        return vwap_schedule(data["shares"], data["volume_forecast"])
    return run


def pricing(method):
    def run(data, given):
        p = params(given, {"steps": 200} if method == "american_crr" else {})
        option = data["option"]
        keys = {"S", "K", "T", "r", "q", "sigma", "flag", "exercise"}
        if not isinstance(option, Mapping) or set(option) != keys:
            raise ValueError(f"option must contain exactly {sorted(keys)}")
        expected = "american" if method == "american_crr" else "european"
        if option["exercise"] != expected:
            raise ValueError(f"this algorithm requires exercise={expected!r}")
        if option["flag"] not in ("c", "p"):
            raise ValueError("option.flag must be c or p")
        values = {k: number(option[k], f"option.{k}") for k in ("S", "K", "T", "r", "q", "sigma")}
        if any(values[k] <= 0 for k in ("S", "K", "T", "sigma")):
            raise ValueError("option S, K, T and sigma must be positive")
        from fin_skills.models.option_models import bsm_price, crr_price
        if method == "american_crr":
            return crr_price(**values, flag=option["flag"], american=True,
                             steps=integer(p["steps"], "steps", maximum=2_000))
        return bsm_price(**values, flag=option["flag"])
    return run


def supervised(method):
    def run(data, given):
        from .models import supervised_estimator
        estimator = supervised_estimator(method, given)
        # Fit every transform on training X only; prediction data never enters fit.
        estimator.fit(data["X"], data["y"])
        return np.asarray(estimator.predict(data["X_predict"]))
    return run


from .contracts import preflight  # shared checks, after runtime helpers are defined
