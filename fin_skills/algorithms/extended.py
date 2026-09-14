"""Optional backend adapters. Imports occur only when the adapter is executed."""
from __future__ import annotations

import datetime as dt
import threading

import numpy as np
import pandas as pd

from .runtime import integer, number, params

_QL_LOCK = threading.RLock()


def auto_arima(data, given):
    from statsforecast.models import AutoARIMA
    p = params(given, {"horizon": 1})
    model = AutoARIMA(season_length=data["seasonal_period"])
    return model.fit(y=np.asarray(data["series"])).predict(h=p["horizon"])["mean"]


def arch_volatility(method):
    def run(data, given):
        from arch import arch_model
        p = params(given, {"periods_per_year": 1})
        # arch fits percentage returns; convert forecast variance back to decimal units.
        model = arch_model(np.asarray(data["returns"]) * 100, mean="Zero",
                           vol=method.upper(), p=1, o=1 if method == "egarch" else 0,
                           q=1, rescale=False)
        fitted = model.fit(disp="off", show_warning=False)
        if fitted.convergence_flag != 0:
            raise RuntimeError(f"{method} optimization did not converge")
        variance = float(fitted.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]) / 10_000
        if not np.isfinite(variance) or variance < 0:
            raise ValueError("invalid variance forecast")
        return {"volatility": float(np.sqrt(variance * p["periods_per_year"])),
                "periods_per_year": p["periods_per_year"], "method": method,
                "horizon": 1, "converged": True}
    return run


def engle_granger(data, given):
    from statsmodels.tsa.stattools import coint
    params(given, {})
    X = np.asarray(data["X"])
    if (X.var(axis=0) <= 1e-16).any():
        raise ValueError("cointegration requires nonconstant series")
    statistic, pvalue, critical = coint(X[:, 0], X[:, 1], trend="c", autolag="aic")
    if not np.isfinite(statistic):
        raise ValueError("degenerate cointegration statistic (possibly collinear series)")
    return {"statistic": float(statistic), "pvalue": float(pvalue),
            "critical_values": np.asarray(critical).tolist(),
            "null": "no cointegration", "assumption": "both input series are I(1)",
            "timing": "full supplied training history; not a trading signal"}


def gaussian_hmm(data, given):
    from hmmlearn.hmm import GaussianHMM
    p = params(given, {"n_states": 2, "seed": 0})
    X = np.asarray(data["X"])
    model = GaussianHMM(n_components=p["n_states"], covariance_type="diag",
                        n_iter=200, random_state=p["seed"]).fit(X)
    history = list(model.monitor_.history)
    if len(history) < 2 or history[-1] - history[-2] > model.tol:
        raise RuntimeError("HMM did not converge within its iteration budget")
    return {"states": model.predict(X).tolist(), "means": model.means_.tolist(),
            "transition_matrix": model.transmat_.tolist(), "converged": True,
            "timing": "retrospective full-sample decoding; unavailable for historical trading",
            "label_meaning": "arbitrary state IDs, not bullish/bearish labels"}


def change_points(data, given):
    import ruptures as rpt
    p = params(given, {"penalty": 10.0, "min_size": 5})
    ends = rpt.Pelt(model="l2", min_size=p["min_size"], jump=1).fit(
        np.asarray(data["series"])).predict(pen=p["penalty"])
    return {"segment_ends_exclusive": ends, "timing": "retrospective; final end is sample length"}


def validate_heston(opt, hp):
    expected = {"v0", "kappa", "theta", "sigma", "rho", "valuation_date", "expiry_date"}
    if not isinstance(hp, dict) or set(hp) != expected:
        raise ValueError(f"heston_parameters requires exactly {sorted(expected)}")
    values = {k: number(hp[k], k) for k in ("v0", "kappa", "theta", "sigma", "rho")}
    if any(values[k] <= 0 for k in ("v0", "kappa", "theta", "sigma")) or abs(values["rho"]) >= 1:
        raise ValueError("Heston variances, speeds and vol-of-vol must be positive; abs(rho)<1")
    ev, ex = (dt.date.fromisoformat(hp[k]) for k in ("valuation_date", "expiry_date"))
    if ex <= ev or abs((ex - ev).days / 365 - opt["T"]) > 1e-10:
        raise ValueError("option.T must equal (expiry_date - valuation_date).days / 365")
    return values, ev, ex


def heston(data, given):
    params(given, {})
    opt = data["option"]
    values, ev, ex = validate_heston(opt, data["heston_parameters"])
    from fin_skills.bridges.quantlib import evaluation_date
    with _QL_LOCK, evaluation_date(ev, curve_ref=ev, expiry=ex) as ql:
        ref, expiry = ql.Date(ev.day, ev.month, ev.year), ql.Date(ex.day, ex.month, ex.year)
        day_count = ql.Actual365Fixed()
        rf = ql.YieldTermStructureHandle(ql.FlatForward(ref, opt["r"], day_count))
        div = ql.YieldTermStructureHandle(ql.FlatForward(ref, opt["q"], day_count))
        process = ql.HestonProcess(rf, div, ql.QuoteHandle(ql.SimpleQuote(opt["S"])),
                                  *(values[k] for k in ("v0", "kappa", "theta", "sigma", "rho")))
        instrument = ql.VanillaOption(ql.PlainVanillaPayoff(
            ql.Option.Call if opt["flag"] == "c" else ql.Option.Put, opt["K"]),
            ql.EuropeanExercise(expiry))
        instrument.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(process)))
        price = float(instrument.NPV())
    if not np.isfinite(price) or price < 0:
        raise ValueError("invalid Heston price")
    return price


class _FrameDataset:
    """In-memory implementation of Qlib LinearModel's public prepare protocol."""
    def __init__(self, X, y=None):
        self.X = pd.DataFrame(X).copy()
        self.y = y

    def prepare(self, segment, col_set, data_key):
        if segment == "test" and col_set == "feature":
            return self.X.copy()
        if segment == "train" and col_set == ["feature", "label"] and self.y is not None:
            label = pd.DataFrame({"target": np.asarray(self.y)}, index=self.X.index)
            return pd.concat({"feature": self.X, "label": label}, axis=1)
        raise ValueError("unsupported Qlib dataset request")


class QlibRidge:
    """Qlib's real LinearModel, supplied with explicit precomputed features and labels."""
    def fit(self, X, y):
        from qlib.contrib.model.linear import LinearModel
        self.model = LinearModel(estimator="ridge", alpha=1.0, fit_intercept=True,
                                 include_valid=False)
        self.model.fit(_FrameDataset(X, y))
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(_FrameDataset(X)))
