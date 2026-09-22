"""Thin, lazy native adapters. No fallback implementations and no broker side effects."""
from collections.abc import Mapping
import importlib
import inspect

import numpy as np
import pandas as pd

from fin_skills.algorithms.runtime import _array, integer
from .base import ModelArtifact
from .upstream_catalog import (TA_FUNCTIONS, SKLEARN, TIME_SERIES, RISK_MEASURES,
                               RISK_SPECIAL, FRONTIERS, CVX, COMPARISONS, SPLITS)


def _data(data, required, optional=()):
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")
    missing, unknown = set(required) - set(data), set(data) - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"missing data fields: {sorted(missing)}; unknown: {sorted(unknown)}")
    return data


def _returns(value):
    frame = pd.DataFrame(_array(value, "asset_returns", 2))
    if len(frame) < 5 or (frame < -1).any().any():
        raise ValueError("asset_returns requires at least 5 observations of decimal simple returns")
    return frame


class UpstreamModel(ModelArtifact):
    def __init__(self, model_id, parameters):
        from .catalog import model_catalog
        card = next(c for c in model_catalog() if c["id"] == model_id)
        self.model_id, self.parameters = model_id, dict(parameters)
        self.dependencies = (card["library"],)
        self.estimator = None
        self.training = None
        self.columns = None

    def fit(self, data):
        id_, p = self.model_id, dict(self.parameters)
        if id_ in SKLEARN:
            module, name, task, scale = SKLEARN[id_]
            unsupervised = task in ("clustering", "decomposition")
            _data(data, ["X"] if unsupervised else ["X", "y"], ["X_predict"])
            X = _array(data["X"], "X", 1 if task == "calibration" else 2)
            self.columns = list(X.columns) if isinstance(X, pd.DataFrame) else None
            y = None if unsupervised else _array(data["y"], "y", 1)
            if y is not None and len(y) != len(X):
                raise ValueError("X and y lengths differ")
            if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series) and not X.index.equals(y.index):
                raise ValueError("X and y indices differ")
            cls = getattr(importlib.import_module("sklearn." + module), name)
            seed = p.pop("seed", 0)
            if "random_state" in inspect.signature(cls).parameters:
                p.setdefault("random_state", seed)
            if id_ in ("mlp", "hist_gradient_boosting"):
                if p.get("early_stopping", False) is not False:
                    raise ValueError("random early-stopping splits are disabled; use temporal validation")
                p["early_stopping"] = False
            if id_ == "isotonic_calibration":
                p.setdefault("out_of_bounds", "clip")
            estimator = cls(**p)
            if scale:
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                estimator = make_pipeline(StandardScaler(), estimator)
            self.estimator = estimator.fit(X, y)
            self.training = np.asarray(X)
            return self
        if id_ not in TIME_SERIES or TIME_SERIES[id_][0] != "forecast":
            raise ValueError("method has no persistent fit/predict interface; use run")
        field = TIME_SERIES[id_][1]
        _data(data, [field], ["exog"])
        X = np.asarray(_array(data[field], field, 1 if field == "series" else 2))
        self.training = X.copy()
        fit_parameters = dict(p.pop("fit_parameters", {}))
        from statsmodels.tsa.api import (SARIMAX, VAR, VECM, UnobservedComponents,
                                        DynamicFactor, ExponentialSmoothing)
        if id_ == "sarimax":
            p.setdefault("order", (1, 0, 0))
            exog = None if "exog" not in data else _array(data["exog"], "exog", 2)
            cls, args = SARIMAX, {"exog": exog}
        elif "exog" in data:
            raise ValueError("this wrapper supports exog only for SARIMAX")
        elif id_ == "var":
            cls, args = VAR, {}
            fit_parameters.setdefault("maxlags", p.pop("maxlags", 1))
        elif id_ == "vecm":
            cls, args = VECM, {}
            p.setdefault("k_ar_diff", 1)
        elif id_ == "unobserved_components":
            cls, args = UnobservedComponents, {}
            p.setdefault("level", "local level")
        elif id_ == "dynamic_factor":
            cls, args = DynamicFactor, {}
            p.setdefault("k_factors", 1)
            p.setdefault("factor_order", 1)
        else:
            cls, args = ExponentialSmoothing, {}
        if id_ in ("sarimax", "unobserved_components", "dynamic_factor"):
            fit_parameters.setdefault("disp", False)
        self.estimator = cls(X, **args, **p).fit(**fit_parameters)
        if hasattr(self.estimator, "mle_retvals") and not self.estimator.mle_retvals.get("converged", True):
            raise RuntimeError("native fit did not converge; revise training specification")
        return self

    def predict(self, X=None, *, horizon=1):
        if self.estimator is None:
            raise ValueError("fit before prediction")
        id_ = self.model_id
        if id_ in SKLEARN:
            if id_ in ("dbscan_regime", "spectral_regime"):
                raise ValueError("transductive clustering has no native out-of-sample prediction")
            if self.columns is not None and (not isinstance(X, pd.DataFrame) or list(X.columns) != self.columns):
                raise ValueError("prediction columns must match training columns in order")
            X = _array(X, "X_predict", 1 if id_ == "isotonic_calibration" else 2)
            task = SKLEARN[id_][2]
            method = self.estimator.transform if task == "decomposition" else self.estimator.predict
            return method(X)
        h = integer(horizon, "horizon", maximum=10000)
        if id_ == "var":
            if X is not None:
                raise ValueError("VAR predicts from stored training history")
            return self.estimator.forecast(self.training[-max(1, self.estimator.k_ar):], steps=h)
        if id_ == "vecm":
            if X is not None:
                raise ValueError("VECM predicts from stored training history")
            return self.estimator.predict(steps=h)
        if id_ == "sarimax":
            exog = None if X is None else _array(X, "future_exog", 2)
            return np.asarray(self.estimator.forecast(h, exog=exog))
        if X is not None:
            raise ValueError("forecast uses stored history; refit explicitly for new observations")
        return np.asarray(self.estimator.forecast(h))

    def run(self, data):
        id_, p = self.model_id, dict(self.parameters)
        if id_ in TA_FUNCTIONS:
            return self._technical(data, p)
        if id_ in SKLEARN:
            task = SKLEARN[id_][2]
            if task not in ("clustering", "decomposition") and "X_predict" not in data:
                raise ValueError("run requires held-out X_predict; use fit for stateful training")
            self.fit(data)
            if id_ in ("dbscan_regime", "spectral_regime"):
                final = self.estimator.steps[-1][1]
                return {"labels": final.labels_.copy(), "timing": "retrospective training labels"}
            if task in ("clustering", "decomposition"):
                return self.predict(data.get("X_predict", data["X"]))
            return self.predict(data["X_predict"])
        if id_ in TIME_SERIES:
            if TIME_SERIES[id_][0] == "forecast":
                raise ValueError("forecast requires fit followed by predict(horizon=...)")
            return self._time_analysis(data, p)
        if id_ in RISK_MEASURES or id_ in RISK_SPECIAL:
            return self._riskfolio(data, p)
        if id_ in FRONTIERS:
            return self._frontier(data, p)
        if id_ in CVX:
            from fin_skills.bridges.cvxportfolio import native_policy
            _data(data, ["holdings", "market_data", "t"])
            policy = native_policy(id_, **p)
            return policy.execute(data["holdings"], data["market_data"], t=data["t"])
        if id_ in COMPARISONS:
            return self._comparison(data, p)
        if id_ in SPLITS:
            _data(data, ["X"], ["groups"])
            X = _array(data["X"], "X", 2)
            from sklearn import model_selection
            splitter = getattr(model_selection, SPLITS[id_])(**p)
            return [{"train": train, "test": test} for train, test in
                    splitter.split(X, groups=data.get("groups"))]
        raise KeyError(id_)

    def _technical(self, data, p):
        from talib import abstract
        _data(data, ["inputs"])
        source = data["inputs"]
        if isinstance(source, pd.DataFrame):
            source = {col: source[col] for col in source.columns}
        if not isinstance(source, Mapping):
            raise TypeError("inputs must be a mapping or DataFrame")
        meta = TA_FUNCTIONS[self.model_id]
        missing = set(meta["inputs"]) - set(source)
        if missing:
            raise ValueError(f"missing TA-Lib inputs: {sorted(missing)}")
        arrays = {k: _array(source[k], k, 1) for k in meta["inputs"]}
        if len({len(v) for v in arrays.values()}) != 1:
            raise ValueError("input lengths differ")
        indices = [v.index for v in arrays.values() if isinstance(v, pd.Series)]
        if indices and any(not index.equals(indices[0]) for index in indices):
            raise ValueError("input indices differ")
        function = abstract.Function(meta["function"])
        if set(p) - set(meta["parameters"]):
            raise ValueError("unknown native TA-Lib parameters")
        function.set_parameters(p)
        output = function({k: np.asarray(v, dtype=float) for k, v in arrays.items()})
        values = [output] if len(meta["outputs"]) == 1 else output
        frame = pd.DataFrame(dict(zip(meta["outputs"], values)),
                             index=indices[0] if indices else None)
        return {"values": frame, "lookback": int(function.lookback),
                "native_function": meta["function"], "warmup": "preserved as NaN"}

    def _time_analysis(self, data, p):
        id_ = self.model_id
        field = TIME_SERIES[id_][1]
        _data(data, [field])
        X = np.asarray(_array(data[field], field, 1 if field == "series" else 2))
        from statsmodels.tsa import stattools
        if id_ == "adf_test":
            result = stattools.adfuller(X, **p)
            return dict(statistic=float(result[0]), pvalue=float(result[1]),
                        used_lag=int(result[2]), nobs=int(result[3]), critical_values=result[4])
        if id_ == "kpss_test":
            result = stattools.kpss(X, **p)
            return dict(statistic=float(result[0]), pvalue=float(result[1]),
                        lags=int(result[2]), critical_values=result[3])
        if id_ == "granger_test":
            if X.shape[1] != 2:
                raise ValueError("Granger test requires exactly two columns: target, predictor")
            p.setdefault("maxlag", 1)
            result = stattools.grangercausalitytests(X, **p)
            return {str(lag): tests for lag, (tests, _) in result.items()}
        if id_ == "stl_decomposition":
            from statsmodels.tsa.seasonal import STL
            if "period" not in p:
                raise ValueError("declare period explicitly")
            result = STL(X, **p).fit()
            return dict(trend=result.trend, seasonal=result.seasonal, residual=result.resid,
                        timing="retrospective decomposition")
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
        fit_parameters = dict(p.pop("fit_parameters", {}))
        p.setdefault("k_regimes", 2)
        cls = MarkovRegression if id_ == "markov_regression" else MarkovAutoregression
        if id_ == "markov_autoregression":
            p.setdefault("order", 1)
        fit_parameters.setdefault("disp", False)
        result = cls(X, **p).fit(**fit_parameters)
        if not result.mle_retvals.get("converged", False):
            raise RuntimeError("native regime fit did not converge")
        return dict(filtered_probabilities=np.asarray(result.filtered_marginal_probabilities),
                    smoothed_probabilities=np.asarray(result.smoothed_marginal_probabilities),
                    timing="parameters fitted on entire supplied sample; not historical signals")

    def _riskfolio(self, data, p):
        import riskfolio as rp
        _data(data, ["asset_returns"], ["factors", "risk_budget", "owa_weights"])
        returns = _returns(data["asset_returns"])
        stats = dict(p.pop("stats", {}))
        portfolio_parameters = dict(p.pop("portfolio", {}))
        if "returns" in portfolio_parameters or "factors" in portfolio_parameters:
            raise ValueError("supply returns/factors as data, not parameters")
        portfolio = rp.Portfolio(returns=returns, **portfolio_parameters)
        portfolio.assets_stats(**stats)
        id_ = self.model_id
        if id_ in RISK_MEASURES:
            p.setdefault("obj", "MinRisk")
            result = portfolio.optimization(rm=RISK_MEASURES[id_], **p)
        elif id_ == "risk_budget_allocation":
            budget = data.get("risk_budget")
            if budget is not None:
                budget = np.asarray(_array(budget, "risk_budget", 1)).reshape(-1, 1)
                if len(budget) != returns.shape[1] or (budget <= 0).any() or not np.isclose(budget.sum(), 1):
                    raise ValueError("positive asset risk budgets must sum to one")
            result = portfolio.rp_optimization(b=budget, **p)
        elif id_ == "factor_risk_budget":
            if "factors" not in data:
                raise ValueError("factor risk budgeting requires factors")
            factors = pd.DataFrame(_array(data["factors"], "factors", 2))
            if not factors.index.equals(returns.index):
                raise ValueError("factors and asset returns must share the same index")
            portfolio.factors = factors
            portfolio.factors_stats()
            budget = data.get("risk_budget")
            if budget is not None:
                budget = np.asarray(_array(budget, "risk_budget", 1)).reshape(-1, 1)
                if len(budget) != factors.shape[1] or (budget <= 0).any() or not np.isclose(budget.sum(), 1):
                    raise ValueError("positive factor risk budgets must sum to one")
            result = portfolio.rp_optimization(model="FC", b_f=budget, **p)
        elif id_ == "worst_case_allocation":
            # Deterministic uncertainty sets; no implicit bootstrap workload.
            uncertainty = dict(p.pop("uncertainty", {"box": "d", "ellip": "n"}))
            portfolio.wc_stats(**uncertainty)
            p.setdefault("obj", "MinRisk")
            result = portfolio.wc_optimization(**p)
        else:
            p.setdefault("obj", "MinRisk")
            weights = data.get("owa_weights")
            weights = None if weights is None else np.asarray(_array(weights, "owa_weights", 1)).reshape(-1, 1)
            result = portfolio.owa_optimization(owa_w=weights, **p)
        if result is None or not np.isfinite(result.to_numpy()).all():
            raise RuntimeError("native portfolio optimizer returned no finite solution")
        if not portfolio.sht and (result.to_numpy() < -1e-5).any():
            raise RuntimeError("native optimizer violated long-only constraint; factor models may "
                               "require explicit portfolio={'sht': True}")
        return result

    def _frontier(self, data, p):
        import pypfopt
        _data(data, ["asset_returns"])
        returns = _returns(data["asset_returns"])
        objective = p.pop("objective", "min_risk")
        target = p.pop("target_return", None)
        cls = getattr(pypfopt, FRONTIERS[self.model_id])
        # Mean and risk have per-observation units, not implicit annualization.
        if self.model_id == "semivariance_frontier":
            p.setdefault("frequency", 1)
        ef = cls(returns.mean(), returns, **p)
        if objective == "min_risk":
            method = {"semivariance_frontier": "min_semivariance", "cvar_frontier": "min_cvar",
                      "cdar_frontier": "min_cdar"}[self.model_id]
            getattr(ef, method)()
        elif objective == "efficient_return" and target is not None:
            ef.efficient_return(target)
        else:
            raise ValueError("objective must be min_risk or efficient_return with target_return")
        return pd.Series(ef.weights, index=returns.columns, name="weight")

    def _comparison(self, data, p):
        from arch import bootstrap
        _data(data, ["losses"], ["benchmark_losses"])
        losses = _array(data["losses"], "losses", 2)
        p.setdefault("seed", 0)
        p.setdefault("reps", 1000)
        integer(p["reps"], "reps", maximum=100000)
        cls = getattr(bootstrap, COMPARISONS[self.model_id])
        if self.model_id == "model_confidence_set":
            p.setdefault("size", .05)
            model = cls(losses, **p)
        else:
            if "benchmark_losses" not in data:
                raise ValueError("SPA/StepM requires benchmark_losses")
            benchmark = _array(data["benchmark_losses"], "benchmark_losses", 1)
            if len(benchmark) != len(losses):
                raise ValueError("benchmark and losses lengths differ")
            if isinstance(losses, pd.DataFrame) and isinstance(benchmark, pd.Series) and not losses.index.equals(benchmark.index):
                raise ValueError("benchmark and losses indices differ")
            model = cls(benchmark, losses, **p)
        model.compute()
        return ({"superior_models": model.superior_models} if self.model_id == "stepm_selection"
                else {"pvalues": model.pvalues})
