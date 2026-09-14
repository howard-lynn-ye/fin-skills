"""Fitted forecasting and supervised models with explicit trusted-artifact persistence."""
from __future__ import annotations

import hashlib
import json
import pickle
import platform
import zipfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from .runtime import _array, integer, params, validate_data

ARTIFACT_SCHEMA = 1


def environment(library="fin-skills"):
    result = {"python": platform.python_version()}
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode("utf-8") + b"\0" + source.read_bytes())
    result["algorithm_implementation_sha256"] = digest.hexdigest()
    for name in dict.fromkeys(("fin-skills", "numpy", "pandas", "scipy", library)):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def supervised_estimator(method, given):
    p = params(given, {"seed": 0})
    seed = integer(p["seed"], "seed", maximum=2**32 - 1, minimum=0)
    if method == "qlib_alpha":
        from .extended import QlibRidge
        return QlibRidge()
    if method.startswith(("lightgbm", "xgboost")):
        if method.startswith("lightgbm"):
            from lightgbm import LGBMClassifier, LGBMRegressor
            cls = LGBMClassifier if method.endswith("classification") else LGBMRegressor
            return cls(n_estimators=100, max_depth=6, random_state=seed, n_jobs=1, verbosity=-1)
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if method.endswith("classification") else XGBRegressor
        return cls(n_estimators=100, max_depth=6, random_state=seed, n_jobs=1)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    constructors = {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=1000, random_state=seed)),
        "random_forest_regression": lambda: RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=1),
        "random_forest_classification": lambda: RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=1),
    }
    if method not in constructors:
        raise NotImplementedError(f"no fitted-model constructor for {method}")
    return constructors[method]()


@dataclass
class FittedModel:
    algorithm: str
    task: str
    estimator: object
    metadata: dict

    def predict(self, X=None, *, horizon=1):
        if self.task == "forecast":
            if X is not None:
                raise TypeError("forecast predict accepts horizon, not X")
            h = integer(horizon, "horizon", maximum=10_000)
            if self.algorithm == "arima":
                result = self.estimator.forecast(h)
            elif self.algorithm == "auto_arima":
                result = self.estimator.predict(h=h)["mean"]
            else:
                from .runtime import forecast
                result = forecast(self.algorithm)(self.estimator, {"horizon": h})
            expected = h
        else:
            X = _array(X, "X_predict", 2)
            if X.shape[1] != self.metadata["n_features"]:
                raise ValueError("prediction feature count differs from training")
            columns = self.metadata["feature_columns"]
            if columns is not None:
                if not isinstance(X, pd.DataFrame) or list(X.columns) != columns:
                    raise ValueError("prediction requires training DataFrame columns in order")
            result = self.estimator.predict(X)
            expected = len(X)
        result = np.asarray(result)
        if result.shape != (expected,) or not np.isfinite(result).all():
            raise ValueError("model returned nonfinite or incorrectly shaped predictions")
        return result

    def save(self, path):
        """Write an exclusive ZIP artifact. Loading requires trusted=True (pickle inside)."""
        payload = pickle.dumps(self, protocol=5)
        manifest = {"schema_version": ARTIFACT_SCHEMA, "algorithm": self.algorithm,
                    "environment": self.metadata["environment"],
                    "sha256": hashlib.sha256(payload).hexdigest()}
        with zipfile.ZipFile(Path(path), "x", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, allow_nan=False))
            archive.writestr("model.pkl", payload)
        return Path(path)


def load_model(path, *, trusted=False, strict_versions=True):
    """Load only a trusted local artifact. Hashes detect corruption, not malicious pickle."""
    if trusted is not True:
        raise ValueError("model contains pickle; load only your own artifact with trusted=True")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["schema_version"] != ARTIFACT_SCHEMA:
            raise ValueError("unsupported model artifact schema")
        expected = manifest["environment"]
        current = environment()
        for name in expected:
            if name not in current:
                current.update(environment(name))
        if strict_versions and any(current.get(k) != v for k, v in expected.items()):
            raise ValueError("model environment differs; use its recorded dependency versions")
        payload = archive.read("model.pkl")
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise ValueError("model artifact checksum mismatch")
    model = pickle.loads(payload)
    if not isinstance(model, FittedModel) or model.algorithm != manifest["algorithm"]:
        raise ValueError("unexpected model artifact content")
    return model


def fit(algorithm_id, data, *, registry=None, **parameters):
    """Fit once on chronological training data; predict never refits transformations."""
    if registry is None:
        from . import _DEFAULT
        registry = _DEFAULT
    spec = registry.get(algorithm_id)
    if spec.task not in ("forecast", "regression", "classification"):
        raise ValueError("fit supports forecast, regression and classification")
    if "X_predict" in data:
        raise ValueError("fit accepts training data only; pass X_predict to model.predict")
    raw = dict(data)
    if spec.task != "forecast":
        if "X" not in raw:
            raise ValueError("X is required")
        raw["X_predict"] = raw["X"].iloc[:1] if isinstance(raw["X"], pd.DataFrame) else raw["X"][:1]
    clean, n = validate_data(registry, spec.task, raw)
    if set(spec.inputs) - set(clean) or n is None or n < spec.min_observations:
        raise ValueError("missing training inputs or insufficient observations")
    status = registry.status(algorithm_id)
    if status == "catalog_only":
        raise NotImplementedError(f"{algorithm_id} has no execution adapter")
    if status != "ready":
        raise ImportError(f"{algorithm_id} requires {spec.library}")
    if algorithm_id in registry._checks:
        registry._checks[algorithm_id](clean, parameters)
    metadata = {"environment": environment(spec.library), "parameters": dict(parameters),
                "n_observations": n, "n_features": None, "feature_columns": None}
    from .research import fingerprint
    metadata["training_input_hash"] = fingerprint({k: v for k, v in clean.items() if k != "X_predict"})
    history = clean["series"] if spec.task == "forecast" else clean["X"]
    metadata["last_training_index"] = str(history.index[-1]) if hasattr(history, "index") else n - 1
    if spec.task == "forecast":
        if parameters:
            raise TypeError("set forecast horizon on predict, not fit")
        if algorithm_id == "arima":
            from statsmodels.tsa.arima.model import ARIMA
            estimator = ARIMA(np.asarray(clean["series"]), order=(1, 1, 0)).fit()
        elif algorithm_id == "auto_arima":
            from statsforecast.models import AutoARIMA
            estimator = AutoARIMA(season_length=clean["seasonal_period"]).fit(
                y=np.asarray(clean["series"]))
        elif algorithm_id in ("naive", "mean", "drift", "seasonal_naive"):
            estimator = clean
        else:
            raise NotImplementedError("custom forecasts must provide their own fitted lifecycle")
    else:
        estimator = supervised_estimator(algorithm_id, parameters)
        estimator.fit(clean["X"], clean["y"])
        metadata.update(n_features=clean["X"].shape[1], feature_columns=(
            list(clean["X"].columns) if isinstance(clean["X"], pd.DataFrame) else None))
    return FittedModel(algorithm_id, spec.task, estimator, metadata)
