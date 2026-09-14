"""Shared, side-effect-free adapter checks used by selection and execution."""
import numpy as np

from .runtime import integer, number, params


def defaults(algorithm_id, task):
    if task == "portfolio":
        return {"linkage": None} if algorithm_id.endswith("hrp") else {}
    if task == "forecast":
        return {"horizon": 1}
    if task == "volatility":
        return dict(periods_per_year=1, **({"decay": 0.94} if
                    algorithm_id == "ewma_volatility" else {}))
    if task == "risk":
        return {"confidence": 0.95}
    if task in ("regression", "classification"):
        return {"seed": 0}
    if task == "signal":
        return {"lookback": 20} if algorithm_id == "momentum" else {"fast": 5, "slow": 20}
    if algorithm_id == "american_crr":
        return {"steps": 200}
    if algorithm_id == "gaussian_hmm":
        return {"n_states": 2, "seed": 0}
    if algorithm_id == "change_points":
        return {"penalty": 10.0, "min_size": 5}
    return {}


def preflight(algorithm_id, task):
    def check(data, given):
        p = params(given, defaults(algorithm_id, task))
        for name, value in p.items():
            if name in ("horizon", "steps", "fast", "slow", "lookback", "periods_per_year",
                        "n_states", "min_size"):
                cap = 10_000 if name == "horizon" else 2_000 if name == "steps" else 100_000
                integer(value, name, maximum=cap)
            if name == "seed":
                integer(value, name, maximum=2**32 - 1, minimum=0)
            if name in ("confidence", "decay") and not 0 < number(value, name) < 1:
                raise ValueError(f"{name} must be strictly between 0 and 1")
        if task == "portfolio":
            a = np.asarray(data["asset_returns"])
            if algorithm_id != "equal_weight" and (a.var(axis=0, ddof=1) <= 1e-16).any():
                raise ValueError("allocation requires nonconstant assets")
            if algorithm_id.endswith("hrp"):
                from fin_skills.bridges.optimizers import require_linkage
                require_linkage("HRP", p["linkage"])
                if a.shape[1] < 2:
                    raise ValueError("HRP requires at least two assets")
        if task == "forecast" and "seasonal_period" in data:
            if algorithm_id in ("seasonal_naive", "auto_arima"):
                if len(data["series"]) < data["seasonal_period"]:
                    raise ValueError("history must cover the seasonal period")
        if task == "classification":
            classes = np.unique(data["y"])
            if len(classes) < 2 or not np.equal(classes, np.floor(classes)).all():
                raise ValueError("classification requires at least two integer class labels")
            if algorithm_id.startswith(("xgboost", "lightgbm")):
                if not np.array_equal(classes, np.arange(len(classes))):
                    raise ValueError("boosted classifiers require contiguous labels starting at zero")
        if task == "signal":
            window = p.get("lookback", p.get("slow"))
            source = data["returns"] if algorithm_id == "momentum" else data["prices"]
            if len(source) <= window:
                raise ValueError("need more observations than the signal window for a lagged signal")
            if algorithm_id == "momentum" and (np.asarray(source) <= -1).any():
                raise ValueError("momentum requires returns strictly greater than -1")
            if algorithm_id != "momentum" and p["fast"] >= p["slow"]:
                raise ValueError("fast must be less than slow")
        if algorithm_id == "vwap":
            v = np.asarray(data["volume_forecast"])
            if (v < 0).any() or v.sum() <= 0:
                raise ValueError("volume_forecast must be nonnegative with positive total")
        if task == "pricing":
            if "heston_parameters" in data and algorithm_id != "quantlib_heston":
                raise ValueError("supplied Heston parameters require the Heston adapter")
            from collections.abc import Mapping
            option = data["option"]
            keys = {"S", "K", "T", "r", "q", "sigma", "flag", "exercise"}
            if not isinstance(option, Mapping) or set(option) != keys:
                raise ValueError(f"option must contain exactly {sorted(keys)}")
            expected = "american" if algorithm_id == "american_crr" else "european"
            if option["exercise"] != expected:
                raise ValueError(f"this algorithm requires exercise={expected!r}")
            if option["flag"] not in ("c", "p"):
                raise ValueError("option.flag must be c or p")
            for key in ("S", "K", "T", "r", "q", "sigma"):
                value = number(option[key], f"option.{key}")
                if key in ("S", "K", "T", "sigma") and value <= 0:
                    raise ValueError(f"option.{key} must be positive")
        if algorithm_id == "engle_granger" and np.shape(data["X"])[1] != 2:
            raise ValueError("Engle-Granger requires exactly two series in X")
        if algorithm_id == "quantlib_heston":
            from .extended import validate_heston
            validate_heston(data["option"], data["heston_parameters"])
        if algorithm_id == "gaussian_hmm" and not 2 <= p["n_states"] <= 10:
            raise ValueError("n_states must be in 2..10")
        if algorithm_id == "gaussian_hmm":
            if len(np.unique(np.asarray(data["X"]), axis=0)) < p["n_states"]:
                raise ValueError("HMM requires at least n_states distinct observations")
        if algorithm_id == "change_points":
            if number(p["penalty"], "penalty") <= 0:
                raise ValueError("penalty must be positive")
    return check
