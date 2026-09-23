"""Authored financial method-selection tasks; synthetic development, never a sealed benchmark."""
from __future__ import annotations

import hashlib
import json
import math
from statistics import NormalDist

import numpy as np

PAIRS = {
    "volatility": ("historical_volatility", "ewma_volatility"),
    "tail_risk": ("historical_var_es", "normal_var_es"),
    "allocation": ("equal_weight", "inverse_volatility"),
    "forecast": ("naive", "drift"),
}
PARAMETERS = {
    "historical_volatility": {"periods_per_year": 252},
    "ewma_volatility": {"periods_per_year": 252, "decay": .94},
    "historical_var_es": {"confidence": .95}, "normal_var_es": {"confidence": .95},
    "equal_weight": {}, "inverse_volatility": {}, "naive": {"horizon": 1},
    "drift": {"horizon": 1},
}
RULES = {
    "historical_volatility": "Use sample standard deviation (ddof=1), annualized by sqrt(periods_per_year).",
    "ewma_volatility": "Use zero-mean EWMA variance, initialized by the first squared return, then annualize.",
    "historical_var_es": "Use empirical losses (negative returns); VaR is the linear-interpolated quantile; ES averages losses >= VaR.",
    "normal_var_es": "Fit Gaussian losses using sample standard deviation; report parametric VaR and ES.",
    "equal_weight": "Allocate equal capital to all assets, regardless of their volatility.",
    "inverse_volatility": "Allocate capital proportional to inverse sample volatility, normalized to sum to one.",
    "naive": "Forecast each future level as the last observed level; do not extrapolate a trend.",
    "drift": "Forecast the last level plus horizon times (last-first)/(n-1).",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def clean(value):
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def cases(seed=101, episodes=4):
    """Eight client contracts, four task families; disjoint synthetic time blocks.

    Two acquisition episodes precede two later evaluation episodes per contract.
    Seeds vary numerical inputs, not the number of independent task mechanisms.
    """
    rng = np.random.default_rng(seed)
    result = []
    for fi, (family, pair) in enumerate(PAIRS.items()):
        for variant, method in enumerate(pair):
            client = f"desk-{fi}-{variant}"
            params = dict(PARAMETERS[method])
            if family == "volatility":
                params["periods_per_year"] = 52 if variant == 0 else 252
                if variant:
                    params["decay"] = .90
            if family == "tail_risk":
                params["confidence"] = .90 if variant == 0 else .975
            if family == "forecast":
                params["horizon"] = 3 if variant == 0 else 5
            for episode in range(episodes):
                returns = rng.standard_t(5, size=(64, 3)) * np.array([.004, .012, .02])
                series = 100 + np.cumsum(rng.normal(.2, .7, 64))
                key = {"volatility": "returns", "tail_risk": "returns",
                       "allocation": "asset_returns", "forecast": "series"}[family]
                data = {key: clean(series if key == "series" else
                                  returns if key == "asset_returns" else returns[:, 1])}
                day = 1 + episode * 70
                result.append(dict(id=f"{client}-e{episode}", client=client, family=family,
                    episode=episode, phase="acquisition" if episode < 2 else "later_evaluation",
                    observed_block=[day, day + 63], decision_day=day + 64,
                    feedback_available_day=day + 65, data=data, data_sha256=digest(data),
                    method=method, parameters=params,
                    policy=RULES[method] + " Contract parameters: " + json.dumps(params)))
    return result


def reference(case):
    """Independent numerical reference; does not invoke the FinSkills dispatcher."""
    method, p = case["method"], case["parameters"]
    x = np.asarray(next(iter(case["data"].values())), dtype=float)
    if method == "historical_volatility":
        return [float(np.std(x, ddof=1) * math.sqrt(p["periods_per_year"]))]
    if method == "ewma_volatility":
        weights = (1-p["decay"]) * p["decay"] ** np.arange(len(x)-1, -1, -1)
        weights[0] = p["decay"] ** (len(x)-1)
        return [float(math.sqrt(float(weights @ (x*x)) * p["periods_per_year"]))]
    if method == "historical_var_es":
        losses = sorted(-float(v) for v in x)
        index = (len(losses)-1) * p["confidence"]
        lo, hi = math.floor(index), math.ceil(index)
        var = losses[lo] + (index-lo) * (losses[hi]-losses[lo])
        tail = [v for v in losses if v >= var]
        return [var, sum(tail)/len(tail)]
    if method == "normal_var_es":
        z = NormalDist().inv_cdf(p["confidence"])
        mu, sd = -float(np.mean(x)), float(np.std(x, ddof=1))
        return [mu+sd*z, mu+sd*math.exp(-z*z/2)/math.sqrt(2*math.pi)/(1-p["confidence"])]
    if method == "equal_weight":
        return [1/x.shape[1]] * x.shape[1]
    if method == "inverse_volatility":
        w = [1/math.sqrt(sum((v-float(np.mean(col)))**2 for v in col)/(len(col)-1))
             for col in x.T]
        return [v/sum(w) for v in w]
    slope = 0 if method == "naive" else (x[-1]-x[0])/(len(x)-1)
    return [float(x[-1]+slope*h) for h in range(1, p["horizon"]+1)]


def execute(case, method, parameters):
    from fin_skills.algorithms import run
    if method not in PARAMETERS or not isinstance(parameters, dict):
        raise ValueError("unknown method or malformed parameters")
    if set(parameters) != set(PARAMETERS[method]):
        raise ValueError("all and only advertised parameters must be supplied")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or abs(v) > 1000 for v in parameters.values()):
        raise ValueError("invalid bounded numerical parameter")
    value = run(method, {k: np.asarray(v) for k, v in case["data"].items()}, **parameters)
    if isinstance(value, dict):
        vector = [value["volatility"]] if "volatility" in value else [value["var"], value["expected_shortfall"]]
    else:
        vector = clean(value.to_numpy() if hasattr(value, "to_numpy") else value)
    receipt = dict(method=method, parameters=parameters, values=vector,
                   data_sha256=case["data_sha256"], observed_through=case["observed_block"][1])
    receipt["sha256"] = digest(receipt)
    return receipt


def grade(case, action, final, receipt):
    expected = reference(case)
    method_ok = action.get("method") == case["method"]
    parameter_ok = action.get("parameters") == case["parameters"]
    values = final.get("values")
    try:
        numeric_ok = (isinstance(values, list) and len(values) == len(expected)
                      and all(math.isfinite(float(v)) for v in values)
                      and np.allclose(values, expected, rtol=1e-4, atol=1e-6))
    except (ValueError, TypeError):
        numeric_ok = False
    uses_output = bool(receipt and final.get("receipt_sha256") == receipt["sha256"])
    return dict(method_correct=method_ok, parameters_correct=parameter_ok,
        numeric_correct=bool(numeric_ok), used_output=uses_output,
        correct=bool(method_ok and parameter_ok and numeric_ok and uses_output),
        expected=expected, exposure="author-constructed synthetic development contract")
