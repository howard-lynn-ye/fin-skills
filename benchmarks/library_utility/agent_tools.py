"""Read-only tool dispatcher bound to one past-only market snapshot.

The model chooses calls and parameters. This adapter chooses neither an algorithm nor
a portfolio on its behalf. No dates, future data, arbitrary paths or live news are exposed.
"""
import hashlib
import math

import numpy as np
import pandas as pd

from benchmarks.library_utility.evaluate import digest


def definitions():
    from fin_skills.algorithms.technical import SIGNAL_DEFAULTS
    from fin_skills.algorithms.strategy import REGIME_DEFAULTS
    return {
        "equal_weight": {}, "inverse_volatility": {}, "min_variance": {},
        "hrp": {"linkage": "single"},
        "cross_sectional_momentum": {"lookback": 60, "top_k": 3},
        **SIGNAL_DEFAULTS, "rolling_market_state": REGIME_DEFAULTS,
        "historical_volatility": {"periods_per_year": 252},
        "ewma_volatility": {"periods_per_year": 252, "decay": .94},
        "historical_var_es": {"confidence": .95}, "normal_var_es": {"confidence": .95},
        "naive": {"horizon": 1}, "mean": {"horizon": 1}, "drift": {"horizon": 1},
    }


def catalog_rows():
    from fin_skills.algorithms import catalog
    defaults = definitions()
    return [dict(id=r["id"], task=r["task"], description=r["name"],
                 caveat=r["caveat"], skill=r["skill"], defaults=defaults[r["id"]])
            for r in catalog() if r["id"] in defaults and r["status"] == "ready"]


def tool_manifest():
    return [
        {"name": "list_algorithms", "arguments": {},
         "description": "Discover available algorithms, parameters, caveats and related skills."},
        {"name": "read_skill", "arguments": {"name": "skill ID", "offset": "integer, default 0"},
         "description": "Read a 4000-character page of a related skill. IDs from list_algorithms; "
                        "also portfolio-and-risk, backtest-validation, trend-following-models."},
        {"name": "run_algorithm", "arguments": {"algorithm": "ID from list_algorithms",
         "asset": "asset_0..asset_3 for single-series algorithms; omit for portfolio",
         "parameters": "optional object overriding advertised defaults"},
         "description": "Execute the selected algorithm on this decision's past-only data. "
                        "Portfolio outputs are weights; signals are the last three lagged values; "
                        "regime outputs are the last three rows. No future performance returned."},
        {"name": "recommend_strategy", "arguments": {"asset": "asset_0..asset_3",
         "target_vol": "positive number, default 0.10", "max_exposure": "number in (0,1], default 1"},
         "description": "Optional heuristic regime/strategy adviser for a selected asset; "
                        "returns candidates and risk cap. Its recommendation is not binding."},
    ]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def execute(tool, arguments, past):
    """Only `past` is accessible; callers bind it using context(prices, decision_index)."""
    from fin_skills.algorithms import run, recommend_strategy
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        raise ValueError("tool must be a string and arguments an object")
    allowed = {"list_algorithms": set(), "read_skill": {"name", "offset"},
               "run_algorithm": {"algorithm", "asset", "parameters"},
               "recommend_strategy": {"asset", "target_vol", "max_exposure"}}
    if tool not in allowed or set(arguments) - allowed[tool]:
        raise ValueError("unknown tool or argument; use the advertised schema")
    rows = catalog_rows()
    if tool == "list_algorithms":
        result = rows
    elif tool == "read_skill":
        import fin_skills
        names = {r["skill"] for r in rows} | {
            "portfolio-and-risk", "backtest-validation", "trend-following-models"}
        name, offset = arguments.get("name"), arguments.get("offset", 0)
        if not isinstance(name, str) or name not in names:
            raise ValueError("choose a skill ID from the catalog")
        if type(offset) is not int or not 0 <= offset <= 100000:
            raise ValueError("offset must be an integer in 0..100000")
        text = fin_skills.load(name)
        result = {"name": name, "text": text[offset:offset + 4000], "offset": offset,
                  "next_offset": offset + 4000 if offset + 4000 < len(text) else None,
                  "sha256": hashlib.sha256(text.encode()).hexdigest()}
    else:
        asset = arguments.get("asset")
        if tool == "recommend_strategy":
            if not isinstance(asset, str) or asset not in past:
                raise ValueError("choose an asset from the supplied asset_order")
            cap = arguments.get("max_exposure", 1.)
            if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 0 < cap <= 1:
                raise ValueError("max_exposure must be in (0,1]")
            report = recommend_strategy(past[asset], as_of=past.index[-1],
                target_vol=arguments.get("target_vol", .1), max_exposure=cap, allow_short=False)
            result = {k: report[k] for k in ("regime", "indicators", "candidates", "selected",
                       "exposure_cap", "target_exposure", "warnings")}
        else:
            algorithm = arguments.get("algorithm")
            row = next((r for r in rows if r["id"] == algorithm), None)
            if row is None:
                raise ValueError("algorithm is not available; call list_algorithms")
            given = arguments.get("parameters", {})
            if not isinstance(given, dict) or set(given) - set(row["defaults"]):
                raise ValueError("parameters must match this algorithm's advertised defaults")
            parameters = dict(row["defaults"], **given)
            for key, val in parameters.items():
                if key == "linkage":
                    if val not in ("single", "complete", "average", "ward"):
                        raise ValueError("unsupported linkage")
                elif isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or abs(val) > 1000:
                    raise ValueError("numeric parameter outside bounded adapter contract")
            if parameters.get("horizon", 1) > 21:
                raise ValueError("forecast horizon must be at most 21")
            if row["task"] == "portfolio":
                if asset is not None:
                    raise ValueError("portfolio algorithms use all assets; omit asset")
                data = {"asset_returns": past.pct_change(fill_method=None).iloc[1:]}
            else:
                if not isinstance(asset, str) or asset not in past:
                    raise ValueError("single-series algorithms require asset")
                key = ("returns" if row["task"] in ("volatility", "risk") else
                       "series" if row["task"] == "forecast" else "prices")
                data = {key: past[asset].pct_change(fill_method=None).iloc[1:]
                        if key == "returns" else past[asset]}
            value = run(algorithm, data, **parameters)
            if isinstance(value, pd.Series):
                value = value.to_dict() if row["task"] == "portfolio" else {
                    "last_three_values": value.tail(3).tolist(), "lag_bars": value.attrs.get("lag_bars")}
            elif isinstance(value, pd.DataFrame):
                value = {"last_three_rows": value.tail(3).to_dict(orient="records")}
            result = {"algorithm": algorithm, "parameters": parameters, "output": value}
    result = clean(result)
    receipt = {"tool": tool, "arguments": arguments, "result": result,
               "past_prices_sha256": digest(past.to_numpy().tolist()),
               "observed_through": past.index[-1].isoformat()}
    receipt["sha256"] = digest(receipt)
    return result, receipt
