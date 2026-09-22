"""JSON/MCP entry points for the algorithm registry; no dynamic code loading."""
from collections.abc import Mapping

import numpy as np
import pandas as pd

from fin_skills.tools import payloads


def list_algorithms(task=None):
    from fin_skills.algorithms import catalog, TASKS
    return {"algorithms": catalog(task), "tasks": list(TASKS),
            "status_meaning": {"ready": "adapter present and dependency discoverable",
                               "missing_dependency": "adapter present; dependency absent",
                               "catalog_only": "capability documented; adapter not implemented"}}


def recommend_algorithms(request, data=None, parameters=None):
    from fin_skills.algorithms import recommend, Request
    if not isinstance(request, Mapping):
        raise TypeError("request must be an object")
    return recommend(Request(**request), data=_decode(data) if data is not None else None,
                     parameters=parameters).as_dict()


def _decode(data):
    if not isinstance(data, Mapping):
        raise TypeError("data must be an object")
    return {key: payloads.decode(value, key) for key, value in data.items()}


def _encode(result):
    # The guard encoder normally truncates evidence. Algorithm outputs must be complete.
    def check(value):
        if isinstance(value, (np.ndarray, pd.Series, pd.DataFrame)) and value.size > 10_000:
            raise ValueError("algorithm result exceeds 10,000 values; use the Python API")
        if isinstance(value, Mapping):
            for v in value.values():
                check(v)
        elif isinstance(value, (list, tuple)):
            if len(value) > 10_000:
                raise ValueError("algorithm result exceeds 10,000 items; use the Python API")
            for v in value:
                check(v)
    check(result)
    return payloads.encode(result, rows=10_000, items=10_000, chars=10_000)


def run_algorithm(algorithm_id, data, parameters=None):
    from fin_skills.algorithms import run
    return {"algorithm": algorithm_id,
            "result": _encode(run(algorithm_id, _decode(data), **dict(parameters or {})))}


def auto_algorithm(task, data, constraints=None, parameters=None):
    from fin_skills.algorithms import auto_run
    return _encode(auto_run(task, _decode(data), parameters=parameters, **dict(constraints or {})))


def compare_forecast_algorithms(series, initial_train, horizon=1, gap=0,
                                candidates=("naive", "mean", "drift"), metric="mae",
                                seasonal_period=None):
    from fin_skills.algorithms import walk_forward
    return walk_forward(payloads.decode(series, "series"), initial_train=initial_train,
                        horizon=horizon, gap=gap, candidates=candidates, metric=metric,
                        seasonal_period=seasonal_period)


def profile_algorithm_data(task, data):
    from fin_skills.algorithms import profile_data
    return profile_data(task, _decode(data))


def research_algorithms(task, data, evaluation=None, parameters=None, constraints=None,
                        cost_bps=0.0, provenance=None, audit_inputs=None, guards=None):
    from fin_skills.algorithms import research
    evaluation = dict(evaluation or {})
    allowed = {"initial_train", "horizon", "gap", "holdout", "candidates", "metric"}
    if set(evaluation) - allowed:
        raise ValueError("unknown evaluation fields")
    result = research(task, _decode(data), parameters=parameters, constraints=constraints,
                      cost_bps=cost_bps, provenance=provenance,
                      audit_bundle=_decode(audit_inputs) if audit_inputs is not None else None,
                      guards=guards, **evaluation)
    # Check original arrays before converting to JSON to preserve complete-output limits.
    return _encode(result.to_dict())


def recommend_trading_strategy(prices, as_of, news=None, constraints=None):
    from fin_skills.algorithms import recommend_strategy
    return _encode(recommend_strategy(payloads.decode(prices, "prices"), as_of=as_of,
                                      news=news, **dict(constraints or {})))


FUNCTIONS = {f.__name__: f for f in (list_algorithms, recommend_algorithms, run_algorithm,
                                   auto_algorithm, compare_forecast_algorithms,
                                   profile_algorithm_data, research_algorithms,
                                   recommend_trading_strategy)}


def definitions():
    from fin_skills.algorithms import TASKS
    strings = {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
    task = {"type": "string", "description": "Task name or documented Chinese alias.",
            "examples": list(TASKS)}
    constraints = {
        "objective": {"type": "string"}, "preferences": strings,
        "required_capabilities": strings, "allowed_libraries": strings,
        "max_complexity": {"type": "string", "enum": ["low", "medium", "high"]},
        "executable_only": {"type": "boolean"},
    }
    request = {"type": "object", "properties": dict(constraints, task=task,
        available_inputs=strings, n_observations={"type": "integer", "minimum": 0}),
        "required": ["task", "available_inputs"], "additionalProperties": False}
    data = {"type": "object", "description": "Fields from list_algorithms.inputs. Numeric arrays "
            "or Series/Frame payloads are supported. Returns are simple decimal returns; "
            "training history must be chronological. Pricing option requires S,K,T,r,q,sigma,flag,exercise.",
            "additionalProperties": True}
    parameters = {"type": "object", "description": "Named adapter parameters; unknown names raise. "
                  "HRP requires linkage; forecasts accept horizon; signals accept windows.",
                  "additionalProperties": True}
    rows = [
        ("recommend_trading_strategy", "Infer causal market state and propose next-bar research "
         "strategy targets with evidence and news risk screening. Requires timestamped prices. "
         "No live orders; rankings are explicit rules, not measured profitability.",
         {"prices": {"type": "object", "description": "Series payload with timezone-aware close timestamps"},
          "as_of": {"type": "string", "description": "Decision timestamp with timezone"},
          "news": {"type": "array", "items": {"type": "object"}, "maxItems": 1000},
          "constraints": {"type": "object", "additionalProperties": False, "properties": {
              "allow_short": {"type": "boolean"}, "target_vol": {"type": "number", "exclusiveMinimum": 0},
              "max_exposure": {"type": "number", "exclusiveMinimum": 0},
              "max_price_age_days": {"type": "number", "exclusiveMinimum": 0},
              "regime_parameters": {"type": "object"}, "news_keywords": strings, "risk_terms": strings}}},
         ["prices", "as_of"]),
        ("list_algorithms", "List algorithms, backends, inputs, objectives and execution status. "
         "Catalog-only entries have no execution adapter.", {"task": task}, []),
        ("recommend_algorithms", "Rank algorithms using explicit suitability rules and return "
         "reasons and exclusions. Scores are policy preferences, not measured performance.",
         {"request": request, "data": data, "parameters": parameters}, ["request"]),
        ("run_algorithm", "Run a named research algorithm. Optional packages are not installed "
         "automatically; errors propagate. Execution algorithms return schedules only.",
         {"algorithm_id": {"type": "string"}, "data": data, "parameters": parameters},
         ["algorithm_id", "data"]),
        ("auto_algorithm", "Derive data availability and sample size, select a compatible "
         "executable algorithm, and run it with an explanation. Does not estimate trading profit.",
         {"task": task, "data": data, "parameters": parameters, "constraints": {
             "type": "object", "properties": constraints, "additionalProperties": False}},
         ["task", "data"]),
        ("compare_forecast_algorithms", "Choose a forecast algorithm by expanding-window MAE "
         "or RMSE with an optional temporal gap. Return all validation scores and a refit "
         "forecast. These scores are used for selection, not an untouched test estimate.",
         {"series": {"description": "Chronological numeric series or Series payload.",
                     "oneOf": [{"type": "array", "items": {"type": "number"}}, {"type": "object"}]},
          "initial_train": {"type": "integer", "minimum": 1},
          "horizon": {"type": "integer", "minimum": 1, "maximum": 1000},
          "gap": {"type": "integer", "minimum": 0, "maximum": 1000},
          "candidates": dict(strings, minItems=1, maxItems=20),
          "metric": {"type": "string", "enum": ["mae", "rmse"]},
          "seasonal_period": {"type": "integer", "minimum": 1}}, ["series", "initial_train"]),
    ]
    rows.extend([
        ("profile_algorithm_data", "Validate algorithm data and summarize chronological index, "
         "shape and constant features.", {"task": task, "data": data}, ["task", "data"]),
        ("research_algorithms", "Run an auditable algorithm study. Optional expanding-window "
         "selection freezes the winner before a reserved final holdout; errors and missing "
         "audit coverage remain visible. Does not write files or submit orders.",
         {"task": task, "data": data, "parameters": {"type": "object", "description":
             "Algorithm ID to parameter object.", "additionalProperties": {"type": "object"}},
          "constraints": {"type": "object", "properties": constraints, "additionalProperties": False},
          "evaluation": {"type": "object", "additionalProperties": False, "properties": {
              "initial_train": {"type": "integer", "minimum": 1},
              "horizon": {"type": "integer", "minimum": 1, "maximum": 1000},
              "gap": {"type": "integer", "minimum": 0, "maximum": 1000},
              "holdout": {"type": "integer", "minimum": 0, "maximum": 100000},
              "candidates": dict(strings, minItems=1, maxItems=20),
              "metric": {"type": "string"}}},
          "cost_bps": {"type": "number", "minimum": 0, "maximum": 100},
          "provenance": {"type": "object"}, "audit_inputs": {"type": "object"},
          "guards": strings}, ["task", "data"]),
    ])
    return [{"name": name, "description": description, "input_schema": {
        "type": "object", "properties": properties, "required": required,
        "additionalProperties": False}} for name, description, properties, required in rows]
