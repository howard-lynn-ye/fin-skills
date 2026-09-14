"""Research workflow: preflight, temporal selection, untouched holdout and audit record."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .core import Request, strings, task_name
from .models import environment
from .runtime import integer, number, validate_data

SCHEMA_VERSION = 1
METRICS = {"forecast": ("mae", "rmse"), "regression": ("mae", "rmse"),
           "classification": ("error_rate",), "portfolio": ("variance", "negative_mean_return"),
           "volatility": ("qlike",), "risk": ("quantile_loss",),
           "signal": ("negative_mean_return",)}


def json_value(value):
    """Lossless-sized JSON representation; missing warmup values are explicit nulls."""
    if isinstance(value, pd.DataFrame):
        return {"index": json_value(list(value.index)), "columns": json_value(list(value.columns)),
                "values": json_value(value.to_numpy())}
    if isinstance(value, pd.Series):
        return {"index": json_value(list(value.index)), "values": json_value(value.to_numpy()),
                "name": json_value(value.name), "attrs": json_value(value.attrs)}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported research record value: {type(value).__name__}")


def fingerprint(data):
    from fin_skills.data.provenance import content_hash
    # Use the existing provenance hash on an explicit representation that preserves
    # feature order and every array value (never numpy's truncated string rendering).
    return content_hash(json_value(data))


def profile_data(task, data, *, registry=None):
    if registry is None:
        from . import _DEFAULT
        registry = _DEFAULT
    clean, n = validate_data(registry, task_name(task), data)
    arrays = {}
    for key, value in clean.items():
        if isinstance(value, (np.ndarray, pd.Series, pd.DataFrame)):
            a = np.asarray(value)
            arrays[key] = {"shape": list(a.shape), "indexed": isinstance(value, (pd.Series, pd.DataFrame)),
                           "constant_columns": np.flatnonzero(np.var(a, axis=0) <= 1e-16).tolist()
                           if a.ndim == 2 else [],
                           "start": str(value.index[0]) if hasattr(value, "index") else None,
                           "end": str(value.index[-1]) if hasattr(value, "index") else None}
    return {"task": task_name(task), "observations": n, "inputs": arrays,
            "units": "simple decimal returns; annual decimal option rates; expiry in years",
            "time_contract": "rows in chronological order; point-in-time availability is caller supplied"}


@dataclass
class ResearchResult:
    task: str
    status: str
    selected: str | None
    result: object
    selection: dict
    validation: dict | None
    audit: dict
    metadata: dict
    schema_version: int = SCHEMA_VERSION

    def to_dict(self):
        return json_value(asdict(self))

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)

    def save(self, path):
        with Path(path).open("x", encoding="utf-8") as stream:
            stream.write(self.to_json() + "\n")
        return Path(path)

    def render(self):
        lines = [f"Research: {self.task}", f"Status: {self.status}", f"Selected: {self.selected}"]
        if self.validation:
            lines.append(f"Metric: {self.validation['metric']} (lower is better)")
            lines.append(f"Holdout: {self.validation.get('holdout')}")
        lines.extend([f"Audit: {self.audit['status']}", f"Input hash: {self.metadata['input_hash']}"])
        return "\n".join(lines)


def _slice(data, start, end):
    history = {"series", "returns", "prices", "X", "y", "asset_returns"}
    return {k: (v.iloc[start:end].copy() if hasattr(v, "iloc") else v[start:end].copy())
            if k in history else v for k, v in data.items() if k != "X_predict"}


def _portfolio_returns(weights, returns, cost_bps):
    """Daily target rebalancing, entry from cash and terminal liquidation; proportional fees."""
    w = np.asarray(weights, float)
    R = np.asarray(returns, float)
    if w.shape != (R.shape[1],) or not np.isfinite(w).all() or (w < -1e-8).any():
        raise ValueError("portfolio adapter must return finite long-only weights")
    if abs(w.sum() - 1) > 2e-5:
        raise ValueError("portfolio weights must sum to one")
    w = w / w.sum()
    held = np.zeros_like(w)
    net, turnover = [], []
    for row in R:
        turn = float(np.abs(w - held).sum())
        gross = float(w @ row)
        net.append((1 - turn * cost_bps / 10_000) * (1 + gross) - 1)
        turnover.append(turn)
        if gross <= -1:
            raise ValueError("portfolio exhausted its capital")
        held = w * (1 + row) / (1 + gross)
    exit_turn = float(np.abs(held).sum())
    net[-1] = (1 + net[-1]) * (1 - exit_turn * cost_bps / 10_000) - 1
    turnover[-1] += exit_turn
    return np.asarray(net), np.asarray(turnover)


def _fold(registry, id, task, data, train_end, start, end, parameters, metric, cost_bps):
    train, actual = _slice(data, 0, train_end), _slice(data, start, end)
    if task == "forecast":
        p = dict(parameters, horizon=end - train_end)
        raw = np.asarray(registry.run(id, train, **p), float)
        if raw.shape != (end - train_end,):
            raise ValueError("forecast has incorrect horizon")
        prediction, target = raw[start - train_end:], np.asarray(actual["series"])
    elif task in ("regression", "classification"):
        prediction = np.asarray(registry.run(id, dict(train, X_predict=actual["X"]), **parameters))
        target = np.asarray(actual["y"])
    elif task == "portfolio":
        weights = registry.run(id, train, **parameters)
        net, turnover = _portfolio_returns(weights, actual["asset_returns"], cost_bps)
        score = float(np.var(net)) if metric == "variance" else -float(np.mean(net))
        return score, {"weights": weights, "net_returns": net, "turnover": turnover}
    elif task == "signal":
        if id not in ("momentum", "ma_crossover"):
            raise ValueError("signal evaluation requires a verified prefix-causal built-in")
        # Each position is computed with a prefix; no later test row is ever supplied.
        positions = [float(registry.run(id, _slice(data, 0, t + 1), **parameters).iloc[-1])
                     for t in range(start, end)]
        if "returns" in data:
            R = np.asarray(actual["returns"])
        else:
            prices = np.asarray(data["prices"])
            R = prices[start:end] / prices[start - 1:end - 1] - 1
        positions = np.asarray(positions)
        turnover = np.abs(np.diff(np.r_[0., positions]))
        net = positions * R - turnover * cost_bps / 10_000
        net[-1] -= abs(positions[-1]) * cost_bps / 10_000
        if not np.isfinite(net).all():
            raise ValueError("nonfinite signal or net returns")
        return -float(net.mean()), {"positions": positions, "net_returns": net}
    else:
        output = registry.run(id, train, **parameters)
        R = np.asarray(actual["returns"])
        if task == "volatility":
            variance = output["volatility"] ** 2 / output["periods_per_year"]
            if not np.isfinite(variance) or variance <= 0:
                raise ValueError("QLIKE requires positive finite forecast variance")
            score = float(np.mean(np.log(variance) + R**2 / variance))
        else:
            residual = -R - output["var"]
            score = float(np.mean(np.maximum(output["confidence"] * residual,
                                            (output["confidence"] - 1) * residual)))
            output = dict(output, violation_rate=float(np.mean(-R > output["var"])))
        if not np.isfinite(score):
            raise ValueError("nonfinite validation score")
        return score, output
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("prediction must be finite and match the target shape")
    residual = prediction - target
    score = (float(np.mean(prediction != target)) if metric == "error_rate" else
             float(np.sqrt(np.mean(residual**2))) if metric == "rmse" else
             float(np.mean(np.abs(residual))))
    if not np.isfinite(score):
        raise ValueError("nonfinite validation score")
    return score, {"prediction": prediction, "actual": target}


def research(task, data, *, initial_train=None, horizon=1, gap=0, holdout=0,
             candidates=None, metric=None, parameters=None, constraints=None,
             cost_bps=0.0, provenance=None, ledger_path=None, audit_bundle=None,
             guards=None, registry=None):
    """Execute a research plan. `parameters` maps algorithm IDs to parameter dictionaries.

    With initial_train, select on expanding nonoverlapping development folds, freeze the
    winner, fit up to the holdout gap and evaluate it once on the reserved final rows.
    Holdout results never select/reselect an algorithm. Without validation, use suitability.
    Runtime failures remain failures; no unrecorded fallback. No orders are submitted.
    """
    if registry is None:
        from . import _DEFAULT
        registry = _DEFAULT
    started = time.perf_counter()
    task = task_name(task)
    raw = dict(data)
    if task in ("regression", "classification") and "X_predict" not in raw and "X" in raw:
        raw["X_predict"] = raw["X"].iloc[:1] if hasattr(raw["X"], "iloc") else raw["X"][:1]
        if initial_train is None:
            raise ValueError("unvalidated supervised execution requires explicit X_predict")
    clean, n = validate_data(registry, task, raw)
    horizon = integer(horizon, "horizon", maximum=1000)
    gap = integer(gap, "gap", maximum=1000, minimum=0)
    holdout = integer(holdout, "holdout", maximum=100_000, minimum=0)
    cost_bps = number(cost_bps, "cost_bps", minimum=0, maximum=100)
    parameters = dict(parameters or {})
    if dict(constraints or {}).get("executable_only", True) is not True:
        raise ValueError("research requires executable_only=True")
    for id, p in parameters.items():
        if registry.get(id).task != task or not isinstance(p, dict):
            raise ValueError("parameters must map task algorithm IDs to parameter dictionaries")
    if initial_train is None and holdout:
        raise ValueError("holdout requires initial_train")
    if initial_train is None and (gap or metric is not None):
        raise ValueError("gap and metric require temporal validation via initial_train")
    if initial_train is not None and task == "forecast" and any("horizon" in p for p in parameters.values()):
        raise ValueError("set temporal forecast horizon on research, not per-algorithm parameters")
    if initial_train is not None and "X_predict" in data:
        raise ValueError("temporal evaluation accepts historical X/y only, not X_predict")
    development_end = n - holdout if n is not None else None
    if holdout and (development_end <= gap or holdout < 1):
        raise ValueError("holdout leaves no development history")
    development = _slice(clean, 0, development_end) if initial_train is not None else clean
    if task in ("regression", "classification"):
        development = dict(development, X_predict=clean["X_predict"])
    candidate_ids = strings(candidates, "candidates") if candidates is not None else None
    if candidate_ids is not None:
        if not 1 <= len(candidate_ids) <= 20:
            raise ValueError("candidates must contain 1..20 IDs")
        for id in candidate_ids:
            if registry.get(id).task != task:
                raise ValueError(f"{id} is not a {task} algorithm")
    selection = registry.recommend(Request(task, tuple(development), development_end,
                                           **dict(constraints or {}))).as_dict()
    ids, rejected = [], dict(selection["rejected"])
    for row in selection["candidates"]:
        id = row["algorithm"]["id"]
        if candidate_ids is not None and id not in candidate_ids:
            continue
        try:
            if id in registry._checks:
                registry._checks[id](development, parameters.get(id, {}))
            ids.append(id)
        except (TypeError, ValueError) as exc:
            rejected[id] = [f"preflight: {exc}"]
    selection.update(selected=ids[0] if ids else None, eligible=ids, rejected=rejected)
    if initial_train is not None and len(ids) > 20:
        raise ValueError("validation supports at most 20 candidates; specify candidates")
    from fin_skills.core.trial_ledger import TrialLedger
    from fin_skills.data.provenance import scrub, utc_now
    ledger = TrialLedger(ledger_path) if ledger_path else None
    metadata = {"created_at": utc_now(), "input_hash": fingerprint(clean),
                "environment": environment(), "parameters": parameters,
                "candidate_environments": {id: environment(registry.get(id).library) for id in ids},
                "profile": profile_data(task, development, registry=registry),
                "provenance": provenance.to_dict() if hasattr(provenance, "to_dict") else scrub(provenance),
                "cost_bps": cost_bps, "ledger_path": str(ledger_path) if ledger else None,
                "trial_ids": {}, "errors": {},
                "time_contract": "train_end and interval ends are exclusive; gap rows never enter fit",
                "cost_contract": "portfolio: daily target rebalance plus entry/exit; signal: position changes plus exit; proportional bps, no impact/borrow"}
    validation, output = None, None
    selected = ids[0] if ids else None

    def register_trial(id):
        if ledger:
            tid = ledger.record(id, {"parameters": parameters.get(id, {}),
                "input_hash": metadata["input_hash"], "initial_train": initial_train,
                "horizon": horizon, "gap": gap, "holdout": holdout, "metric": metric,
                "cost_bps": cost_bps})
            metadata["trial_ids"][id] = tid

    if initial_train is not None:
        if task not in METRICS:
            raise ValueError(f"temporal comparison is not defined for {task}")
        metric = metric or METRICS[task][0]
        if metric not in METRICS[task]:
            raise ValueError(f"metric must be one of {METRICS[task]}")
        initial_train = integer(initial_train, "initial_train")
        ends = list(range(initial_train, development_end - gap - horizon + 1, horizon))
        if not 2 <= len(ends) <= 200:
            raise ValueError("research requires 2..200 complete development folds")
        folds = [{"train_end": e, "start": e + gap, "end": e + gap + horizon} for e in ends]
        ranking, failed = [], {}
        for id in ids:
            register_trial(id)
            scores = []
            try:
                for fold in folds:
                    score, _ = _fold(registry, id, task, clean, **fold,
                        parameters=parameters.get(id, {}), metric=metric, cost_bps=cost_bps)
                    scores.append(score)
                score = float(np.mean(scores))
                ranking.append({"algorithm": id, "score": score, "fold_scores": scores})
                if ledger:
                    ledger.complete(metadata["trial_ids"][id], {"validation_loss": score})
            except Exception as exc:
                failed[id] = {"error": f"{type(exc).__name__}: {exc}", "completed_folds": len(scores)}
                if ledger:
                    ledger.abandon(metadata["trial_ids"][id], failed[id]["error"])
        ranking.sort(key=lambda row: (row["score"], row["algorithm"]))
        selected = ranking[0]["algorithm"] if ranking else None
        validation = {"metric": metric, "ranking": ranking, "rejected": failed, "folds": folds,
                      "development_end": development_end, "holdout": None,
                      "unused_development_tail": development_end - folds[-1]["end"],
                      "score_definition": "mean fold loss; RMSE is averaged per fold",
                      "selection_frozen_before_holdout": True}
        if selected and holdout:
            try:
                score, output = _fold(registry, selected, task, clean,
                    development_end - gap, development_end, n, parameters.get(selected, {}), metric, cost_bps)
                validation["holdout"] = {"start": development_end, "end": n,
                    "train_end": development_end - gap, "score": score, "algorithm": selected}
            except Exception as exc:
                metadata["errors"][selected] = f"holdout: {type(exc).__name__}: {exc}"
        elif selected:
            metadata["errors"]["holdout"] = "No untouched test period reserved; selection scores only"
    elif selected:
        register_trial(selected)
        try:
            execution_parameters = dict(parameters.get(selected, {}))
            if task == "forecast":
                execution_parameters.setdefault("horizon", horizon)
            output = registry.run(selected, clean, **execution_parameters)
            metadata["execution_parameters"] = execution_parameters
            if ledger:
                ledger.complete(metadata["trial_ids"][selected], {"executed": True})
        except Exception as exc:
            metadata["errors"][selected] = f"{type(exc).__name__}: {exc}"
            if ledger:
                ledger.abandon(metadata["trial_ids"][selected], metadata["errors"][selected])
    if selected:
        metadata["environment"] = environment(registry.get(selected).library)
    audit = {"status": "not_evaluated", "reason": "no audit inputs supplied"}
    if audit_bundle is not None or task in ("portfolio", "signal") and isinstance(output, dict):
        from fin_skills.api import Bundle
        bundle = audit_bundle if isinstance(audit_bundle, Bundle) else Bundle(**dict(audit_bundle or {}))
        if isinstance(output, dict) and "net_returns" in output:
            index = next((v.index for v in clean.values() if isinstance(v, (pd.Series, pd.DataFrame))), None)
            index = index[development_end:n] if index is not None else pd.RangeIndex(development_end, n)
            bundle = bundle.with_(returns=pd.Series(output["net_returns"], index=index))
        if guards is not None:
            report = bundle.check(guards=guards)
            audit = {"status": "failed" if not report.passed or report.rejected else
                     "partial" if report.skipped or not report else "passed",
                     "report": report.summary(), "skipped": report.skipped, "rejected": report.rejected}
        else:
            from fin_skills.api.guards.research_audit import reality_check
            report = reality_check(bundle)
            missing = {s.name: s.unavailable for s in report.stages if s.unavailable}
            audit = {"status": "failed" if not report.passed else "partial" if missing else "passed",
                     "report": report.summary(), "missing": missing}
    status = "failed" if selected is None or selected in metadata["errors"] else "completed"
    if audit["status"] == "failed":
        status = "audit_failed"
    metadata["elapsed_seconds"] = time.perf_counter() - started
    selection["policy_selected"] = selection["selected"]
    selection["selected"] = selected
    if validation is not None:
        selection["basis"] = "expanding-window development loss; holdout excluded from selection"
    return ResearchResult(task, status, selected, output, selection, validation, audit, metadata)
