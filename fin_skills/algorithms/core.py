"""Typed, extensible algorithm registry and deterministic suitability routing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Any, Callable, Mapping

TASKS = ("portfolio", "forecast", "volatility", "risk", "signal", "regression",
         "classification", "execution", "pricing", "stat_arb", "regime")
ALIASES = {"组合优化": "portfolio", "预测": "forecast", "波动率": "volatility",
           "风险": "risk", "信号": "signal", "回归": "regression", "分类": "classification",
           "执行": "execution", "定价": "pricing", "统计套利": "stat_arb", "状态识别": "regime"}
COMPLEXITY = {"low": 0, "medium": 1, "high": 2}


def task_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("task must be a string")
    value = ALIASES.get(value, value)
    if value not in TASKS:
        raise ValueError(f"task must be one of {TASKS}; got {value!r}")
    return value


def strings(value, field: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (tuple, list)):
        raise TypeError(f"{field} must be a list or tuple of strings")
    if any(not isinstance(v, str) or not v for v in value):
        raise TypeError(f"{field} must contain nonempty strings")
    return tuple(dict.fromkeys(value))


@dataclass(frozen=True)
class Algorithm:
    """One algorithm/backend pair. A source URL documents capability, not ranking.

    min_observations is an explicit routing policy floor, not a statistical guarantee.
    An adapter is executable only if its handler and optional dependency are available.
    """

    id: str
    name: str
    task: str
    library: str
    inputs: tuple[str, ...]
    objectives: tuple[str, ...]
    tags: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    complexity: str = "low"
    min_observations: int = 0
    priority: int = 0
    module: str | None = None
    skill: str = ""
    source: str = ""
    verified_on: str = "2026-09-14"
    caveat: str = "Validate on later, unseen data before using results."

    def __post_init__(self):
        if not self.id or not self.name or not self.library or not self.source:
            raise ValueError("algorithm id, name, library and source are required")
        object.__setattr__(self, "task", task_name(self.task))
        for field in ("inputs", "objectives", "tags", "capabilities"):
            object.__setattr__(self, field, strings(getattr(self, field), field))
        if not self.inputs or not self.objectives:
            raise ValueError("algorithm inputs and objectives must not be empty")
        if self.complexity not in COMPLEXITY:
            raise ValueError("invalid complexity")
        if type(self.min_observations) is not int or self.min_observations < 0:
            raise ValueError("min_observations must be a nonnegative integer")


@dataclass(frozen=True)
class Request:
    task: str
    available_inputs: tuple[str, ...]
    n_observations: int | None = None
    objective: str | None = None
    preferences: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    allowed_libraries: tuple[str, ...] = ()
    max_complexity: str = "high"
    executable_only: bool = True

    def __post_init__(self):
        object.__setattr__(self, "task", task_name(self.task))
        for field in ("available_inputs", "preferences", "required_capabilities",
                      "allowed_libraries"):
            object.__setattr__(self, field, strings(getattr(self, field), field))
        if self.n_observations is not None and (
                type(self.n_observations) is not int or self.n_observations < 0):
            raise ValueError("n_observations must be a nonnegative integer or None")
        if self.objective is not None and not isinstance(self.objective, str):
            raise TypeError("objective must be a string or None")
        if self.max_complexity not in COMPLEXITY:
            raise ValueError("max_complexity must be low, medium or high")
        if type(self.executable_only) is not bool:
            raise TypeError("executable_only must be boolean")


@dataclass(frozen=True)
class Candidate:
    algorithm: Algorithm
    score: int
    reasons: tuple[str, ...]
    executable: bool
    status: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Selection:
    request: Request
    candidates: tuple[Candidate, ...]
    rejected: Mapping[str, tuple[str, ...]]
    basis: str = "deterministic suitability rules; not measured performance"

    @property
    def selected(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def as_dict(self) -> dict:
        return {"request": asdict(self.request), "selected": (
                    self.selected.algorithm.id if self.selected else None),
                "candidates": [c.as_dict() for c in self.candidates],
                "rejected": dict(self.rejected), "basis": self.basis}


Handler = Callable[[Mapping[str, Any], Mapping[str, Any]], Any]


class Registry:
    """Custom adapters are trusted Python callables; JSON cannot register/import code."""

    def __init__(self):
        self._specs: dict[str, Algorithm] = {}
        self._handlers: dict[str, Handler] = {}
        self._checks: dict[str, Handler] = {}

    def register(self, algorithm: Algorithm, handler: Handler | None = None,
                 *, preflight: Handler | None = None) -> None:
        if not isinstance(algorithm, Algorithm):
            raise TypeError("algorithm must be an Algorithm")
        if algorithm.id in self._specs:
            raise ValueError(f"duplicate algorithm: {algorithm.id}")
        if handler is not None and not callable(handler):
            raise TypeError("handler must be callable")
        if preflight is not None and not callable(preflight):
            raise TypeError("preflight must be callable")
        self._specs[algorithm.id] = algorithm
        if handler is not None:
            self._handlers[algorithm.id] = handler
        if preflight is not None:
            self._checks[algorithm.id] = preflight

    def get(self, algorithm_id: str) -> Algorithm:
        try:
            return self._specs[algorithm_id]
        except KeyError:
            raise KeyError(f"unknown algorithm: {algorithm_id!r}") from None

    def algorithms(self, task: str | None = None) -> tuple[Algorithm, ...]:
        task = task_name(task) if task is not None else None
        return tuple(a for _, a in sorted(self._specs.items()) if task is None or a.task == task)

    def status(self, algorithm_id: str) -> str:
        a = self.get(algorithm_id)
        if a.id not in self._handlers:
            return "catalog_only"
        if a.module:
            try:
                if find_spec(a.module) is None:
                    return "missing_dependency"
            except (ImportError, ValueError, AttributeError):
                return "missing_dependency"
        return "ready"

    def catalog(self, task: str | None = None) -> list[dict]:
        return [dict(asdict(a), status=self.status(a.id)) for a in self.algorithms(task)]

    def recommend(self, request: Request, *, data=None, parameters=None) -> Selection:
        if not isinstance(request, Request):
            raise TypeError("request must be a Request")
        if data is not None:
            from .runtime import validate_data
            data, n = validate_data(self, request.task, data)
            if set(data) != set(request.available_inputs) or n != request.n_observations:
                raise ValueError("request inputs/sample count must match supplied data")
        specs = self.algorithms(request.task)
        vocab = {
            "available_inputs": {v for a in specs for v in a.inputs},
            "preferences": {v for a in specs for v in a.tags},
            "required_capabilities": {v for a in specs for v in a.capabilities},
            "allowed_libraries": {a.library for a in self.algorithms()},
        }
        for field, valid in vocab.items():
            unknown = set(getattr(request, field)) - valid
            if unknown:
                raise ValueError(f"unknown {field}: {sorted(unknown)}; accepted: {sorted(valid)}")
        objectives = {v for a in specs for v in a.objectives}
        if request.objective is not None and request.objective not in objectives:
            raise ValueError(f"objective must be one of {sorted(objectives)}")
        candidates, rejected = [], {}
        for a in specs:
            why = []
            missing = set(a.inputs) - set(request.available_inputs)
            if missing:
                why.append(f"missing inputs: {', '.join(sorted(missing))}")
            if request.objective is not None and request.objective not in a.objectives:
                why.append(f"does not implement objective: {request.objective}")
            if request.n_observations is None and a.min_observations:
                why.append("n_observations is required to check the sample floor")
            elif request.n_observations is not None and request.n_observations < a.min_observations:
                why.append(f"routing policy requires >= {a.min_observations} observations")
            if set(request.required_capabilities) - set(a.capabilities):
                why.append("does not support all required capabilities")
            if request.allowed_libraries and a.library not in request.allowed_libraries:
                why.append("library is outside allowed_libraries")
            if COMPLEXITY[a.complexity] > COMPLEXITY[request.max_complexity]:
                why.append("exceeds max_complexity")
            status = self.status(a.id)
            if request.executable_only and status != "ready":
                why.append(status + (f": install {a.library} separately" if
                                    status == "missing_dependency" else ": no adapter registered"))
            if not why and data is not None and a.id in self._checks:
                try:
                    self._checks[a.id](data, dict(parameters or {}))
                except (ValueError, TypeError) as exc:
                    why.append(f"preflight: {exc}")
            if why:
                rejected[a.id] = tuple(why)
                continue
            matches = sorted(set(request.preferences) & set(a.tags))
            score = a.priority + 10 * len(matches) - COMPLEXITY[a.complexity]
            reasons = ["required inputs and routing constraints satisfied",
                       f"policy priority {a.priority}; complexity {a.complexity}"]
            if request.objective:
                reasons.append(f"implements objective: {request.objective}")
            reasons.extend(f"matches preference: {m}" for m in matches)
            candidates.append(Candidate(a, score, tuple(reasons), status == "ready", status))
        candidates.sort(key=lambda c: (-c.score, c.algorithm.id))
        return Selection(request, tuple(candidates), rejected)

    def run(self, algorithm_id: str, data: Mapping[str, Any], **parameters) -> Any:
        """Run a named adapter. Backend failures propagate; there is no silent fallback."""
        from .runtime import validate_data
        a = self.get(algorithm_id)
        status = self.status(a.id)
        if status == "catalog_only":
            raise NotImplementedError(f"{a.id} is catalog_only; no execution adapter")
        if status == "missing_dependency":
            raise ImportError(f"{a.id} requires optional library: pip install {a.library}")
        clean, n = validate_data(self, a.task, data)
        missing = set(a.inputs) - set(clean)
        if missing:
            raise ValueError(f"missing inputs: {sorted(missing)}")
        if a.min_observations and (n is None or n < a.min_observations):
            raise ValueError(f"{a.id} requires >= {a.min_observations} observations")
        if a.id in self._checks:
            self._checks[a.id](clean, parameters)
        return self._handlers[a.id](clean, parameters)

    def auto_run(self, task: str, data: Mapping[str, Any], *,
                 parameters: Mapping[str, Any] | None = None, **constraints) -> dict:
        from .runtime import validate_data
        if "available_inputs" in constraints or "n_observations" in constraints:
            raise TypeError("auto_run derives available_inputs and n_observations from data")
        if constraints.get("executable_only", True) is not True:
            raise ValueError("auto_run requires executable_only=True")
        clean, n = validate_data(self, task_name(task), data)
        if task_name(task) == "pricing" and isinstance(clean.get("option"), Mapping):
            exercise = clean["option"].get("exercise")
            if exercise in ("american", "european"):
                required = strings(constraints.get("required_capabilities", ()),
                                   "required_capabilities")
                constraints["required_capabilities"] = tuple(dict.fromkeys((*required, exercise)))
        request = Request(task, tuple(clean), n, **constraints)
        selection = self.recommend(request, data=clean, parameters=parameters)
        if selection.selected is None:
            return {"selection": selection.as_dict(), "result": None, "executed": False}
        algorithm_id = selection.selected.algorithm.id
        result = self.run(algorithm_id, clean, **dict(parameters or {}))
        return {"selection": selection.as_dict(), "result": result, "executed": True}
