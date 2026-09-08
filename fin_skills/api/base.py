"""The one interface every guard shares: Guard.run(**inputs) -> GuardResult.

A guard wraps the core function of one generated skill module and turns its outcome
into a uniform verdict. Three rules hold for every guard in the registry:

  * run() NEVER raises because a check failed. A failed check is a GuardResult with
    passed=False and at least one error-level Finding. The wrapped scripts raise
    AssertionError / ValueError / custom errors to signal a failure; the guard catches
    those and converts them.
  * run() raises TypeError on bad inputs: an unknown keyword, a missing required one,
    or a value the wrapped function rejects (its ValueError / KeyError is re-raised as
    TypeError with the original message attached).
  * everything printed is ASCII, so it survives a stock Windows console.

Writing a guard:

    from fin_skills.api.base import Guard, Outcome, register

    @register
    class MyGuard(Guard):
        name = "my_check"                 # registry key
        skill = "backtest-validation"     # owning skill (fin_skills.load(skill) is its text)
        summary = "one line"
        wraps = ("fin_skills.core.some_module.some_function",)
        required = ("returns",)
        optional = ("tol",)

        def check(self, returns, tol=1e-9) -> Outcome:
            out = Outcome()
            ...
            out.error("what is wrong", where="row 7")
            out.note(evidence_key=value)
            return out
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator, Literal, Mapping

Severity = Literal["error", "warning", "info"]
_SEVERITIES: tuple[str, ...] = ("error", "warning", "info")


def ascii_only(text: str) -> str:
    """Replace anything outside ASCII with '?', so summaries print on any console."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _one_line(text: str) -> str:
    return " | ".join(part.strip() for part in str(text).splitlines() if part.strip())


@dataclass(frozen=True)
class Finding:
    """One thing a guard noticed. `error` fails the guard; `warning` and `info` do not."""

    severity: Severity
    message: str
    where: str = ""

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITIES:
            raise ValueError(f"severity must be one of {_SEVERITIES}, got {self.severity!r}")

    def __str__(self) -> str:
        loc = f"{self.where}: " if self.where else ""
        return ascii_only(f"{self.severity}: {loc}{_one_line(self.message)}")


@dataclass
class GuardResult:
    """What Guard.run returns. `passed` is True iff no finding has severity 'error'."""

    guard: str
    skill: str
    passed: bool
    findings: list[Finding]
    evidence: dict[str, Any]
    elapsed_s: float

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def summary(self) -> str:
        """Status line plus one line per finding. ASCII only."""
        head = (f"{'PASS' if self.passed else 'FAIL'}  {self.guard}  [{self.skill}]  "
                f"{self.elapsed_s:.3f}s")
        lines = [head] + [f"  {f}" for f in self.findings]
        return ascii_only("\n".join(lines))

    def __str__(self) -> str:
        return self.summary()


class Outcome:
    """Collector a guard's check() fills in: findings plus free-form evidence."""

    __slots__ = ("findings", "evidence")

    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.evidence: dict[str, Any] = {}

    def add(self, severity: Severity, message: str, where: str = "") -> Finding:
        f = Finding(severity, str(message), str(where))
        self.findings.append(f)
        return f

    def error(self, message: str, where: str = "") -> Finding:
        return self.add("error", message, where)

    def warning(self, message: str, where: str = "") -> Finding:
        return self.add("warning", message, where)

    def info(self, message: str, where: str = "") -> Finding:
        return self.add("info", message, where)

    def note(self, **evidence: Any) -> None:
        """Attach evidence (numbers, frames, the wrapped function's own result object)."""
        self.evidence.update(evidence)

    @property
    def passed(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)


class Guard(ABC):
    """Base class. Subclasses set the class attributes and implement check()."""

    name: ClassVar[str] = ""
    skill: ClassVar[str] = ""
    summary: ClassVar[str] = ""
    wraps: ClassVar[tuple[str, ...]] = ()
    required: ClassVar[tuple[str, ...]] = ()
    optional: ClassVar[tuple[str, ...]] = ()

    # ------------------------------------------------------------------ inputs
    def missing(self, inputs: Mapping[str, Any]) -> list[str]:
        """Required inputs absent from `inputs`. Override for either/or requirements."""
        return [k for k in self.required if k not in inputs]

    def accepts(self, inputs: Mapping[str, Any]) -> bool:
        """True when run(**inputs) has what it needs (extra keys are ignored here)."""
        return not self.missing(inputs)

    @property
    def inputs(self) -> tuple[str, ...]:
        return tuple(self.required) + tuple(self.optional)

    # ------------------------------------------------------------------ running
    def run(self, **inputs: Any) -> GuardResult:
        """Run the check. Never raises for a failed check; TypeError for bad inputs."""
        unknown = sorted(set(inputs) - set(self.inputs))
        if unknown:
            raise TypeError(f"{self.name}.run() got unexpected input(s) {unknown}; "
                            f"required={list(self.required)} optional={list(self.optional)}")
        missing = self.missing(inputs)
        if missing:
            raise TypeError(f"{self.name}.run() is missing required input(s) {missing}; "
                            f"required={list(self.required)} optional={list(self.optional)}")
        t0 = time.perf_counter()
        try:
            out = self.check(**inputs)
        except TypeError:
            raise
        except (ValueError, KeyError) as exc:
            # The wrapped script rejected the inputs. Its message is the useful part.
            raise TypeError(f"{self.name}: bad inputs - {type(exc).__name__}: "
                            f"{_one_line(str(exc))}") from exc
        elapsed = time.perf_counter() - t0
        if not isinstance(out, Outcome):
            raise TypeError(f"{type(self).__name__}.check() must return an Outcome")
        return GuardResult(guard=self.name, skill=self.skill, passed=out.passed,
                           findings=list(out.findings), evidence=dict(out.evidence),
                           elapsed_s=elapsed)

    @abstractmethod
    def check(self, **inputs: Any) -> Outcome:
        """The check itself. Convert the wrapped function's failure signals into findings."""

    # ------------------------------------------------------------------ display
    def describe(self) -> str:
        return ascii_only(f"{self.name} [{self.skill}]: {self.summary}\n"
                          f"  required: {list(self.required)}\n"
                          f"  optional: {list(self.optional)}\n"
                          f"  wraps:    {list(self.wraps)}")

    def __repr__(self) -> str:
        return f"<Guard {self.name} ({self.skill})>"


# ---------------------------------------------------------------------- registry
_REGISTRY: dict[str, type[Guard]] = {}


def register(cls: type[Guard]) -> type[Guard]:
    """Class decorator: validate the guard's metadata and add it to the registry."""
    if not (isinstance(cls, type) and issubclass(cls, Guard)):
        raise TypeError("register() expects a Guard subclass")
    for attr in ("name", "skill", "summary"):
        if not getattr(cls, attr, ""):
            raise TypeError(f"{cls.__name__} must set a non-empty class attribute {attr!r}")
    if not cls.required and not cls.optional:
        raise TypeError(f"{cls.__name__} declares no inputs; a guard must take something")
    clash = set(cls.required) & set(cls.optional)
    if clash:
        raise TypeError(f"{cls.__name__} lists {sorted(clash)} as both required and optional")
    have = _REGISTRY.get(cls.name)
    if have is not None and have is not cls:
        raise ValueError(f"guard name {cls.name!r} is already registered by {have.__name__}")
    _REGISTRY[cls.name] = cls
    return cls


def _ensure_loaded() -> None:
    """Import the guard modules so the registry is populated on first use."""
    import fin_skills.api.guards  # noqa: F401  (registers on import)


def registry() -> list[type[Guard]]:
    """Every registered guard class, sorted by name."""
    _ensure_loaded()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def get(name: str) -> Guard:
    """An instance of the guard registered under `name`."""
    _ensure_loaded()
    try:
        return _REGISTRY[name]()
    except KeyError:
        close = [n for n in sorted(_REGISTRY) if name.lower() in n or n in name.lower()]
        hint = f" - did you mean {close}?" if close else ""
        raise KeyError(f"no guard named {name!r}{hint}") from None


def input_names() -> dict[str, list[str]]:
    """{input name: [guard names that accept it]} - the shared vocabulary run_all relies on."""
    out: dict[str, list[str]] = {}
    for cls in registry():
        for k in cls.required + cls.optional:
            out.setdefault(k, []).append(cls.name)
    return dict(sorted(out.items()))


# ------------------------------------------------------------------------ run_all
class RunReport(list):
    """A list[GuardResult] that also remembers which guards were skipped and why."""

    def __init__(self, results: list[GuardResult] | None = None,
                 skipped: dict[str, list[str]] | None = None) -> None:
        super().__init__(results or [])
        self.skipped: dict[str, list[str]] = dict(skipped or {})

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self)

    @property
    def failed(self) -> list[GuardResult]:
        return [r for r in self if not r.passed]

    @property
    def ran(self) -> list[str]:
        return [r.guard for r in self]

    def summary(self) -> str:
        n_fail = len(self.failed)
        lines = [f"ran {len(self)} guard(s), {len(self) - n_fail} passed, {n_fail} failed, "
                 f"{len(self.skipped)} skipped"]
        lines += [r.summary() for r in self]
        for name, missing in self.skipped.items():
            lines.append(f"SKIP  {name}  (missing {missing})")
        return ascii_only("\n".join(lines))

    def __iter__(self) -> Iterator[GuardResult]:  # typing aid only
        return super().__iter__()


def run_all(**inputs: Any) -> RunReport:
    """Run every registered guard whose required inputs are present; skip the rest.

    Each guard receives only the inputs it declares. The same keyword means the same
    thing in every guard (see `input_names()`), so one call can feed many checks:

        report = run_all(left=signals, right=quotes, on="time", by="symbol",
                         tolerance="5min", returns=strategy, turnover=turn)
        print(report.summary())          # includes which guards were skipped and why
    """
    report = RunReport()
    for cls in registry():
        guard = cls()
        missing = guard.missing(inputs)
        if missing:
            report.skipped[guard.name] = missing
            continue
        report.append(guard.run(**{k: v for k, v in inputs.items() if k in guard.inputs}))
    return report


__all__ = ["Finding", "GuardResult", "Guard", "Outcome", "RunReport", "Severity",
           "ascii_only", "get", "input_names", "register", "registry", "run_all"]
