"""fin_skills.api - one interface over the executable guards, and the conventions behind them.

The generated modules under fin_skills.<namespace> are the skill scripts, verbatim. This
layer wraps the CORE function of each one in a Guard with a single calling convention,
the way PyOD gives every detector one fit / decision_function:

    from fin_skills.api import get, run_all, registry, conventions

    r = get("assert_causal").run(fn=lambda d: d.close.rolling(20).mean(), df=bars, k=200)
    r.passed            # False when the check found the defect
    r.findings          # [Finding(severity, message, where), ...]
    r.evidence          # the wrapped function's own numbers and frames
    print(r.summary())  # ASCII, one status line plus one line per finding

    for cls in registry():          # every guard: name, owning skill, wrapped functions
        print(cls.name, cls.skill, cls.wraps)

    report = run_all(left=signals, right=quotes, on="time", by="symbol", tolerance="5min",
                     returns=gross, turnover=turn)
    print(report.summary())         # which guards ran, which were skipped and why

    conventions.annualize_sharpe(daily, "crypto")     # sqrt(365)
    conventions.pip_value("USDJPY", 100_000, 150.25)  # 6.66 USD per pip

Rules every guard follows: run() never raises for a failed check (that is a
GuardResult with passed=False), raises TypeError on bad inputs, and prints ASCII only.
The same keyword means the same thing in every guard - `input_names()` lists them.
"""
from __future__ import annotations

from fin_skills.api import conventions
from fin_skills.api.base import (Finding, Guard, GuardResult, Outcome, RunReport, get,
                                 input_names, register, registry, run_all)

# Importing the guards package populates the registry.
from fin_skills.api import guards  # noqa: F401,E402  (registration side effect)

__all__ = [
    "Finding", "Guard", "GuardResult", "Outcome", "RunReport", "conventions", "get",
    "input_names", "register", "registry", "run_all",
]
