"""fin_skills.api - one interface over the executable guards, and the conventions behind them.

The generated modules under fin_skills.<namespace> are the skill scripts, verbatim. This
layer wraps the CORE function of each one in a Guard with a single calling convention,
the way PyOD gives every detector one fit / decision_function. PyOD's uniformity comes
from a uniform data container (every detector is fit(X)); here the container is a Bundle:

    from fin_skills.api import Bundle, check

    b = Bundle(returns=strategy, turnover=turn, rf=0.05, bars=bars,
               signal_fn=lambda d: d.close.rolling(20).mean())
    print(b.coverage().summary())   # which guards are ready, what the others still need
    report = check(b)               # every ready guard, one call; skipped ones say why
    print(report.summary())

One slot feeds every guard that means the same thing by it (`returns` reaches cost_curve,
rf_convention and regime_coverage; `close` reaches adjustment_check, reconcile_sources and
warmup_probe). `slots()` lists the vocabulary. The per-guard form is still there:

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
from fin_skills.api.bundle import (Bundle, Coverage, Slot, Suite, check, coverage, slots,
                                   vocabulary)

# Importing the guards package populates the registry.
from fin_skills.api import guards  # noqa: F401,E402  (registration side effect)

__all__ = [
    "Bundle", "Coverage", "Finding", "Guard", "GuardResult", "Outcome", "RunReport", "Slot",
    "Suite", "check", "conventions", "coverage", "get", "input_names", "register",
    "registry", "run_all", "slots", "vocabulary",
]
