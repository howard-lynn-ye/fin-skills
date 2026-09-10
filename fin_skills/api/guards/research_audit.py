"""Guard + assembly: the reality check, run in the order a sceptic would run it.

`check(bundle)` runs every ready guard and reports them all. That is the right default and
the wrong thing for THIS question, because the checks are not independent:

    deflating a Sharpe that was computed on leaked data is meaningless arithmetic.

So `reality_check(bundle)` runs five stages IN ORDER and STOPS at the first one that fails:

    1. integrity   did the backtest see the future, trade a universe it could not have
                   known, or score a signal that repaints? (the existing guards)
    2. cost        does the edge survive the cost you actually pay?
    3. deflation   is the Sharpe still there after the number of things you tried -
                   DSR, the multiple-testing correction, and the Minimum Backtest Length?
    4. overfitting is the in-sample winner better than a coin flip out of sample - PBO
                   via CSCV over the panel of EVERY configuration you ran?
    5. regime      did the test window contain more than one regime?

The report says which stage it stopped at and why. A stage with nothing in the bundle to
run is "not evaluated", which is reported and, with on_unknown="stop", also stops the
audit - `research-integrity-guards` treats unknown as failed, and the ordering only means
anything if you know the earlier stage really passed.

    from fin_skills.api import Bundle
    from fin_skills.api.guards.research_audit import reality_check

    print(reality_check(bundle).summary())

The guard registered here, `research_audit`, is stages 3 and 4 - the two checks no other
guard in the registry performs. Everything else in the ordering is an existing guard,
reached through the Bundle's own vocabulary.

`result_manifest` is deliberately NOT a stage: it is what you fill in AFTER the audit
passes, not a gate the audit can clear on your behalf.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from fin_skills.api.base import Finding, Guard, GuardResult, Outcome, ascii_only, get, register
from fin_skills.api.guards._common import require_int, require_number
from fin_skills.core.overfitting import credible, cscv, expected_max_sharpe, null_pbo
from fin_skills.core.research_history import (METHODS, compare, from_ledger, haircut_sharpe,
                                              p_from_t, t_from_annual_sharpe)

PAPER_PBO_THRESHOLD = 0.05
"""Bailey, Borwein, Lopez de Prado & Zhu recommend rejecting a model whose PBO exceeds
0.05. That is far stricter than the 0.5 no-skill line, so it is a warning here, not the
failing threshold - see `pbo_max`."""


# ------------------------------------------------------------------------------- stages
@dataclass(frozen=True)
class Stage:
    """One rung of the audit: what it asks, and which guards answer it."""

    name: str
    question: str
    guards: tuple[str, ...] = ()
    native: str = ""          # "" | "deflation" | "pbo" - handled by this module


STAGES: tuple[Stage, ...] = (
    Stage("integrity",
          "did the backtest see the future, or trade a universe it could not have known?",
          ("assert_causal", "warmup_probe", "fold_leak_test", "purge_effect", "safe_asof",
           "join_asof_sortedness", "adjustment_check", "reconcile_sources", "pit_universe",
           "pit_fundamentals", "survivorship_audit", "regime_lookahead",
           "contamination_probe", "rf_convention")),
    Stage("cost",
          "does the edge survive the cost you actually pay, at the size you actually trade?",
          ("cost_curve", "cost_plausibility")),
    Stage("deflation",
          "is the Sharpe still there after the number of things you tried?",
          ("trial_ledger", "spa_test"), native="deflation"),
    Stage("overfitting",
          "is the in-sample winner better than a coin flip out of sample?",
          (), native="pbo"),
    Stage("regime",
          "did the test window contain more than one regime?",
          ("regime_coverage",)),
)

STAGE_NAMES: tuple[str, ...] = tuple(s.name for s in STAGES)


# ------------------------------------------------------------------- the two native checks
def deflation_outcome(best_sharpe: float, n_obs: int, periods_per_year: int = 252,
                      n_trials: int | None = None, sharpes: Sequence[float] | None = None,
                      ledger: Any = None, alpha: float = 0.05,
                      correction: str = "holm") -> Outcome:
    """Multiple-testing deflation of ONE reported Sharpe, plus the Minimum Backtest Length.

    `best_sharpe` is PER-PERIOD, on the same frequency as `n_obs` - the convention
    `trial_ledger.deflated_sharpe` already uses. The annualised figure is
    best_sharpe * sqrt(periods_per_year), and that is what MinBTL and the haircut need.

    Fails when the honest trial count leaves the claim unsupportable: the sample is shorter
    than MinBTL, or the multiple-testing adjusted p-value does not clear `alpha`.
    """
    out = Outcome()
    best_sharpe = require_number(best_sharpe, "best_sharpe")
    n_obs = require_int(n_obs, "n_obs", minimum=2)
    ppy = require_int(periods_per_year, "periods_per_year", minimum=1)
    alpha = require_number(alpha, "alpha")
    if correction not in METHODS[1:]:
        raise ValueError(f"correction must be one of {list(METHODS[1:])}, got {correction!r}")

    hist = None
    if ledger is not None:
        hist = from_ledger(ledger)
        counted = hist.m_registered
        source = f"ledger at {hist.source} (registered count)"
    elif sharpes is not None:
        counted = len(list(sharpes))
        source = "the list of trial Sharpes you passed"
    else:
        counted = None
        source = ""
    if n_trials is not None:
        n_trials = require_int(n_trials, "n_trials", minimum=1)
        if counted is not None and n_trials < counted:
            out.warning(f"n_trials={n_trials} is below the {counted} trials in {source}; "
                        f"using {counted}", where="n_trials")
            n_trials = counted
        else:
            source = "the n_trials you declared"
    elif counted is not None:
        n_trials = counted
    else:
        raise ValueError("deflation needs a trial count: pass n_trials, sharpes, or a ledger")

    sr_annual = best_sharpe * np.sqrt(ppy)
    years = n_obs / ppy
    cred = credible(sr_annual, max(n_trials, 2), n_obs, ppy)
    t = t_from_annual_sharpe(sr_annual, years)
    p = p_from_t(t, n_obs)
    cut = haircut_sharpe(sr_annual, years, max(n_trials, 1), method="bonferroni", alpha=alpha)
    out.note(n_trials=n_trials, trial_count_source=source, sharpe_annual=float(sr_annual),
             sample_years=years, minbtl=cred, t_stat=t, pvalue=p, haircut=cut,
             expected_max_sharpe_from_noise=float(
                 expected_max_sharpe(max(n_trials, 2), 1.0 / np.sqrt(years))) if years > 0
             else float("inf"))

    if not cred["credible"]:
        out.error(f"{years:.1f} years of data cannot support an annualised Sharpe of "
                  f"{sr_annual:.2f} found in {n_trials} trials: the minimum backtest length "
                  f"is {cred['min_backtest_length_years']:.2f} years, short by "
                  f"{cred['shortfall_years']:.2f}. Expected max Sharpe from noise alone at "
                  f"this length is {cred['expected_max_sharpe_at_this_length']:.2f}",
                  where="minbtl")
    else:
        out.info(f"sample {years:.1f}y >= MinBTL {cred['min_backtest_length_years']:.2f}y "
                 f"at {n_trials} trials", where="minbtl")

    if not cut["survives"]:
        out.error(f"annualised Sharpe {sr_annual:.2f} over {years:.1f}y is t={t:.2f}, "
                  f"p={p:.2e}; adjusted for {n_trials} trials that is "
                  f"p={cut['adjusted_pvalue']:.3f} > {alpha:.2f}. Haircut Sharpe "
                  f"{cut['haircut_sharpe']:.2f} ({cut['haircut_pct']:.0f}% taken off)",
                  where="haircut")
    else:
        out.info(f"adjusted p={cut['adjusted_pvalue']:.2e} clears {alpha:.2f}; haircut "
                 f"Sharpe {cut['haircut_sharpe']:.2f} ({cut['haircut_pct']:.0f}% off)",
                 where="haircut")

    if hist is not None and hist.m_scored > 0:
        table = hist.summary(alpha)
        survivors = int(table.loc[correction, "survivors"])
        out.note(family=table)
        msg = (f"across the whole ledger ({hist.m_registered} registered, "
               f"{hist.m_scored} scored), {correction} leaves {survivors} standing")
        (out.info if survivors else out.warning)(msg, where="family")
    elif sharpes is not None and len(list(sharpes)) > 1:
        vals = np.asarray(list(sharpes), dtype=float)
        p_family = np.array([p_from_t(t_from_annual_sharpe(v * np.sqrt(ppy), years), n_obs)
                             for v in vals])
        table = compare(p_family, alpha)
        out.note(family=table)
        survivors = int(table.loc[correction, "survivors"])
        msg = f"across the {vals.size} trial Sharpes, {correction} leaves {survivors} standing"
        (out.info if survivors else out.warning)(msg, where="family")
    return out


def pbo_outcome(model_returns, n_blocks: int = 16, pbo_max: float | None = None) -> Outcome:
    """PBO via CSCV over the (T, N) panel of EVERY configuration tried.

    Fails when PBO reaches the no-skill line for this N - 0.5 for even N, (N+1)/2N for odd
    N - or `pbo_max` when you set one. Warns above the source paper's own 0.05 threshold.
    """
    out = Outcome()
    res = cscv(model_returns, n_blocks=int(n_blocks))
    line = res.null_pbo if pbo_max is None else float(pbo_max)
    panel = np.asarray(model_returns, dtype=float)
    if panel.shape[1] > 1:
        cc = np.corrcoef(panel, rowvar=False)
        rho = float(np.abs(cc[~np.eye(cc.shape[0], dtype=bool)]).mean())
    else:
        rho = float("nan")
    out.note(pbo=res.pbo, no_skill_line=res.null_pbo, threshold=line,
             n_configs=res.n_configs, n_blocks=res.n_blocks,
             n_combinations=res.n_combinations, mean_relative_rank=res.mean_relative_rank,
             prob_loss=res.prob_loss, slope=res.slope, mean_abs_correlation=rho,
             report=res.report())
    if res.pbo >= line:
        out.error(f"PBO {res.pbo:.3f} at or above the {line:.3f} line over "
                  f"{res.n_combinations} splits of {res.n_configs} configurations: the "
                  f"in-sample winner is a coin flip out of sample", where="pbo")
    else:
        out.info(f"PBO {res.pbo:.3f} below the {line:.3f} line ({res.n_configs} "
                 f"configurations, S={res.n_blocks})", where="pbo")
        if res.pbo > PAPER_PBO_THRESHOLD:
            out.warning(f"PBO {res.pbo:.3f} exceeds the 0.05 the source paper recommends "
                        f"rejecting above", where="pbo")
    if np.isfinite(rho) and rho > 0.5:
        out.warning(f"mean absolute correlation between configurations is {rho:.2f}: these "
                    f"are far fewer effective trials than {res.n_configs}, and one panel's "
                    f"PBO is correspondingly noisier", where="correlation")
    if res.n_configs < 5:
        out.warning(f"only {res.n_configs} configurations in the panel - if you tried more "
                    f"than that, the ones you dropped are the search PBO exists to measure",
                    where="panel")
    return out


# -------------------------------------------------------------------------------- guard
@register
class ResearchAuditGuard(Guard):
    """Deflate a claimed Sharpe by the honest trial count, then ask PBO whether the winner
    was a coin flip. The two checks the rest of the registry does not perform.

    Inputs
        best_sharpe      : the PER-PERIOD Sharpe you intend to report (same frequency as
                           n_obs). The annualised figure is best_sharpe * sqrt(ppy).
        n_obs            : observations behind it.
        periods_per_year : 252 equities, 365 crypto, 12 monthly. Default 252.
        n_trials         : the honest count of EVERY configuration tried, abandoned ones
                           included. Taken from `ledger` or `sharpes` when not given.
        ledger           : path to trials.jsonl, or a TrialLedger. Its REGISTERED count is
                           the trial count, and its p-values give the family correction.
        sharpes          : instead of a ledger, the per-period Sharpe of every trial.
        model_returns    : (T, N) per-period returns of every configuration tried. Without
                           it the PBO half is skipped and only the deflation half runs.
        n_blocks         : S for CSCV, even. Default 16 (the source paper's recommendation).
        pbo_max          : PBO threshold. Default: the no-skill line for this N.
        correction       : which multiple-testing correction to report over the family.
                           Default "holm".
        alpha            : significance level. Default 0.05.

    Fails when the sample is shorter than the Minimum Backtest Length for this claim and
    trial count, when the multiple-testing adjusted p-value does not clear alpha, or when
    PBO reaches the no-skill line.
    """

    name = "research_audit"
    skill = "backtest-overfitting"
    summary = ("Deflates a Sharpe by the honest trial count (MinBTL + multiple testing) and "
               "runs PBO via CSCV over the panel of every configuration tried.")
    wraps = ("fin_skills.core.overfitting.cscv",
             "fin_skills.core.overfitting.credible",
             "fin_skills.core.research_history.haircut_sharpe",
             "fin_skills.core.research_history.from_ledger")
    required = ("best_sharpe", "n_obs")
    optional = ("periods_per_year", "n_trials", "ledger", "sharpes", "model_returns",
                "n_blocks", "pbo_max", "correction", "alpha")

    def missing(self, inputs: Mapping[str, Any]) -> list[str]:
        miss = [k for k in self.required if k not in inputs]
        if not any(k in inputs for k in ("n_trials", "ledger", "sharpes")):
            miss.append("n_trials or ledger or sharpes")
        return miss

    def check(self, best_sharpe: float, n_obs: int, periods_per_year: int = 252,
              n_trials: int | None = None, ledger: Any = None,
              sharpes: Sequence[float] | None = None, model_returns: Any = None,
              n_blocks: int = 16, pbo_max: float | None = None, correction: str = "holm",
              alpha: float = 0.05) -> Outcome:
        out = deflation_outcome(best_sharpe, n_obs, periods_per_year, n_trials, sharpes,
                                ledger, alpha, correction)
        if model_returns is None:
            out.warning("no model_returns panel, so PBO was not computed; a trial count "
                        "alone cannot see whether the winner was a coin flip", where="pbo")
            return out
        pbo_out = pbo_outcome(model_returns, n_blocks, pbo_max)
        out.findings.extend(pbo_out.findings)
        out.evidence.update(pbo_out.evidence)
        return out


# ----------------------------------------------------------------------------- assembly
_UNKNOWN_MODES = ("warn", "stop")


@dataclass
class StageResult:
    """One stage's verdict: what ran, what it found, and what it still needed."""

    name: str
    question: str
    status: str                       # pass | fail | not evaluated | not reached
    results: list[GuardResult] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    unavailable: dict[str, list[str]] = field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        out = [f for r in self.results for f in r.errors]
        return out + [f for f in self.findings if f.severity == "error"]

    @property
    def ran(self) -> list[str]:
        return [r.guard for r in self.results] + (["research_audit"] if self.findings else [])


@dataclass
class AuditReport:
    """The staged reality check. `stopped_at` is the whole point of the object."""

    stages: list[StageResult]
    stopped_at: str | None = None
    reason: str = ""
    on_unknown: str = "warn"

    @property
    def passed(self) -> bool:
        return self.stopped_at is None

    @property
    def by_name(self) -> dict[str, StageResult]:
        return {s.name: s for s in self.stages}

    @property
    def unevaluated(self) -> list[str]:
        return [s.name for s in self.stages if s.status == "not evaluated"]

    def summary(self) -> str:
        head = ("PASS - every evaluated stage cleared" if self.passed
                else f"STOPPED AT STAGE '{self.stopped_at}' - {self.reason}")
        lines = ["RESEARCH AUDIT (sceptic's order; stops at the first failure)", "=" * 74,
                 "  " + head, "-" * 74]
        for i, s in enumerate(self.stages, 1):
            lines.append(f"  {i}. {s.name:<13} {s.status.upper():<15} {s.question}")
            for r in s.results:
                mark = "ok  " if r.passed else "FAIL"
                lines.append(f"       {mark} {r.guard}")
                for f in r.errors:
                    lines.append(f"            {f}")
            for f in s.findings:
                lines.append(f"       {f}")
            if s.unavailable and s.status == "not evaluated":
                for g, why in s.unavailable.items():
                    lines.append(f"       -    {g} needs {why}")
            elif s.unavailable and s.status in ("pass", "fail"):
                # A stage that passed on ONE guard is not the same as a stage that passed
                # on all of them. Say which questions the bundle could not answer.
                names = sorted(s.unavailable)
                shown = ", ".join(names[:4]) + (f", +{len(names) - 4} more"
                                                if len(names) > 4 else "")
                lines.append(f"       -    {len(names)} guard(s) had no input: {shown}")
        lines.append("-" * 74)
        if self.unevaluated:
            lines.append(f"  not evaluated: {', '.join(self.unevaluated)} "
                         f"(on_unknown={self.on_unknown!r}; "
                         f"'unknown' is not the same as 'passed')")
        if not self.passed:
            lines.append("  Later stages were not run on purpose: deflating a Sharpe that "
                         "failed an")
            lines.append("  earlier stage measures nothing.")
        return ascii_only("\n".join(lines))

    def __str__(self) -> str:
        return self.summary()


def _touched(bundle, guard: Guard) -> bool:
    """True when the bundle actually supplies something this guard reads. A guard with no
    required inputs is otherwise 'ready' on an empty bundle and would report on nothing."""
    return bool(bundle.inputs_for(guard))


def reality_check(bundle=None, *, alpha: float = 0.05, on_unknown: str = "warn",
                  stages: Iterable[str] | None = None, **slots: Any) -> AuditReport:
    """Run the five stages over a Bundle, in order, stopping at the first failure.

        report = reality_check(b)
        report.stopped_at        # None, or the name of the stage that failed
        print(report.summary())

    on_unknown="warn"  a stage with nothing to run is reported and the audit continues.
    on_unknown="stop"  it stops there instead - the posture `research-integrity-guards`
                       prescribes, where an unanswerable check counts as failed.
    """
    from fin_skills.api.bundle import Bundle          # local: bundle imports the registry

    if on_unknown not in _UNKNOWN_MODES:
        raise ValueError(f"on_unknown must be one of {list(_UNKNOWN_MODES)}, "
                         f"got {on_unknown!r}")
    if bundle is None:
        b = Bundle(**slots)
    else:
        b = bundle if isinstance(bundle, Bundle) else Bundle(**dict(bundle))
        if slots:
            b = b.with_(**slots)
    wanted = set(STAGE_NAMES if stages is None else stages)
    unknown_stage = wanted - set(STAGE_NAMES)
    if unknown_stage:
        raise ValueError(f"unknown stage(s) {sorted(unknown_stage)}; "
                         f"stages are {list(STAGE_NAMES)}")

    report = AuditReport(stages=[], on_unknown=on_unknown)
    for stage in STAGES:
        if stage.name not in wanted:
            continue
        if report.stopped_at is not None:
            report.stages.append(StageResult(stage.name, stage.question, "not reached"))
            continue
        sr = StageResult(stage.name, stage.question, "not evaluated")
        for gname in stage.guards:
            g = get(gname)
            miss = b.missing_for(g)
            if miss or not _touched(b, g):
                sr.unavailable[gname] = miss or ["any of " + str(list(g.inputs))]
                continue
            try:
                sr.results.append(g.run(**b.inputs_for(g)))
            except TypeError as exc:
                sr.unavailable[gname] = [ascii_only(str(exc))]
        if stage.native:
            guard = get("research_audit")
            miss = guard.missing(b.slots())
            need_panel = stage.native == "pbo"
            if miss or (need_panel and "model_returns" not in b):
                sr.unavailable["research_audit"] = miss or ["model_returns"]
            else:
                kw = b.inputs_for(guard)
                kw.setdefault("alpha", alpha)
                if stage.native == "deflation":
                    kw.pop("model_returns", None)
                    for k in ("n_blocks", "pbo_max"):
                        kw.pop(k, None)
                    out = deflation_outcome(
                        kw["best_sharpe"], kw["n_obs"], kw.get("periods_per_year", 252),
                        kw.get("n_trials"), kw.get("sharpes"), kw.get("ledger"),
                        kw["alpha"], kw.get("correction", "holm"))
                else:
                    out = pbo_outcome(kw["model_returns"], kw.get("n_blocks", 16),
                                      kw.get("pbo_max"))
                sr.findings.extend(out.findings)
                sr.evidence.update(out.evidence)

        if sr.results or sr.findings:
            sr.status = "fail" if sr.errors else "pass"
        if sr.status == "fail":
            first = sr.errors[0]
            report.stopped_at = stage.name
            report.reason = ascii_only(f"{stage.question} -> {first}")
        elif sr.status == "not evaluated" and on_unknown == "stop":
            report.stopped_at = stage.name
            report.reason = (f"{stage.question} -> nothing in the bundle answers it, and "
                             f"on_unknown='stop' treats unknown as failed")
        report.stages.append(sr)
    return report


__all__ = ["AuditReport", "PAPER_PBO_THRESHOLD", "STAGES", "STAGE_NAMES", "ResearchAuditGuard",
           "Stage", "StageResult", "deflation_outcome", "pbo_outcome", "reality_check"]
