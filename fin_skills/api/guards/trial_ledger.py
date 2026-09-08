"""Guard: the deflated Sharpe ratio on an honest trial count (backtest-validation / trial_ledger.py)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_int, require_number
from fin_skills.core.trial_ledger import TrialLedger


@register
class TrialLedgerGuard(Guard):
    """Deflate the best Sharpe by the number of trials it took to find it.

    Inputs (either `ledger` or `sharpes`)
        ledger      : path to an append-only trials.jsonl, or a TrialLedger.
        sharpes     : instead of a ledger, the Sharpe of every configuration you tried.
                      A temporary ledger is built from them - it counts only what you
                      list, so abandoned trials you forgot are not in it.
        best_sharpe : the Sharpe you intend to report.
        n_obs       : observations behind that Sharpe.
        skew, kurt  : return moments (kurt is NON-excess; 3.0 = normal).
        alpha       : DSR must exceed 1 - alpha to survive. Default 0.05.

    Fails when the DSR verdict is "NOT distinguishable from noise", when the ledger has
    fewer than two completed trials with a Sharpe (nothing to deflate against), or when
    the reported Sharpe beats every trial in the ledger (the ledger is incomplete).
    """

    name = "trial_ledger"
    skill = "backtest-validation"
    summary = "Deflated Sharpe ratio from the ledger's honest trial count; fails when the best is noise."
    wraps = ("fin_skills.core.trial_ledger.TrialLedger.deflated_sharpe",
             "fin_skills.core.trial_ledger.TrialLedger.summary")
    required = ("best_sharpe", "n_obs")
    optional = ("ledger", "sharpes", "skew", "kurt", "alpha")

    def missing(self, inputs: Mapping[str, Any]) -> list[str]:
        miss = [k for k in self.required if k not in inputs]
        if "ledger" not in inputs and "sharpes" not in inputs:
            miss.append("ledger or sharpes")
        return miss

    def check(self, best_sharpe: float, n_obs: int, ledger: str | Path | TrialLedger | None = None,
              sharpes: Sequence[float] | None = None, skew: float = 0.0, kurt: float = 3.0,
              alpha: float = 0.05) -> Outcome:
        out = Outcome()
        best_sharpe = require_number(best_sharpe, "best_sharpe")
        n_obs = require_int(n_obs, "n_obs", minimum=2)
        if ledger is None and sharpes is None:
            raise TypeError("give a ledger (path or TrialLedger) or a list of trial sharpes")
        if isinstance(ledger, TrialLedger):
            led = ledger
        elif ledger is not None:
            led = TrialLedger(Path(ledger))
        else:
            vals = [require_number(s, "sharpes[i]") for s in list(sharpes or [])]
            if not vals:
                raise TypeError("sharpes is empty")
            led = TrialLedger(Path(tempfile.mkdtemp(prefix="fin_skills_ledger_")) / "trials.jsonl")
            for i, s in enumerate(vals):
                tid = led.record("listed", {"i": i})
                led.complete(tid, {"sharpe": float(s), "n_obs": n_obs})
            out.warning("ledger built from the list you passed; it counts only those trials",
                        where="ledger")

        summary = led.summary()
        dsr = led.deflated_sharpe(best_sharpe, n_obs, skew=skew, kurt=kurt)
        out.note(summary=summary, dsr=dsr, ledger_path=str(led.path))
        if "error" in dsr:
            out.error(f"{dsr['error']} (n_trials={dsr['n_trials']})", where="ledger")
            return out
        if summary["best_sharpe"] is not None and best_sharpe > summary["best_sharpe"] + 1e-12:
            out.error(f"reported Sharpe {best_sharpe:.3f} exceeds the best recorded trial "
                      f"{summary['best_sharpe']:.3f}: the ledger does not contain the "
                      f"configuration you are reporting, so its trial count is a fiction",
                      where="ledger")
        p = float(dsr["deflated_sharpe_ratio"])
        if p <= 1.0 - alpha:
            out.error(f"DSR {p:.3f} after {dsr['n_trials']} trials: expected max Sharpe from "
                      f"noise alone is {dsr['expected_max_sharpe_from_noise']:.3f} vs "
                      f"observed {best_sharpe:.3f} - NOT distinguishable from noise", where="dsr")
        else:
            out.info(f"DSR {p:.3f} after {dsr['n_trials']} trials: survives the search",
                     where="dsr")
        return out
