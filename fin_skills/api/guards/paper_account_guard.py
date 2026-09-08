"""Guard: prove the account is paper (broker-execution-apis / paper_account_guard.py)."""
from __future__ import annotations

from typing import Any, Mapping

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_str
from fin_skills.core.paper_account_guard import NotPaperError, assert_paper


@register
class PaperAccountGuard(Guard):
    """Refuse unless a SERVER-RETURNED fact proves the session is paper.

    Inputs
        broker     : 'ib', 'alpaca', 'schwab', 'ccxt:<venue>' (aliases accepted).
        account_id : IB - the id from ib.managedAccounts(); DU/DF is paper, U is live.
        base_url   : Alpaca - the RESOLVED base url the client will call.
        extra      : broker-specific facts: {'paper': flag} for Alpaca;
                     {'sandbox_mode', 'urls', 'headers'} for ccxt venues.

    Fails on anything that is not positive proof: a live account id, a live host, a
    flag that disagrees with the host, a venue with no sandbox (Schwab), a missing
    fact, or an unknown broker. "Could not prove paper" and "is live" have the same
    consequence, so they get the same verdict.
    """

    name = "paper_account_guard"
    skill = "broker-execution-apis"
    summary = "Fails closed unless a server-returned account id / host / header proves the session is paper."
    wraps = ("fin_skills.core.paper_account_guard.assert_paper",)
    required = ("broker",)
    optional = ("account_id", "base_url", "extra")

    def check(self, broker: str, account_id: str | None = None, base_url: str | None = None,
              extra: Mapping[str, Any] | None = None) -> Outcome:
        out = Outcome()
        broker = require_str(broker, "broker")
        if extra is not None and not isinstance(extra, Mapping):
            raise TypeError("extra must be a mapping of server-returned facts")
        try:
            verdict = assert_paper(broker, account_id=account_id, base_url=base_url, extra=extra)
        except NotPaperError as exc:
            out.error(str(exc), where=broker)
            out.note(is_paper=False)
        else:
            out.note(is_paper=True, broker=verdict.broker, evidence_text=verdict.evidence,
                     rule=verdict.rule)
            out.info(str(verdict), where=broker)
        return out
