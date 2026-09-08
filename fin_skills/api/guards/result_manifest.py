"""Guard: the result card's provenance (research-integrity-guards / result_manifest.py)."""
from __future__ import annotations

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.core.result_manifest import ResultCard


@register
class ResultManifestGuard(Guard):
    """A bare Sharpe is not a result; the card must carry everything that qualifies it.

    Inputs
        card : a fin_skills.core.result_manifest.ResultCard.

    Fails on every problem ResultCard.problems() reports: excluded delistings, a
    Sharpe without its annualisation and rf convention, a cost curve with fewer than
    three points, a trial count of one, no ledger path, no benchmark, no factor alpha,
    no regime coverage, no falsifier. Evidence carries the non-strict rendering.
    """

    name = "result_manifest"
    skill = "research-integrity-guards"
    summary = "Refuses a strategy result card that is missing any piece of its provenance."
    wraps = ("fin_skills.core.result_manifest.ResultCard.problems",
             "fin_skills.core.result_manifest.ResultCard.render")
    required = ("card",)
    optional = ()

    def check(self, card: ResultCard) -> Outcome:
        out = Outcome()
        if not isinstance(card, ResultCard):
            raise TypeError("card must be a fin_skills.core.result_manifest.ResultCard")
        problems = card.problems()
        out.note(problems=problems, verdict=card.verdict(), rendered=card.render(strict=False))
        for p in problems:
            out.error(p, where=card.strategy_id)
        if not problems:
            out.info(f"complete result card; {card.verdict()}", where=card.strategy_id)
        return out
