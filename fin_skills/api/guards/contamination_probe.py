"""Guard: LLM training-cutoff contamination (llm-finance-agents / contamination_probe.py)."""
from __future__ import annotations

from datetime import date
from typing import Callable, Sequence

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import as_date
from fin_skills.llm.contamination_probe import ProbeItem, score_probe, window_overlap


@register
class ContaminationGuard(Guard):
    """Does the backtest window overlap the model's training data? Structural first, then behavioural.

    Inputs
        cutoff, test_start, test_end : dates (str or date).
        items : optional list of ProbeItem from build_probe_set, straddling the cutoff.
        ask   : optional model call str -> str; with `items` it runs the accuracy split.
        match : optional (answer, truth) -> bool comparator.

    Fails when any part of the test window predates the cutoff (the model may be
    recalling outcomes), or when probe accuracy collapses at the cutoff (gap > 0.25;
    a gap > 0.10 is a warning). A pass means "not caught", not "clean".
    """

    name = "contamination_probe"
    skill = "llm-finance-agents"
    summary = "Fails a backtest window that overlaps the LLM's training cutoff; optional accuracy-split probe."
    wraps = ("fin_skills.llm.contamination_probe.window_overlap",
             "fin_skills.llm.contamination_probe.score_probe")
    required = ("cutoff", "test_start", "test_end")
    optional = ("items", "ask", "match")

    def check(self, cutoff: str | date, test_start: str | date, test_end: str | date,
              items: Sequence[ProbeItem] | None = None,
              ask: Callable[[str], str] | None = None,
              match: Callable[[str, str], bool] | None = None) -> Outcome:
        out = Outcome()
        c, s, e = as_date(cutoff, "cutoff"), as_date(test_start, "test_start"), \
            as_date(test_end, "test_end")
        w = window_overlap(c, s, e)
        out.note(**w)
        v = w["verdict"]
        if v.startswith("INVALID") or v.startswith("CONTAMINATED"):
            out.error(v, where="window")
        else:
            out.info(v, where="window")

        if items is not None or ask is not None:
            if items is None or ask is None:
                raise TypeError("items and ask must be given together")
            if not callable(ask):
                raise TypeError("ask must be callable str -> str")
            items = list(items)
            if not items or not all(isinstance(i, ProbeItem) for i in items):
                raise TypeError("items must be a non-empty list of ProbeItem")
            sc = score_probe(items, ask, match)
            out.note(probe=sc)
            pv = sc["verdict"]
            if pv.startswith("MEMORIZATION"):
                out.error(pv + f" (pre {sc['pre_cutoff_accuracy']:.0%} vs post "
                          f"{sc['post_cutoff_accuracy']:.0%})", where="probe")
            elif pv.startswith("SUSPICIOUS"):
                out.warning(pv + f" (gap {sc['gap']:+.0%})", where="probe")
            else:
                out.info(pv, where="probe")
        return out
