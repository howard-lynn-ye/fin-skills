"""fin_skills.synthesis - combining information of DIFFERENT kinds into one research view.

`fin_skills.data` fetches, `fin_skills.discovery` finds, `fin_skills.engine` backtests and
`fin_skills.api` audits. What none of them does is COMBINE: prices plus fundamentals plus a
macro series plus a filing, each with its own availability clock, its own vintage and its
own identifier - which may not refer to the same entity on the date you are joining.
`reconcile_sources` compares two price series; that is one kind from two sources. This
layer is the other half.

    from fin_skills.synthesis import Dossier, Fact, Timeline, price_fact, combine
    from fin_skills.api import check

    tl = Timeline([price_fact("close", "cik:0000320193", 189.4, "2024-02-02",
                              source="vendor-a")])
    tl.add(Fact(field="revenue", entity="cik:0000320193", value=1.19e11,
                period_end="2023-12-30", filed_at="2024-02-01",
                available_at="2024-02-02 16:31", kind="fundamental", source="edgar"))
    signal = tl.combine("earnings_yield", tl.as_of("2024-02-05"), 0.062)
    d = Dossier(entity="cik:0000320193", as_of="2024-02-05", timeline=tl)
    print(d.render())
    print(d.explain("earnings_yield"))     # every input, transitively, with its vintage
    print(check(d.to_bundle()).summary())  # the guards run on a SYNTHESISED view

Three rules hold everywhere in this package:

  * **A combined fact is knowable only when its LAST input is.** `available_at =
    max(inputs)` - never the earliest, never the date the join ran. `timeline.combine()`
    refuses anything earlier; `Timeline.check_availability()` finds the ones some other
    code path built.
  * **A pick between disagreeing sources is written down.** `merge.merge_series()` resolves
    by a DECLARED ranking and emits a `Disagreement` for every timestamp beyond the
    threshold. `combine_first` resolves by nullity and emits nothing.
  * **A merge that cannot be correct is refused, not attempted.** Different declared
    `Adjustment`, or a symbology boundary the caller has not resolved, raise `MergeRefused`
    before a number exists.

Nothing here fetches anything. No vendor library is imported, at module scope or anywhere
else, and no network call is reachable from this package.
"""
from __future__ import annotations

from fin_skills.synthesis.dossier import (Dossier, Explanation, Step, result_card_kwargs)
from fin_skills.synthesis.merge import (Disagreement, MergePolicy, MergeRefused,
                                        MergeResult, SourceSeries, check_mergeable,
                                        merge_series, unrecorded_disagreements)
from fin_skills.synthesis.timeline import (AvailabilityError, Entity, Fact, KINDS,
                                           PERMANENT_SCHEMES, Timeline, as_entity,
                                           combine, combine_values, price_fact)

__all__ = [
    "AvailabilityError", "Disagreement", "Dossier", "Entity", "Explanation", "Fact",
    "KINDS", "MergePolicy", "MergeRefused", "MergeResult", "PERMANENT_SCHEMES",
    "SourceSeries", "Step", "Timeline", "as_entity", "check_mergeable", "combine",
    "combine_values", "merge_series", "price_fact", "result_card_kwargs",
]
