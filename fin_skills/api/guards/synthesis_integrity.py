"""Guard: a synthesised view is only as honest as its clocks (combining-data-sources)."""
from __future__ import annotations

from fin_skills.api.base import Guard, Outcome, register


def _synthesis():
    """Import the layer lazily.

    `fin_skills.synthesis` imports `fin_skills.data`, which imports `fin_skills.api.base`,
    which initialises `fin_skills.api` - and that imports this module. A module-scope
    import here would therefore see a half-built package. The registry only needs the
    class; the types are needed when the guard actually runs.
    """
    from fin_skills.synthesis import dossier as _dossier                # noqa: PLC0415
    from fin_skills.synthesis import merge as _merge                    # noqa: PLC0415
    from fin_skills.synthesis import timeline as _timeline              # noqa: PLC0415
    return _timeline, _merge, _dossier


@register
class SynthesisIntegrityGuard(Guard):
    """Fail a Timeline or a Dossier whose combined numbers cannot be traced back.

    Inputs
        timeline : a fin_skills.synthesis.Timeline - facts of mixed kinds with their
                   availability clocks. Required.
        dossier  : the fin_skills.synthesis.Dossier assembled from it, when there is one;
                   adds the per-field provenance-chain check.
        tol_bps  : agreement threshold in basis points. Two merged sources that differ by
                   more than this must have left a Disagreement record. Default 10.

    Three failures, each an error:

      1. **A combined fact claims an availability EARLIER than its last input.** A price
         known at t joined to a filing knowable at t+45 is not a fact about t. The
         combined row looks complete on date t, and half of it was published six weeks
         later.
      2. **Two merged sources disagree beyond the threshold with no Disagreement
         recorded.** The spread is recomputed from the fact's own inputs, so a silent
         `combine_first` cannot hide behind its output.
      3. **A field's provenance chain is incomplete** - a root fact that names no source
         and carries no Provenance, so `Dossier.explain()` stops at nothing.

    Passes on a dossier built with `timeline.combine()` and `merge.merge_series()`, which
    cannot produce (1) and always produce the record for (2).
    """

    name = "synthesis_integrity"
    skill = "combining-data-sources"
    summary = ("Fails a synthesised view whose combined facts claim an earlier clock than "
               "their inputs, whose merges resolved silently, or whose provenance stops "
               "short of a source.")
    wraps = ("fin_skills.synthesis.timeline.Timeline.check_availability",
             "fin_skills.synthesis.merge.unrecorded_disagreements",
             "fin_skills.synthesis.dossier.Dossier.incomplete")
    required = ("timeline",)
    optional = ("dossier", "tol_bps")

    def check(self, timeline: "Timeline", dossier: "Dossier | None" = None,
              tol_bps: float = 10.0) -> Outcome:
        tl_mod, merge_mod, dossier_mod = _synthesis()
        out = Outcome()

        if dossier is not None and isinstance(timeline, dossier_mod.Dossier):
            raise TypeError("the `timeline` slot holds a Dossier; pass the Dossier as "
                            "`dossier=` and its `.timeline` as `timeline=`")
        if isinstance(timeline, dossier_mod.Dossier):     # a Dossier alone is accepted
            dossier, timeline = timeline, timeline.timeline
        if not isinstance(timeline, tl_mod.Timeline):
            raise TypeError(f"timeline must be a fin_skills.synthesis.Timeline, got "
                            f"{type(timeline).__name__}")
        if dossier is not None and not isinstance(dossier, dossier_mod.Dossier):
            raise TypeError(f"dossier must be a fin_skills.synthesis.Dossier, got "
                            f"{type(dossier).__name__}")
        if dossier is not None and dossier.timeline is not timeline:
            out.warning("the dossier was assembled from a DIFFERENT timeline than the one "
                        "checked; the two views can disagree", where="inputs")
        tol = float(tol_bps)

        # ------------------------------------------------------ 1. availability clock
        violations = timeline.check_availability()
        out.note(n_facts=len(timeline), n_availability_violations=len(violations),
                 tol_bps=tol)
        for fact, need in violations:
            late = max(fact.inputs, key=lambda f: f.available_at)
            out.error(
                f"{fact.field}[{fact.entity.key}] is stamped knowable "
                f"{fact.available_at.date()} but its last input ({late.field} from "
                f"{late.source or '?'}) is knowable {need.date()}. A combined fact is "
                f"knowable only when its LAST input is - taking the earliest backdates "
                f"{(need - fact.available_at).days} day(s) of information.",
                where=f"{fact.field}@{fact.available_at.date()}")

        # ------------------------------------------------------ 2. silent merge picks
        silent = merge_mod.unrecorded_disagreements(timeline, tol_bps=tol)
        out.note(n_silent_disagreements=len(silent),
                 n_recorded_disagreements=sum(len(f.disagreements) for f in timeline))
        for fact, spread in silent:
            names = ", ".join(f"{f.source or '?'}={f.value!r}" for f in fact.inputs
                              if f.field == fact.field)
            out.error(
                f"{fact.field}[{fact.entity.key}] on {fact.period_end.date()} was merged "
                f"from sources differing by {spread:.1f} bps (threshold {tol:.1f}) with no "
                f"Disagreement recorded: {names}. A silent pick is not a merge - "
                f"merge_series() records the ones it takes and the ones it does not.",
                where=f"{fact.field}@{fact.period_end.date()}")

        # ------------------------------------------------- 3. provenance completeness
        rootless = timeline.incomplete_provenance()
        out.note(n_rootless_facts=len(rootless))
        for fact, why in rootless:
            out.error(f"{why}. Nothing downstream can say where "
                      f"{fact.field}[{fact.entity.key}] came from.",
                      where=fact.field)

        if dossier is not None:
            incomplete = dossier.incomplete()
            out.note(n_fields=len(dossier.fields()), n_incomplete_fields=len(incomplete),
                     n_source_rows=len(dossier.provenance_block()))
            for name, problems in incomplete.items():
                for p in problems:
                    out.error(f"dossier field {name!r}: {p}", where=name)
            if not incomplete and dossier.fields():
                out.info(f"{len(dossier.fields())} field(s) traced to "
                         f"{len(dossier.provenance_block())} source row(s)")
        return out
