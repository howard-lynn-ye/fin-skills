"""One entity, one date, every kind of fact - and a function that names where each came from.

A Dossier is the deliverable of this layer: what is knowable about ONE entity as of ONE
date - prices, fundamentals, macro context, documents - assembled from a `Timeline`, with
each field still carrying the chain it came down. Three things it does that a dict of
numbers cannot:

    render()             a compact ASCII block an agent can read, with the clock and the
                         source next to every value
    provenance_block()   `list[result_manifest.DataSource]` - the `data=` argument of a
                         ResultCard, built from the roots rather than typed in again
    explain(field)       every input that field depends on, TRANSITIVELY, and the vintage
                         of each

`explain()` is the traceability claim, and it is a function rather than a docstring on
purpose: a claim in prose about where a number came from is unfalsifiable, and this one
returns a tree that either reaches a vendor or does not. When it does not,
`Dossier.incomplete()` says which field, and the `synthesis_integrity` guard fails.

The as-of filter is the same one everywhere in this library: `available_at <= as_of`,
latest vintage per (field, entity, period). A field with nothing knowable yet is ABSENT
from the dossier - not NaN, not carried forward from last quarter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from fin_skills.core.result_manifest import DataSource
from fin_skills.synthesis.timeline import Entity, Fact, Timeline, as_entity


def _key_for(fact: Fact, subject: Entity) -> str:
    """'revenue' for the subject's own facts, 'cpi_yoy@series_id:CPIAUCSL' for context."""
    if fact.entity.key == subject.key:
        return fact.field
    return f"{fact.field}@{fact.entity.key}"


# ------------------------------------------------------------------------- explanation
@dataclass(frozen=True)
class Step:
    """One node of a lineage: what it is, where it came from, and when it was knowable."""

    depth: int
    field: str
    entity: str
    kind: str
    value: Any
    source: str
    vintage: str
    available_at: pd.Timestamp
    is_root: bool

    def line(self) -> str:
        val = self.value
        shown = f"{val:.6g}" if isinstance(val, float) else str(val)
        if len(shown) > 22:
            shown = shown[:19] + "..."
        mark = "*" if self.is_root else "+"
        return (f"  {'  ' * self.depth}{mark} {self.field:<16} {shown:<22} "
                f"{self.kind:<12} knowable {self.available_at.date()}  {self.vintage}")


@dataclass(frozen=True)
class Explanation:
    """What `Dossier.explain(field)` returns: the whole chain, not a summary of it."""

    field: str
    fact: Fact
    steps: tuple[Step, ...]
    complete: bool
    problems: tuple[str, ...] = ()

    @property
    def inputs(self) -> tuple[Fact, ...]:
        """Every fact this one depends on, transitively."""
        return self.fact.lineage()

    @property
    def roots(self) -> tuple[Fact, ...]:
        return self.fact.roots()

    def sources(self) -> list[str]:
        """The distinct vendors at the bottom of the chain, sorted."""
        return sorted({f.source for f in self.roots if f.source})

    def vintages(self) -> dict[str, str]:
        """{'field@entity': vintage string} for every node, root and derived."""
        return {f"{s.field}@{s.entity}": s.vintage for s in self.steps}

    def binding_input(self) -> Fact | None:
        """The input whose clock the answer inherited - the one that set `available_at`."""
        chain = self.fact.lineage()
        if not chain:
            return None
        return max(chain, key=lambda f: (f.available_at, f.field))

    def render(self) -> str:
        head = (f"explain({self.field!r}) -> {self.fact.value!r}, knowable "
                f"{self.fact.available_at.date()}")
        lines = [head, f"  depends on {len(self.inputs)} input(s), "
                       f"{len(self.roots)} at the source"]
        binding = self.binding_input()
        if binding is not None:
            lines.append(f"  clock set by  : {binding.field} from "
                         f"{binding.source or '?'} ({binding.available_at.date()})")
        lines.extend(s.line() for s in self.steps)
        if self.problems:
            lines.append("  INCOMPLETE:")
            lines.extend(f"    ! {p}" for p in self.problems)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


# ------------------------------------------------------------------------------ Dossier
@dataclass(frozen=True)
class Dossier:
    """What is known about one entity as of one date, with every field's provenance."""

    entity: Entity
    as_of: pd.Timestamp
    timeline: Timeline
    title: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity", as_entity(self.entity))
        object.__setattr__(self, "as_of", pd.Timestamp(self.as_of))
        if not isinstance(self.timeline, Timeline):
            raise TypeError(f"Dossier.timeline must be a Timeline, got "
                            f"{type(self.timeline).__name__}")

    # ---------------------------------------------------------------- construction
    @classmethod
    def assemble(cls, entity: Any, as_of: Any, facts: Iterable[Fact],
                 *, title: str = "") -> "Dossier":
        """Build from loose facts of any kind - the one-call form."""
        return cls(entity=as_entity(entity), as_of=pd.Timestamp(as_of),
                   timeline=Timeline(facts), title=title)

    # -------------------------------------------------------------------- fields
    def fields(self) -> dict[str, Fact]:
        """{name: the latest fact knowable as of this date}, deterministic order.

        Facts about other entities (a macro series, a peer) keep their entity in the key,
        so a dossier can carry context without pretending the context is about the subject.
        """
        best: dict[str, Fact] = {}
        for f in self.timeline.as_of(self.as_of):
            name = _key_for(f, self.entity)
            have = best.get(name)
            if have is None or (f.period_end, f.available_at) >= (have.period_end,
                                                                  have.available_at):
                best[name] = f
        return {k: best[k] for k in sorted(best)}

    def get(self, field: str) -> Fact:
        have = self.fields()
        try:
            return have[field]
        except KeyError:
            close = [k for k in have if field in k or k.startswith(field)]
            hint = f" - did you mean {close}?" if close else ""
            raise KeyError(f"no field {field!r} knowable as of "
                           f"{self.as_of.date()}{hint}") from None

    def value(self, field: str) -> Any:
        return self.get(field).value

    def __contains__(self, field: object) -> bool:
        return field in self.fields()

    # ------------------------------------------------------------------- explain
    def explain(self, field: str) -> Explanation:
        """Name every input `field` depends on, transitively, and the vintage of each.

        Deepest input first, then up to the answer. A root that names no source is a
        problem, not a blank: it is reported in `problems` and `complete` is False.
        """
        fact = self.get(field)
        steps: list[Step] = []
        problems: list[str] = []
        depth_of: dict[int, int] = {id(fact): 0}

        def walk(node: Fact, depth: int) -> None:
            for parent in node.inputs:
                depth_of[id(parent)] = max(depth_of.get(id(parent), 0), depth + 1)
                walk(parent, depth + 1)

        walk(fact, 0)
        for node in list(fact.lineage()) + [fact]:
            is_root = not node.inputs
            if is_root and not node.source and node.provenance is None:
                problems.append(f"{node.field}[{node.entity.key}] is a root with no source "
                                f"and no Provenance - the chain stops at nothing")
            steps.append(Step(depth=depth_of.get(id(node), 0), field=node.field,
                              entity=node.entity.key, kind=node.kind, value=node.value,
                              source=node.source, vintage=node.vintage(),
                              available_at=node.available_at, is_root=is_root))
        steps.sort(key=lambda s: (-s.depth, s.field, s.entity))
        need = fact.availability_violation()
        if need is not None:
            problems.append(f"available_at {fact.available_at.date()} is earlier than its "
                            f"last input ({need.date()})")
        return Explanation(field=field, fact=fact, steps=tuple(steps),
                           complete=not problems, problems=tuple(problems))

    def incomplete(self) -> dict[str, tuple[str, ...]]:
        """{field: why its provenance chain does not reach a source}, for the guard."""
        out: dict[str, tuple[str, ...]] = {}
        for name in self.fields():
            ex = self.explain(name)
            if not ex.complete:
                out[name] = ex.problems
        return out

    # ---------------------------------------------------------------- provenance
    def provenance_block(self) -> list[DataSource]:
        """The `data=` list a `result_manifest.ResultCard` wants, built from the roots.

        One row per distinct (source, convention) actually used, with the retrieval stamp
        from the Provenance where there is one and the availability clock where there is
        not. A dossier that reached four vendors produces four rows; a card that lists one
        is a card whose data section is a guess.
        """
        rows: dict[tuple[str, str], DataSource] = {}
        for fact in self.fields().values():
            for root in fact.roots():
                name = root.source or "UNDECLARED"
                prov = root.provenance
                convention = root.note or _convention_of(root)
                stamp = prov.retrieved_at if prov is not None \
                    else root.available_at.isoformat()
                rows.setdefault((name, convention),
                                DataSource(name=name, retrieved_at=stamp,
                                           adjustment=convention))
        return [rows[k] for k in sorted(rows)]

    def data_sources(self) -> list[DataSource]:
        """Alias for `provenance_block()` - the name `ResultCard(data=...)` uses."""
        return self.provenance_block()

    # -------------------------------------------------------------------- bundle
    def to_bundle(self, **extra: Any):
        """-> `fin_skills.api.check()`. The guards then run on a SYNTHESISED view.

        Fills `dossier`, `timeline` and `as_of`; anything else the caller has (a price
        series, a returns series) goes in through `**extra` and reaches the guards that
        want it, so one call checks the synthesis and the backtest together.
        """
        from fin_skills.api.bundle import Bundle                       # noqa: PLC0415
        slots: dict[str, Any] = {"dossier": self, "timeline": self.timeline,
                                 "as_of": self.as_of}
        slots.update(extra)
        return Bundle(**slots)

    def check(self, **extra: Any):
        """`fin_skills.api.check(self.to_bundle())` - every ready guard, one call."""
        from fin_skills.api.bundle import check                        # noqa: PLC0415
        return check(self.to_bundle(**extra))

    # -------------------------------------------------------------------- render
    def render(self, width: int = 96) -> str:
        """A compact ASCII block: value, clock, lag and source for every field."""
        have = self.fields()
        head = self.title or f"DOSSIER {self.entity.key}"
        lines = [f"{head} as of {self.as_of.date()}",
                 f"  entity      : {self.entity.entity_id} [{self.entity.scheme}]"
                 + (f" {self.entity.name}" if self.entity.name else "")
                 + ("" if self.entity.is_resolved else "  <- UNRESOLVED: a ticker is not "
                                                      "an entity"),
                 f"  facts       : {len(have)} knowable of {len(self.timeline)} held",
                 f"  {'field':<18} {'value':<16} {'kind':<12} {'knowable':<11} "
                 f"{'lag':>6}  source"]
        for name, fact in have.items():
            val = fact.value
            shown = f"{val:.6g}" if isinstance(val, float) else str(val)
            if len(shown) > 16:
                shown = shown[:13] + "..."
            src = fact.source or "UNDECLARED"
            lines.append(f"  {name[:18]:<18} {shown:<16} {fact.kind:<12} "
                         f"{str(fact.available_at.date()):<11} "
                         f"{fact.lag_days:>5.0f}d  {src}")
        bad = self.timeline.check_availability()
        missing = self.incomplete()
        lines.append(f"  provenance  : {len(self.provenance_block())} source row(s); "
                     f"{len(missing)} field(s) with an incomplete chain")
        if bad:
            lines.append(f"  ! {len(bad)} fact(s) claim a clock EARLIER than their last "
                         f"input")
        for name, why in list(missing.items())[:3]:
            lines.append(f"  ! {name}: {why[0]}")
        return "\n".join(line[:width] if width else line for line in lines)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return (f"Dossier({self.entity.key!r}, as_of={self.as_of.date()}, "
                f"fields={len(self.fields())})")


def _convention_of(fact: Fact) -> str:
    """The `DataSource.adjustment` string for a root fact, from its kind."""
    return {"price": "as declared by the source",
            "fundamental": "PIT, available_at<=as_of",
            "macro": "vintage, realtime_start<=as_of",
            "document": "as published",
            "forecast": "forecast, not an observation",
            "derived": "combined"}.get(fact.kind, "unknown")


def result_card_kwargs(dossier: Dossier) -> dict[str, Any]:
    """The two `ResultCard` arguments a Dossier can fill honestly: `data` and `strategy_id`.

    Nothing else is invented here. A result card wants a universe, a split, a cost model
    and a trial count, and a data layer that filled those in would be fabricating the parts
    of the card that are about the RESEARCH rather than about the data.
    """
    return {"strategy_id": dossier.title or dossier.entity.key,
            "data": dossier.provenance_block()}


__all__ = ["Dossier", "Explanation", "Step", "result_card_kwargs"]
