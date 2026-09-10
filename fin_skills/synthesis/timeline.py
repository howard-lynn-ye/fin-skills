"""The availability clock: one Fact, three timestamps, and the rule for combining them.

`fin_skills.data.schema` already insists that a number carries more than one date -
`Fundamentals` stores `period_end`, `filed_at` and `acceptance_at` and USES only
`available_at`; `Macro` stores `obs_date` and `realtime_start`. Each of those is one KIND
of data with its own clock. This module is what happens when several kinds meet.

A `Fact` is one value with the three timestamps spelled out:

    period_end     the period the value DESCRIBES (a quarter end, a bar's session, an
                   observation month). Never a knowability date. Joining on it is a
                   30-90 day look-ahead for a filing and a 3-8 week one for a macro print.
    filed_at       the moment it was filed, released or stamped by the source.
    available_at   the moment it became PUBLICLY KNOWABLE. The only one that may ever
                   decide whether a row enters a signal.

The rule this module exists to enforce, in one line:

    a combined fact is knowable only when its LAST input is
    -> available_at = max(inputs), never min(inputs) and never the date the join ran.

`min` is the seductive one. A price known at t joined to a fundamental knowable at t+45
LOOKS like a fact about t, because one half of it is: the join produces a row on date t,
the row has a price in it, and every column is populated. Nothing about the shape of that
row says half of it was published six weeks later. The join date is the other error and it
fails in the opposite direction: stamping the whole panel with the moment the research ran
makes every fact unknowable before that moment, and an `as_of` filter then silently returns
nothing rather than raising.

Construction refuses the impossible orderings (a value knowable before it was filed, a
period filed before it ended) but does NOT refuse a hand-built combined fact with a
back-dated clock: `combine()` refuses that, and `Timeline.check_availability()` finds the
ones that were built some other way. That split is deliberate - the guard needs a
violation it can actually be handed.

Nothing here fetches, fills or repairs anything. A fact that is not knowable yet is absent,
not carried forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from fin_skills.data.provenance import Provenance

#: what a fact is about, in the vocabulary the rest of the library already uses.
#: `forecast` is the one kind whose value may be filed BEFORE the period it describes ends.
KINDS: tuple[str, ...] = ("price", "fundamental", "macro", "document", "forecast",
                          "derived")

#: identifier schemes that name an ENTITY rather than a place in a quote stream. A ticker
#: is not on this list on purpose: see `Entity.is_resolved`.
PERMANENT_SCHEMES: frozenset[str] = frozenset(
    {"cik", "figi", "isin", "lei", "permno", "gvkey", "series_id", "internal"})


class AvailabilityError(ValueError):
    """A combined fact was stamped knowable before its last input was."""


# ------------------------------------------------------------------------------ Entity
@dataclass(frozen=True)
class Entity:
    """WHO a fact is about, at the resolution a join actually needs.

    `scheme` matters more than `entity_id`. A ticker is a slot in an exchange's namespace,
    not a company: it is reassigned after a delisting, it changes on a rename, and two
    venues can point it at different issuers on the same day. Joining two sources on a
    ticker therefore joins whatever each source happened to mean by it - which is a
    different question from whether the numbers agree.
    """

    entity_id: str
    scheme: str = "ticker"
    name: str = ""

    def __post_init__(self) -> None:
        if not str(self.entity_id):
            raise ValueError("Entity needs a non-empty entity_id")
        object.__setattr__(self, "scheme", str(self.scheme).lower())

    @property
    def is_resolved(self) -> bool:
        """True when the scheme names a permanent entity rather than a quote-stream slot."""
        return self.scheme in PERMANENT_SCHEMES

    @property
    def key(self) -> str:
        return f"{self.scheme}:{self.entity_id}"

    def __str__(self) -> str:
        return self.key


def as_entity(value: Any) -> Entity:
    """Accept an Entity, or a 'scheme:id' / bare-ticker string, and return an Entity."""
    if isinstance(value, Entity):
        return value
    text = str(value)
    scheme, sep, ident = text.partition(":")
    if sep and scheme.lower() in PERMANENT_SCHEMES:
        return Entity(ident, scheme.lower())
    return Entity(text, "ticker")


# -------------------------------------------------------------------------------- Fact
def _ts(value: Any, what: str) -> pd.Timestamp:
    out = pd.Timestamp(value)
    if pd.isna(out):
        raise ValueError(f"{what} must be a timestamp, got {value!r}")
    return out


@dataclass(frozen=True)
class Fact:
    """One value, the entity it is about, the period it describes, and its three clocks.

    `inputs` is what makes a Fact more than a row: a derived fact keeps the facts it was
    computed from, so `available_at` can be checked against them and `Dossier.explain()`
    can walk back to the vendors. `disagreements` holds the records `merge.merge_series()`
    produced for this value, if any - kept as a tuple of opaque records so this module
    does not depend on the merge module.
    """

    field: str
    entity: Entity
    value: Any
    period_end: pd.Timestamp
    filed_at: pd.Timestamp
    available_at: pd.Timestamp
    kind: str = "price"
    period_start: pd.Timestamp | None = None
    source: str = ""
    provenance: Provenance | None = None
    inputs: tuple["Fact", ...] = ()
    disagreements: tuple[Any, ...] = ()
    note: str = ""

    # ------------------------------------------------------------------ construction
    def __post_init__(self) -> None:
        if not str(self.field):
            raise ValueError("Fact needs a non-empty field name")
        object.__setattr__(self, "field", str(self.field))
        object.__setattr__(self, "entity", as_entity(self.entity))
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {list(KINDS)}, got {self.kind!r}")
        pe = _ts(self.period_end, "period_end")
        fa = _ts(self.filed_at, "filed_at")
        av = _ts(self.available_at, "available_at")
        if av < fa:
            raise ValueError(
                f"{self.field}: available_at {av} is before filed_at {fa}. A number cannot "
                f"be knowable before it is published; if the source stamps the release in "
                f"another zone, convert it - do not move the clock back.")
        if fa < pe and self.kind != "forecast":
            raise ValueError(
                f"{self.field}: filed_at {fa} is before period_end {pe}. A completed period "
                f"cannot be filed before it ends; pass kind='forecast' if the value really "
                f"is about a period that has not finished.")
        object.__setattr__(self, "period_end", pe)
        object.__setattr__(self, "filed_at", fa)
        object.__setattr__(self, "available_at", av)
        if self.period_start is not None:
            object.__setattr__(self, "period_start", _ts(self.period_start, "period_start"))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "disagreements", tuple(self.disagreements))
        for f in self.inputs:
            if not isinstance(f, Fact):
                raise TypeError(f"Fact.inputs must hold Facts, got {type(f).__name__}")

    # ---------------------------------------------------------------------- identity
    @property
    def key(self) -> tuple[str, str, Any, Any]:
        """The point-in-time key: one value per (field, entity, period). Vintages share it."""
        return (self.field, self.entity.key, self.period_start, self.period_end)

    @property
    def lag_days(self) -> float:
        """Calendar days between the period ending and the value becoming knowable."""
        return float((self.available_at - self.period_end) / pd.Timedelta(days=1))

    @property
    def is_derived(self) -> bool:
        return bool(self.inputs)

    def known_at(self, ts: Any) -> bool:
        return self.available_at <= pd.Timestamp(ts)

    # ----------------------------------------------------------------------- lineage
    def lineage(self) -> tuple["Fact", ...]:
        """Every fact this one depends on, transitively, deepest first, each exactly once.

        Deterministic: depth-first in `inputs` order, de-duplicated by identity. This is
        what makes `Dossier.explain()` a function rather than a claim.
        """
        seen: dict[int, Fact] = {}
        out: list[Fact] = []

        def walk(f: Fact) -> None:
            for parent in f.inputs:
                if id(parent) in seen:
                    continue
                seen[id(parent)] = parent
                walk(parent)
                out.append(parent)

        walk(self)
        return tuple(out)

    def roots(self) -> tuple["Fact", ...]:
        """The facts at the bottom of the chain - the ones a vendor actually supplied."""
        if not self.inputs:
            return (self,)
        return tuple(f for f in self.lineage() if not f.inputs)

    def vintage(self) -> str:
        """'source@available_at' plus the content hash when one is recorded."""
        av = self.available_at
        stamp = av.strftime("%Y-%m-%d" if av == av.normalize() else "%Y-%m-%d %H:%M")
        head = f"{self.source or '?'}@{stamp}"
        if self.provenance is not None:
            head += f" #{self.provenance.content_sha256[:8]}"
        elif self.is_derived:
            head += " (derived)"
        return head

    def required_available_at(self) -> pd.Timestamp | None:
        """The earliest clock this fact is ALLOWED to carry: max over its inputs.

        None for a root fact - a fact with no inputs is whatever the source says it is.
        """
        if not self.inputs:
            return None
        return max(f.available_at for f in self.inputs)

    def availability_violation(self) -> pd.Timestamp | None:
        """The clock this fact should have carried, when it carries an earlier one."""
        need = self.required_available_at()
        if need is not None and self.available_at < need:
            return need
        return None

    # ----------------------------------------------------------------------- display
    def describe(self) -> str:
        val = self.value
        shown = f"{val:.6g}" if isinstance(val, (int, float, np.floating)) else str(val)
        return (f"{self.field:<18} {shown:<14} {self.entity.key:<20} "
                f"period_end={self.period_end.date()} avail={self.available_at.date()} "
                f"[{self.kind}] {self.vintage()}")

    def __repr__(self) -> str:
        return (f"Fact({self.field!r}, {self.entity.key!r}, value={self.value!r}, "
                f"period_end={self.period_end.date()}, available_at="
                f"{self.available_at.date()}, kind={self.kind!r}, "
                f"inputs={len(self.inputs)})")


# ------------------------------------------------------------------------- combining
def combine(field: str, inputs: Sequence[Fact], value: Any, *,
            entity: Any = None, kind: str = "derived", period_end: Any = None,
            period_start: Any = None, filed_at: Any = None, available_at: Any = None,
            source: str = "", note: str = "",
            disagreements: Sequence[Any] = ()) -> Fact:
    """Build the combined fact, with `available_at` forced to the LATEST input.

    Pass `available_at` only to make it LATER (an execution lag, a next-open rule). An
    earlier one raises `AvailabilityError`, because that is the bug this whole module is
    about: the combined row looks like a fact about the join date and half of it was
    published weeks after.
    """
    facts = tuple(inputs)
    if not facts:
        raise ValueError("combine() needs at least one input fact; a derived value with no "
                         "inputs has no clock to inherit and no lineage to explain")
    for f in facts:
        if not isinstance(f, Fact):
            raise TypeError(f"combine() inputs must be Facts, got {type(f).__name__}")
    latest = max(f.available_at for f in facts)
    if available_at is None:
        stamp = latest
    else:
        stamp = _ts(available_at, "available_at")
        if stamp < latest:
            late = max(facts, key=lambda f: f.available_at)
            raise AvailabilityError(
                f"{field}: available_at={stamp} is earlier than its last input "
                f"({late.field} from {late.source or '?'} became knowable {latest}). A "
                f"combined fact is knowable only when its LAST input is; taking the "
                f"earliest (or the join date) is the look-ahead this module exists to "
                f"stop. Move the decision date to {latest} or later.")
    if period_end is None:
        period_end = max(f.period_end for f in facts)
    if filed_at is None:
        filed_at = max(f.filed_at for f in facts)
    filed_at = min(_ts(filed_at, "filed_at"), stamp)
    ent = as_entity(entity) if entity is not None else facts[0].entity
    return Fact(field=field, entity=ent, value=value, period_end=period_end,
                filed_at=filed_at, available_at=stamp, kind=kind,
                period_start=period_start, source=source or "combined",
                inputs=facts, note=note, disagreements=tuple(disagreements))


def combine_values(field: str, inputs: Sequence[Fact], fn: Callable[..., Any],
                   **kwargs: Any) -> Fact:
    """`combine()` with the value computed from the inputs' values, in order."""
    facts = tuple(inputs)
    return combine(field, facts, fn(*[f.value for f in facts]), **kwargs)


# ---------------------------------------------------------------------------- Timeline
class Timeline:
    """Facts of mixed kinds under one clock, answering `as_of(ts)` for all of them at once.

    A Timeline is a container, not a store: it holds whatever facts were put in it, in
    insertion order, and every query is a pure function of that list. Two facts with the
    same `key` are two VINTAGES of one number, and `as_of` picks the latest one that was
    knowable - the same `sort_values('available_at').groupby(key).tail(1)` rule
    `Fundamentals.as_of` uses, never `drop_duplicates(keep='last')`.
    """

    __slots__ = ("_facts",)

    def __init__(self, facts: Iterable[Fact] = ()) -> None:
        self._facts: list[Fact] = []
        for f in facts:
            self.add(f)

    # ----------------------------------------------------------------- construction
    def add(self, fact: Fact) -> "Timeline":
        if not isinstance(fact, Fact):
            raise TypeError(f"Timeline holds Facts, got {type(fact).__name__}")
        self._facts.append(fact)
        return self

    def extend(self, facts: Iterable[Fact]) -> "Timeline":
        for f in facts:
            self.add(f)
        return self

    def combine(self, field: str, inputs: Sequence[Fact], value: Any, **kwargs: Any) -> Fact:
        """`combine()` the inputs, add the result to this timeline and return it."""
        fact = combine(field, inputs, value, **kwargs)
        self.add(fact)
        return fact

    # --------------------------------------------------------------------- mapping
    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self) -> Iterator[Fact]:
        return iter(self._facts)

    def __getitem__(self, i: int) -> Fact:
        return self._facts[i]

    @property
    def facts(self) -> tuple[Fact, ...]:
        return tuple(self._facts)

    @property
    def fields(self) -> list[str]:
        return sorted({f.field for f in self._facts})

    @property
    def entities(self) -> list[Entity]:
        seen: dict[str, Entity] = {}
        for f in self._facts:
            seen.setdefault(f.entity.key, f.entity)
        return [seen[k] for k in sorted(seen)]

    @property
    def kinds(self) -> list[str]:
        return sorted({f.kind for f in self._facts})

    # ------------------------------------------------------------------ point in time
    def select(self, *, field: str | None = None, entity: Any = None,
               kind: str | None = None) -> list[Fact]:
        ent = as_entity(entity).key if entity is not None else None
        return [f for f in self._facts
                if (field is None or f.field == field)
                and (ent is None or f.entity.key == ent)
                and (kind is None or f.kind == kind)]

    def as_of(self, ts: Any, *, field: str | None = None, entity: Any = None,
              kind: str | None = None) -> list[Fact]:
        """Every fact KNOWABLE at `ts`, latest vintage per (field, entity, period).

        One call for facts of every kind: prices, filings, macro prints and documents come
        back through the same filter, because they share the only clock that matters.
        """
        stamp = pd.Timestamp(ts)
        latest: dict[tuple, Fact] = {}
        for f in self.select(field=field, entity=entity, kind=kind):
            if f.available_at > stamp:
                continue
            have = latest.get(f.key)
            if have is None or f.available_at >= have.available_at:
                latest[f.key] = f
        return sorted(latest.values(),
                      key=lambda f: (f.field, f.entity.key, f.period_end, f.available_at))

    def latest(self, field: str, ts: Any, *, entity: Any = None) -> Fact | None:
        """The single most recent knowable fact for one field - the Dossier's accessor."""
        rows = self.as_of(ts, field=field, entity=entity)
        if not rows:
            return None
        return max(rows, key=lambda f: (f.period_end, f.available_at))

    def vintages(self, field: str, *, entity: Any = None) -> list[Fact]:
        """Every version of one field, oldest clock first - what a restatement looks like."""
        rows = self.select(field=field, entity=entity)
        return sorted(rows, key=lambda f: (f.period_end, f.available_at))

    def panel(self, field: str, sessions: Iterable, *,
              entities: Sequence[Any] | None = None) -> pd.DataFrame:
        """The value knowable on each session, as a wide panel (sessions x entity).

        An as-of filter, not a fill: a session before the first knowable vintage is NaN and
        stays NaN. This is the shape a cross-sectional signal is built from, and building
        it with `period_end` instead of `available_at` is the measured Sharpe difference in
        the `combining-data-sources` skill.
        """
        idx = pd.DatetimeIndex(pd.to_datetime(list(sessions))).sort_values()
        rows = self.select(field=field)
        if entities is not None:
            wanted = {as_entity(e).key for e in entities}
            rows = [f for f in rows if f.entity.key in wanted]
        cols = sorted({f.entity.key for f in rows})
        out = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float)
        for key in cols:
            group = sorted((f for f in rows if f.entity.key == key),
                           key=lambda f: (f.available_at, f.period_end))
            if not group:
                continue
            avail = np.array([f.available_at.value for f in group], dtype="int64")
            pos = np.searchsorted(avail, idx.asi8, side="right") - 1
            # among everything knowable by each session, the latest PERIOD, then the
            # latest vintage of that period - never the last non-null of each column
            best_val: list[Any] = []
            cur_period = None
            cur_val: Any = np.nan
            for f in group:
                if cur_period is None or f.period_end >= cur_period:
                    cur_period, cur_val = f.period_end, f.value
                best_val.append(cur_val)
            vals = pd.to_numeric(pd.Series(best_val), errors="coerce").to_numpy(dtype=float)
            out[key] = np.where(pos >= 0, vals[np.clip(pos, 0, len(vals) - 1)], np.nan)
        out.index.name = "session"
        out.columns.name = "entity"
        return out

    # ------------------------------------------------------------------- integrity
    def check_availability(self) -> list[tuple[Fact, pd.Timestamp]]:
        """Facts stamped knowable BEFORE their last input - (fact, the clock it needed).

        `combine()` cannot produce one. A hand-built Fact, a deserialised one, or a row
        assembled by code that predates this module can, and that is exactly the defect
        the `synthesis_integrity` guard reports.
        """
        out = []
        for f in self._facts:
            need = f.availability_violation()
            if need is not None:
                out.append((f, need))
        return out

    def incomplete_provenance(self) -> list[tuple[Fact, str]]:
        """Root facts that name no source - the end of a chain that explains nothing."""
        out = []
        for f in self._facts:
            for root in f.roots():
                if not root.source and root.provenance is None:
                    out.append((root, f"root fact {root.field!r} names no source and "
                                      f"carries no Provenance"))
        seen: set[int] = set()
        uniq = []
        for fact, why in out:
            if id(fact) in seen:
                continue
            seen.add(id(fact))
            uniq.append((fact, why))
        return uniq

    # --------------------------------------------------------------------- display
    def frame(self) -> pd.DataFrame:
        """Every fact as a row - for looking at, never for joining on."""
        rows = []
        for f in self._facts:
            rows.append({"field": f.field, "entity": f.entity.key, "kind": f.kind,
                         "value": f.value, "period_start": f.period_start,
                         "period_end": f.period_end, "filed_at": f.filed_at,
                         "available_at": f.available_at, "lag_days": f.lag_days,
                         "source": f.source, "n_inputs": len(f.inputs),
                         "n_disagreements": len(f.disagreements)})
        return pd.DataFrame(rows)

    def to_bundle(self, **extra: Any):
        """-> `fin_skills.api.check()`: the `timeline` slot the synthesis guard reads."""
        from fin_skills.api.bundle import Bundle                       # noqa: PLC0415
        return Bundle(timeline=self, **extra)

    def report(self) -> str:
        bad = self.check_availability()
        missing = self.incomplete_provenance()
        lines = [f"Timeline: {len(self._facts)} fact(s), {len(self.fields)} field(s), "
                 f"{len(self.entities)} entity(ies), kinds {self.kinds}"]
        if self._facts:
            lo = min(f.available_at for f in self._facts)
            hi = max(f.available_at for f in self._facts)
            lines.append(f"  knowable from {lo.date()} to {hi.date()}")
        lines.append(f"  availability violations : {len(bad)}")
        for f, need in bad[:5]:
            lines.append(f"    {f.field} stamped {f.available_at.date()}, "
                         f"needs {need.date()}")
        lines.append(f"  roots without a source  : {len(missing)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"Timeline({len(self._facts)} facts, fields={self.fields}, "
                f"kinds={self.kinds})")


def price_fact(field: str, entity: Any, value: Any, ts: Any, *, source: str = "",
               provenance: Provenance | None = None, note: str = "") -> Fact:
    """A market observation: the three clocks coincide, and saying so is the point.

    A bar is the one kind whose period end, publication and knowability are the same
    instant, which is exactly why it is the input people forget has a clock at all.
    """
    stamp = _ts(ts, "ts")
    return Fact(field=field, entity=entity, value=value, period_end=stamp, filed_at=stamp,
                available_at=stamp, kind="price", source=source, provenance=provenance,
                note=note)


__all__ = ["AvailabilityError", "Entity", "Fact", "KINDS", "PERMANENT_SCHEMES", "Timeline",
           "as_entity", "combine", "combine_values", "price_fact"]
