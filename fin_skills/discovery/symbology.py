"""Resolution keyed on `(identifier, DATE)` - never on the identifier alone.

The trap is not that mappings are hard to build. It is that an identifier is not an
entity, and neither one is stable, so a mapping table without a validity window is a
statement about today wearing the clothes of history:

  * CIK 933136 has carried SIX registrant names since 1995 - Washington Mutual Inc,
    Washington Mutual, Inc, WMI Holdings Corp, WMIH Corp, Mr. Cooper Group Inc and, since
    the Rocket acquisition closed, Maverick Merger Sub 2, LLC. A `companyfacts` series
    keyed on the CIK is one continuous history spanning a failed thrift, an empty
    bankruptcy shell and a mortgage servicer.
  * The SEC's own `formerNames` windows OVERLAP, and they also leave GAPS. Apple's record
    has a window whose `from` equals its `to`, nested inside another, and a five-day
    stretch in 2007 covered by no name at all. So `name_as_of(date)` legitimately returns
    two answers, or zero. `resolve()` returns a list because the truth is a list.
  * `company_tickers.json` has no dates in it whatsoever. The only honest window for a
    mapping built from it opens at the moment it was downloaded - which is why
    `from_company_tickers()` demands a `retrieved_at` and refuses to invent one.

So: `resolve(identifier, scheme, as_of)` -> the candidates whose window covers `as_of`,
and `check_usage()` -> a guard-shaped verdict that FAILS when a mapping was used outside
its window. The check-digit arithmetic lives in the skill script this layer is generated
alongside (`fin_skills.core.identifier_checks`) and is re-exported here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from fin_skills.api.base import Finding, ascii_only
from fin_skills.core.identifier_checks import (CheckResult, check_identifier,
                                               cusip_from_isin, cusip_check_digit,
                                               figi_check_digit, isin_check_digit,
                                               sedol_check_digit)


class Scheme(str, Enum):
    """The identifier namespaces this layer knows how to key on.

    LOCAL is the one that matters in practice: whatever stable surrogate YOU assign is
    the only key that survives a rename, a re-listing and a ticker being recycled. The
    others are all borrowed from somebody who may reassign them.
    """

    TICKER = "ticker"
    CIK = "cik"
    ISIN = "isin"
    FIGI = "figi"
    SEDOL = "sedol"
    CUSIP = "cusip"
    LOCAL = "local"

    def __str__(self) -> str:
        return self.value


#: schemes that carry a self-check digit, so a typo is detectable without any network
CHECKED = (Scheme.ISIN, Scheme.CUSIP, Scheme.SEDOL, Scheme.FIGI)


def _as_scheme(value: Any) -> Scheme:
    if isinstance(value, Scheme):
        return value
    try:
        return Scheme(str(value).lower())
    except ValueError as exc:
        raise ValueError(f"scheme must be one of {[s.value for s in Scheme]}, "
                         f"got {value!r}") from exc


def normalise(identifier: str, scheme: Any) -> str:
    """The canonical form used as a dict key. CIKs are zero-padded to ten digits.

    The padding is not cosmetic: `320193`, `0000320193` and `CIK0000320193` all appear in
    SEC responses for the same company, and a join on the raw string silently misses.
    """
    s = _as_scheme(scheme)
    text = str(identifier).strip()
    if s is Scheme.CIK:
        digits = text.upper().removeprefix("CIK").lstrip("0") or "0"
        return f"{int(digits):010d}"
    if s is Scheme.LOCAL:
        return text
    return text.upper()


def validate(identifier: str, scheme: Any) -> CheckResult:
    """Check-digit and shape validation, for the four schemes that have one.

    A pass means the identifier was not MISTYPED. It is not evidence that it exists, that
    it is current, or that it refers to what you think - that is what `resolve()` is for.
    """
    s = _as_scheme(scheme)
    if s not in CHECKED:
        return CheckResult(normalise(identifier, s), s.value, True,
                           "")
    return check_identifier(normalise(identifier, s), s.value)


# ------------------------------------------------------------------------- the assignment
@dataclass(frozen=True)
class Assignment:
    """One identifier bound to one entity over the HALF-OPEN window [start, end).

    `entity_id` is the thing that is supposed to be stable; the identifier is not. Two
    assignments of the same identifier to different entities is a recycled ticker, and
    the whole reason this class carries dates.
    """

    identifier: str
    scheme: Scheme
    entity_id: str
    name: str
    start: pd.Timestamp
    end: pd.Timestamp | None         # None = open, still current
    source: str
    note: str = ""
    #: True when the window was derived from a source with NO dates of its own, so it is
    #: an assertion about the moment of download and about nothing else
    snapshot: bool = False

    def __post_init__(self) -> None:
        s = _as_scheme(self.scheme)
        object.__setattr__(self, "scheme", s)
        object.__setattr__(self, "identifier", normalise(self.identifier, s))
        object.__setattr__(self, "start", pd.Timestamp(self.start))
        if self.end is not None:
            end = pd.Timestamp(self.end)
            if end <= self.start:
                raise ValueError(
                    f"{self.identifier}: end {end.date()} is not after start "
                    f"{self.start.date()}; a window that covers no date cannot be used, "
                    f"and a zero-width source record needs converting before it gets here")
            object.__setattr__(self, "end", end)

    def covers(self, ts) -> bool:
        t = pd.Timestamp(ts)
        return t >= self.start and (self.end is None or t < self.end)

    @property
    def open_ended(self) -> bool:
        return self.end is None

    def row(self) -> dict[str, Any]:
        return {"identifier": self.identifier, "scheme": self.scheme.value,
                "entity_id": self.entity_id, "name": self.name, "start": self.start,
                "end": self.end, "source": self.source, "snapshot": self.snapshot,
                "note": self.note}

    def __repr__(self) -> str:
        end = self.end.date() if self.end is not None else "open"
        return (f"Assignment({self.scheme.value}:{self.identifier} -> {self.entity_id!r} "
                f"[{self.start.date()}, {end}) via {self.source})")


class AmbiguousIdentifier(LookupError):
    """More than one entity claims this identifier on this date, which is information."""


class UnknownIdentifier(LookupError):
    """No assignment covers this date. Never resolved to the nearest one instead."""


# ------------------------------------------------------------------------ the master
class SecurityMaster:
    """A set of `(identifier, date) -> entity` assignments, and the checks over them."""

    def __init__(self, assignments: Iterable[Assignment] = ()) -> None:
        self._rows: list[Assignment] = []
        for a in assignments:
            self.add(a)

    # ----------------------------------------------------------------- construction
    def add(self, assignment: Assignment) -> "SecurityMaster":
        if not isinstance(assignment, Assignment):
            raise TypeError("SecurityMaster holds Assignment objects; a bare tuple has "
                            "no validity window and is what this class exists to refuse")
        self._rows.append(assignment)
        return self

    def extend(self, assignments: Iterable[Assignment]) -> "SecurityMaster":
        for a in assignments:
            self.add(a)
        return self

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=list(Assignment.__dataclass_fields__))
        return (pd.DataFrame([a.row() for a in self._rows])
                .sort_values(["scheme", "identifier", "start"], kind="mergesort")
                .reset_index(drop=True))

    # ---------------------------------------------------------------------- lookup
    def resolve(self, identifier: str, scheme: Any, as_of) -> list[Assignment]:
        """Every assignment of `identifier` covering `as_of`. A LIST, possibly empty.

        Empty is a real answer and the caller must handle it: the identifier existed
        before the mapping opened, after it closed, or in one of the gaps the primary
        source leaves. Returning the nearest window instead is an invention.
        """
        s = _as_scheme(scheme)
        key = normalise(identifier, s)
        ts = pd.Timestamp(as_of)
        return sorted((a for a in self._rows
                       if a.scheme is s and a.identifier == key and a.covers(ts)),
                      key=lambda a: a.start)

    def resolve_one(self, identifier: str, scheme: Any, as_of) -> Assignment:
        """The single assignment, or an exception naming which of the two problems it is."""
        found = self.resolve(identifier, scheme, as_of)
        if not found:
            hist = self.history(identifier, scheme)
            when = (f"; it is mapped over "
                    f"{[(str(a.start.date()), str(a.end.date()) if a.end else 'open') for a in hist]}"
                    if hist else "; this master has no assignment for it at all")
            raise UnknownIdentifier(
                f"no mapping for {_as_scheme(scheme).value}:{normalise(identifier, scheme)} "
                f"as of {pd.Timestamp(as_of).date()}{when}")
        if len(found) > 1:
            raise AmbiguousIdentifier(
                f"{len(found)} entities claim {_as_scheme(scheme).value}:"
                f"{normalise(identifier, scheme)} on {pd.Timestamp(as_of).date()}: "
                f"{[a.entity_id for a in found]}")
        return found[0]

    def history(self, identifier: str, scheme: Any) -> list[Assignment]:
        s = _as_scheme(scheme)
        key = normalise(identifier, s)
        return sorted((a for a in self._rows if a.scheme is s and a.identifier == key),
                      key=lambda a: a.start)

    def entity(self, entity_id: str, as_of) -> list[Assignment]:
        """Every identifier that pointed at one entity on a date - the reverse lookup."""
        ts = pd.Timestamp(as_of)
        return sorted((a for a in self._rows
                       if a.entity_id == entity_id and a.covers(ts)),
                      key=lambda a: (a.scheme.value, a.identifier))

    # --------------------------------------------------------------------- integrity
    def overlaps(self) -> pd.DataFrame:
        """Windows of one identifier that both cover some date - a recycled identifier."""
        rows = []
        by: dict[tuple[str, str], list[Assignment]] = {}
        for a in self._rows:
            by.setdefault((a.scheme.value, a.identifier), []).append(a)
        for (scheme, ident), group in sorted(by.items()):
            group = sorted(group, key=lambda a: a.start)
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    a_end = a.end if a.end is not None else pd.Timestamp.max
                    b_end = b.end if b.end is not None else pd.Timestamp.max
                    if b.start < a_end and a.start < b_end:
                        rows.append({"scheme": scheme, "identifier": ident,
                                     "a_entity": a.entity_id, "a_name": a.name,
                                     "b_entity": b.entity_id, "b_name": b.name,
                                     "same_entity": a.entity_id == b.entity_id,
                                     "from": max(a.start, b.start),
                                     "to": min(a_end, b_end)})
        return pd.DataFrame(rows)

    def gaps(self) -> pd.DataFrame:
        """Stretches an identifier is known but unmapped - where `resolve` returns []."""
        rows = []
        by: dict[tuple[str, str], list[Assignment]] = {}
        for a in self._rows:
            by.setdefault((a.scheme.value, a.identifier), []).append(a)
        for (scheme, ident), group in sorted(by.items()):
            reach = None
            for a in sorted(group, key=lambda x: x.start):
                if reach is not None and a.start > reach:
                    rows.append({"scheme": scheme, "identifier": ident,
                                 "gap_from": reach, "gap_to": a.start,
                                 "days": int((a.start - reach).days)})
                if a.end is None:
                    break
                reach = a.end if reach is None or a.end > reach else reach
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------- the guard
    def check_usage(self, usages: Iterable[Mapping]) -> "WindowReport":
        """Guard-shaped: FAILS when a mapping was used outside its validity window.

        `usages` is anything with `identifier`, `scheme` and `as_of` - a list of dicts, a
        DataFrame's `to_dict("records")`, or the join keys of the frame you are about to
        build. Each row produces at most one finding, and an error finding fails the
        report the same way an error `Finding` fails a `GuardResult`.
        """
        findings: list[Finding] = []
        checked = 0
        for u in usages:
            checked += 1
            ident, scheme = u["identifier"], u.get("scheme", Scheme.TICKER)
            as_of = pd.Timestamp(u["as_of"])
            where = f"{_as_scheme(scheme).value}:{normalise(ident, scheme)} @ {as_of.date()}"
            shape = validate(ident, scheme)
            if not shape.ok:
                findings.append(Finding("error", f"identifier is malformed: {shape.reason}",
                                        where))
                continue
            found = self.resolve(ident, scheme, as_of)
            if not found:
                hist = self.history(ident, scheme)
                if hist:
                    spans = ", ".join(
                        f"[{a.start.date()}, {a.end.date() if a.end else 'open'}) "
                        f"-> {a.entity_id}" for a in hist)
                    findings.append(Finding(
                        "error",
                        f"used outside every validity window; this identifier is mapped "
                        f"over {spans}", where))
                else:
                    findings.append(Finding(
                        "error", "no assignment for this identifier in this master, so "
                        "the join key is unverified rather than merely unmatched", where))
            elif len(found) > 1:
                findings.append(Finding(
                    "error", f"{len(found)} entities claim it on this date "
                    f"({[a.entity_id for a in found]}); the join would pick one silently",
                    where))
            elif found[0].snapshot:
                findings.append(Finding(
                    "warning", f"the only mapping comes from a dateless current snapshot "
                    f"({found[0].source}), so it asserts something about "
                    f"{found[0].start.date()} and nothing about any other date", where))
        return WindowReport(checked=checked, findings=findings)


@dataclass(frozen=True)
class WindowReport:
    """The verdict of `check_usage`. `passed` is True iff nothing is error-level."""

    checked: int
    findings: list[Finding]

    @property
    def passed(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    def summary(self) -> str:
        head = (f"{'PASS' if self.passed else 'FAIL'}  symbology_windows  "
                f"{self.checked} usage(s), {len(self.errors)} outside their window")
        return ascii_only("\n".join([head] + [f"  {f}" for f in self.findings]))

    def __str__(self) -> str:
        return self.summary()


#: the master `resolve()` reaches when none is passed. Starts EMPTY on purpose: this
#: library ships no mapping data, so an empty answer is the correct first answer.
DEFAULT_MASTER = SecurityMaster()


def resolve(identifier: str, scheme: Any, as_of, *,
            master: SecurityMaster | None = None) -> list[Assignment]:
    """Candidates for `(identifier, scheme)` that cover `as_of`, with their windows."""
    return (master or DEFAULT_MASTER).resolve(identifier, scheme, as_of)


# --------------------------------------------------------------------- primary sources
def from_sec_former_names(cik: str, current_name: str,
                          former_names: Sequence[Mapping], *,
                          entity_id: str | None = None,
                          source: str = "sec:submissions") -> list[Assignment]:
    """SEC `formerNames` + the current name -> dated assignments of the CIK.

    Two conversions the SEC's JSON forces and does not document:

      * `to` is the LAST DAY the name was in effect, not an exclusive bound, so the window
        ends at `to + 1 day`. Read as exclusive, Apple's `from == to == 1997-07-28`
        record becomes a zero-width window that covers nothing and vanishes.
      * the current name has no `from` anywhere in the file, so its window opens where the
        newest former name closes. That is an inference and it is labelled as one in
        `note`.

    The `entity_id` stays constant across every window, which is the point: the CIK is one
    filer, and the names are what changed. Whether the ENTITY behind it is the same
    business is a judgement the data cannot make for you - see CIK 933136.
    """
    eid = entity_id or f"cik:{normalise(cik, Scheme.CIK)}"
    out: list[Assignment] = []
    for f in former_names:
        end = f.get("to")
        out.append(Assignment(
            identifier=cik, scheme=Scheme.CIK, entity_id=eid, name=str(f["name"]),
            start=pd.Timestamp(str(f["from"])[:10]),
            end=(pd.Timestamp(str(end)[:10]) + pd.Timedelta(days=1)) if end else None,
            source=source,
            note="SEC formerNames; `to` is inclusive and was converted to half-open"))
    out.sort(key=lambda a: a.start)
    opens = max((a.end for a in out if a.end is not None), default=None)
    out.append(Assignment(
        identifier=cik, scheme=Scheme.CIK, entity_id=eid, name=str(current_name),
        start=opens if opens is not None else pd.Timestamp("1900-01-01"), end=None,
        source=source,
        note="current name; its start is INFERRED - the SEC's file gives it no date"))
    return out


def from_company_tickers(rows: Iterable[Mapping], retrieved_at, *,
                         source: str = "sec:company_tickers") -> list[Assignment]:
    """company_tickers.json -> assignments valid from the moment you downloaded it.

    `retrieved_at` has no default and cannot be inferred, because the file contains no
    date of any kind. Every window it produces therefore opens at the download and stays
    open, so `resolve("JPM", TICKER, "2008-01-01")` correctly returns NOTHING and
    `check_usage` fails on any backtest date before the download. That is the intended
    behaviour: a current snapshot cannot answer a historical question, and this is what
    refusing to pretend looks like.
    """
    ts = pd.Timestamp(retrieved_at)
    out = []
    for r in rows:
        cik = normalise(r["cik_str"], Scheme.CIK)
        out.append(Assignment(
            identifier=str(r["ticker"]), scheme=Scheme.TICKER, entity_id=f"cik:{cik}",
            name=str(r.get("title", "")), start=ts, end=None, source=source,
            snapshot=True,
            note=f"current snapshot downloaded {ts.date()}; the file carries no dates, "
                 f"so this window cannot extend backwards"))
    return out


def widen(assignment: Assignment, *, start=None, end=None, why: str = "") -> Assignment:
    """Extend a window BACKWARDS, deliberately and with a reason recorded.

    The only sanctioned way to make a snapshot-derived mapping cover history: you assert,
    in `why`, the evidence that the ticker meant the same thing then. The assertion is
    stored so a reviewer can disagree with it.
    """
    if not why:
        raise ValueError("widen() needs a `why`: a mapping stretched over dates its "
                         "source never covered is an assertion, and an unrecorded "
                         "assertion is indistinguishable from data")
    return replace(assignment, start=pd.Timestamp(start) if start else assignment.start,
                   end=pd.Timestamp(end) if end else assignment.end,
                   snapshot=False,          # it is now an assertion, not a snapshot
                   note=f"{assignment.note} | widened: {why}".strip(" |"))


__all__ = ["AmbiguousIdentifier", "Assignment", "CHECKED", "CheckResult",
           "DEFAULT_MASTER", "Scheme", "SecurityMaster", "UnknownIdentifier",
           "WindowReport", "check_identifier", "cusip_check_digit", "cusip_from_isin",
           "figi_check_digit", "from_company_tickers", "from_sec_former_names",
           "isin_check_digit", "normalise", "resolve", "sedol_check_digit", "validate",
           "widen"]
