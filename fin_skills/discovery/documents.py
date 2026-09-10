"""Primary documents, retrieved with the point-in-time discipline prices already get.

A document has three dates and they are not interchangeable:

    period_end       what it COVERS. Months before the document exists. Stamping a
                     number on it is the 30-90 day look-ahead every fundamental study
                     is warned about.
    filed_at         the date the SEC STAMPS, assigned by its own cutoff rules
                     (17:30 ET for periodic reports, 22:00 ET for Forms 3/4/5). It is
                     wrong in BOTH directions: a filing accepted at 16:30 ET keeps that
                     day, and one accepted at 18:03 ET is stamped the NEXT day.
    acceptance_at    the wall-clock instant the document became public. The only one of
                     the three that is an instant rather than a date, and the truth.

`available_at` is derived, never supplied: acceptance converted to exchange-local and
rolled to the next session when it lands at or after the close. It is the only column a
backtest may join on, and `DocumentSet.get(..., as_of=...)` REFUSES a document whose
`available_at` is after `as_of` rather than returning it with a warning.

The SEC's own two APIs disagree about the timezone of the acceptance stamp - the
Submissions API's `acceptanceDateTime` is UTC while the Financial Statement Data Sets'
`sub.txt.accepted` is Eastern - so `tz_of_record` is declared per source and never
assumed. `research-integrity-guards` owns the audit side of all this; what is here is the
retrieval side.

Full-text search hits are PROVISIONAL by construction: efts.sec.gov returns `file_date`
and no acceptance stamp at all, so a document built from a hit carries
`provisional=True`, and asking for it as of a date raises unless you enrich it first or
say explicitly that you accept the filing date as a proxy.

Earnings-call transcripts are deliberately NOT implemented here - see `TRANSCRIPTS`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import time as _time
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from fin_skills.market_data.pit_fundamentals import available_at as _availability
from fin_skills.data.ratelimit import PerSecond
from fin_skills.discovery.search import (SEC_PACE_PER_S, SearchQuery, Transport, _json,
                                         sec_identity)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/{doc}"

#: the exchange this layer rolls post-close acceptances against
EXCHANGE_TZ = "America/New_York"

#: Why there is no transcript backend, stated rather than left as an absence.
TRANSCRIPTS = """\
Earnings-call transcripts are NOT implemented here, and the reason is a data trap rather
than an engineering one.

There is no free, licensable, point-in-time transcript source. Every vendor that sells
them stamps each transcript with a `date` field, and that field is almost always the
date of the CALL - not the date the transcript was published. Publication lags the call
by hours for an automated feed and by days for a human-reviewed one.

So an NLP feature computed from a transcript and stamped on the vendor's `date` is a
same-day look-ahead: it trades on the sentiment of a call using text that did not exist
until after the close, and often not until the following session. The measurement is
identical in shape to the acceptanceDateTime work above - take the vendor's `date`,
find the publication timestamp, and report the distribution of the gap - and it is a
measurement you must run on YOUR vendor, because the gap is a property of their pipeline.

Until you have that timestamp, treat a transcript's availability as UNKNOWN, which in
this library means the feature cannot be used point-in-time at all. The audio is public
at the moment of the call; the text is not.
"""


class LookAheadError(RuntimeError):
    """A document was requested as of a date before it was public."""


# ------------------------------------------------------------------------------ Document
@dataclass(frozen=True)
class Document:
    """One primary document, carrying all three of its dates plus the derived one."""

    doc_id: str                       # the accession number
    entity_id: str
    entity_scheme: str
    form: str
    filed_at: pd.Timestamp
    acceptance_at: pd.Timestamp | None      # tz-aware UTC, or None when the source has none
    available_at: pd.Timestamp
    period_start: pd.Timestamp | None = None
    period_end: pd.Timestamp | None = None
    title: str = ""
    url: str = ""
    primary_document: str = ""
    source: str = ""
    tz_of_record: str = "UTC"
    post_close: bool = False
    #: True when `available_at` fell back to `filed_at` because the source gave no
    #: acceptance instant. A provisional document is not point-in-time.
    provisional: bool = False

    def __post_init__(self) -> None:
        for name in ("filed_at", "available_at"):
            object.__setattr__(self, name, pd.Timestamp(getattr(self, name)))
        for name in ("period_start", "period_end"):
            v = pd.to_datetime(getattr(self, name), errors="coerce")
            object.__setattr__(self, name, None if pd.isna(v) else pd.Timestamp(v))
        if self.acceptance_at is not None:
            ts = pd.Timestamp(self.acceptance_at)
            ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
            object.__setattr__(self, "acceptance_at", ts)
        elif not self.provisional:
            raise ValueError(
                f"{self.doc_id}: a document with no acceptance instant must be marked "
                f"provisional=True. available_at then falls back to filed_at, which is "
                f"the SEC's stamp rather than the moment the document appeared.")

    def known_at(self, ts) -> bool:
        return pd.Timestamp(ts) >= self.available_at

    @property
    def period_lag_days(self) -> int:
        """filed_at - period_end, in days. The look-ahead a period stamp buys for free."""
        if self.period_end is None:
            return -1
        return int((self.filed_at.normalize() - self.period_end.normalize()).days)

    def row(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "entity_id": self.entity_id, "form": self.form,
                "period_end": self.period_end, "filed_at": self.filed_at,
                "acceptance_at": self.acceptance_at, "available_at": self.available_at,
                "post_close": self.post_close, "provisional": self.provisional,
                "period_lag_days": self.period_lag_days, "url": self.url}

    def __repr__(self) -> str:
        return (f"Document({self.form} {self.doc_id} period_end="
                f"{self.period_end.date() if self.period_end is not None else None} "
                f"filed={self.filed_at.date()} available={self.available_at.date()}"
                f"{' PROVISIONAL' if self.provisional else ''})")


def build_document(*, doc_id: str, entity_id: str, form: str, filed_at,
                   acceptance_at=None, period_end=None, period_start=None,
                   entity_scheme: str = "cik", tz_of_record: str = "UTC",
                   sessions: Sequence | None = None, source: str = "",
                   title: str = "", url: str = "", primary_document: str = "",
                   close_hour: int = 16) -> Document:
    """Assemble a Document, DERIVING `available_at` rather than accepting one.

    With an acceptance instant this is `pit_fundamentals.available_at()` - the same
    function the EDGAR adapter uses, so a document and a fundamental row agree by
    construction. Without one, `available_at` falls back to `filed_at` and the document
    is marked provisional.
    """
    filed = pd.Timestamp(filed_at)
    if acceptance_at:
        av = _availability(acceptance_at, exchange_tz=EXCHANGE_TZ,
                           sessions=pd.DatetimeIndex(pd.to_datetime(list(sessions)))
                           if sessions is not None else None,
                           close=_time(close_hour, 0), tz_in=tz_of_record)
        return Document(doc_id=doc_id, entity_id=entity_id, entity_scheme=entity_scheme,
                        form=form, filed_at=filed, acceptance_at=av.acceptance_utc,
                        available_at=av.first_tradeable_session, period_start=period_start,
                        period_end=period_end, title=title, url=url,
                        primary_document=primary_document, source=source,
                        tz_of_record=tz_of_record, post_close=bool(av.post_close),
                        provisional=False)
    return Document(doc_id=doc_id, entity_id=entity_id, entity_scheme=entity_scheme,
                    form=form, filed_at=filed, acceptance_at=None, available_at=filed,
                    period_start=period_start, period_end=period_end, title=title,
                    url=url, primary_document=primary_document, source=source,
                    tz_of_record=tz_of_record, post_close=False, provisional=True)


# --------------------------------------------------------------------------- DocumentSet
class DocumentSet:
    """A set of documents that will not hand one back before it was public."""

    def __init__(self, documents: Iterable[Document] = (), *, source: str = "",
                 former_names: Sequence[Mapping] = (),
                 file_types: Mapping[str, str] | None = None) -> None:
        self.documents: tuple[Document, ...] = tuple(documents)
        self.source = source
        #: the registrant's name history, carried straight through from the Submissions
        #: API so `symbology.from_sec_former_names` can key the documents by entity
        self.former_names: list[dict] = [dict(f) for f in former_names]
        #: full-text hits only: accession -> the EXHIBIT type the hit came from. A hit
        #: whose file_type is EX-19 is not the 10-K body even though its form says 10-K.
        self.file_types: dict[str, str] = dict(file_types or {})

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def frame(self) -> pd.DataFrame:
        if not self.documents:
            return pd.DataFrame(columns=["doc_id", "entity_id", "form", "period_end",
                                         "filed_at", "acceptance_at", "available_at",
                                         "post_close", "provisional", "period_lag_days",
                                         "url"])
        return (pd.DataFrame([d.row() for d in self.documents])
                .sort_values("available_at", kind="mergesort").reset_index(drop=True))

    def as_of(self, ts, *, forms: Sequence[str] = (),
              allow_provisional: bool = False) -> "DocumentSet":
        """Only what was PUBLIC at `ts`. Provisional documents are dropped by default.

        Dropping them is the strict reading and the right default: a provisional document
        has no acceptance instant, so "was it public at ts" cannot be answered, and an
        unanswerable question must not resolve to yes.
        """
        t = pd.Timestamp(ts)
        keep = [d for d in self.documents
                if d.known_at(t) and (allow_provisional or not d.provisional)
                and (not forms or d.form in tuple(forms))]
        return DocumentSet(keep, source=self.source)

    def get(self, doc_id: str, *, as_of, allow_provisional: bool = False) -> Document:
        """One document, or an exception. Never a silently-early document.

        This is the refusal the module exists for: a document requested as of a date
        before its `available_at` raises `LookAheadError` naming both dates, instead of
        coming back with a warning nobody reads.
        """
        t = pd.Timestamp(as_of)
        for d in self.documents:
            if d.doc_id != doc_id:
                continue
            if d.provisional and not allow_provisional:
                raise LookAheadError(
                    f"{doc_id} is PROVISIONAL: it came from a source that supplies "
                    f"file_date and no acceptanceDateTime, so whether it was public on "
                    f"{t.date()} is unknown. Enrich it from the Submissions API, or pass "
                    f"allow_provisional=True to accept filed_at as a proxy and record "
                    f"that you did.")
            if not d.known_at(t):
                raise LookAheadError(
                    f"{doc_id} ({d.form}) was not public on {t.date()}: accepted "
                    f"{d.acceptance_at} and first tradeable {d.available_at.date()}"
                    + (" - it was accepted at or after the close, so the filing date is "
                       "NOT the availability date" if d.post_close else ""))
            return d
        raise KeyError(f"no document {doc_id!r} in this set ({len(self)} documents)")

    def lookahead_report(self) -> pd.DataFrame:
        """Per document: what each of the three dates would have bought you."""
        rows = []
        for d in self.documents:
            rows.append({
                "doc_id": d.doc_id, "form": d.form,
                "period_end": d.period_end, "filed_at": d.filed_at,
                "available_at": d.available_at,
                "period_lag_days": d.period_lag_days,
                "filed_vs_available_days": int((d.available_at.normalize()
                                                - d.filed_at.normalize()).days),
                "post_close": d.post_close, "provisional": d.provisional})
        return pd.DataFrame(rows)

    def summary(self) -> str:
        f = self.frame()
        if not len(f):
            return "DocumentSet(empty)"
        prov = int(f["provisional"].sum())
        post = int(f["post_close"].sum())
        return (f"DocumentSet({len(f)} documents from {self.source or 'unknown'}; "
                f"{post} accepted at/after the close, {prov} provisional; "
                f"available_at {f['available_at'].min().date()} .. "
                f"{f['available_at'].max().date()})")

    def __repr__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------------- EDGAR
class EdgarDocuments:
    """Filing retrieval by CIK / form / date range, plus full-text search.

    The transport is injected exactly as in `search.py`, so every path here is testable
    offline. Both endpoints need the SEC's declared User-Agent and share the one
    per-USER rate limit - the ceiling is 10 requests per second regardless of how many
    machines you run, so a second instance buys nothing and the limiter is shared.
    """

    #: shared across instances on purpose: the SEC's limit is per user, not per process
    limiter = PerSecond(SEC_PACE_PER_S)

    def __init__(self, identity: str | None = None, transport: Transport | None = None,
                 sessions: Sequence | None = None) -> None:
        self._identity = identity
        self._transport = transport
        self.sessions = sessions

    @property
    def transport(self) -> Transport:
        if self._transport is None:
            from fin_skills.discovery.search import urllib_transport   # noqa: PLC0415
            self._transport = urllib_transport()
        return self._transport

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": sec_identity(self._identity)}

    # ------------------------------------------------------------------- submissions
    def filings(self, cik: str | int, *, forms: Sequence[str] = (), start=None, end=None,
                tz_of_record: str = "UTC") -> DocumentSet:
        """Every filing of one CIK in a window, with all three dates.

        `start`/`end` filter on `filed_at`, because that is the field the SEC indexes on.
        Filtering on `available_at` instead would silently drop the post-close filings
        that are the whole point, so the filter is stated and the caller re-filters with
        `as_of()`.
        """
        n = int(str(cik).upper().removeprefix("CIK").lstrip("0") or 0)
        self.limiter.acquire()
        payload = _json(self.transport, SUBMISSIONS_URL.format(cik=n), self._headers())
        recent = (payload.get("filings", {}) or {}).get("recent", {}) or {}
        names = payload.get("formerNames", [])
        entity = payload.get("name", "")
        docs = []
        count = len(recent.get("accessionNumber", []))
        lo = pd.Timestamp(start) if start is not None else None
        hi = pd.Timestamp(end) if end is not None else None

        def col(name: str, i: int, default=None):
            values = recent.get(name) or []
            return values[i] if i < len(values) else default

        for i in range(count):
            form = str(col("form", i, ""))
            if forms and form not in tuple(forms):
                continue
            filed = pd.Timestamp(col("filingDate", i))
            if (lo is not None and filed < lo) or (hi is not None and filed > hi):
                continue
            accn = str(col("accessionNumber", i, ""))
            doc = str(col("primaryDocument", i, "") or "")
            docs.append(build_document(
                doc_id=accn, entity_id=f"{n:010d}", form=form, filed_at=filed,
                acceptance_at=col("acceptanceDateTime", i),
                period_end=col("reportDate", i) or None,
                tz_of_record=tz_of_record, sessions=self.sessions,
                source="sec:submissions", title=entity,
                primary_document=doc,
                url=ARCHIVE_URL.format(cik=n, accn_nodash=accn.replace("-", ""), doc=doc)))
        return DocumentSet(docs, source="sec:submissions", former_names=names)

    # -------------------------------------------------------------------- full text
    def full_text(self, text: str, *, forms: Sequence[str] = (), start=None, end=None,
                  limit: int = 20) -> DocumentSet:
        """efts.sec.gov hits as PROVISIONAL documents.

        Provisional is not a hedge: the response carries `file_date` and no acceptance
        stamp of any kind, so the moment a hit became public is genuinely unknown until
        `enrich()` fetches it. A hit is also a DOCUMENT rather than a filing - an
        exhibit's `form` is still the parent's - so `file_type` is carried through.
        """
        from fin_skills.discovery.search import EdgarFullTextSearch   # noqa: PLC0415
        q = SearchQuery(text=text, limit=limit,
                        extra={k: v for k, v in (("forms", tuple(forms) or None),
                                                 ("startdt", start), ("enddt", end))
                               if v})
        backend = EdgarFullTextSearch(transport=self._transport, identity=self._identity)
        hits = backend.search(q)
        docs = [build_document(
            doc_id=h.identifier, entity_id=str(h.detail.get("cik", "")),
            form=str(h.detail.get("form", "")), filed_at=h.as_of,
            acceptance_at=None,
            period_end=h.detail.get("period_ending") or None,
            source="sec:efts", title=h.name,
            primary_document=str(h.detail.get("document", "")),
        ) for h in hits]
        return DocumentSet(
            docs, source="sec:efts",
            file_types={h.identifier: str(h.detail.get("file_type", "")) for h in hits})

    # ----------------------------------------------------------------------- enrich
    def enrich(self, docs: DocumentSet, *, tz_of_record: str = "UTC") -> DocumentSet:
        """Fill in the acceptance instant of provisional documents, one CIK at a time.

        This is the step that turns full-text hits into point-in-time documents, and it
        costs one Submissions request per distinct CIK - which is why it is explicit
        rather than automatic.
        """
        ciks = sorted({d.entity_id for d in docs if d.provisional and d.entity_id})
        acceptance: dict[str, Document] = {}
        for cik in ciks:
            for known in self.filings(cik, tz_of_record=tz_of_record):
                acceptance[known.doc_id] = known
        out = []
        for d in docs:
            hit = acceptance.get(d.doc_id)
            if d.provisional and hit is not None and hit.acceptance_at is not None:
                out.append(replace(d, acceptance_at=hit.acceptance_at,
                                   available_at=hit.available_at,
                                   post_close=hit.post_close, provisional=False,
                                   tz_of_record=tz_of_record))
            else:
                out.append(d)
        return DocumentSet(out, source=docs.source + "+submissions",
                           file_types=docs.file_types)

    # ------------------------------------------------------------------ transcripts
    def transcripts(self, *_args: Any, **_kw: Any):
        """Not implemented, on purpose. `documents.TRANSCRIPTS` is the whole reason."""
        raise NotImplementedError(TRANSCRIPTS)


def from_submission_rows(rows: Iterable[Mapping], *, entity_id: str = "",
                         tz_of_record: str = "UTC",
                         sessions: Sequence | None = None) -> DocumentSet:
    """Rows shaped like the Submissions API's `filings.recent` -> a DocumentSet.

    Offline, so a saved copy of the JSON is as good as a live call. This is the seam the
    tests use, and the seam a user with a cached submissions file should use too.
    """
    docs = [build_document(
        doc_id=str(r["accessionNumber"]), entity_id=entity_id or str(r.get("cik", "")),
        form=str(r.get("form", "")), filed_at=r["filingDate"],
        acceptance_at=r.get("acceptanceDateTime"),
        period_end=r.get("reportDate") or None, tz_of_record=tz_of_record,
        sessions=sessions, source="sec:submissions",
        primary_document=str(r.get("primaryDocument", ""))) for r in rows]
    return DocumentSet(docs, source="sec:submissions")


__all__ = ["ARCHIVE_URL", "Document", "DocumentSet", "EXCHANGE_TZ", "EdgarDocuments",
           "LookAheadError", "SUBMISSIONS_URL", "TRANSCRIPTS", "build_document",
           "from_submission_rows"]
