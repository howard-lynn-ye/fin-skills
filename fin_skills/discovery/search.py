"""Search: the half `fin_skills.data` does not have.

Every method on the Adapter protocol - bars, fundamentals, macro, actions, universe -
takes an identifier you must already know. There is no `find`. So an agent that does not
already know `DGS10`, `0000320193` or `ES=F` cannot start, and an agent that guesses is
worse than stuck: a wrong series id returns an empty frame rather than an error, and an
empty frame is indistinguishable from "no data" once it is written to disk.

This module is the mirror image of `fin_skills.data.declare`. A source registers a
`SearchCapability` before it can be reached, and the field that earns its place is
`cannot`: three of the four shipped backends cannot answer "as of a past date" at all, so
anything they return is a CURRENT snapshot however you label it. That is the same
survivorship axis `Declaration.includes_delisted` exists for, applied to discovery.

Four rules hold everywhere in this package:

  * **Transport is injected.** Every backend takes `transport=(url, headers) -> bytes`.
    The default builds a urllib transport at CALL time, so importing this module opens no
    socket and the tests run the whole registry offline against canned bytes.
  * **Nothing is imported until it is used.** No vendor library is imported at module
    scope; `import fin_skills.discovery` works with none of them installed.
  * **A credential is read, used and dropped.** It comes from the environment variable the
    capability names, at the moment of the call. No URL carrying one is stored, returned,
    logged or put in an exception message - `_redact()` runs on every URL that reaches a
    message.
  * **A source that cannot run says so.** `search()` never raises because a backend is
    unavailable; it returns a `SearchReport` whose `skipped` maps each source to the
    reason. With every backend absent the report is empty and complete.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd

from fin_skills.data.ratelimit import PerInstanceDelay, PerSecond, RateLimit, Unpublished

#: The SEC's Internet Security Policy, verified 2026-09-09 in fin_skills.data.adapters
#: and re-read 2026-09-10: "no more than 10 requests per second, regardless of the number
#: of machines used to submit requests", with a ten-minute block after a breach.
SEC_CEILING_PER_S = 10
#: what this layer actually paces at - a margin under the ceiling, since the penalty for
#: crossing it is a ten-minute wall rather than a rejected request
SEC_PACE_PER_S = 8

#: environment variable holding the declared SEC User-Agent. Not a credential: it is a
#: public statement of who is asking, which is why it may be passed as an argument.
SEC_IDENTITY_ENV = "EDGAR_IDENTITY"

Transport = Callable[[str, Mapping[str, str]], bytes]

_SECRET_QS = re.compile(r"(?i)([?&](?:api_?key|apikey|token|access_key|secret)=)[^&]*")


def _redact(url: str) -> str:
    """A URL with any credential-shaped query value replaced. Applied to every message."""
    return _SECRET_QS.sub(r"\1<redacted>", str(url))


def _flat(text: str) -> str:
    """One line, so a multi-line reason does not break a report's layout."""
    return " ".join(str(text).split())


def urllib_transport(timeout: float = 30.0) -> Transport:
    """The default transport, built lazily so importing this module opens nothing."""

    def fetch(url: str, headers: Mapping[str, str]) -> bytes:
        import urllib.error                                          # noqa: PLC0415
        import urllib.request                                        # noqa: PLC0415
        req = urllib.request.Request(url, headers=dict(headers))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:                        # pragma: no cover
            raise SearchUnavailable(
                f"HTTP {exc.code} from {_redact(url)}") from None
        except Exception as exc:                                     # pragma: no cover
            raise SearchUnavailable(
                f"{type(exc).__name__} reaching {_redact(url)}") from None

    return fetch


class SearchUnavailable(RuntimeError):
    """This source cannot answer right now, and the message says why.

    Raised for a missing credential, a missing library, a refused network - never for
    "the query matched nothing", which is a legitimate empty result.
    """


# --------------------------------------------------------------------------- the vocabulary
@dataclass(frozen=True)
class SearchQuery:
    """Free text, plus the three narrowings a source may or may not honour.

    `as_of` is the one that matters. A source whose capability says `as_of_supported` is
    False does not silently ignore it - `search()` records a warning naming the source, so
    a result set that cannot be historical never passes for one.
    """

    text: str
    asset_class: str | None = None     # equity | etf | fund | macro | crypto | filing
    market: str | None = None          # "US" | "XNYS" | "FRED" | "binance" | a venue id
    as_of: pd.Timestamp | None = None
    limit: int = 20
    extra: dict = field(default_factory=dict)   # source-specific hints; unknown keys ignored

    def __post_init__(self) -> None:
        if not str(self.text).strip():
            raise ValueError("SearchQuery needs some text to search for")
        if int(self.limit) < 1:
            raise ValueError("SearchQuery.limit must be at least 1")
        if self.as_of is not None:
            object.__setattr__(self, "as_of", pd.Timestamp(self.as_of))


@dataclass(frozen=True)
class SearchResult:
    """One candidate. `confidence` orders results WITHIN one source and nowhere else.

    It is the source's own ranking normalised to [0, 1] - an exact identifier match is
    1.0, an exact name match 0.9, everything else decays with rank. It is not a
    probability and two sources' confidences are not comparable, which is why
    `SearchReport.frame()` keeps `source` next to it.
    """

    identifier: str
    scheme: str                # ticker | cik | series_id | market_symbol | accession
    name: str
    market: str
    asset_class: str
    source: str
    confidence: float
    active: bool | None        # None = the source does not say, which is not False
    as_of: str = ""            # the date the ANSWER is as of, as the source reports it
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        c = float(self.confidence)
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {c}")
        object.__setattr__(self, "confidence", c)

    def row(self) -> dict[str, Any]:
        return {"identifier": self.identifier, "scheme": self.scheme, "name": self.name,
                "market": self.market, "asset_class": self.asset_class,
                "source": self.source, "confidence": round(self.confidence, 3),
                "active": self.active, "as_of": self.as_of}


@dataclass(frozen=True)
class SearchCapability:
    """What a source can search, and - the load-bearing half - what it cannot.

    Same posture as `data.declare.Declaration`: no field has a default except `notes`,
    because a source that will not state whether it can answer as of a past date is a
    source whose results cannot be dated.
    """

    name: str
    endpoint: str
    searches: tuple[str, ...]        # what the free text is matched AGAINST
    returns: tuple[str, ...]         # identifier schemes it can hand back
    asset_classes: tuple[str, ...]
    cannot: tuple[str, ...]
    as_of_supported: bool
    requires_key: bool
    key_env_var: str
    key_sharing: str                 # byok-required | byok-blessed | unstated | n/a
    user_agent_required: bool
    rate_limit: RateLimit
    terms_url: str
    verified_on: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.requires_key and not self.key_env_var:
            raise ValueError(f"{self.name}: requires_key=True needs a key_env_var naming "
                             f"the variable the USER sets")
        if not self.requires_key and self.key_env_var:
            raise ValueError(f"{self.name}: key_env_var set but requires_key is False")
        if not self.cannot:
            raise ValueError(f"{self.name}: state at least one thing this source cannot "
                             f"search; a capability with no limits has not been read")
        pd.Timestamp(self.verified_on)          # raises on a non-date

    def row(self) -> dict[str, Any]:
        return {"source": self.name, "searches": "; ".join(self.searches),
                "returns": "/".join(self.returns), "as_of": self.as_of_supported,
                "key": self.key_env_var or "-", "rate_limit": self.rate_limit.describe(),
                "cannot": len(self.cannot), "verified_on": self.verified_on}


@runtime_checkable
class SearchSource(Protocol):
    """What a backend offers. `search` returns [] for no match and raises
    SearchUnavailable when it cannot run at all - the two are never conflated."""

    capability: SearchCapability

    def search(self, query: SearchQuery) -> list[SearchResult]: ...


# ------------------------------------------------------------------------------ registry
_REGISTRY: dict[str, tuple[type, SearchCapability]] = {}


def register_source(source: type, capability: SearchCapability) -> None:
    """Accept a backend, or refuse it. The only way into `sources()`."""
    if not isinstance(capability, SearchCapability):
        raise TypeError("register_source() needs a SearchCapability; a source that will "
                        "not declare what it cannot do is not usable")
    if source is not None and not isinstance(source, type):
        raise TypeError("register_source() takes the CLASS, not an instance")
    _REGISTRY[capability.name] = (source, capability)


def unregister_source(name: str) -> None:
    _REGISTRY.pop(name, None)


def _registered_names() -> list[str]:
    return sorted(_REGISTRY)


def sources() -> list[str]:
    return _registered_names()


def capabilities() -> list[SearchCapability]:
    return [c for _, c in sorted(_REGISTRY.values(), key=lambda kv: kv[1].name)]


def capability(name: str) -> SearchCapability:
    if name not in _REGISTRY:
        raise KeyError(f"no search source {name!r}; registered: {sources()}")
    return _REGISTRY[name][1]


def get_source(name: str, **kw: Any) -> SearchSource:
    if name not in _REGISTRY:
        raise KeyError(f"no search source {name!r}; registered: {sources()}")
    return _REGISTRY[name][0](**kw)


def catalogue() -> pd.DataFrame:
    """Every registered source and what it declares - the table to read BEFORE searching."""
    return pd.DataFrame([c.row() for c in capabilities()])


def describe(name: str | None = None) -> str:
    """The capability table plus every `cannot`, as ASCII text."""
    caps = capabilities() if name is None else [capability(name)]
    lines: list[str] = []
    for c in caps:
        lines.append(f"{c.name}  ({c.endpoint}, verified {c.verified_on})")
        lines.append(f"  searches      : {'; '.join(c.searches)}")
        lines.append(f"  returns       : {'/'.join(c.returns)}   "
                     f"asset classes: {list(c.asset_classes)}")
        lines.append(f"  as of a date  : {'yes' if c.as_of_supported else 'NO'}")
        lines.append(f"  credential    : "
                     f"{('$' + c.key_env_var) if c.requires_key else 'none required'}"
                     f"   ({c.key_sharing})"
                     + ("   declared User-Agent REQUIRED" if c.user_agent_required else ""))
        lines.append(f"  rate limit    : {c.rate_limit.describe()}")
        for w in c.cannot:
            lines.append(f"  ! cannot {w}")
        if c.notes:
            for chunk in str(c.notes).splitlines():
                lines.append(f"  . {chunk}")
        lines.append("")
    return "\n".join(lines).encode("ascii", "replace").decode("ascii")


# -------------------------------------------------------------------------------- report
@dataclass(frozen=True)
class SearchReport:
    """What a fan-out produced, including every source that could NOT answer and why."""

    query: SearchQuery
    results: tuple[SearchResult, ...]
    skipped: dict[str, str]
    warnings: tuple[str, ...] = ()

    @property
    def ran(self) -> list[str]:
        return sorted({r.source for r in self.results})

    def frame(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(columns=["identifier", "scheme", "name", "market",
                                         "asset_class", "source", "confidence", "active",
                                         "as_of"])
        return (pd.DataFrame([r.row() for r in self.results])
                .sort_values(["source", "confidence"], ascending=[True, False],
                             kind="mergesort").reset_index(drop=True))

    def summary(self) -> str:
        lines = [f"query {self.query.text!r}"
                 + (f" as of {self.query.as_of.date()}" if self.query.as_of is not None else "")
                 + f" -> {len(self.results)} result(s) from {len(self.ran)} source(s)"]
        for w in self.warnings:
            lines.append(f"  ! {_flat(w)}")
        for name, why in sorted(self.skipped.items()):
            lines.append(f"  - {name}: {_flat(why)}")
        return "\n".join(lines).encode("ascii", "replace").decode("ascii")

    def __str__(self) -> str:
        return self.summary()


def search(query: SearchQuery | str, *, sources: Sequence[str] | None = None,
           instances: Mapping[str, SearchSource] | None = None,
           **construct: Any) -> SearchReport:
    """Fan out across the registered sources and collect what came back.

    This never raises because a source is unavailable. A missing key, a missing library
    and a refused socket all land in `report.skipped` with the reason, so the same call
    works with every backend absent and the report says so rather than looking empty.

    `instances` injects already-built sources, which is how the tests run the whole
    registry offline.
    """
    q = SearchQuery(query) if isinstance(query, str) else query
    names = list(sources) if sources is not None else _registered_names()
    out: list[SearchResult] = []
    skipped: dict[str, str] = {}
    warnings: list[str] = []
    for name in names:
        try:
            cap = capability(name)
        except KeyError as exc:
            skipped[name] = str(exc)
            continue
        if q.as_of is not None and not cap.as_of_supported:
            warnings.append(f"{name} cannot answer as of {q.as_of.date()}: its results are "
                            f"a CURRENT snapshot, whatever the query said")
        try:
            src = (instances or {}).get(name) or get_source(name, **construct)
            found = src.search(q)
        except SearchUnavailable as exc:
            skipped[name] = str(exc)
            continue
        except (ImportError, KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            skipped[name] = f"{type(exc).__name__}: {_redact(str(exc))}"
            continue
        out.extend(found)
    return SearchReport(query=q, results=tuple(out), skipped=skipped,
                        warnings=tuple(warnings))


# ------------------------------------------------------------------------------- helpers
def _json(transport: Transport, url: str, headers: Mapping[str, str]) -> Any:
    raw = transport(url, headers)
    try:
        return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SearchUnavailable(f"{_redact(url)} did not return JSON: {exc}") from None


def _rank(hits: Sequence[Any], text: str, key: Callable[[Any], tuple[str, str]]) -> list[float]:
    """Confidence per hit: 1.0 exact identifier, 0.9 exact name, then rank decay."""
    t = str(text).strip().lower()
    out = []
    for i, h in enumerate(hits):
        ident, name = key(h)
        if str(ident).lower() == t:
            out.append(1.0)
        elif str(name).lower() == t:
            out.append(0.9)
        else:
            out.append(round(max(0.1, 0.8 * (0.97 ** i)), 4))
    return out


def credential(cap: SearchCapability) -> str:
    """Read the key from the variable the capability names. Read, used, never stored.

    No function in this package takes a key as an argument and no dataclass field can hold
    one, for the reason FRED states on its own API-key page: "All users of an application
    shall use their own API key." The code path that would ship or proxy one does not exist.
    """
    if not cap.requires_key:
        return ""
    value = os.environ.get(cap.key_env_var) or ""
    if not value:
        raise SearchUnavailable(
            f"{cap.name} needs ${cap.key_env_var} and it is not set. Get your own key at "
            f"{cap.terms_url} and export it; this library never ships or proxies one "
            f"({cap.key_sharing}).")
    return value


def sec_identity(explicit: str | None = None) -> str:
    """The declared User-Agent every sec.gov request needs, or a message saying so."""
    ident = explicit or os.environ.get(SEC_IDENTITY_ENV) or ""
    if not ident:
        raise SearchUnavailable(
            "sec.gov returns 403 without a declared User-Agent. Pass "
            "identity='Your Name you@example.com' or set $" + SEC_IDENTITY_ENV + ". The "
            "SEC checks the header for presence, not content, so a fake browser "
            "User-Agent is a policy violation rather than a workaround.")
    return ident


# ============================================================== the four free backends
class _Backend:
    """Shared plumbing: an injected transport and the capability's own limiter.

    Unknown keyword arguments are ignored on purpose: `search()` fans one set of
    construction arguments out to every source, so `search(q, identity=...)` must not
    blow up on the source that has no identity. Credential-shaped names are the exception
    and are refused by name.

    The limiter comes from the CAPABILITY, so every instance of a source shares it. That
    is right for the SEC, whose ceiling is per user "regardless of the number of machines",
    and wrong for ccxt, whose bucket lives on the exchange instance - so that one
    overrides it.
    """

    capability: SearchCapability

    def __init__(self, transport: Transport | None = None, **_ignored: Any) -> None:
        bad = [k for k in _ignored if re.search(r"(?i)key|token|secret|password", k)]
        if bad:
            raise TypeError(
                f"{sorted(bad)} cannot be passed to a search source. A credential is read "
                f"from the environment variable the capability names, at the moment it is "
                f"used; there is deliberately no argument through which one could travel.")
        self._transport = transport
        self.limiter = self.capability.rate_limit

    @property
    def transport(self) -> Transport:
        if self._transport is None:
            self._transport = urllib_transport()
        return self._transport

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.capability.name}>"


# ------------------------------------------------------------------------------ FRED
FRED_SEARCH_URL = "https://api.stlouisfed.org/fred/series/search"

FRED_CAP = SearchCapability(
    name="fred_series",
    endpoint=FRED_SEARCH_URL,
    searches=("series title", "units", "frequency", "tags - stemmed word matching",
              "the series id itself, but only with search_type='series_id'"),
    returns=("series_id",),
    asset_classes=("macro",),
    cannot=(
        "match a series id in the DEFAULT search_type='full_text' mode - the documented "
        "behaviour is to match title, units, frequency and tags, so searching 'DGS10' "
        "full-text can miss the series it names",
        "return observations: search answers about the series, never about its values",
        "run without a key, and a shared key cannot ship - FRED's API-key page says all "
        "users of an application shall use their own",
    ),
    as_of_supported=True,          # realtime_start / realtime_end, default today
    requires_key=True,
    key_env_var="FRED_API_KEY",
    key_sharing="byok-required",
    user_agent_required=False,
    rate_limit=Unpublished(courtesy_per_s=2.0),
    terms_url="https://fred.stlouisfed.org/docs/api/api_key.html",
    verified_on="2026-09-10",
    notes=("limit defaults to 1000 and is capped at 1000; order_by defaults to "
           "search_rank (fred/series/search docs, read 2026-09-10).\n"
           "realtime_start/realtime_end default to TODAY, so the default answer is which "
           "series exist now - pass query.as_of to ask which existed then.\n"
           "The rate limit has no single published shape: the v1 errors page says 120 "
           "requests/minute, the v2 page says 2/second. Pacing at 2/s satisfies both."),
)


class FredSeriesSearch(_Backend):
    """`fred/series/search` - the only way to turn 'ten year treasury' into DGS10."""

    capability = FRED_CAP

    def search(self, query: SearchQuery) -> list[SearchResult]:
        from urllib.parse import urlencode                           # noqa: PLC0415
        key = credential(self.capability)                            # read; never stored
        params = {"search_text": query.text, "file_type": "json",
                  "limit": min(int(query.limit), 1000),
                  "search_type": str(query.extra.get("search_type", "full_text"))}
        if query.as_of is not None:
            stamp = query.as_of.strftime("%Y-%m-%d")
            params["realtime_start"] = params["realtime_end"] = stamp
        for k in ("filter_variable", "filter_value", "tag_names", "order_by"):
            if k in query.extra:
                params[k] = str(query.extra[k])
        self.limiter.acquire()
        url = f"{FRED_SEARCH_URL}?{urlencode(dict(params, api_key=key))}"
        payload = _json(self.transport, url, {})
        del url, key                                     # the only place either existed
        hits = list(payload.get("seriess", []))[: int(query.limit)]
        conf = _rank(hits, query.text, lambda h: (h.get("id", ""), h.get("title", "")))
        return [SearchResult(
            identifier=str(h.get("id", "")), scheme="series_id",
            name=str(h.get("title", "")), market="FRED", asset_class="macro",
            source=self.capability.name, confidence=c,
            active=None,               # FRED does not flag discontinued series here
            as_of=str(h.get("realtime_start", "")),
            detail={k: h.get(k) for k in ("frequency", "units", "seasonal_adjustment",
                                          "observation_start", "observation_end",
                                          "last_updated", "popularity")},
        ) for h, c in zip(hits, conf)]


# -------------------------------------------------------------------- EDGAR company file
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

EDGAR_COMPANY_CAP = SearchCapability(
    name="edgar_company",
    endpoint=SEC_TICKERS_URL,
    searches=("ticker", "company title - substring, case-insensitive, LOCALLY after one "
              "download of the whole file"),
    returns=("ticker", "cik"),
    asset_classes=("equity", "etf", "etn"),
    cannot=(
        "find a company that no longer trades: the file is a CURRENT snapshot, so a "
        "universe built from it carries the survivorship bias EDGAR's own filings do not",
        "say WHEN a ticker mapped to that CIK - the file has no date field at all, which "
        "is why every mapping it produces opens at the moment you downloaded it",
        "separate a common share from a preferred line or an ETN: one CIK's ticker list "
        "mixes them, so ticker -> CIK -> fundamentals can attach a bank's income "
        "statement to a note",
    ),
    as_of_supported=False,
    requires_key=False,
    key_env_var="",
    key_sharing="n/a",
    user_agent_required=True,
    rate_limit=PerSecond(SEC_PACE_PER_S),
    terms_url="https://www.sec.gov/about/privacy-information#security",
    verified_on="2026-09-10",
    notes=(f"The SEC ceiling is {SEC_CEILING_PER_S} requests/second REGARDLESS of the "
           f"number of machines, so parallelism buys no throughput; this source paces at "
           f"{SEC_PACE_PER_S}/s to keep a margin under a limit whose penalty is a "
           f"ten-minute block.\n"
           "The whole file is fetched once per instance and searched in memory: one "
           "request answers every query, which is the only polite way to use it."),
)


class EdgarCompanySearch(_Backend):
    """company_tickers.json, downloaded once per instance and searched in memory."""

    capability = EDGAR_COMPANY_CAP

    def __init__(self, transport: Transport | None = None, identity: str | None = None,
                 **kw: Any) -> None:
        super().__init__(transport, **kw)
        self._identity = identity
        self._rows: list[dict] | None = None

    def rows(self) -> list[dict]:
        """The file, fetched at most once. `retrieved_at` is the only date it has."""
        if self._rows is None:
            headers = {"User-Agent": sec_identity(self._identity)}
            self.limiter.acquire()
            payload = _json(self.transport, SEC_TICKERS_URL, headers)
            data = payload.values() if isinstance(payload, dict) else payload
            self._rows = [dict(r) for r in data]
        return self._rows

    def search(self, query: SearchQuery) -> list[SearchResult]:
        t = query.text.strip().lower()
        rows = self.rows()
        exact = [r for r in rows if str(r.get("ticker", "")).lower() == t]
        loose = [r for r in rows
                 if r not in exact and (t in str(r.get("ticker", "")).lower()
                                        or t in str(r.get("title", "")).lower())]
        hits = (exact + loose)[: int(query.limit)]
        conf = _rank(hits, query.text,
                     lambda h: (h.get("ticker", ""), h.get("title", "")))
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        return [SearchResult(
            identifier=str(h.get("ticker", "")), scheme="ticker",
            name=str(h.get("title", "")), market="US", asset_class="equity",
            source=self.capability.name, confidence=c,
            active=True,               # everything in this file is currently listed
            as_of=stamp,               # the ONLY date available: when it was downloaded
            detail={"cik": f"{int(h['cik_str']):010d}" if h.get("cik_str") is not None
                    else "", "snapshot": "current - not point-in-time"},
        ) for h, c in zip(hits, conf)]


# ----------------------------------------------------------------- EDGAR full-text search
EDGAR_FTS_URL = "https://efts.sec.gov/LATEST/search-index"

EDGAR_FTS_CAP = SearchCapability(
    name="edgar_fulltext",
    endpoint=EDGAR_FTS_URL,
    searches=("the full text of filings AND of every exhibit attached to them, "
              "2001 onwards",),
    returns=("accession", "cik"),
    asset_classes=("filing",),
    cannot=(
        "reach a filing made before 2001 - the index does not cover 1994-2000",
        "tell you when a hit became PUBLIC: a hit carries file_date and no "
        "acceptanceDateTime, so it is NOT point-in-time until it is enriched from the "
        "Submissions API",
        "distinguish the primary document from an exhibit: a forms=10-K query returns "
        "EX-19 and EX-99 attachments whose `form` is still 10-K",
        "report a total above 10000 - the count saturates rather than paginating further",
    ),
    as_of_supported=True,     # startdt/enddt - but on the FILING date, not availability
    requires_key=False,
    key_env_var="",
    key_sharing="n/a",
    user_agent_required=True,
    rate_limit=PerSecond(SEC_PACE_PER_S),
    terms_url="https://www.sec.gov/about/privacy-information#security",
    verified_on="2026-09-10",
    notes=("as_of narrows on `file_date`, which is the date the SEC STAMPS, not the "
           "instant the document appeared. Treat a hit as provisional and enrich it "
           "before using it point-in-time - fin_skills.discovery.documents does.\n"
           "The hit id is '<accession>:<document>', so two hits can share one filing."),
)


class EdgarFullTextSearch(_Backend):
    """efts.sec.gov - free, no key, and it searches exhibits as well as filings."""

    capability = EDGAR_FTS_CAP

    def __init__(self, transport: Transport | None = None, identity: str | None = None,
                 **kw: Any) -> None:
        super().__init__(transport, **kw)
        self._identity = identity

    def search(self, query: SearchQuery) -> list[SearchResult]:
        from urllib.parse import urlencode                           # noqa: PLC0415
        params: dict[str, str] = {"q": f'"{query.text}"'}
        forms = query.extra.get("forms")
        if forms:
            params["forms"] = ",".join(forms) if not isinstance(forms, str) else forms
        start = query.extra.get("startdt")
        end = query.extra.get("enddt") or (query.as_of.strftime("%Y-%m-%d")
                                           if query.as_of is not None else None)
        if start or end:
            params["dateRange"] = "custom"
            params["startdt"] = str(start or "2001-01-01")
            params["enddt"] = str(end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
        headers = {"User-Agent": sec_identity(self._identity)}
        self.limiter.acquire()
        payload = _json(self.transport, f"{EDGAR_FTS_URL}?{urlencode(params)}", headers)
        hits = list(payload.get("hits", {}).get("hits", []))[: int(query.limit)]
        conf = _rank(hits, query.text, lambda h: ("", ""))
        out = []
        for h, c in zip(hits, conf):
            s = h.get("_source", {})
            names = s.get("display_names") or [""]
            out.append(SearchResult(
                identifier=str(s.get("adsh", "")), scheme="accession",
                name=str(names[0]), market="US", asset_class="filing",
                source=self.capability.name, confidence=c,
                active=None,
                as_of=str(s.get("file_date", "")),
                detail={"cik": (s.get("ciks") or [""])[0], "form": s.get("form", ""),
                        "root_forms": s.get("root_forms", []),
                        "file_type": s.get("file_type", ""),
                        "document": str(h.get("_id", "")).split(":", 1)[-1],
                        "period_ending": s.get("period_ending", ""),
                        "available_at": None,
                        "provisional": "file_date only - no acceptanceDateTime in FTS"}))
        return out


# ------------------------------------------------------------------------------- ccxt
CCXT_CAP = SearchCapability(
    name="ccxt_markets",
    endpoint="ccxt <venue>.load_markets()",
    searches=("unified symbol", "base", "quote", "the venue's own market id - LOCALLY, "
              "over the market map the instance already holds"),
    returns=("market_symbol",),
    asset_classes=("crypto", "perpetual", "future", "option"),
    cannot=(
        "find a DELISTED pair: load_markets() returns what the venue lists right now, so "
        "a pair list built from it is survivorship-biased in a market where listings turn "
        "over faster than they do in equities",
        "answer as of a past date - there is no historical market map in ccxt at all",
        "search across venues in one call: the market map belongs to ONE exchange "
        "instance, and constructing a fresh instance per call resets its rate limiter",
    ),
    as_of_supported=False,
    requires_key=False,
    key_env_var="",
    key_sharing="n/a",
    user_agent_required=False,
    rate_limit=PerInstanceDelay(),
    terms_url="https://github.com/ccxt/ccxt#licence",
    verified_on="2026-09-10",
    notes=("Verified in ccxt's own source (python/ccxt/base/exchange.py at 4.5.78, read "
           "2026-09-10): load_markets(reload=False) returns self.markets without a "
           "request when they are already loaded, and otherwise issues fetch_currencies() "
           "AND fetch_markets() - up to two calls through the instance's own limiter.\n"
           "This source holds ONE instance per venue for its lifetime, mirroring the "
           "leaky bucket that lives on that instance."),
)


class CcxtMarketSearch(_Backend):
    """One venue's market map, loaded once and searched in memory.

    `markets_loader` is injected in the tests; in production it lazily imports ccxt,
    constructs the venue once, and reuses it - the same discipline the ccxt adapter keeps,
    for the same reason.
    """

    capability = CCXT_CAP

    def __init__(self, venue: str = "binance", transport: Transport | None = None,
                 markets_loader: Callable[[str], Mapping[str, Mapping]] | None = None,
                 **kw: Any) -> None:
        super().__init__(transport, **kw)
        self.venue = str(venue)
        # ccxt's bucket belongs to the exchange INSTANCE, so this source gets its own
        # rather than the shared one on the capability
        self.limiter = PerInstanceDelay()
        self._loader = markets_loader
        self._markets: Mapping[str, Mapping] | None = None
        self._exchange = None

    def _default_loader(self, venue: str) -> Mapping[str, Mapping]:
        try:
            import ccxt                                              # noqa: PLC0415
        except ImportError:
            raise SearchUnavailable(
                "this source needs 'ccxt', which is not installed:\n    pip install ccxt\n"
                "fin_skills.discovery itself depends only on numpy and pandas.") from None
        if self._exchange is None:
            if not hasattr(ccxt, venue):
                raise SearchUnavailable(f"ccxt has no exchange {venue!r}")
            self._exchange = getattr(ccxt, venue)()      # ONE instance, held
        self.limiter.acquire()
        return self._exchange.load_markets()

    def markets(self) -> Mapping[str, Mapping]:
        if self._markets is None:
            loader = self._loader or self._default_loader
            self._markets = dict(loader(self.venue))
        return self._markets

    def search(self, query: SearchQuery) -> list[SearchResult]:
        if query.market and str(query.market).lower() != self.venue.lower():
            return []                     # this instance is one venue; say nothing of others
        t = query.text.strip().lower()
        hits = []
        for symbol, m in self.markets().items():
            fields = (str(symbol), str(m.get("id", "")), str(m.get("base", "")),
                      str(m.get("quote", "")))
            if any(t == f.lower() for f in fields):
                hits.insert(0, (symbol, m))
            elif any(t in f.lower() for f in fields):
                hits.append((symbol, m))
        if query.asset_class:
            want = str(query.asset_class).lower()
            hits = [(s, m) for s, m in hits
                    if want in (str(m.get("type", "")).lower(), "crypto")]
        hits = hits[: int(query.limit)]
        conf = _rank(hits, query.text, lambda h: (h[0], h[1].get("id", "")))
        return [SearchResult(
            identifier=str(symbol), scheme="market_symbol",
            name=f"{m.get('base', '')}/{m.get('quote', '')}", market=self.venue,
            asset_class=str(m.get("type", "spot")), source=self.capability.name,
            confidence=c,
            active=m.get("active"),      # the venue's own flag; None means it did not say
            as_of="",                    # deliberately blank: there is no as-of here
            detail={"id": m.get("id", ""), "settle": m.get("settle"),
                    "expiry": m.get("expiry"), "contract": m.get("contract"),
                    "linear": m.get("linear"), "inverse": m.get("inverse")},
        ) for (symbol, m), c in zip(hits, conf)]


register_source(FredSeriesSearch, FRED_CAP)
register_source(EdgarCompanySearch, EDGAR_COMPANY_CAP)
register_source(EdgarFullTextSearch, EDGAR_FTS_CAP)
register_source(CcxtMarketSearch, CCXT_CAP)


__all__ = ["CCXT_CAP", "EDGAR_COMPANY_CAP", "EDGAR_FTS_CAP", "EDGAR_FTS_URL",
           "FRED_CAP", "FRED_SEARCH_URL", "SEC_CEILING_PER_S", "SEC_IDENTITY_ENV",
           "SEC_PACE_PER_S", "SEC_TICKERS_URL", "CcxtMarketSearch", "EdgarCompanySearch",
           "EdgarFullTextSearch", "FredSeriesSearch", "SearchCapability", "SearchQuery",
           "SearchReport", "SearchResult", "SearchSource", "SearchUnavailable",
           "Transport", "capabilities", "capability", "catalogue", "credential",
           "describe", "get_source", "register_source", "search", "sec_identity",
           "sources", "unregister_source", "urllib_transport"]
