"""fin_skills.discovery - FIND the identifier, then resolve what you found.

`fin_skills.data` fetches given an identifier. Every method on its Adapter protocol -
bars, fundamentals, macro, actions, universe - takes a symbol, a CIK or a series id you
must already know, and there is no `find`. This package is that missing half:

    from fin_skills.discovery import SearchQuery, search, describe

    print(describe())                                   # who can search WHAT, and cannot
    report = search(SearchQuery("ten year treasury", asset_class="macro"))
    print(report.summary())        # results, plus every source that could not run and why

    from fin_skills.discovery import SecurityMaster, from_sec_former_names, Scheme
    master = SecurityMaster(from_sec_former_names("933136", "Maverick Merger Sub 2, LLC",
                                                  former_names))
    master.resolve("933136", Scheme.CIK, "2007-06-29")   # -> WASHINGTON MUTUAL, INC

    from fin_skills.discovery import EdgarDocuments
    docs = EdgarDocuments(identity="You you@example.com").filings(320193, forms=["10-Q"])
    docs.get("0000320193-24-000081", as_of="2024-08-01")   # raises LookAheadError

Four properties hold across the package:

  * **Import is inert.** `import fin_skills.discovery` imports no vendor library and
    opens no socket. Every network call is lazy and behind an injected transport, which
    is also how the whole thing is tested offline.
  * **A source declares what it cannot do.** `SearchCapability.cannot` has no default and
    cannot be empty. Three of the four shipped backends cannot answer "as of a past
    date", so anything they return is a current snapshot - and `search()` says so in the
    report rather than letting the label pass.
  * **The key is `(identifier, DATE)`.** `resolve()` returns a LIST of candidates with
    their windows, because the SEC's own records overlap and leave gaps, so the honest
    answer is sometimes two and sometimes none. `check_usage()` FAILS on a mapping used
    outside its window.
  * **A document carries all three of its dates.** period, filed and acceptance - and a
    derived `available_at` that is the only one a backtest may join on. Asking for a
    document as of a date before it was public raises rather than returning it.

No credential can travel through this package: a key is read from the environment
variable its capability names, at the moment of use, and no URL carrying one is stored,
returned or put in a message.

One naming note: `fin_skills.discovery.search` is both the submodule and the re-exported
function, and the function wins as an attribute of the package. `from
fin_skills.discovery.search import ...` reaches the module as usual; `import
fin_skills.discovery.search as x` binds the function instead, so use
`importlib.import_module` when you want the module object itself.
"""
from __future__ import annotations

from fin_skills.discovery.documents import (Document, DocumentSet, EdgarDocuments,
                                            LookAheadError, TRANSCRIPTS, build_document,
                                            from_submission_rows)
from fin_skills.discovery.search import (CcxtMarketSearch, EdgarCompanySearch,
                                         EdgarFullTextSearch, FredSeriesSearch,
                                         SearchCapability, SearchQuery, SearchReport,
                                         SearchResult, SearchSource, SearchUnavailable,
                                         capabilities, capability, catalogue, describe,
                                         get_source, register_source, search,
                                         sec_identity, sources, unregister_source,
                                         urllib_transport)
from fin_skills.discovery.symbology import (AmbiguousIdentifier, Assignment, Scheme,
                                            SecurityMaster, UnknownIdentifier,
                                            WindowReport, check_identifier,
                                            cusip_from_isin, from_company_tickers,
                                            from_sec_former_names, normalise, resolve,
                                            validate, widen)

__version__ = "0.1.0"

__all__ = [
    "AmbiguousIdentifier", "Assignment", "CcxtMarketSearch", "Document", "DocumentSet",
    "EdgarCompanySearch", "EdgarDocuments", "EdgarFullTextSearch", "FredSeriesSearch",
    "LookAheadError", "Scheme", "SearchCapability", "SearchQuery", "SearchReport",
    "SearchResult", "SearchSource", "SearchUnavailable", "SecurityMaster", "TRANSCRIPTS",
    "UnknownIdentifier", "WindowReport", "build_document", "capabilities", "capability",
    "catalogue", "check_identifier", "cusip_from_isin", "describe",
    "from_company_tickers", "from_sec_former_names", "from_submission_rows",
    "get_source", "normalise", "register_source", "resolve", "search", "sec_identity",
    "sources", "unregister_source", "urllib_transport", "validate", "widen",
]
