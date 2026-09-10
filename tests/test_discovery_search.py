"""fin_skills.discovery.search - the registry, the capabilities, and four offline backends.

Every backend is exercised against canned bytes through the injected transport, so the
whole file runs with no network, no key and no vendor library. The payload shapes are
copied verbatim from live responses on 2026-09-10 (the SEC and FRED shapes) or from
ccxt's own MarketInterface TypedDict, so a vendor changing its shape breaks a test rather
than a user.
"""
from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import fin_skills.discovery as D

# `fin_skills.discovery.search` is BOTH the module and the re-exported function, and the
# function wins as an attribute of the package - so the module is reached explicitly.
S = importlib.import_module("fin_skills.discovery.search")

MODULE = Path(S.__file__).resolve()


def canned(payload) -> S.Transport:
    """A transport that answers every URL with the same JSON, and records the calls."""
    calls: list[tuple[str, dict]] = []

    def fetch(url, headers):
        calls.append((url, dict(headers)))
        return json.dumps(payload).encode("utf-8")

    fetch.calls = calls                                       # type: ignore[attr-defined]
    return fetch


# ------------------------------------------------------------------ import is inert
def test_importing_reaches_no_vendor_and_no_socket():
    assert "ccxt" not in sys.modules
    src = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {getattr(n, "module", "") or "" for n in top_level}
    names |= {a.name for n in top_level if isinstance(n, ast.Import) for a in n.names}
    for banned in ("ccxt", "requests", "urllib.request", "httpx", "fredapi", "edgar"):
        assert not any(str(m).startswith(banned) for m in names), \
            f"{banned} is imported at module scope"


def test_the_package_docstring_promises_what_the_module_delivers():
    assert D.__doc__ and "no socket" in D.__doc__
    assert set(D.__all__) == set(sorted(D.__all__))


# -------------------------------------------------------------------- the vocabulary
def test_a_query_needs_text_and_a_sane_limit():
    with pytest.raises(ValueError):
        S.SearchQuery("   ")
    with pytest.raises(ValueError):
        S.SearchQuery("x", limit=0)
    q = S.SearchQuery("x", as_of="2020-01-02")
    assert q.as_of == pd.Timestamp("2020-01-02")


def test_confidence_is_bounded():
    with pytest.raises(ValueError):
        S.SearchResult("X", "ticker", "n", "US", "equity", "s", 1.5, True)
    r = S.SearchResult("X", "ticker", "n", "US", "equity", "s", 0.5, None)
    assert r.row()["active"] is None                # None is not False


def test_a_capability_must_declare_a_limit():
    kw = dict(name="x", endpoint="e", searches=("a",), returns=("b",),
              asset_classes=("c",), as_of_supported=False, requires_key=False,
              key_env_var="", key_sharing="n/a", user_agent_required=False,
              rate_limit=S.PerSecond(1) if hasattr(S, "PerSecond") else None,
              terms_url="t", verified_on="2026-09-10")
    from fin_skills.data.ratelimit import PerSecond
    kw["rate_limit"] = PerSecond(1)
    with pytest.raises(ValueError, match="cannot search"):
        S.SearchCapability(cannot=(), **kw)
    ok = S.SearchCapability(cannot=("fly",), **kw)
    assert ok.row()["cannot"] == 1


def test_a_capability_that_needs_a_key_must_name_the_variable():
    from fin_skills.data.ratelimit import PerSecond
    with pytest.raises(ValueError, match="key_env_var"):
        S.SearchCapability(name="x", endpoint="e", searches=("a",), returns=("b",),
                           asset_classes=("c",), cannot=("d",), as_of_supported=False,
                           requires_key=True, key_env_var="", key_sharing="byok-required",
                           user_agent_required=False, rate_limit=PerSecond(1),
                           terms_url="t", verified_on="2026-09-10")


# ---------------------------------------------------------------------- the registry
def test_the_four_backends_are_registered_and_all_declare_limits():
    assert S.sources() == ["ccxt_markets", "edgar_company", "edgar_fulltext",
                           "fred_series"]
    for cap in S.capabilities():
        assert cap.cannot and cap.verified_on == "2026-09-10"
    assert len(S.catalogue()) == 4
    assert S.describe().isascii()


def test_register_refuses_an_instance_and_a_bare_capability():
    with pytest.raises(TypeError):
        S.register_source(object(), S.FRED_CAP)
    with pytest.raises(TypeError):
        S.register_source(S.FredSeriesSearch, {"name": "nope"})


def test_registry_round_trip_leaves_the_shipped_sources_intact():
    from fin_skills.data.ratelimit import Unpublished
    cap = S.SearchCapability(name="_probe", endpoint="e", searches=("a",),
                             returns=("b",), asset_classes=("c",), cannot=("d",),
                             as_of_supported=False, requires_key=False, key_env_var="",
                             key_sharing="n/a", user_agent_required=False,
                             rate_limit=Unpublished(), terms_url="t",
                             verified_on="2026-09-10")

    class Probe:
        capability = cap

        def __init__(self, **kw):
            pass

        def search(self, query):
            return []

    S.register_source(Probe, cap)
    try:
        assert "_probe" in S.sources()
        assert isinstance(S.get_source("_probe"), S.SearchSource)
    finally:
        S.unregister_source("_probe")
    assert "_probe" not in S.sources()
    with pytest.raises(KeyError):
        S.capability("_probe")


# ----------------------------------------------------- the whole registry, unavailable
def test_search_with_every_backend_absent_reports_why_and_never_raises(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    report = D.search("ten year treasury")
    assert report.results == ()
    assert set(report.skipped) == set(S.sources())
    assert "FRED_API_KEY" in report.skipped["fred_series"]
    assert "User-Agent" in report.skipped["edgar_company"]
    assert "ccxt" in report.skipped["ccxt_markets"]
    assert len(report.frame()) == 0
    assert report.summary().isascii()
    assert "\n  - fred_series:" in report.summary()          # one line per source


def test_an_as_of_query_warns_on_every_source_that_cannot_honour_it(monkeypatch):
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    report = D.search(S.SearchQuery("AAPL", as_of="2008-01-02"))
    warned = {w.split()[0] for w in report.warnings}
    assert warned == {"edgar_company", "ccxt_markets"}       # the two with no as-of
    assert all("CURRENT snapshot" in w for w in report.warnings)


def test_a_missing_source_name_lands_in_skipped_rather_than_raising():
    report = D.search("x", sources=["not_a_source"])
    assert list(report.skipped) == ["not_a_source"]


# ------------------------------------------------------------------------- FRED
FRED_PAYLOAD = {"realtime_start": "2026-09-10", "seriess": [
    {"id": "DGS10", "title": "Market Yield on U.S. Treasury Securities at 10-Year "
                             "Constant Maturity, Quoted on an Investment Basis",
     "frequency": "Daily", "units": "Percent", "seasonal_adjustment": "Not Applicable",
     "observation_start": "1962-01-02", "realtime_start": "2026-09-10", "popularity": 93},
    {"id": "DGS10YR", "title": "A plausible series that does not exist",
     "frequency": "Daily", "units": "Percent", "realtime_start": "2026-09-10"},
]}


def test_fred_needs_a_key_and_says_where_to_get_one(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(S.SearchUnavailable, match="FRED_API_KEY"):
        S.FredSeriesSearch(transport=canned(FRED_PAYLOAD)).search(S.SearchQuery("x"))


def test_fred_search_parses_series_and_never_leaks_the_key(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "0123456789abcdef0123456789abcdef")
    t = canned(FRED_PAYLOAD)
    out = S.FredSeriesSearch(transport=t).search(S.SearchQuery("DGS10", limit=5))
    assert [r.identifier for r in out] == ["DGS10", "DGS10YR"]
    assert out[0].confidence == 1.0 and out[1].confidence < 1.0   # exact id wins
    assert out[0].scheme == "series_id" and out[0].market == "FRED"
    assert out[0].detail["frequency"] == "Daily"
    # the key is in the URL the transport saw and in nothing this layer returns
    url = t.calls[0][0]
    assert "api_key=" in url
    for r in out:
        assert "0123456789abcdef" not in json.dumps(r.row())
        assert "0123456789abcdef" not in json.dumps(r.detail, default=str)
    assert "<redacted>" in S._redact(url) and "0123456789abcdef" not in S._redact(url)


def test_fred_as_of_becomes_a_realtime_window(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k" * 32)
    t = canned(FRED_PAYLOAD)
    S.FredSeriesSearch(transport=t).search(S.SearchQuery("gdp", as_of="2014-01-30"))
    url = t.calls[0][0]
    assert "realtime_start=2014-01-30" in url and "realtime_end=2014-01-30" in url


def test_fred_series_id_mode_is_opt_in(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k" * 32)
    t = canned(FRED_PAYLOAD)
    S.FredSeriesSearch(transport=t).search(S.SearchQuery("DGS*"))
    assert "search_type=full_text" in t.calls[0][0]           # the DEFAULT, and the trap
    t2 = canned(FRED_PAYLOAD)
    S.FredSeriesSearch(transport=t2).search(
        S.SearchQuery("DGS*", extra={"search_type": "series_id"}))
    assert "search_type=series_id" in t2.calls[0][0]


# ------------------------------------------------------------------- EDGAR companies
TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 19617, "ticker": "JPM", "title": "JPMORGAN CHASE & CO"},
    "2": {"cik_str": 19617, "ticker": "JPM-PC", "title": "JPMORGAN CHASE & CO"},
    "3": {"cik_str": 19617, "ticker": "VYLD", "title": "JPMORGAN CHASE & CO"},
}


def test_edgar_company_search_needs_a_declared_user_agent(monkeypatch):
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    with pytest.raises(S.SearchUnavailable, match="403"):
        S.EdgarCompanySearch(transport=canned(TICKERS_PAYLOAD)).search(S.SearchQuery("AAPL"))


def test_edgar_company_search_matches_ticker_then_name_and_sends_the_identity():
    t = canned(TICKERS_PAYLOAD)
    src = S.EdgarCompanySearch(transport=t, identity="Tester test@example.com")
    out = src.search(S.SearchQuery("JPM"))
    assert out[0].identifier == "JPM" and out[0].confidence == 1.0
    # the title matches too, so one CIK hands back the common share, a preferred line and
    # an ETN under one name - which is exactly why the capability says it cannot tell them
    # apart and why a ticker->CIK->fundamentals join misattributes
    assert {r.identifier for r in out} == {"JPM", "JPM-PC", "VYLD"}
    assert len({r.detail["cik"] for r in out}) == 1
    assert out[0].detail["cik"] == "0000019617"              # zero-padded to ten
    assert t.calls[0][1]["User-Agent"] == "Tester test@example.com"
    src.search(S.SearchQuery("Apple"))
    assert len(t.calls) == 1, "the file is fetched once per instance, not once per query"


def test_edgar_company_results_are_stamped_with_the_download_not_the_query():
    src = S.EdgarCompanySearch(transport=canned(TICKERS_PAYLOAD), identity="T t@e.com")
    out = src.search(S.SearchQuery("AAPL", as_of="2008-01-02"))
    assert out[0].as_of == pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    assert out[0].detail["snapshot"].startswith("current")
    assert S.EDGAR_COMPANY_CAP.as_of_supported is False


# ---------------------------------------------------------------- EDGAR full text
FTS_PAYLOAD = {"hits": {"total": {"value": 707, "relation": "eq"}, "hits": [
    {"_id": "0001739566-24-000054:a20231110utzinsidertrading.htm",
     "_source": {"adsh": "0001739566-24-000054", "ciks": ["0001739566"],
                 "display_names": ["Utz Brands, Inc.  (UTZ)  (CIK 0001739566)"],
                 "form": "10-K", "root_forms": ["10-K"], "file_type": "EX-19",
                 "file_date": "2024-02-29", "period_ending": "2023-12-31"}},
]}}


def test_full_text_search_returns_documents_with_no_acceptance_stamp():
    src = S.EdgarFullTextSearch(transport=canned(FTS_PAYLOAD), identity="T t@e.com")
    out = src.search(S.SearchQuery("supply chain finance", extra={"forms": ("10-K",)}))
    assert out[0].scheme == "accession"
    assert out[0].identifier == "0001739566-24-000054"
    assert out[0].as_of == "2024-02-29"                  # file_date, and nothing else
    assert out[0].detail["available_at"] is None
    assert "no acceptanceDateTime" in out[0].detail["provisional"]
    assert out[0].detail["file_type"] == "EX-19"         # an exhibit, not the 10-K body
    assert out[0].detail["form"] == "10-K"               # which the form field hides


def test_full_text_search_passes_the_form_and_date_narrowings():
    t = canned(FTS_PAYLOAD)
    S.EdgarFullTextSearch(transport=t, identity="T t@e.com").search(
        S.SearchQuery("x", extra={"forms": ("10-K", "10-Q"), "startdt": "2024-01-01",
                                  "enddt": "2024-03-31"}))
    url = t.calls[0][0]
    assert "forms=10-K%2C10-Q" in url and "startdt=2024-01-01" in url
    assert "dateRange=custom" in url
    assert "q=%22x%22" in url                            # the phrase stays quoted


# --------------------------------------------------------------------------- ccxt
CCXT_MARKETS = {
    "BTC/USDT": {"id": "BTCUSDT", "base": "BTC", "quote": "USDT", "active": True,
                 "type": "spot", "contract": False, "settle": None, "expiry": None},
    "BTC/USDT:USDT": {"id": "BTCUSDT", "base": "BTC", "quote": "USDT", "active": True,
                      "type": "swap", "contract": True, "settle": "USDT",
                      "linear": True, "inverse": False, "expiry": None},
    "ETH/USDT": {"id": "ETHUSDT", "base": "ETH", "quote": "USDT", "active": True,
                 "type": "spot", "contract": False},
}


def test_ccxt_search_is_local_over_an_injected_market_map():
    loaded = []

    def loader(venue):
        loaded.append(venue)
        return CCXT_MARKETS

    src = S.CcxtMarketSearch(venue="binance", markets_loader=loader)
    out = src.search(S.SearchQuery("BTC"))
    assert {r.identifier for r in out} == {"BTC/USDT", "BTC/USDT:USDT"}
    assert all(r.market == "binance" for r in out)
    assert {r.asset_class for r in out} == {"spot", "swap"}
    src.search(S.SearchQuery("ETH"))
    assert loaded == ["binance"], "load_markets is called once per instance"


def test_ccxt_search_is_silent_about_other_venues():
    src = S.CcxtMarketSearch(venue="binance", markets_loader=lambda v: CCXT_MARKETS)
    assert src.search(S.SearchQuery("BTC", market="kraken")) == []


def test_ccxt_reports_the_venues_own_active_flag_and_never_invents_one():
    markets = dict(CCXT_MARKETS)
    markets["OLD/USDT"] = {"id": "OLDUSDT", "base": "OLD", "quote": "USDT",
                           "type": "spot"}                # no `active` key at all
    src = S.CcxtMarketSearch(venue="binance", markets_loader=lambda v: markets)
    out = src.search(S.SearchQuery("OLD"))
    assert out[0].active is None                          # not False
    assert out[0].as_of == ""                             # there is no as-of here


def test_ccxt_is_skipped_with_a_pip_line_when_the_library_is_absent():
    src = S.CcxtMarketSearch(venue="binance")
    with pytest.raises(S.SearchUnavailable, match="pip install ccxt"):
        src.search(S.SearchQuery("BTC"))


# ------------------------------------------------------------------ credential hygiene
@pytest.mark.parametrize("kw", ["api_key", "apikey", "token", "secret", "password"])
def test_no_backend_accepts_a_credential_as_an_argument(kw):
    for cls in (S.FredSeriesSearch, S.EdgarCompanySearch, S.EdgarFullTextSearch,
                S.CcxtMarketSearch):
        with pytest.raises(TypeError, match="cannot be passed"):
            cls(**{kw: "x"})


def test_search_fans_construction_arguments_out_without_choking(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    report = D.search("AAPL", identity="Tester t@e.com",
                      instances={"edgar_company": S.EdgarCompanySearch(
                          transport=canned(TICKERS_PAYLOAD), identity="T t@e.com")})
    assert [r.identifier for r in report.results] == ["AAPL"]
    assert "fred_series" in report.skipped


def test_no_source_file_contains_a_hardcoded_key_shaped_string():
    import re
    for path in Path(S.__file__).resolve().parent.glob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert not re.search(r"['\"][0-9a-f]{32}['\"]", src), f"{path.name}"
