"""fin_skills.discovery.documents - three dates per document, and the fetch that refuses.

The fixtures are Apple's own filing metadata and one efts.sec.gov hit, both copied
verbatim on 2026-09-10, so the assertions are about what the SEC actually returns rather
than about a shape somebody imagined. Everything runs through the injected transport, so
nothing here touches the network.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from fin_skills.discovery.documents import (Document, DocumentSet, EdgarDocuments,
                                            LookAheadError, TRANSCRIPTS, build_document,
                                            from_submission_rows)

# Apple's filings.recent, verbatim from data.sec.gov/submissions/CIK0000320193.json
AAPL_RECENT = {
    "accessionNumber": ["0000320193-25-000077", "0000320193-24-000081",
                        "0001140361-24-040659", "0000320193-25-000079"],
    "filingDate": ["2025-10-30", "2024-08-02", "2024-09-10", "2025-10-31"],
    "reportDate": ["2025-10-30", "2024-06-29", "2024-09-10", "2025-09-27"],
    "acceptanceDateTime": ["2025-10-30T20:30:35.000Z", "2024-08-01T22:03:34.000Z",
                           "2024-09-10T13:06:34.000Z", "2025-10-31T10:01:26.000Z"],
    "form": ["8-K", "10-Q", "8-K", "10-K"],
    "primaryDocument": ["a.htm", "aapl-20240629.htm", "c.htm", "aapl-20250927.htm"],
}
SUBMISSIONS_PAYLOAD = {
    "name": "Apple Inc.", "cik": "320193",
    "formerNames": [{"name": "APPLE INC", "from": "2007-01-10T05:00:00.000Z",
                     "to": "2019-08-05T04:00:00.000Z"}],
    "filings": {"recent": AAPL_RECENT, "files": []},
}
FTS_PAYLOAD = {"hits": {"total": {"value": 707, "relation": "eq"}, "hits": [
    {"_id": "0000320193-24-000081:aapl-20240629.htm",
     "_source": {"adsh": "0000320193-24-000081", "ciks": ["0000320193"],
                 "display_names": ["Apple Inc.  (AAPL)  (CIK 0000320193)"],
                 "form": "10-Q", "root_forms": ["10-Q"], "file_type": "10-Q",
                 "file_date": "2024-08-02", "period_ending": "2024-06-29"}},
    {"_id": "0001739566-24-000054:a20231110utzinsidertrading.htm",
     "_source": {"adsh": "0001739566-24-000054", "ciks": ["0001739566"],
                 "display_names": ["Utz Brands, Inc.  (UTZ)  (CIK 0001739566)"],
                 "form": "10-K", "root_forms": ["10-K"], "file_type": "EX-19",
                 "file_date": "2024-02-29", "period_ending": "2023-12-31"}},
]}}


def rows() -> list[dict]:
    n = len(AAPL_RECENT["form"])
    return [{k: AAPL_RECENT[k][i] for k in AAPL_RECENT} for i in range(n)]


def canned(*payloads):
    """A transport that answers each call with the next payload, then repeats the last."""
    seq = list(payloads)
    calls: list[tuple[str, dict]] = []

    def fetch(url, headers):
        calls.append((url, dict(headers)))
        return json.dumps(seq[min(len(calls) - 1, len(seq) - 1)]).encode("utf-8")

    fetch.calls = calls                                       # type: ignore[attr-defined]
    return fetch


# --------------------------------------------------------------------- the three dates
def test_a_document_carries_all_three_dates_plus_the_derived_one():
    d = from_submission_rows(rows(), entity_id="0000320193").frame()
    for col in ("period_end", "filed_at", "acceptance_at", "available_at"):
        assert d[col].notna().all(), col
    assert d["acceptance_at"].dt.tz is not None                # an instant, not a date
    assert not isinstance(d["available_at"].dtype, pd.DatetimeTZDtype)  # a session date


def test_available_at_rolls_a_post_close_acceptance_to_the_next_session():
    ds = from_submission_rows(rows(), entity_id="0000320193")
    d = {x.doc_id: x for x in ds}
    earnings = d["0000320193-25-000077"]                       # 16:30:35 ET
    assert earnings.post_close
    assert earnings.filed_at == pd.Timestamp("2025-10-30")
    assert earnings.available_at == pd.Timestamp("2025-10-31")


def test_the_filing_date_is_also_wrong_in_the_other_direction():
    d = {x.doc_id: x for x in from_submission_rows(rows())}
    q = d["0000320193-24-000081"]                              # accepted 18:03 ET
    assert q.acceptance_at == pd.Timestamp("2024-08-01T22:03:34Z")
    assert q.filed_at == pd.Timestamp("2024-08-02")            # stamped the NEXT day
    assert q.available_at == pd.Timestamp("2024-08-02")
    assert q.period_end == pd.Timestamp("2024-06-29")
    assert q.period_lag_days == 34


def test_an_intraday_acceptance_is_available_the_same_day():
    d = {x.doc_id: x for x in from_submission_rows(rows())}
    assert not d["0001140361-24-040659"].post_close
    assert d["0001140361-24-040659"].available_at == pd.Timestamp("2024-09-10")


def test_the_acceptance_timezone_is_declared_not_assumed():
    """One string, two source conventions, two different tradeable sessions.

    The Submissions API's `acceptanceDateTime` is UTC; the Financial Statement Data Sets'
    `sub.txt.accepted` is Eastern. 19:30 read as UTC is 15:30 ET - inside the session.
    Read as Eastern it is 19:30 ET - three and a half hours after the close.
    """
    stamp = "2025-10-30T19:30:00"
    as_utc = build_document(doc_id="a", entity_id="1", form="8-K",
                            filed_at="2025-10-30", acceptance_at=stamp,
                            tz_of_record="UTC")
    as_eastern = build_document(doc_id="a", entity_id="1", form="8-K",
                                filed_at="2025-10-30", acceptance_at=stamp,
                                tz_of_record="America/New_York")
    assert not as_utc.post_close and as_utc.available_at == pd.Timestamp("2025-10-30")
    assert as_eastern.post_close
    assert as_eastern.available_at == pd.Timestamp("2025-10-31")
    assert as_utc.available_at != as_eastern.available_at
    assert as_utc.acceptance_at != as_eastern.acceptance_at   # both stored in UTC


def test_a_supplied_session_calendar_skips_a_holiday():
    sessions = pd.DatetimeIndex(["2025-11-26", "2025-11-28"])   # Thanksgiving closed
    d = build_document(doc_id="a", entity_id="1", form="8-K", filed_at="2025-11-26",
                       acceptance_at="2025-11-26T21:30:00.000Z", sessions=sessions)
    assert d.available_at == pd.Timestamp("2025-11-28")


# --------------------------------------------------------------------- the refusal
def test_a_document_is_refused_as_of_a_date_before_it_was_public():
    ds = from_submission_rows(rows(), entity_id="0000320193")
    with pytest.raises(LookAheadError) as exc:
        ds.get("0000320193-25-000077", as_of="2025-10-30")
    assert "was not public on 2025-10-30" in str(exc.value)
    assert "first tradeable 2025-10-31" in str(exc.value)
    assert "accepted at or after the close" in str(exc.value)


def test_the_same_document_comes_back_one_day_later():
    ds = from_submission_rows(rows(), entity_id="0000320193")
    got = ds.get("0000320193-25-000077", as_of="2025-10-31")
    assert got.form == "8-K" and got.known_at("2025-10-31")


def test_as_of_filters_the_whole_set_and_an_unknown_id_is_a_keyerror():
    ds = from_submission_rows(rows(), entity_id="0000320193")
    assert len(ds.as_of("2024-09-10")) == 2                     # the two 2024 filings
    assert len(ds.as_of("2020-01-01")) == 0
    assert {d.form for d in ds.as_of("2025-12-31", forms=["10-K"])} == {"10-K"}
    with pytest.raises(KeyError):
        ds.get("nope", as_of="2026-01-01")


def test_an_empty_set_reports_itself_rather_than_raising():
    ds = DocumentSet([])
    assert len(ds) == 0 and ds.summary() == "DocumentSet(empty)"
    assert list(ds.frame().columns)[:3] == ["doc_id", "entity_id", "form"]
    assert ds.lookahead_report().empty


# ------------------------------------------------------------------- provisional docs
def test_a_document_with_no_acceptance_must_declare_itself_provisional():
    with pytest.raises(ValueError, match="provisional=True"):
        Document(doc_id="a", entity_id="1", entity_scheme="cik", form="10-K",
                 filed_at=pd.Timestamp("2024-01-02"), acceptance_at=None,
                 available_at=pd.Timestamp("2024-01-02"))


def test_a_full_text_hit_is_provisional_and_cannot_be_used_point_in_time():
    edgar = EdgarDocuments(identity="Tester t@e.com", transport=canned(FTS_PAYLOAD))
    ds = edgar.full_text("supply chain finance", forms=["10-K"])
    assert len(ds) == 2
    assert all(d.provisional for d in ds)
    assert all(d.acceptance_at is None for d in ds)
    assert ds.file_types["0001739566-24-000054"] == "EX-19"    # an exhibit, form says 10-K
    with pytest.raises(LookAheadError, match="PROVISIONAL"):
        ds.get("0000320193-24-000081", as_of="2026-01-01")
    # accepting filed_at as a proxy is possible, but only by saying so
    assert ds.get("0000320193-24-000081", as_of="2026-01-01",
                  allow_provisional=True).doc_id == "0000320193-24-000081"
    assert len(ds.as_of("2026-01-01")) == 0                    # dropped by default
    assert len(ds.as_of("2026-01-01", allow_provisional=True)) == 2


def test_enrich_turns_a_hit_into_a_point_in_time_document():
    edgar = EdgarDocuments(identity="Tester t@e.com",
                           transport=canned(FTS_PAYLOAD, SUBMISSIONS_PAYLOAD))
    hits = edgar.full_text("apple")
    enriched = edgar.enrich(hits)
    got = {d.doc_id: d for d in enriched}["0000320193-24-000081"]
    assert not got.provisional
    assert got.acceptance_at == pd.Timestamp("2024-08-01T22:03:34Z")
    assert got.available_at == pd.Timestamp("2024-08-02")
    # the Utz hit belongs to another CIK the canned transport also answers for; what
    # matters is that nothing was invented for a document with no acceptance instant
    assert enriched.file_types == hits.file_types


# ---------------------------------------------------------------------- EDGAR fetch
def test_filings_needs_a_declared_user_agent(monkeypatch):
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    from fin_skills.discovery.search import SearchUnavailable
    with pytest.raises(SearchUnavailable, match="403"):
        EdgarDocuments(transport=canned(SUBMISSIONS_PAYLOAD)).filings(320193)


def test_filings_pads_the_cik_filters_forms_and_dates_and_carries_former_names():
    t = canned(SUBMISSIONS_PAYLOAD)
    edgar = EdgarDocuments(identity="Tester t@e.com", transport=t)
    ds = edgar.filings("CIK320193", forms=["10-Q", "10-K"], start="2024-01-01",
                       end="2025-12-31")
    assert "CIK0000320193.json" in t.calls[0][0]
    assert t.calls[0][1]["User-Agent"] == "Tester t@e.com"
    assert {d.form for d in ds} == {"10-Q", "10-K"}
    assert [f["name"] for f in ds.former_names] == ["APPLE INC"]
    assert all(d.url.startswith("https://www.sec.gov/Archives/edgar/data/320193/")
               for d in ds)
    assert "000032019324000081" in {d.url.split("/")[-2] for d in ds}


def test_the_sec_limiter_is_shared_across_instances_because_the_limit_is_per_user():
    a = EdgarDocuments(identity="t t@e.com")
    b = EdgarDocuments(identity="t t@e.com")
    assert a.limiter is b.limiter
    assert a.limiter.describe() == "PerSecond(8/s)"             # below the 10/s ceiling


# ---------------------------------------------------------------------- transcripts
def test_transcripts_are_documented_as_absent_rather_than_stubbed():
    edgar = EdgarDocuments(identity="t t@e.com")
    with pytest.raises(NotImplementedError) as exc:
        edgar.transcripts("AAPL")
    assert "same-day look-ahead" in str(exc.value)
    assert "date of the CALL" in TRANSCRIPTS
    assert "publication" in TRANSCRIPTS.lower()
    assert TRANSCRIPTS.isascii()


# ------------------------------------------------------------------------- reporting
def test_the_lookahead_report_prices_each_shortcut():
    ds = from_submission_rows(rows(), entity_id="0000320193")
    r = ds.lookahead_report().set_index("doc_id")
    assert r.loc["0000320193-25-000079", "period_lag_days"] == 34      # 10-K
    assert r.loc["0000320193-25-000077", "filed_vs_available_days"] == 1
    assert int(r["post_close"].sum()) == 2
    assert ds.summary().isascii() and "accepted at/after the close" in ds.summary()
