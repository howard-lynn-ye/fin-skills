"""fin_skills.discovery.symbology - resolution keyed on (identifier, DATE).

The load-bearing assertions are the ones where the answer is not one: the same ticker
resolves to different entities at two dates, an identifier resolves to NOTHING outside
its window, and `check_usage` fails when a mapping is used outside it. Every fixture is
copied verbatim from data.sec.gov on 2026-09-10.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fin_skills.discovery.symbology import (AmbiguousIdentifier, Assignment,
                                            DEFAULT_MASTER, Scheme, SecurityMaster,
                                            UnknownIdentifier, from_company_tickers,
                                            from_sec_former_names, normalise, resolve,
                                            validate, widen)

WMI_FORMER = [
    {"name": "Mr. Cooper Group Inc.", "from": "2018-10-10T00:00:00.000Z",
     "to": "2025-10-01T00:00:00.000Z"},
    {"name": "WMIH CORP.", "from": "2015-05-13T00:00:00.000Z",
     "to": "2018-09-14T00:00:00.000Z"},
    {"name": "WMI HOLDINGS CORP.", "from": "2012-01-25T00:00:00.000Z",
     "to": "2015-05-08T00:00:00.000Z"},
    {"name": "WASHINGTON MUTUAL, INC", "from": "2006-09-19T00:00:00.000Z",
     "to": "2012-03-20T00:00:00.000Z"},
    {"name": "WASHINGTON MUTUAL INC", "from": "1995-03-29T00:00:00.000Z",
     "to": "2006-10-12T00:00:00.000Z"},
]
AAPL_FORMER = [
    {"name": "APPLE INC", "from": "2007-01-10T05:00:00.000Z",
     "to": "2019-08-05T04:00:00.000Z"},
    {"name": "APPLE COMPUTER INC", "from": "1994-01-26T05:00:00.000Z",
     "to": "2007-01-04T05:00:00.000Z"},
    {"name": "APPLE COMPUTER INC/ FA", "from": "1997-07-28T04:00:00.000Z",
     "to": "1997-07-28T04:00:00.000Z"},
]
TICKER_ROWS = [{"cik_str": 19617, "ticker": "JPM", "title": "JPMORGAN CHASE & CO"},
               {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}]


# ------------------------------------------------------------------------ normalising
@pytest.mark.parametrize("raw", ["320193", "0000320193", "CIK0000320193", " 320193 "])
def test_every_shape_of_cik_normalises_to_the_same_key(raw):
    assert normalise(raw, Scheme.CIK) == "0000320193"


def test_tickers_and_identifiers_upper_case_but_local_ids_are_left_alone():
    assert normalise("aapl", Scheme.TICKER) == "AAPL"
    assert normalise("us0378331005", Scheme.ISIN) == "US0378331005"
    assert normalise("my-Key-1", Scheme.LOCAL) == "my-Key-1"


def test_an_unknown_scheme_is_refused_by_name():
    with pytest.raises(ValueError, match="scheme must be one of"):
        normalise("X", "permno")


def test_validate_only_speaks_for_the_schemes_that_carry_a_check_digit():
    assert validate("US0378331005", Scheme.ISIN).ok
    assert not validate("US0378331006", Scheme.ISIN).ok
    assert validate("ANYTHING", Scheme.TICKER).ok        # no check digit exists to fail
    assert validate("0000320193", Scheme.CIK).ok


# ------------------------------------------------------------------------- assignments
def test_a_window_that_covers_no_date_is_refused_at_construction():
    with pytest.raises(ValueError, match="not after start"):
        Assignment("AAPL", Scheme.TICKER, "e", "n", "2020-01-01", "2020-01-01", "test")


def test_windows_are_half_open():
    a = Assignment("AAPL", Scheme.TICKER, "e", "n", "2020-01-01", "2021-01-01", "test")
    assert a.covers("2020-01-01") and a.covers("2020-12-31")
    assert not a.covers("2021-01-01") and not a.covers("2019-12-31")
    assert not a.open_ended


# ------------------------------------------------------- the same identifier, two dates
def test_one_cik_resolves_to_different_registrants_at_two_dates():
    m = SecurityMaster(from_sec_former_names("933136", "Maverick Merger Sub 2, LLC",
                                             WMI_FORMER))
    then = m.resolve_one("933136", Scheme.CIK, "2007-06-29")
    later = m.resolve_one("933136", Scheme.CIK, "2019-12-31")
    assert then.name == "WASHINGTON MUTUAL, INC"
    assert later.name == "Mr. Cooper Group Inc."
    assert then.name != later.name
    assert then.entity_id == later.entity_id            # one CIK, and that is the trap
    assert len(m.history("933136", Scheme.CIK)) == 6


def test_a_recycled_ticker_resolves_to_two_different_entities():
    m = SecurityMaster([
        Assignment("XYZ", Scheme.TICKER, "local:old", "Old Corp",
                   "1998-01-02", "2009-03-01", "test"),
        Assignment("XYZ", Scheme.TICKER, "local:new", "New Corp Inc",
                   "2014-06-02", None, "test"),
    ])
    assert m.resolve_one("XYZ", Scheme.TICKER, "2005-01-03").entity_id == "local:old"
    assert m.resolve_one("XYZ", Scheme.TICKER, "2020-01-03").entity_id == "local:new"
    assert m.resolve("XYZ", Scheme.TICKER, "2011-01-03") == []      # nobody, for 5 years
    assert len(m.gaps()) == 1 and m.gaps().iloc[0]["days"] > 1900


def test_the_sec_name_history_can_give_two_answers_on_one_day():
    m = SecurityMaster(from_sec_former_names("320193", "Apple Inc.", AAPL_FORMER))
    found = m.resolve("320193", Scheme.CIK, "1997-07-28")
    assert {a.name for a in found} == {"APPLE COMPUTER INC", "APPLE COMPUTER INC/ FA"}
    with pytest.raises(AmbiguousIdentifier, match="2 entities claim"):
        m.resolve_one("320193", Scheme.CIK, "1997-07-28")
    ov = m.overlaps()
    assert len(ov) == 1 and bool(ov.iloc[0]["same_entity"]) is True


def test_the_sec_name_history_can_also_give_none():
    m = SecurityMaster(from_sec_former_names("320193", "Apple Inc.", AAPL_FORMER))
    assert m.resolve("320193", Scheme.CIK, "2007-01-07") == []
    with pytest.raises(UnknownIdentifier, match="no mapping for"):
        m.resolve_one("320193", Scheme.CIK, "2007-01-07")
    gaps = m.gaps()
    assert len(gaps) == 1 and gaps.iloc[0]["days"] == 5


def test_the_inclusive_to_is_converted_so_a_one_day_record_survives():
    m = SecurityMaster(from_sec_former_names("320193", "Apple Inc.", AAPL_FORMER))
    fa = [a for a in m.history("320193", Scheme.CIK) if a.name.endswith("/ FA")][0]
    assert fa.start == pd.Timestamp("1997-07-28")
    assert fa.end == pd.Timestamp("1997-07-29")
    assert "inclusive" in fa.note


def test_the_current_name_start_is_inferred_and_says_so():
    m = SecurityMaster(from_sec_former_names("933136", "Maverick Merger Sub 2, LLC",
                                             WMI_FORMER))
    current = m.history("933136", Scheme.CIK)[-1]
    assert current.open_ended and current.start == pd.Timestamp("2025-10-02")
    assert "INFERRED" in current.note


def test_the_reverse_lookup_finds_every_identifier_of_one_entity():
    m = SecurityMaster([
        Assignment("AAPL", Scheme.TICKER, "e1", "Apple", "2020-01-01", None, "t"),
        Assignment("US0378331005", Scheme.ISIN, "e1", "Apple", "2020-01-01", None, "t"),
        Assignment("MSFT", Scheme.TICKER, "e2", "Microsoft", "2020-01-01", None, "t"),
    ])
    got = m.entity("e1", "2021-06-01")
    assert {(a.scheme.value, a.identifier) for a in got} == {
        ("isin", "US0378331005"), ("ticker", "AAPL")}
    assert m.entity("e1", "2019-01-01") == []


# ------------------------------------------------------- the dateless current snapshot
def test_company_tickers_needs_a_retrieval_date_and_refuses_history():
    m = SecurityMaster(from_company_tickers(TICKER_ROWS, "2026-09-10"))
    assert m.resolve_one("JPM", Scheme.TICKER, "2026-09-10").entity_id == "cik:0000019617"
    assert m.resolve("JPM", Scheme.TICKER, "2008-01-02") == []
    assert all(a.snapshot for a in m)
    with pytest.raises(TypeError):
        from_company_tickers(TICKER_ROWS)                     # no retrieved_at


def test_widen_demands_a_reason_and_records_it():
    a = from_company_tickers(TICKER_ROWS, "2026-09-10")[0]
    with pytest.raises(ValueError, match="needs a `why`"):
        widen(a, start="2008-01-02")
    wider = widen(a, start="2008-01-02", why="CRSP PERMNO 47896 covers 1980-2026")
    assert wider.covers("2008-01-02") and not wider.snapshot
    assert "widened: CRSP" in wider.note


# ---------------------------------------------------------------- the guard-shaped check
def test_the_guard_fails_on_a_mapping_used_outside_its_window():
    m = SecurityMaster(from_company_tickers(TICKER_ROWS, "2026-09-10"))
    report = m.check_usage([{"identifier": "JPM", "scheme": "ticker",
                             "as_of": "2008-01-02"}])
    assert report.passed is False
    assert len(report.errors) == 1
    assert "outside every validity window" in report.errors[0].message
    assert "2026-09-10" in report.errors[0].message           # it names the real window
    assert "FAIL" in report.summary() and report.summary().isascii()


def test_the_guard_passes_inside_the_window_and_still_warns_about_the_snapshot():
    m = SecurityMaster(from_company_tickers(TICKER_ROWS, "2026-09-10"))
    report = m.check_usage([{"identifier": "JPM", "scheme": "ticker",
                             "as_of": "2026-09-11"}])
    assert report.passed is True
    assert [f.severity for f in report.findings] == ["warning"]
    assert "dateless current snapshot" in report.findings[0].message


def test_the_guard_fails_on_an_ambiguous_date_and_on_a_malformed_identifier():
    m = SecurityMaster(from_sec_former_names("320193", "Apple Inc.", AAPL_FORMER))
    m.add(Assignment("US0378331005", Scheme.ISIN, "cik:0000320193", "Apple",
                     "1994-01-26", None, "test"))
    report = m.check_usage([
        {"identifier": "320193", "scheme": "cik", "as_of": "1997-07-28"},
        {"identifier": "US0378331006", "scheme": "isin", "as_of": "2020-01-02"},
        {"identifier": "US0378331005", "scheme": "isin", "as_of": "2020-01-02"},
    ])
    assert report.checked == 3 and not report.passed
    assert len(report.errors) == 2
    assert "2 entities claim it" in report.errors[0].message
    assert "malformed" in report.errors[1].message


def test_an_identifier_the_master_has_never_seen_fails_rather_than_passing():
    m = SecurityMaster(from_company_tickers(TICKER_ROWS, "2026-09-10"))
    report = m.check_usage([{"identifier": "NOPE", "scheme": "ticker",
                             "as_of": "2026-09-10"}])
    assert not report.passed
    assert "unverified rather than merely unmatched" in report.errors[0].message


def test_a_clean_master_passes():
    m = SecurityMaster([Assignment("AAPL", Scheme.TICKER, "e1", "Apple",
                                   "1980-12-12", None, "crsp")])
    report = m.check_usage([{"identifier": "AAPL", "scheme": "ticker",
                             "as_of": "2020-01-02"}])
    assert report.passed and report.findings == []
    assert "PASS" in report.summary()


# ------------------------------------------------------------------------- plumbing
def test_the_default_master_ships_empty_because_no_data_ships():
    assert len(DEFAULT_MASTER) == 0
    assert resolve("AAPL", Scheme.TICKER, "2020-01-02") == []


def test_a_master_holds_assignments_not_tuples():
    with pytest.raises(TypeError, match="no validity window"):
        SecurityMaster().add(("AAPL", "2020-01-01"))


def test_frame_round_trips_every_declared_field():
    m = SecurityMaster(from_sec_former_names("933136", "Maverick Merger Sub 2, LLC",
                                             WMI_FORMER))
    f = m.frame()
    assert len(f) == 6
    assert set(f.columns) >= {"identifier", "scheme", "entity_id", "name", "start",
                              "end", "source", "snapshot", "note"}
    assert f["start"].is_monotonic_increasing
    assert SecurityMaster().frame().empty
