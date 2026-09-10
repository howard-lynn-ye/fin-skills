"""fin_skills.core.find_data_source - the search routing table and the three dates.

The routing table's value is the rows that route NOWHERE. A table that claims a free
source for every need is the failure mode this skill exists to prevent, so the empty
answers are asserted as hard as the populated ones.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fin_skills.core.find_data_source import (AAPL_FILINGS, FTS_HIT, PERIODIC_FORMS,
                                              ROUTES, SOURCES, document_dates,
                                              lookahead_summary, source_table,
                                              where_to_search)


# ------------------------------------------------------------------------- the table
def test_every_source_states_something_it_cannot_do():
    assert len(SOURCES) == 4
    for s in SOURCES:
        assert s.cannot, f"{s.name} declares no limits"
        assert s.searches and s.returns and s.rate_limit and s.as_of
        assert s.endpoint


def test_three_of_the_four_cannot_answer_as_of_a_past_date():
    no_as_of = [s.name for s in SOURCES if s.as_of.startswith("NO")
                or "not when it became public" in s.as_of]
    assert set(no_as_of) == {"sec_company_tickers", "ccxt_markets",
                             "sec_full_text_search"}


def test_source_table_is_one_row_per_source():
    t = source_table()
    assert len(t) == len(SOURCES)
    assert set(t.columns) == {"source", "searches", "returns", "as_of", "cannot"}
    assert (t["cannot"] > 0).all()


# ----------------------------------------------------------------------- the routing
@pytest.mark.parametrize("need,expected", [
    ("a macro series id", ("fred_series_search",)),
    ("a company's CIK from its ticker", ("sec_company_tickers",)),
    ("a filing that mentions a phrase", ("sec_full_text_search",)),
    ("a crypto pair or perpetual symbol", ("ccxt_markets",)),
])
def test_a_need_routes_to_the_source_that_can_answer(need, expected):
    assert where_to_search(need) == expected


@pytest.mark.parametrize("need", ["a delisted company", "a futures contract code",
                                  "the ticker a CIK used in 2008"])
def test_the_needs_no_free_source_serves_route_nowhere(need):
    assert where_to_search(need) == ()


def test_an_unknown_need_returns_nothing_rather_than_guessing():
    assert where_to_search("the winning lottery numbers") == ()
    assert where_to_search("") == ()


def test_routes_only_name_sources_that_exist():
    known = {s.name for s in SOURCES}
    for _, names in ROUTES:
        assert set(names) <= known


# ------------------------------------------------------------------- the three dates
def test_all_three_dates_survive_and_available_at_is_derived():
    d = document_dates(AAPL_FILINGS)
    assert len(d) == len(AAPL_FILINGS)
    for col in ("period_end", "filed_at", "acceptance_local", "available_at"):
        assert col in d.columns and d[col].notna().all()
    assert (d["available_at"] >= d["acceptance_local"].dt.normalize()).all()


def test_a_post_close_acceptance_rolls_to_the_next_session():
    d = document_dates(AAPL_FILINGS).set_index("accession")
    # Apple's earnings 8-K, accepted 16:30:35 ET on the day it is stamped
    row = d.loc["0000320193-25-000077"]
    assert row["post_close"]
    assert row["filed_at"] == pd.Timestamp("2025-10-30")
    assert row["available_at"] == pd.Timestamp("2025-10-31")
    assert row["filed_matches_acceptance"]


def test_the_filing_date_is_wrong_in_the_other_direction_too():
    d = document_dates(AAPL_FILINGS).set_index("accession")
    # accepted 18:03 ET, after the 17:30 cutoff, so the SEC stamped the NEXT day
    row = d.loc["0000320193-24-000081"]
    assert row["acceptance_local"] == pd.Timestamp("2024-08-01 18:03:34")
    assert row["filed_at"] == pd.Timestamp("2024-08-02")
    assert not row["filed_matches_acceptance"]


def test_an_intraday_acceptance_is_available_the_same_day():
    d = document_dates(AAPL_FILINGS).set_index("accession")
    row = d.loc["0001140361-24-040659"]                      # 09:06 ET
    assert not row["post_close"]
    assert row["available_at"] == pd.Timestamp("2024-09-10")


def test_availability_never_lands_on_a_weekend():
    rows = [{"form": "8-K", "accessionNumber": "x", "reportDate": "2025-05-02",
             "filingDate": "2025-05-02",
             "acceptanceDateTime": "2025-05-02T21:00:00.000Z"}]     # Friday 17:00 ET
    d = document_dates(rows)
    assert d.iloc[0]["available_at"] == pd.Timestamp("2025-05-05")  # the Monday
    assert d.iloc[0]["available_at"].weekday() < 5


def test_lookahead_summary_pools_only_periodic_forms_for_the_lag():
    d = document_dates(AAPL_FILINGS)
    s = lookahead_summary(d)
    assert s["n"] == len(AAPL_FILINGS)
    assert s["n_periodic"] == sum(1 for f in AAPL_FILINGS if f["form"] in PERIODIC_FORMS)
    assert s["post_close"] == int(d["post_close"].sum())
    assert s["median_period_lag_days"] >= 30            # a 10-K/10-Q lag, not an 8-K's
    assert lookahead_summary(document_dates([]))["n"] == 0


def test_a_full_text_hit_is_an_exhibit_wearing_the_parent_form():
    assert FTS_HIT["form"] == "10-K"
    assert FTS_HIT["file_type"] == "EX-19"               # not the 10-K body
    assert "acceptanceDateTime" not in FTS_HIT          # the whole point
    assert FTS_HIT["_id"].split(":")[0] == FTS_HIT["adsh"]


# ------------------------------------------------------------------------------ demo
def test_demo_runs_and_stays_ascii(run_main):
    out = run_main("fin_skills.core.find_data_source")
    assert out.isascii()
    assert "NOTHING FREE CAN DO THIS" in out
    assert "Only available_at is safe to join on." in out
