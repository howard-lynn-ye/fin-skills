"""fin_skills.core.vendor_decision - the three flags, and the snapshot that cannot drift.

The script carries a dated copy of each source's decisive flags so it runs offline on a
bare install. A copy is a liability unless something checks it, so the test that matters
here is the cross-check against the live `Declaration` objects.
"""
from __future__ import annotations

import pytest

from fin_skills.core.vendor_decision import (REQUESTS, SNAPSHOT, answer, cross_check,
                                             serves)
from fin_skills.data import declare
from fin_skills.data.advise import COVERAGE, Need, recommend


def test_the_snapshot_still_matches_the_declarations_it_was_taken_from():
    result = cross_check()
    assert result is not None, "fin_skills is importable in the test suite"
    assert result["checked"] == len(SNAPSHOT) == len(declare.declarations())
    assert result["drift"] == [], result["drift"]
    assert set(SNAPSHOT) == set(COVERAGE)


def test_the_counts_the_skill_quotes_are_the_ones_the_script_produces():
    n_delisted = sum(1 for r in SNAPSHOT.values() if r["delisted"])
    n_pit = sum(1 for r in SNAPSHOT.values() if r["pit"])
    n_redis = sum(1 for r in SNAPSHOT.values() if r["redistribution"] != "prohibited")
    bars = [r for r in SNAPSHOT.values() if "bars" in r["serves"]]
    assert (len(SNAPSHOT), n_delisted, n_pit, n_redis) == (8, 1, 2, 2)
    assert len(bars) == 6 and sum(1 for r in bars if r["delisted"]) == 0, \
        "of the sources that serve PRICE BARS, none includes delisted names"


def test_delisted_us_equity_bars_resolve_to_nothing_and_name_four_refusals():
    ok, refused = answer(cls="equity", market="US", method="bars", years=10.0,
                         delisted=True)
    assert ok == []
    on_flag = sorted(k for k, v in refused.items() if v == "delisted")
    assert on_flag == ["alphavantage", "stooq", "tiingo", "yfinance"]


def test_the_script_and_the_advisor_agree_on_every_canonical_request():
    """Two independent implementations of the same decision - a hand-written filter over a
    dated snapshot, and the advisor over the live registry. If they ever disagree, one of
    them is wrong and the SKILL.md is quoting whichever it was."""
    fields = {"cls": "asset_class", "market": "market", "years": "history_years",
              "delisted": "delisted", "pit": "point_in_time",
              "redistribute": "redistribute", "store": "store", "key_ok": "key_ok",
              "method": "method"}
    for label, kw in REQUESTS:
        mine, _ = answer(**kw)
        need = Need(frequency="1d", **{fields[k]: v for k, v in kw.items()})
        assert sorted(mine) == sorted(recommend(need).names()), label


def test_a_source_that_does_not_answer_a_script_is_refused_on_that_and_not_on_price():
    assert SNAPSHOT["stooq"]["reachable"] is False
    assert not SNAPSHOT["stooq"]["key"], "it is genuinely keyless; that is not the problem"
    ok, refused = answer(cls="equity", market="US", method="bars", years=10.0)
    assert "stooq" not in ok and refused["stooq"] == "unreachable"


def test_the_filters_apply_in_the_order_the_skill_documents():
    row = dict(SNAPSHOT["tiingo"])
    base = dict(cls="equity", market="US", method="bars", years=1.0)
    assert serves(row, **base) == ""
    assert serves(row, **base, delisted=True) == "delisted"
    assert serves(row, **base, pit=True) == "point_in_time"
    assert serves(row, **base, redistribute=True) == "redistribution"
    assert serves(row, **base, store=True) == "cache"
    assert serves(row, **base, key_ok=False) == "key"
    assert serves(row, **{**base, "years": 999.0}) == "history"
    assert serves(row, cls="crypto", market="US", method="bars") == "asset_class"
    assert serves(row, cls="equity", market="JP", method="bars") == "market"
    assert serves(row, cls="equity", market="US", method="macro") == "method"


def test_every_snapshot_row_names_where_it_was_verified():
    for name, row in SNAPSHOT.items():
        assert row["source"] and ", 2026-09-" in row["source"], name
        assert row["free"], name
        assert row["redistribution"] in declare.REDISTRIBUTION, name
        assert row["cache"] in declare.CACHE_POLICY, name


def test_the_demo_runs_and_prints_its_takeaway(run_main):
    out = run_main("fin_skills.core.vendor_decision")
    assert out.isascii(), "it must survive a stock Windows console"
    assert "TAKEAWAY" in out
    assert "refused on includes_delisted=False: 4" in out
    assert "of the 6 that serve PRICE BARS, 0 include delisted names." in out
    assert "no drift" in out, "the cross-check runs inside the test suite"
    assert "EODHD ALL-IN-ONE, $99.99/month" in out


@pytest.mark.parametrize("name", sorted(SNAPSHOT))
def test_no_snapshot_row_invents_a_source_that_is_not_registered(name):
    assert declare.registered(name)
