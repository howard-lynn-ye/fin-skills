"""fin_skills.data.advise - and above all, what it does when the answer is NOTHING.

The case this module exists for is the one every free-tier recommender gets wrong:
*delisted US equity daily bars, on a free tier*. There is no such source. A recommender
that quietly returns the least-bad option turns a licensing-and-coverage dead end into a
survivorship-biased backtest, so the empty answer is asserted here first and hardest.

Nothing in this file reaches a network - the autouse fixture in conftest.py refuses every
socket - and nothing imports a vendor library.
"""
from __future__ import annotations

import pytest

from fin_skills.data import declare
from fin_skills.data.advise import (ASSET_CLASSES, COVERAGE, MARKETS, METHODS, Need,
                                    PAID, coverage_frame, recommend)


# --------------------------------------------------------- the empty answer, first
def test_delisted_us_equity_dailies_on_a_free_tier_returns_NOTHING():
    """The headline case. Every free adapter here declares includes_delisted=False, so
    the ranked list must be EMPTY, the reason must name the flag, and a paid vendor must
    be named - not a silently substituted second-best."""
    rec = recommend(Need(asset_class="equity", market="US", frequency="1d",
                         history_years=10, delisted=True))
    assert rec.empty and rec.ranked == () and rec.names() == []
    assert rec.best is None
    assert rec.to_frame().empty and list(rec.to_frame().columns)   # shape, not rows

    assert "includes_delisted=False" in rec.empty_reason
    assert "NO free source of delisted US equity price history" in rec.empty_reason
    assert "upper bound" in rec.empty_reason

    # every price source was refused, and each refusal says which flag did it
    delisted_refusals = {r.adapter for r in rec.refused if r.flag == "delisted"}
    assert delisted_refusals == {"yfinance", "tiingo", "alphavantage", "stooq"}

    # and a paid vendor is named, with the page and date it was checked at
    vendors = {p.vendor for p in rec.paid}
    assert {"EODHD", "Norgate Data"} <= vendors
    eodhd = next(p for p in rec.paid if p.vendor == "EODHD")
    assert "$99.99/month" in eodhd.price
    assert "eodhd.com/pricing, verified 2026-09-10" in eodhd.source


def test_the_dead_end_still_names_the_half_that_IS_available():
    """Alpha Vantage's LISTING_STATUS is free and point-in-time, so the MEMBERSHIP half
    of survivorship is fixable even though the price half is not. Saying so is the
    difference between a refusal and a useful refusal."""
    rec = recommend(Need(market="US", delisted=True))
    partial = dict(rec.partial)
    assert "alphavantage" in partial
    assert "LISTING_STATUS" in partial["alphavantage"]
    assert "2010-01-01" in partial["alphavantage"]
    assert "MEMBERSHIP" in partial["alphavantage"]
    assert "edgar" in partial and "survivorship-free" in partial["edgar"]


def test_every_unverified_paid_claim_is_marked_as_such():
    for p in PAID:
        assert p.source, p.vendor
        if p.verified:
            assert "verified 2026-09-10" in p.source, p.vendor
        else:
            assert "NOT verified" in p.source, p.vendor
            assert "SECONDHAND" in "\n".join(p.lines()), p.vendor


# ------------------------------------------------------------------ the answers that exist
def test_crypto_ohlcv_with_no_key_resolves_to_exactly_one_adapter():
    rec = recommend(Need(asset_class="crypto", market="CRYPTO", frequency="1d",
                         key_ok=False))
    assert rec.names() == ["ccxt"]
    best = rec.best
    assert "no key and no account" in best.cost[0]
    assert "per exchange instance" in best.cost[1]
    assert any("UPPER BOUND" in c for c in best.cannot)
    assert any("redistribution=prohibited" in c for c in best.licence)
    # ccxt is the only crypto adapter, so everything else must be refused on coverage
    assert {r.flag for r in rec.refused} <= {"asset_class", "method", "market"}


def test_ten_years_of_us_dailies_on_a_free_tier_ranks_tiingo_first():
    """Tiingo wins on the axis that matters most among survivors: its default is
    RAW_PLUS_FACTORS, so a cached copy stays valid, while yfinance's is anchored at the
    present and is rewritten by every new dividend."""
    rec = recommend(Need(market="US", frequency="1d", history_years=10))
    assert rec.names() == ["tiingo", "yfinance"]
    frame = rec.to_frame()
    assert list(frame["adapter"]) == ["tiingo", "yfinance"]
    rewrites = frame.set_index("adapter")["rewrites_history"]
    assert not bool(rewrites["tiingo"]) and bool(rewrites["yfinance"])


def test_a_source_that_does_not_answer_a_script_is_refused_not_ranked():
    """stooq needs no key at all, which would otherwise make it the free default. It was
    checked on 2026-09-10 and answers a proof-of-work page with HTTP 200 instead of the
    data, so it is refused with that reason rather than recommended."""
    rec = recommend(Need(market="US", frequency="1d"))
    assert "stooq" not in rec.names()
    stooq = next(r for r in rec.refused if r.adapter == "stooq")
    assert stooq.flag == "unreachable"
    assert "proof-of-work" in stooq.reason and "2026-09-10" in stooq.reason
    assert not COVERAGE["stooq"].reachable


def test_alphavantage_is_refused_on_history_because_a_free_key_gets_100_bars():
    rec = recommend(Need(market="US", frequency="1d", history_years=10))
    av = next(r for r in rec.refused if r.adapter == "alphavantage")
    assert av.flag == "history_years"
    assert "100 data points" in av.reason
    # ...and it is NOT refused for a window that fits inside those 100 points
    short = recommend(Need(market="US", frequency="1d", history_years=0.25))
    assert "alphavantage" in short.names()


# ------------------------------------------------------------------- the other hard flags
def test_point_in_time_leaves_only_the_vintage_carrying_sources():
    macro = recommend(Need(asset_class="macro", market="US", method="macro",
                           frequency="1d", point_in_time=True))
    assert macro.names() == ["fred"]
    facts = recommend(Need(asset_class="fundamentals", market="US",
                           method="fundamentals", frequency="1d", point_in_time=True))
    assert facts.names() == ["edgar"]
    prices = recommend(Need(market="US", frequency="1d", point_in_time=True))
    assert prices.empty
    assert "vintage-carrying" in prices.empty_reason


def test_redistribution_empties_the_list_and_says_the_code_licence_is_not_the_data_licence():
    rec = recommend(Need(market="US", frequency="1d", redistribute=True))
    assert rec.empty
    assert "code licence of a client is not the licence of the data" in rec.empty_reason
    assert any(r.flag == "redistribute" for r in rec.refused)
    assert any(p.vendor.startswith("Tiingo") for p in rec.paid)


def test_storing_the_result_refuses_the_source_whose_terms_forbid_a_cache():
    rec = recommend(Need(market="US", frequency="1d", store=True))
    tiingo = next(r for r in rec.refused if r.adapter == "tiingo")
    assert tiingo.flag == "store" and "no-persist" in tiingo.reason
    assert "yfinance" in rec.names(), "a persist-ok source is still available"
    assert any("will be STORED" in line for line in rec.best.licence)


def test_a_key_you_will_not_register_for_is_a_refusal_not_a_footnote():
    rec = recommend(Need(market="US", frequency="1d", key_ok=False))
    keyed = {r.adapter for r in rec.refused if r.flag == "key_ok"}
    assert "tiingo" in keyed and "alphavantage" in keyed
    assert rec.names() == ["yfinance"], "stooq is keyless but unreachable"


def test_an_interval_the_vendor_does_not_serve_is_a_refusal():
    rec = recommend(Need(market="US", frequency="1m"))
    assert "tiingo" in {r.adapter for r in rec.refused if r.flag == "frequency"}
    assert rec.names() == ["yfinance"], "only yfinance declares 1m support here"


def test_paid_options_stay_quiet_until_they_are_asked_for_or_needed():
    """Naming a paid vendor when a free one already serves the request is advertising.
    They appear when nothing free serves it, or when the caller said money is available -
    and even then only for a need the caller actually stated."""
    free = recommend(Need(market="US", frequency="1d", history_years=10))
    assert free.ranked and free.paid == (), "a free source serves it; say nothing"

    asked = recommend(Need(market="US", frequency="1d", history_years=10, paid_ok=True))
    assert [p.vendor for p in asked.paid] == ["Alpha Vantage premium"]
    assert "$49.99/month" in asked.paid[0].price

    quiet = recommend(Need(market="US", frequency="1d", history_years=1, paid_ok=True))
    assert quiet.paid == (), "paid_ok alone invents no need"


# ---------------------------------------------------------------- the table cannot drift
def test_every_registered_adapter_has_a_coverage_row_with_a_source():
    names = {d.name for d in declare.declarations()}
    assert set(COVERAGE) == names, "an adapter with no COVERAGE row is invisible here"
    for cov in COVERAGE.values():
        assert cov.source and len(cov.source) > 20, cov.adapter
        assert set(cov.asset_classes) <= set(ASSET_CLASSES), cov.adapter
        assert set(cov.markets) <= set(MARKETS), cov.adapter
        assert set(cov.serves) <= set(METHODS), cov.adapter
        assert cov.serves, cov.adapter


def test_an_adapter_with_no_coverage_row_fails_loudly_rather_than_vanishing():
    saved = COVERAGE.pop("yfinance")
    try:
        with pytest.raises(KeyError, match="no advise.COVERAGE row"):
            recommend(Need(market="US"))
    finally:
        COVERAGE["yfinance"] = saved


def test_a_method_an_adapter_does_not_serve_actually_refuses_when_called():
    """COVERAGE.serves is hand-declared, so it is checked against behaviour: every method
    an adapter does NOT claim raises rather than returning something. The sockets are
    blocked in this suite, so a method that DID reach the network would fail here too -
    which is why only the not-served half can be asserted."""
    import os

    os.environ.setdefault("EDGAR_IDENTITY", "fin-skills test test@example.com")
    args = {"bars": (["X"], "2024-01-01", "2024-02-01"), "fundamentals": (["X"],),
            "macro": (["X"],), "universe": ("US", "2024-01-01"),
            "actions": (["X"], "2024-01-01", "2024-02-01")}
    from fin_skills.data.adapters import get

    for name, cov in sorted(COVERAGE.items()):
        adapter = get(name)
        for method in METHODS:
            if method in cov.serves:
                continue
            with pytest.raises(NotImplementedError):
                getattr(adapter, method)(*args[method])


def test_the_coverage_table_prints_and_carries_its_sources():
    frame = coverage_frame()
    assert len(frame) == len(COVERAGE)
    assert set(frame.columns) >= {"adapter", "asset_classes", "markets", "serves",
                                  "max_history_years", "reachable", "source"}
    assert (frame["max_history_years"] == "unpublished").any(), \
        "unpublished is not a synonym for unlimited, and is recorded as itself"


# --------------------------------------------------------------------------- the CLI
def test_the_advise_command_exits_3_when_nothing_can_serve_the_request(capsys):
    from fin_skills.data.__main__ import main

    assert main(["advise", "--market", "US", "--delisted"]) == 3
    out = capsys.readouterr().out
    assert out.isascii(), "the CLI must survive a stock Windows console"
    assert "NO ADAPTER IN THIS LIBRARY CAN SERVE THIS REQUEST." in out
    assert "EODHD" in out and "Norgate Data" in out
    assert max(len(line) for line in out.splitlines()) <= 100, "it wraps for a terminal"

    assert main(["advise", "--asset-class", "crypto", "--market", "CRYPTO"]) == 0
    crypto = capsys.readouterr().out
    assert "ccxt" in crypto and "1. ccxt" in crypto


def test_the_advise_command_prints_a_grid_and_a_coverage_table(capsys):
    from fin_skills.data.__main__ import main

    assert main(["advise", "--market", "US", "--table"]) == 0
    grid = capsys.readouterr().out
    assert "rewrites_history" in grid and "tiingo" in grid

    assert main(["advise", "--coverage"]) == 0
    cov = capsys.readouterr().out
    assert cov.isascii() and "alphavantage" in cov and "unpublished" in cov


def test_a_need_rejects_a_vocabulary_it_cannot_filter_on():
    with pytest.raises(ValueError, match="asset_class"):
        Need(asset_class="collectibles")
    with pytest.raises(ValueError, match="market"):
        Need(market="MARS")
    with pytest.raises(ValueError, match="method"):
        Need(method="everything")
    with pytest.raises(ValueError, match="negative"):
        Need(history_years=-1)
    assert Need(market="us").market == "US", "the market is normalised, not rejected"
    with pytest.raises(TypeError, match="not both"):
        recommend(Need(), market="US")
