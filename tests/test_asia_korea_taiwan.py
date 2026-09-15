"""Dated eligibility and explicit liquidity evidence must survive an OHLC backtest."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fin_skills.asia.korea_taiwan import (
    daily_limit_pct, fill_rate, reachable, settlement_date, short_selling_allowed,
    synthetic_limited_series, twse_tick,
)


@pytest.mark.parametrize("market,before,after,old,new", [
    ("KRX", "2015-06-14", "2015-06-15", .15, .30),
    ("TWSE", "2015-05-31", "2015-06-01", .07, .10),
])
def test_limit_transition_is_dated(market, before, after, old, new):
    assert daily_limit_pct(market, before) == old
    assert daily_limit_pct(market, after) == new
    with pytest.raises(ValueError):
        daily_limit_pct(market, after, ordinary_share=False)


@pytest.mark.parametrize("price,tick", [(9.99,.01),(10.,.05),(50.,.1),(100.,.5),
                                        (500.,1.),(1000.,5.)])
def test_twse_tick_boundaries(price, tick):
    assert twse_tick(price) == tick


def test_short_sale_partial_reopening_uses_historical_membership():
    assert not short_selling_allowed("2021-05-02", index_member=True)[0]
    assert short_selling_allowed("2021-05-03", index_member=True)[0]
    assert not short_selling_allowed("2021-05-03", index_member=False)[0]
    assert not short_selling_allowed("2023-11-06", index_member=True)[0]
    assert not short_selling_allowed("2025-03-30", index_member=True)[0]
    assert short_selling_allowed("2025-03-31")[0]


def test_old_financial_stock_exception_and_broad_bans_are_separate():
    assert short_selling_allowed("2010-01-04")[0]
    assert not short_selling_allowed("2010-01-04", financial_stock=True)[0]
    assert not short_selling_allowed("2013-11-13", financial_stock=True)[0]
    assert short_selling_allowed("2013-11-14", financial_stock=True)[0]
    with pytest.raises(ValueError):
        short_selling_allowed("2001-01-01")


def test_an_at_limit_bar_does_not_establish_absent_liquidity():
    bar = dict(price=110., lower_limit=90., upper_limit=110., volume=500.)
    assert reachable(bar, "buy")[0] is None
    assert reachable(dict(bar, volume=0.), "buy")[0] is None
    assert reachable(dict(bar, ask_qty=0.), "buy")[0] is False
    assert reachable(dict(bar, ask_qty=10.), "buy", quantity=10.)[0] is True
    assert reachable(dict(bar, ask_qty=10.), "buy", quantity=11.)[0] is False
    # Buy capacity says nothing about sell capacity.
    assert reachable(dict(bar, ask_qty=10.), "sell")[0] is None


def test_sell_capacity_is_independent_and_halts_or_out_of_band_prices_block():
    bar = dict(price=90., lower_limit=90., upper_limit=110., volume=100., bid_qty=3.)
    assert reachable(bar, "sell", 3.)[0] is True
    assert reachable(dict(bar, suspended=True), "sell")[0] is False
    assert reachable(dict(bar, price=89.), "sell")[0] is False
    with pytest.raises(ValueError):
        reachable(dict(bar, bid_qty=-1.), "sell")


def test_synthetic_series_carries_same_seed_and_obeys_declared_bands():
    first = synthetic_limited_series()
    pd.testing.assert_frame_equal(first, synthetic_limited_series())
    assert not first.equals(synthetic_limited_series(seed=1))
    assert (first.price >= first.lower_limit).all()
    assert (first.price <= first.upper_limit).all()
    assert first.locked.any()
    for record in first.to_dict("records"):
        if record["ask_qty"] == 0:
            assert record["price"] == record["upper_limit"]
        if record["bid_qty"] == 0:
            assert record["price"] == record["lower_limit"]


def test_demo_fill_counts_are_synthetic_and_unknown_capacity_never_fills():
    krx = fill_rate(synthetic_limited_series(limit_pct=.30))
    twse = fill_rate(synthetic_limited_series(limit_pct=.10))
    assert (krx["signals"],krx["filled"],krx["rejected"]) == (213,210,3)
    assert (twse["signals"],twse["filled"],twse["rejected"]) == (229,185,44)
    unknown = synthetic_limited_series().drop(columns=["ask_qty","bid_qty"])
    assert fill_rate(unknown)["filled"] == 0


def test_settlement_accepts_holidays_and_skips_weekends():
    assert settlement_date("2026-09-10") == date(2026,9,14)
    assert settlement_date("2026-09-10", ["2026-09-14"]) == date(2026,9,15)


def test_demo_is_ascii_and_labels_the_liquidity_assumption(run_main):
    out = run_main("fin_skills.asia.korea_taiwan")
    assert out.isascii()
    assert "not measured exchange liquidity" in out
    assert "Limit bar without depth: (None" in out
    assert "Same bar with offers: (True" in out
