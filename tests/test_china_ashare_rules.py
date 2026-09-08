"""fin_skills.china.ashare_rules - limits are date-keyed, settlement is T+1, costs are one-sided.

Chinese stock names appear only inside test bodies (never in parametrize ids), so test
output stays ASCII.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from fin_skills.china.ashare_rules import (board_of, can_buy, can_sell, daily_limit_pct,
                                           explain_buy, is_st, limit_price, normalise_code,
                                           round_trip_cost, sellable_qty, session_bar_count,
                                           session_index)


def test_code_normalisation_and_boards():
    for raw in ("sh600000", "600000.SH", " 600000 ", "600000.XSHG"):
        assert normalise_code(raw) == "600000"
    with pytest.raises(ValueError, match="Hong Kong"):
        normalise_code("00700")
    assert board_of("688981") == "STAR" and board_of("301236") == "ChiNext"
    assert board_of("430047") == "BSE" and board_of("873527") == "BSE"
    assert board_of("600519") == "SSE-Main" and board_of("002594") == "SZSE-Main"
    assert board_of("900901") == "B-share"
    with pytest.raises(ValueError, match="classify"):
        board_of("100000")


def test_st_is_a_prefix_match():
    assert is_st("*ST茅台") and is_st("ST x") and is_st("S*ST y") and is_st("SST z")
    assert not is_st("贵州茅台") and not is_st("BEST") and not is_st(None)


def test_chinext_limit_switched_on_2020_08_24():
    assert daily_limit_pct("300059", None, "2020-08-21") == 0.10
    assert daily_limit_pct("300059", None, "2020-08-24") == 0.20
    assert daily_limit_pct("300059", None, date(2021, 6, 1)) == 0.20


def test_limits_by_board_and_st_status():
    assert daily_limit_pct("688981", None, "2024-01-02") == 0.20
    assert daily_limit_pct("430047", None, "2024-01-02") == 0.30
    assert daily_limit_pct("600519", "贵州茅台", "2024-01-02") == 0.10
    assert daily_limit_pct("600519", "*ST茅台", "2024-01-02") == 0.05
    assert daily_limit_pct("300750", "ST宁德", "2021-06-01") == 0.20   # ST halves MAIN board only
    assert daily_limit_pct("300750", None, "2021-06-01", days_since_ipo=3) == math.inf
    assert daily_limit_pct("600519", None, "2024-01-02", days_since_ipo=0) == 0.10
    with pytest.raises(ValueError, match="did not exist"):
        daily_limit_pct("688981", None, "2019-01-02")
    with pytest.raises(ValueError, match="REQUIRED"):
        daily_limit_pct("600519", None)


def test_limit_price_rounds_half_up_like_the_exchange():
    assert limit_price(1.05, 0.10, "up") == 1.16          # 1.155 -> 1.16, not banker's 1.15
    assert limit_price(1.05, 0.10, "down") == 0.95        # 0.945 -> 0.95
    assert round(0.945, 2) != 0.95                        # the bug this guards against
    assert limit_price(200.0, 0.20, "up") == 240.0
    assert limit_price(10.0, math.inf, "up") == math.inf and limit_price(10.0, math.inf, "down") == 0.0
    with pytest.raises(ValueError):
        limit_price(10.0, 0.1, "sideways")
    with pytest.raises(ValueError):
        limit_price(0.0, 0.1)


@pytest.fixture
def bars():
    prev, pct = 200.0, 0.20
    up, down = limit_price(prev, pct, "up"), limit_price(prev, pct, "down")
    return pct, {
        "locked_up": {"prev_close": prev, "open": up, "high": up, "low": up, "close": up, "volume": 1_200_000},
        "locked_down": {"prev_close": prev, "open": down, "high": down, "low": down, "close": down, "volume": 900_000},
        "normal": {"prev_close": prev, "open": 202.0, "high": up, "low": 199.0, "close": 239.9, "volume": 8_400_000},
        "suspended": {"prev_close": prev, "open": prev, "high": prev, "low": prev, "close": prev, "volume": 0},
        "paused": {"prev_close": prev, "open": prev, "high": prev, "low": prev, "close": prev, "volume": 10, "paused": True},
    }


def test_limit_locked_bars_are_asymmetric(bars):
    pct, b = bars
    assert not can_buy(b["locked_up"], pct) and can_sell(b["locked_up"], pct)
    assert can_buy(b["locked_down"], pct) and not can_sell(b["locked_down"], pct)
    assert can_buy(b["normal"], pct) and can_sell(b["normal"], pct)
    assert not can_buy(b["suspended"], pct) and not can_sell(b["suspended"], pct)
    assert not can_buy(b["paused"], pct)
    ok, why = explain_buy(b["locked_up"], pct)
    assert not ok and "LOCKED at the up limit" in why
    assert explain_buy(b["normal"], pct) == (True, "tradable")
    with pytest.raises(KeyError, match="prev_close"):
        can_buy({"high": 1, "low": 1, "volume": 1}, pct)


def test_t_plus_1_settlement():
    lots = [{"date": "2024-03-11", "qty": 2_000}, {"date": "2024-03-12", "qty": 1_000}]
    assert sellable_qty(lots, "2024-03-12") == 2_000
    assert sellable_qty(lots, "2024-03-13") == 3_000
    assert sellable_qty([("2024-03-12", 1_000)], "2024-03-12") == 0
    with pytest.raises(ValueError):
        sellable_qty([{"qty": 100}], "2024-03-12")


def test_stamp_duty_is_sell_side_only_and_halved_in_2023():
    before = round_trip_cost(10.0, 10.5, 10_000, "2023-08-01", "2023-08-25")
    after = round_trip_cost(10.0, 10.5, 10_000, "2023-09-01", "2023-09-05")
    assert before["buy_stamp"] == 0.0 == after["buy_stamp"]
    assert before["stamp_rate_used"] == 0.0010 and after["stamp_rate_used"] == 0.0005
    assert before["sell_stamp"] == pytest.approx(105_000 * 0.0010)
    assert after["sell_stamp"] == pytest.approx(105_000 * 0.0005)
    assert before["round_trip"] > after["round_trip"]
    assert after["net_pnl"] == pytest.approx(after["gross_pnl"] - after["round_trip"])
    assert after["buy_transfer"] == pytest.approx(100_000 * 0.00001)
    old = round_trip_cost(10.0, 10.5, 10_000, "2022-04-01", "2022-04-28")
    assert old["buy_transfer"] == pytest.approx(100_000 * 0.00002)


def test_minimum_commission_and_lot_rules():
    small = round_trip_cost(10.0, 10.5, 100, "2023-09-01", "2023-09-05")
    assert small["buy_commission"] == 5.0 and small["sell_commission"] == 5.0
    with pytest.raises(ValueError, match="round lots"):
        round_trip_cost(10.0, 10.5, 150, "2023-09-01")
    with pytest.raises(ValueError, match="precedes"):
        round_trip_cost(10.0, 10.5, 100, "2023-09-05", "2023-09-01")


def test_session_has_240_one_minute_bars_and_no_bar_spans_lunch():
    assert session_bar_count(1) == 240 and session_bar_count(5) == 48
    assert session_bar_count(30) == 8 and session_bar_count(60) == 4
    with pytest.raises(ValueError, match="lunch"):
        session_bar_count(45)
    with pytest.raises(ValueError):
        session_bar_count(0)
    idx = session_index("2024-03-12", 30)
    assert [t.strftime("%H:%M") for t in idx] == ["10:00", "10:30", "11:00", "11:30",
                                                   "13:30", "14:00", "14:30", "15:00"]
    assert len(session_index("2024-03-12", 1)) == 240
