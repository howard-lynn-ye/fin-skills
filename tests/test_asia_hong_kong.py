"""Hong Kong: NAV conservation, board-lot arithmetic, dated rules and quota state."""
from __future__ import annotations

import math
from datetime import date, time

import numpy as np
import pytest

from fin_skills.asia.hong_kong import (AGGREGATE_QUOTA_ABOLISHED_ON, CAS_SUBPERIODS,
                                       DAILY_QUOTA_QUADRUPLED_ON,
                                       HKEX_BOARD_LOT_FREQUENCIES, HKEX_BOARD_LOT_VALUES,
                                       HKEX_DISTINCT_LOTS, HKEX_EQUITY_COUNT,
                                       LUNCH_SHORTENED_ON, NORTHBOUND_DAILY_QUOTA_RMB,
                                       OPEN_MOVED_TO_0930_ON, POS_SUBPERIODS,
                                       SEVERE_WEATHER_TRADING_FROM,
                                       SOUTHBOUND_DAILY_QUOTA_RMB, VCM_BAND_BY_TIER,
                                       VCM_COOLING_OFF_MINUTES, adr_implied_local,
                                       ah_premium, capital_for_weight_error,
                                       connect_buy_allowed, continuous_bar_count,
                                       daily_quota, equal_weight_with_lots, is_odd_lot,
                                       min_ticket, naive_equal_weight, quota_balance,
                                       round_to_lot, session_minutes, settlement_date,
                                       stamp_duty_rate, synthetic_hk_universe,
                                       weather_arrangement)


@pytest.fixture(scope="module")
def universe():
    return synthetic_hk_universe(60, seed=20260910)


# --------------------------------------------------------------------------- board lots
def test_scenario_lot_weights_are_well_formed():
    # Fixed recovered inputs are not a current empirical exchange-coverage assertion.
    assert len(HKEX_BOARD_LOT_VALUES) == len(set(HKEX_BOARD_LOT_VALUES))
    assert all(lot > 0 for lot in HKEX_BOARD_LOT_VALUES)
    assert all(lot in HKEX_BOARD_LOT_VALUES and count > 0
               for lot, count in HKEX_BOARD_LOT_FREQUENCIES)


def test_a_quantity_that_is_not_a_whole_lot_does_not_reach_the_main_book():
    assert is_odd_lot(137, 500) and round_to_lot(137, 500) == 0
    assert not is_odd_lot(500, 500) and round_to_lot(500, 500) == 500
    assert is_odd_lot(1_000, 400) and round_to_lot(1_000, 400) == 800
    assert not is_odd_lot(2_400, 400)
    assert round_to_lot(1_000, 400, "up") == 1_200
    assert round_to_lot(1_100, 400, "nearest") == 1_200
    assert round_to_lot(1_000, 400, "nearest") == 800      # 2.5 lots: numpy rounds to even
    assert round_to_lot(-5, 100) == 0
    for bad in (0, -100):
        with pytest.raises(ValueError):
            round_to_lot(100, bad)
    with pytest.raises(ValueError):
        round_to_lot(100, 100, "sideways")


def test_min_ticket_is_price_times_lot():
    assert min_ticket(64.5, 400) == pytest.approx(25_800.0)
    with pytest.raises(ValueError):
        min_ticket(0.0, 100)
    with pytest.raises(ValueError):
        min_ticket(1.0, 0)


def test_the_scenario_is_deterministic_and_uses_fixed_lot_inputs(universe):
    prices, lots = universe
    again = synthetic_hk_universe(60, seed=20260910)
    np.testing.assert_array_equal(prices, again[0])
    np.testing.assert_array_equal(lots, again[1])
    assert not np.array_equal(lots, synthetic_hk_universe(60, seed=1)[1])
    assert set(lots.tolist()) <= {v for v, _ in HKEX_BOARD_LOT_FREQUENCIES}
    # the coupling is negative: expensive names carry smaller lots
    rho = np.corrcoef(np.argsort(np.argsort(prices)),
                      np.argsort(np.argsort(lots)))[0, 1]
    assert rho < -0.5
    with pytest.raises(ValueError):
        synthetic_hk_universe(10, coupling=1.5)


# --------------------------------------------------------------------------- the trap
def test_lot_rounding_uses_total_account_nav(universe):
    prices, lots = universe
    r = equal_weight_with_lots(1e6, prices, lots)
    assert r["n_unfillable"] == 32
    assert r["max_abs_weight_error_bps"] == pytest.approx(166.6667, abs=.001)
    assert r["cash_drag_pct"] == pytest.approx(.6396, abs=5e-4)
    np.testing.assert_allclose(r["realised_weight"], r["position_value"] / 1e6)
    assert r["realised_weight"].sum() + r["cash_drag_pct"] == pytest.approx(1.)
    assert np.all(r["weight_error"] <= 1e-10)
    assert r["deployed_weight"].sum() == pytest.approx(1.)
    absent = r["unfillable"]
    assert np.all(r["realised_weight"][absent] == 0)
    assert np.allclose(r["weight_error"][absent], -1/60)


def test_a_cash_only_account_has_zero_weights():
    r = equal_weight_with_lots(100., [100., 200.], [100, 100])
    assert r["n_unfillable"] == 2
    assert r["cash_drag"] == 100.
    np.testing.assert_array_equal(r["realised_weight"], [0., 0.])
    np.testing.assert_array_equal(r["deployed_weight"], [0., 0.])


def test_rounding_error_need_not_shrink_monotonically():
    # At 200 the sole target fits two lots exactly; at 250 it leaves 50 uninvested.
    perfect = equal_weight_with_lots(200., [10.], [10])
    residual = equal_weight_with_lots(250., [10.], [10])
    assert perfect["max_abs_weight_error_bps"] == 0.
    assert residual["max_abs_weight_error_bps"] == pytest.approx(2000.)


def test_capital_bound_is_sufficient_even_across_rounding_jumps(universe):
    prices, lots = universe
    cap = capital_for_weight_error(prices, lots, 100.)
    assert cap == pytest.approx(5_792_000.)
    for multiple in (1., 1.001, 1.2, 3.):
        r = equal_weight_with_lots(cap * multiple, prices, lots)
        assert r["max_abs_weight_error_bps"] <= 100. + 1e-9
    assert math.isinf(capital_for_weight_error(prices, lots, 100., lo=1., hi=1.))


def test_the_fractional_engine_reports_a_clean_equal_weight_that_cannot_be_traded(universe):
    prices, lots = universe
    naive = naive_equal_weight(1e6, prices)
    assert np.allclose(naive["realised_weight"], 1.0 / 60)
    assert naive["deployed"] == pytest.approx(1e6)
    # Quantities outside board lots need a separate odd-lot execution model.
    illegal = [i for i in range(60) if is_odd_lot(naive["shares"][i], int(lots[i]))]
    assert len(illegal) >= 55


def test_equal_weight_validates_its_inputs(universe):
    prices, lots = universe
    with pytest.raises(ValueError):
        equal_weight_with_lots(0.0, prices, lots)
    with pytest.raises(ValueError):
        equal_weight_with_lots(1e6, prices[:5], lots)


# --------------------------------------------------------------------------- sessions
def test_the_extended_morning_session_is_the_lunch_break():
    assert session_minutes("2026-09-10") == (150, 180)
    assert continuous_bar_count(1, "2026-09-10") == 330
    for f in (2, 3, 5, 6, 10, 15, 30):
        assert continuous_bar_count(f, "2026-09-10") == 330 // f
    with pytest.raises(ValueError, match="12:00-13:00 break"):
        continuous_bar_count(60, "2026-09-10")
    with pytest.raises(ValueError, match="2012-03-05"):
        session_minutes("2011-06-01")
    assert OPEN_MOVED_TO_0930_ON == date(2011, 3, 7)
    assert LUNCH_SHORTENED_ON == date(2012, 3, 5)


def test_the_auction_subperiods_tile_their_windows():
    assert POS_SUBPERIODS[0][1] == time(9, 0) and POS_SUBPERIODS[-1][2] == time(9, 30)
    assert CAS_SUBPERIODS[0][1] == time(16, 0) and CAS_SUBPERIODS[-1][2] == time(16, 10)
    for subs in (POS_SUBPERIODS, CAS_SUBPERIODS):
        for (_, _, end), (_, start, _) in zip(subs, subs[1:]):
            assert end == start                      # no gap, no overlap
    assert any("5%" in label for label, _, _ in CAS_SUBPERIODS)


def test_settlement_is_t2_in_business_days():
    assert settlement_date("2026-09-08") == date(2026, 9, 10)
    assert settlement_date("2026-09-10") == date(2026, 9, 14)      # Thu -> Mon
    assert settlement_date("2026-09-10", holidays=["2026-09-14"]) == date(2026, 9, 15)


# --------------------------------------------------------------------------- stamp duty
def test_stamp_duty_has_three_regimes_and_both_sides_pay():
    assert stamp_duty_rate("2021-07-31") == 0.001
    assert stamp_duty_rate("2021-08-01") == 0.0013
    assert stamp_duty_rate("2023-11-16") == 0.0013
    assert stamp_duty_rate("2023-11-17") == 0.001
    assert stamp_duty_rate("2026-09-10") == 0.001
    assert 2 * stamp_duty_rate("2026-09-10") == pytest.approx(0.0020)
    assert 2 * stamp_duty_rate("2022-06-01") == pytest.approx(0.0026)


# --------------------------------------------------------------------------- weather
def test_the_weather_rule_reverses_on_2024_09_23():
    assert SEVERE_WEATHER_TRADING_FROM == date(2024, 9, 23)
    old = weather_arrangement("2023-09-08", "Typhoon Signal No. 8")
    new = weather_arrangement("2024-09-23", "Typhoon Signal No. 8")
    assert "SUSPENDED" in old and "CONTINUES" in new
    assert weather_arrangement("2026-07-20", "Black Rainstorm").startswith("trading CONT")
    assert weather_arrangement("2026-07-20", "Amber Rainstorm") == "normal trading"
    # three regimes, not two: the afternoon resumption moved in 2012
    assert "13:30" in weather_arrangement("2011-09-01", "Typhoon Signal No. 8")
    assert "13:00" in weather_arrangement("2013-09-01", "Typhoon Signal No. 8")


# --------------------------------------------------------------------------- Connect
def test_the_connect_quota_is_charged_net_and_sells_restore_it():
    assert daily_quota("northbound", "2026-09-10") == NORTHBOUND_DAILY_QUOTA_RMB == 52e9
    assert daily_quota("southbound", "2026-09-10") == SOUTHBOUND_DAILY_QUOTA_RMB == 42e9
    assert daily_quota("northbound", "2017-06-01") == 13e9
    assert daily_quota("southbound", "2017-06-01") == 10.5e9
    assert DAILY_QUOTA_QUADRUPLED_ON == date(2018, 5, 1)
    assert AGGREGATE_QUOTA_ABOLISHED_ON == date(2016, 8, 16)
    with pytest.raises(ValueError):
        daily_quota("eastbound", "2026-09-10")

    bal = quota_balance("northbound", "2026-09-10", 40e9, 35e9)
    assert bal == pytest.approx(47e9)                 # 52 - 40 + 35
    assert connect_buy_allowed("northbound", "2026-09-10", bal)
    # gross accounting would have refused a trade the exchange accepts
    gross = 52e9 - (40e9 + 35e9)
    assert gross < 0 and not connect_buy_allowed("northbound", "2026-09-10", gross)


def test_exhausting_the_quota_closes_buying_only():
    bal = quota_balance("northbound", "2026-09-10", 52e9, 0.0)
    assert bal == pytest.approx(0.0)
    assert not connect_buy_allowed("northbound", "2026-09-10", bal)
    # Opening-auction balance can recover before any continuous-session latch.
    assert connect_buy_allowed("northbound", "2026-09-10",
                               quota_balance("northbound", "2026-09-10", 52e9, 1e9))


# --------------------------------------------------------------------------- A/H and ADR
def test_the_ah_premium_is_a_same_currency_comparison():
    assert ah_premium(40.0, 42.5, 1.08) == pytest.approx(0.1475, abs=1e-4)
    assert ah_premium(40.0, 40.0 / 1.08, 1.08) == pytest.approx(0.0, abs=1e-12)
    for bad in ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)):
        with pytest.raises(ValueError):
            ah_premium(*bad)


def test_the_adr_ratio_scales_the_level_and_leaves_every_return_correct():
    a = adr_implied_local(21.40, 1.0, 7.80)
    b = adr_implied_local(21.40, 5.0, 7.80)
    assert a == pytest.approx(166.92) and b == pytest.approx(33.384)
    assert a / b == pytest.approx(5.0)
    # the whole point: returns are unchanged by the ratio, so no return check finds it
    r1 = adr_implied_local(22.0, 1.0, 7.8) / adr_implied_local(21.4, 1.0, 7.8)
    r5 = adr_implied_local(22.0, 5.0, 7.8) / adr_implied_local(21.4, 5.0, 7.8)
    assert r1 == pytest.approx(r5)
    with pytest.raises(ValueError):
        adr_implied_local(21.4, 0.0, 7.8)


def test_the_vcm_tiers_widen_with_falling_liquidity():
    bands = list(VCM_BAND_BY_TIER.values())
    assert bands[:3] == [0.10, 0.15, 0.20]
    assert VCM_BAND_BY_TIER["HSCI LargeCap"] < VCM_BAND_BY_TIER["HSCI SmallCap"]
    assert VCM_COOLING_OFF_MINUTES == 5


# --------------------------------------------------------------------------- the demo
def test_run_main_prints_the_rule_and_the_headline_numbers(run_main):
    out = run_main("fin_skills.asia.hong_kong")
    assert "RULE: Hong Kong has no universal round lot" in out
    assert "32/60" in out or "32 names" in out
    assert "63.96%" in out
    assert "47.0 bn RMB left" in out
    assert "exhaustion permits new buy: False" in out
    assert out.isascii(), "skill scripts print ASCII only"


def test_continuous_session_quota_exhaustion_stays_latched():
    restored = quota_balance("northbound", "2026-09-14", 52e9, 1e9, 2e9)
    assert restored == 3e9
    assert connect_buy_allowed("northbound", "2026-09-14", restored)
    assert not connect_buy_allowed("northbound", "2026-09-14", restored,
                                   exhausted_in_continuous=True)


@pytest.mark.parametrize("prices,lots", [([float("nan")],[10]), ([1.],[1.5]),
                                         ([1.],[0]), ([],[])])
def test_nonfinite_prices_or_noninteger_lots_are_rejected(prices, lots):
    with pytest.raises(ValueError):
        equal_weight_with_lots(1e6, prices, lots)
