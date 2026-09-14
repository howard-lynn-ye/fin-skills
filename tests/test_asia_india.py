"""fin_skills.asia.india - a fill outside the price band is not a bad fill, it is no fill.

Each test asserts a documented PROPERTY: a band-crossing fill is rejected, the strategy's
breach rate is a large multiple of the unconditional rate, the circuit-breaker rows use
different afternoon cutoffs, the dynamic band slides rather than widens, STT moved twice,
and the demo prints the rule.
"""
from __future__ import annotations

import math
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from fin_skills.asia.india import (CASH_BOARD_LOT, CAS_BAND, CAS_FROM,
                                   CIRCUIT_BREAKER_HALTS, FNO_CONTRACT_VALUE_RAISED_ON,
                                   LAKH, POST_HALT_PRE_OPEN_MINUTES, PRE_OPEN_SUBPERIODS,
                                   PRE_OPEN_RESTRUCTURED_ON, SLIDING_PRICE_BAND_FROM,
                                   STT_RAISED_AGAIN_ON, STT_RAISED_ON, STT_RATES,
                                   T0_BETA_FROM, T1_ROLLOUT_COMPLETE, T1_ROLLOUT_START,
                                   band_fill_audit, band_for, band_prices, clip_to_band,
                                   dynamic_band_after_flexing, fno_lot_size,
                                   halted_minutes_in_day, index_circuit_breaker,
                                   is_within_band, min_contract_value, phantom_fill_audit,
                                   round_trip_stt_bps, settlement_date, settlement_lag,
                                   stt_cost, synthetic_daily)


@pytest.fixture(scope="module")
def series():
    return synthetic_daily(2_500, seed=20260910)


# --------------------------------------------------------------------------- price bands
def test_a_band_crossing_fill_is_rejected_not_repriced():
    lo, hi = band_prices(100.0, 0.10)
    assert (lo, hi) == (90.0, 110.0)
    assert is_within_band(110.0, 100.0, 0.10)
    assert not is_within_band(110.05, 100.0, 0.10)
    assert not is_within_band(89.95, 100.0, 0.10)
    # the reachable price is the band edge; the rest of the gap did not happen
    assert clip_to_band(118.0, 100.0, 0.10) == 110.0
    assert clip_to_band(72.0, 100.0, 0.10) == 90.0
    assert clip_to_band(105.0, 100.0, 0.10) == 105.0


def test_band_prices_round_to_the_five_paise_tick_half_up():
    lo, hi = band_prices(49.5, 0.05)
    # 49.5 * 1.05 = 51.975 -> 51.975/0.05 = 1039.5 -> HALF UP -> 1040 -> 52.00
    assert hi == pytest.approx(52.00)
    # The LOWER edge is the discriminating case: 47.025/0.05 = 940.5, and banker's
    # rounding sends that tie DOWN to the even 940 (= 47.00) while the exchange's
    # HALF UP gives 941 (= 47.05). One tick, on exactly the bars a band test looks at.
    assert round(940.5) == 940
    assert lo == pytest.approx(47.05) and 940 * 0.05 == pytest.approx(47.00)
    # every band edge lands exactly on a 5-paise tick
    for p in (1.0, 12.35, 49.5, 987.65):
        for edge in band_prices(p, 0.10):
            assert abs(edge / 0.05 - round(edge / 0.05)) < 1e-9


def test_band_prices_validate_their_inputs():
    with pytest.raises(ValueError):
        band_prices(0.0, 0.10)
    with pytest.raises(ValueError):
        band_prices(100.0, 1.5)
    with pytest.raises(ValueError):
        band_prices(100.0, 0.0)
    assert band_prices(100.0, math.inf) == (0.0, math.inf)


def test_a_cash_scrip_has_no_default_band():
    assert band_for("fno") == math.inf
    assert band_for("derivatives") == math.inf
    assert band_for("cash", 0.05) == 0.05
    with pytest.raises(ValueError, match="no default"):
        band_for("cash")


# --------------------------------------------------------------------------- the trap
def test_the_generator_gives_independent_overnight_and_intraday_moves(series):
    """The signal has no edge by construction, so any P&L is the fill model talking."""
    gap = np.log(series["open"] / series["prev_close"])
    intraday = np.log(series["close"] / series["open"])
    assert abs(np.corrcoef(gap, intraday)[0, 1]) < 0.06
    pd.testing.assert_frame_equal(series, synthetic_daily(2_500, seed=20260910))
    assert not synthetic_daily(2_500, seed=11)["close"].equals(series["close"])


def test_the_unconditional_breach_rate_falls_with_a_wider_band(series):
    rates = [band_fill_audit(series, b)["breach_rate"] for b in (0.02, 0.05, 0.10, 0.20)]
    assert rates == sorted(rates, reverse=True)
    a5 = band_fill_audit(series, 0.05)
    assert a5["n_outside"] == 67
    assert a5["breach_rate"] == pytest.approx(0.0268, abs=5e-4)
    assert a5["max_overshoot_bps"] == pytest.approx(1930, rel=0.02)


def test_the_band_binds_on_the_strategy_far_more_often_than_on_the_sample(series):
    """SKILL.md section 1: 2.68% of sessions, but 69% of the trades - 26x."""
    a5 = band_fill_audit(series, 0.05)
    s5 = phantom_fill_audit(series, 0.05)
    assert s5["n_signals"] == 48
    assert s5["n_outside"] == 33
    assert s5["outside_share"] == pytest.approx(0.6875, abs=0.005)
    assert s5["outside_share"] / a5["breach_rate"] == pytest.approx(26, rel=0.1)
    assert s5["mean_phantom_bps"] == pytest.approx(502, rel=0.02)
    assert s5["max_phantom_bps"] == pytest.approx(1930, rel=0.02)
    assert s5["total_phantom_pct"] == pytest.approx(165.6, rel=0.02)


def test_a_wider_band_kills_fewer_of_the_strategys_trades(series):
    shares = [phantom_fill_audit(series, b)["outside_share"]
              for b in (0.02, 0.05, 0.10, 0.20)]
    assert shares == sorted(shares, reverse=True)
    assert shares[0] == 1.0                       # a 2% band forbids EVERY -3% gap
    assert phantom_fill_audit(series, 0.20)["n_outside"] == 1
    # a 20% band leaves the strategy essentially untouched
    assert phantom_fill_audit(series, 0.20)["phantom_bps_per_signal"] < 20


# ------------------------------------------------------------------ circuit breakers
def test_the_ten_and_fifteen_percent_rows_use_different_afternoon_cutoffs():
    """14:15 is the discriminating time: a 15-min halt vs closed for the day."""
    assert index_circuit_breaker(-0.10, "14:15") == ("10%", 15, True)
    assert index_circuit_breaker(-0.155, "14:15") == ("15%", None, False)
    # ... and before 14:00 the 15% row is only a 45-minute halt
    assert index_circuit_breaker(-0.155, "13:30") == ("15%", 45, True)
    # ... while the 10% row is still a 15-minute halt right up to 14:30
    assert index_circuit_breaker(-0.10, "14:29") == ("10%", 15, True)
    assert index_circuit_breaker(-0.10, "14:30") == ("10%", 0, False)


def test_the_breaker_is_symmetric_and_takes_the_deepest_trigger():
    for t in ("11:00", "13:30", "15:00"):
        assert index_circuit_breaker(0.21, t) == index_circuit_breaker(-0.21, t)
    assert index_circuit_breaker(-0.21, "11:00")[0] == "20%"
    assert index_circuit_breaker(-0.155, "11:00")[0] == "15%"
    assert index_circuit_breaker(-0.099, "11:00") == (None, 0, False)
    assert index_circuit_breaker(-0.10, "11:00") == ("10%", 45, True)
    assert set(CIRCUIT_BREAKER_HALTS) == {0.10, 0.15, 0.20}
    assert CIRCUIT_BREAKER_HALTS[0.20] == ()      # any time: rest of the day


def test_the_halt_costs_the_documented_minutes():
    assert halted_minutes_in_day(-0.10, "11:00") == 45 + POST_HALT_PRE_OPEN_MINUTES == 60
    assert halted_minutes_in_day(-0.10, "15:00") == 0
    assert halted_minutes_in_day(-0.09, "11:00") == 0
    assert halted_minutes_in_day(-0.21, "15:00") == 30       # 15:00 to the 15:30 close
    assert halted_minutes_in_day(-0.155, "11:00") == 105 + POST_HALT_PRE_OPEN_MINUTES
    assert halted_minutes_in_day(-0.21, "15:30") == 0
    assert index_circuit_breaker(-0.21, time(11, 0))[0] == "20%"


def test_the_dynamic_band_slides_and_does_not_widen():
    widths = []
    for k in range(4):
        b = dynamic_band_after_flexing(0.10, k)
        widths.append(b["band_width_pct"])
        assert b["upper_edge_pct"] == pytest.approx(0.10 + 0.05 * k)
        assert b["lower_edge_pct"] == pytest.approx(-0.10 + 0.05 * k)
        assert b["minutes_required"] == 15.0 * k
    assert widths == [0.20] * 4                     # the WIDTH never changes
    # two flexes and the lower edge is at the previous close: a resting bid below is gone
    assert dynamic_band_after_flexing(0.10, 2)["lower_edge_pct"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        dynamic_band_after_flexing(0.10, -1)
    assert SLIDING_PRICE_BAND_FROM == date(2024, 11, 18)


# --------------------------------------------------------------------------- settlement
def test_the_t1_rollout_window_has_no_market_wide_answer():
    assert T1_ROLLOUT_START == date(2022, 2, 25)
    assert T1_ROLLOUT_COMPLETE == date(2023, 1, 27)
    assert settlement_lag("2021-06-01") == 2
    assert settlement_lag("2023-02-01") == 1
    assert settlement_lag("2026-09-10") == 1
    with pytest.raises(ValueError, match="rollout"):
        settlement_lag("2022-06-15")
    # ... unless you say which tranche the scrip was in
    assert settlement_lag("2022-06-15", "2022-02-25") == 1
    assert settlement_lag("2022-06-15", "2023-01-27") == 2


def test_settlement_date_counts_business_days():
    assert settlement_date("2026-09-10") == date(2026, 9, 11)
    assert settlement_date("2026-09-11") == date(2026, 9, 14)          # Fri -> Mon
    assert settlement_date("2021-06-01") == date(2021, 6, 3)           # T+2 era
    assert settlement_date("2026-09-10", holidays=["2026-09-11"]) == date(2026, 9, 14)
    assert T0_BETA_FROM == date(2024, 3, 28)


# --------------------------------------------------------------------------- STT
def test_stt_rose_twice_and_the_options_base_is_the_premium():
    assert STT_RAISED_ON == date(2024, 10, 1)
    assert STT_RAISED_AGAIN_ON == date(2026, 4, 1)
    n = 1_000_000.0
    assert stt_cost(n, "futures_sell", "2024-09-30") == pytest.approx(125.0)
    assert stt_cost(n, "futures_sell", "2024-10-01") == pytest.approx(200.0)
    assert stt_cost(n, "futures_sell", "2026-03-31") == pytest.approx(200.0)
    assert stt_cost(n, "futures_sell", "2026-04-01") == pytest.approx(500.0)
    r0, r2 = STT_RATES["before_2024_10_01"], STT_RATES["from_2026_04_01"]
    assert r2["futures_sell"] / r0["futures_sell"] == pytest.approx(4.0)
    assert r2["options_sell_premium"] == 0.0015
    assert r2["options_exercise_buy"] == 0.0015
    # the delivery legs did NOT move
    assert r0["delivery_buy"] == r2["delivery_buy"] == 0.001
    assert r0["intraday_sell"] == r2["intraday_sell"] == 0.00025


def test_the_wrong_options_base_is_125x_too_big():
    prem, strike_notional = 12_000.0, 1_500_000.0
    right = stt_cost(prem, "options_sell_premium", "2026-09-10")
    wrong = stt_cost(strike_notional, "options_sell_premium", "2026-09-10")
    assert right == pytest.approx(18.0)
    assert wrong / right == pytest.approx(strike_notional / prem) == pytest.approx(125.0)


def test_stt_validates_its_inputs():
    with pytest.raises(ValueError, match="unknown STT leg"):
        stt_cost(1.0, "wealth_tax", "2026-09-10")
    with pytest.raises(ValueError):
        stt_cost(-1.0, "delivery_buy", "2026-09-10")
    with pytest.raises(ValueError, match="REQUIRED"):
        stt_cost(1.0, "delivery_buy", None)
    assert round_trip_stt_bps(1e6, "delivery_buy", "delivery_sell",
                              "2026-09-10") == pytest.approx(20.0)


# --------------------------------------------------------------------------- F&O size
def test_the_contract_value_tripled_on_2024_11_20():
    assert FNO_CONTRACT_VALUE_RAISED_ON == date(2024, 11, 20)
    assert min_contract_value("2024-11-19") == 5 * LAKH
    assert min_contract_value("2024-11-20") == 15 * LAKH
    for price in (85.0, 640.0, 2_500.0, 11_400.0):
        before = fno_lot_size(price, "2024-11-19")
        after = fno_lot_size(price, "2024-11-20")
        # 3x up to the 5-unit rounding the exchanges apply to a published lot
        assert after == pytest.approx(3 * before, rel=0.01)
        assert price * before >= 5 * LAKH
        assert price * after >= 15 * LAKH
        assert before % 5 == 0 and after % 5 == 0
    assert fno_lot_size(2_500.0, "2024-11-20") == 600
    with pytest.raises(ValueError):
        fno_lot_size(0.0, "2026-09-10")


def test_ordinary_mainboard_example_uses_one_share_lot():
    assert CASH_BOARD_LOT == 1


# --------------------------------------------------------------------------- 2026 session
def test_the_pre_open_subperiods_tile_the_window_and_the_cas_is_dated():
    assert PRE_OPEN_SUBPERIODS[0][1] == time(9, 0)
    assert PRE_OPEN_SUBPERIODS[-1][2] == time(9, 15)
    for (_, _, end), (_, start, _) in zip(PRE_OPEN_SUBPERIODS, PRE_OPEN_SUBPERIODS[1:]):
        assert end == start
    assert PRE_OPEN_RESTRUCTURED_ON == date(2026, 9, 7)
    assert CAS_FROM == date(2026, 8, 3)
    assert CAS_BAND == 0.03


# --------------------------------------------------------------------------- the demo
def test_run_main_prints_the_rule_and_the_headline_numbers(run_main):
    out = run_main("fin_skills.asia.india")
    assert "RULE: an Indian fill outside the security's price band is not a bad fill" in out
    assert "2.68%" in out
    assert "26x the unconditional rate" in out
    assert "125x too big" in out
    assert "closed for the day" in out
    assert out.isascii(), "skill scripts print ASCII only"


def test_cas_requires_both_effective_date_and_security_eligibility():
    from fin_skills.asia.india import cash_session
    assert cash_session("2026-08-02", cas_eligible=True)["closing_auction"] is None
    assert cash_session("2026-08-03", cas_eligible=False)["closing_auction"] is None
    eligible = cash_session("2026-08-03", cas_eligible=True)
    assert eligible["closing_auction"] == (time(15,15),time(15,35))
    assert eligible["continuous"] == (time(9,15),time(15,15))


def test_band_tick_is_an_explicit_security_input():
    from fin_skills.asia.india import band_prices
    lower, upper = band_prices(101.01, .02, tick=.01)
    assert lower <= upper
    assert lower * 100 == pytest.approx(round(lower * 100))
    assert upper * 100 == pytest.approx(round(upper * 100))
    with pytest.raises(ValueError):
        band_prices(100., .05, tick=0.)
