"""fin_skills.asia.japan - the TSE session is 330 minutes with a hole, and it was 300.

Each test asserts a PROPERTY the SKILL.md documents, not just that the code runs: the
session-aware bar count differs from the wall-clock one by the claimed amount, the two
mixed volatility pipelines bracket the truth by the claimed amount, the divisor keeps a
price-weighted index continuous across a split, and the demo prints the rule.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fin_skills.asia.japan import (CLOSE_EXTENDED_FROM, FINE_TICKS_TOPIX500_FROM,
                                   NIKKEI_PAF_FROM, SHORT_DISCLOSE_THRESHOLD,
                                   SHORT_REPORT_THRESHOLD, SHORT_SALE_TRIGGER_DROP,
                                   T2_SETTLEMENT_FROM, TSE_RESTRUCTURED_ON,
                                   afternoon_close, annualised_vol, cap_weights,
                                   is_break_spanning_bar, lunch_break_minutes,
                                   naive_bar_count, new_divisor, price_weighted_index,
                                   price_weights, round_to_tick, session_bar_count,
                                   session_index, session_minutes, settlement_date,
                                   short_sale_price_restricted, synthetic_nikkei_panel,
                                   synthetic_tse_minutes, tick_size, vol_comparison,
                                   wall_clock_index, wall_clock_minutes,
                                   wall_clock_resample)


# --------------------------------------------------------------------------- sessions
def test_session_is_330_minutes_and_was_300_before_the_2024_change():
    assert session_minutes("2026-09-10") == (150, 180)
    assert session_minutes("2024-11-04") == (150, 150)
    assert session_bar_count(1, "2026-09-10") == 330
    assert session_bar_count(1, CLOSE_EXTENDED_FROM) == 330
    assert session_bar_count(1, "2024-11-04") == 300
    assert lunch_break_minutes() == 60


def test_the_wall_clock_grid_overcounts_by_exactly_60_bars_or_18_18_percent():
    """The documented number: 390 vs 330, +18.18%."""
    n, naive = session_bar_count(1, "2026-09-10"), naive_bar_count(1, "2026-09-10")
    assert (n, naive) == (330, 390)
    assert naive - n == 60
    assert naive / n - 1 == pytest.approx(0.1818, abs=5e-5)
    # and before the change the same 60-bar hole is a bigger share of a shorter day
    assert naive_bar_count(1, "2024-11-04") - session_bar_count(1, "2024-11-04") == 60
    assert naive_bar_count(1, "2024-11-04") / session_bar_count(1, "2024-11-04") - 1 == \
        pytest.approx(0.20, abs=1e-9)


def test_the_2024_close_extension_is_a_10_percent_step_in_the_bar_count():
    before, after = session_bar_count(1, "2024-11-04"), session_bar_count(1, "2024-11-05")
    assert after / before - 1 == pytest.approx(0.10, abs=1e-9)
    assert wall_clock_minutes("2024-11-04") == 360
    assert wall_clock_minutes("2026-09-10") == 390
    assert afternoon_close("2024-11-04").seconds // 3600 == 15
    assert afternoon_close("2026-09-10") > afternoon_close("2024-11-04")


def test_hourly_bars_require_a_partial_bar_convention():
    """60 does not divide the 150-minute morning session - the documented headline."""
    with pytest.raises(ValueError, match="do not tile"):
        session_bar_count(60, "2026-09-10")
    with pytest.raises(ValueError, match="45"):
        session_bar_count(45, "2026-09-10")
    legal = [f for f in range(1, 121) if 150 % f == 0 and 180 % f == 0]
    assert legal == [1, 2, 3, 5, 6, 10, 15, 30]
    for f in legal:
        assert session_bar_count(f, "2026-09-10") == 330 // f
    with pytest.raises(ValueError):
        session_bar_count(0, "2026-09-10")


def test_session_index_skips_the_break_and_labels_bars_by_close():
    idx = session_index("2026-09-10", 30)
    stamps = [t.strftime("%H:%M") for t in idx]
    assert stamps[0] == "09:30" and stamps[-1] == "15:30"
    assert "12:00" not in stamps and "12:30" not in stamps
    assert stamps[4:6] == ["11:30", "13:00"]          # 11:30 -> 13:00, nothing between
    assert len(session_index("2026-09-10", 1)) == 330
    assert len(wall_clock_index("2026-09-10", 1)) == 390


def test_the_first_afternoon_bar_is_the_break_spanning_one():
    assert is_break_spanning_bar(pd.Timestamp("2026-09-10 12:31"))
    assert not is_break_spanning_bar(pd.Timestamp("2026-09-10 12:32"))
    assert not is_break_spanning_bar(pd.Timestamp("2026-09-10 11:30"))
    idx = session_index("2026-09-10", 1)
    assert sum(is_break_spanning_bar(t) for t in idx) == 1


def test_a_date_is_mandatory():
    with pytest.raises(ValueError, match="REQUIRED"):
        session_minutes(None)


# --------------------------------------------------------------------------- volatility
@pytest.fixture(scope="module")
def vol():
    return vol_comparison(synthetic_tse_minutes(days=60, seed=20260910))


def test_the_wall_clock_resample_inserts_sixty_zero_bars_a_session(vol):
    assert vol["n_session_bars"] == 330 and vol["n_wall_bars"] == 390
    assert vol["zero_return_bars"] == 60 * 60          # 60 sessions x 60 stale bars


def test_the_two_mixed_pipelines_bracket_the_truth_by_the_documented_amounts(vol):
    """+8.71% and -8.04%, 18.21% apart - the numbers in SKILL.md section 2."""
    base = vol["session_aware"]
    assert vol["mixed_high"] / base - 1 == pytest.approx(0.0871, abs=5e-4)
    assert vol["mixed_low"] / base - 1 == pytest.approx(-0.0804, abs=5e-4)
    assert vol["mixed_high"] / vol["mixed_low"] - 1 == pytest.approx(0.1821, abs=1e-3)
    # the high-side error is exactly the annualiser ratio and nothing else
    assert vol["mixed_high"] / base == pytest.approx(math.sqrt(390 / 330), abs=1e-12)


def test_the_naive_wall_clock_pipeline_is_right_by_accident(vol):
    """Both halves wrong in the same direction very nearly cancel: |diff| < 0.1%."""
    assert abs(vol["wall_clock"] / vol["session_aware"] - 1) < 0.001


def test_dropping_the_break_spanning_bar_moves_the_answer_by_about_two_percent(vol):
    assert vol["ex_lunch_bar"] / vol["session_aware"] - 1 == pytest.approx(-0.0181,
                                                                          abs=5e-4)


def test_annualised_vol_needs_two_points():
    with pytest.raises(ValueError):
        annualised_vol([0.01], 252)
    r = np.full(1000, 0.01)
    assert annualised_vol(r, 252) == pytest.approx(0.0, abs=1e-12)


def test_the_tape_is_deterministic_and_offline():
    a = synthetic_tse_minutes(days=5, seed=7)
    b = synthetic_tse_minutes(days=5, seed=7)
    pd.testing.assert_series_equal(a, b)
    assert not synthetic_tse_minutes(days=5, seed=8).equals(a)
    assert len(a) == 5 * 330
    filled = wall_clock_resample(a)
    assert len(filled) == 5 * 390


# --------------------------------------------------------------------------- ticks
def test_the_fine_tick_table_is_topix500_and_gives_sub_yen_prices():
    assert tick_size(487.3, topix500=True) == 0.1
    assert tick_size(487.3, topix500=False) == 1.0
    assert tick_size(2_500.4, topix500=True) == 0.5
    assert tick_size(2_500.4, topix500=False) == 1.0
    assert tick_size(9_800.0, topix500=True) == 1.0
    assert tick_size(9_800.0, topix500=False) == 10.0
    assert tick_size(41_260.0, topix500=True) == 10.0
    assert tick_size(41_260.0, topix500=False) == 50.0
    for bad in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError):
            tick_size(bad)


def test_the_ordinary_table_throws_away_a_half_yen_improvement():
    fine = round_to_tick(2_500.4, topix500=True)
    coarse = round_to_tick(2_500.4, topix500=False)
    assert fine == 2_500.5 and coarse == 2_500.0
    assert abs(coarse - fine) == pytest.approx(0.5)
    assert round_to_tick(487.3, topix500=True) == 487.3     # already legal at 0.1
    assert round_to_tick(487.3, topix500=False) == 487.0
    assert round_to_tick(999.9, topix500=False, side="up") == 1_000.0
    assert round_to_tick(999.9, topix500=False, side="down") == 999.0
    with pytest.raises(ValueError):
        round_to_tick(100.0, side="sideways")


def test_the_fine_table_moved_from_topix100_to_topix500_in_2023():
    assert FINE_TICKS_TOPIX500_FROM == date(2023, 6, 5)


# --------------------------------------------------------------------------- settlement
def test_settlement_is_t2_from_2019_07_16_and_t3_before():
    assert T2_SETTLEMENT_FROM == date(2019, 7, 16)
    assert settlement_date("2019-07-15") == date(2019, 7, 18)     # T+3
    assert settlement_date("2019-07-16") == date(2019, 7, 18)     # T+2, same landing day
    assert settlement_date("2026-09-10") == date(2026, 9, 14)     # Thu -> Mon
    assert settlement_date("2026-09-10", holidays=["2026-09-11"]) == date(2026, 9, 15)


def test_the_t3_era_books_cash_one_business_day_later():
    early = settlement_date("2018-05-14")
    assert (early - date(2018, 5, 14)).days == 3


# --------------------------------------------------------------------------- index maths
def test_price_weighting_gives_the_top_priced_name_a_weight_size_cannot_explain():
    prices, shares = synthetic_nikkei_panel(225, seed=20260910)
    pw, cw = price_weights(prices), cap_weights(prices, shares)
    assert pw.sum() == pytest.approx(1.0) and cw.sum() == pytest.approx(1.0)
    top_p = int(np.argmax(pw))
    assert pw[top_p] == pytest.approx(0.0553, abs=5e-4)
    assert cw[top_p] == pytest.approx(0.0004, abs=5e-4)
    assert pw[top_p] / cw[top_p] == pytest.approx(153.4, rel=0.01)
    # the two weight vectors are nearly unrelated ORDERINGS
    rho = pd.Series(pw).corr(pd.Series(cw), method="spearman")
    assert abs(rho) < 0.2
    # while their concentration is almost the same
    assert abs(np.sort(pw)[-10:].sum() - np.sort(cw)[-10:].sum()) < 0.02


def test_the_divisor_keeps_a_price_weighted_index_continuous_across_a_split():
    prices, _ = synthetic_nikkei_panel(225, seed=20260910)
    d0 = 25.0
    lvl0 = price_weighted_index(prices, d0)
    after = prices.copy()
    top = int(np.argmax(prices))
    after[top] = prices[top] / 3.0
    d1 = new_divisor(prices, after, d0)
    assert d1 == pytest.approx(24.0782, abs=5e-5)
    assert price_weighted_index(after, d1) == pytest.approx(lvl0, rel=1e-12)
    # ... and NOT updating it prints a crash that did not happen
    unadj = price_weighted_index(after, d0)
    assert unadj / lvl0 - 1 == pytest.approx(-0.0369, abs=5e-4)


def test_price_adjustment_factors_scale_the_weights():
    p = np.array([100.0, 200.0, 700.0])
    assert price_weights(p)[2] == pytest.approx(0.7)
    half = price_weights(p, adjustments=np.array([1.0, 1.0, 0.5]))
    assert half[2] == pytest.approx(350 / 650)
    assert price_weighted_index(p, 2.0) == pytest.approx(500.0)
    with pytest.raises(ValueError):
        price_weighted_index(p, 0.0)
    with pytest.raises(ValueError):
        price_weighted_index(p, 1.0, adjustments=np.array([1.0]))
    assert NIKKEI_PAF_FROM == date(2021, 10, 1)


# --------------------------------------------------------------------------- short sale
def test_the_trigger_arms_at_exactly_minus_ten_percent_despite_float_error():
    """900/1000 - 1 is -0.09999999999999998; a return-based test misses the boundary."""
    assert 900.0 / 1000.0 - 1.0 > -0.10                # the bug this guards against
    assert short_sale_price_restricted(1_000.0, 900.0)
    assert short_sale_price_restricted(1_000.0, 899.0)
    assert not short_sale_price_restricted(1_000.0, 901.0)
    assert not short_sale_price_restricted(1_000.0, 960.0)
    assert SHORT_SALE_TRIGGER_DROP == 0.10
    with pytest.raises(ValueError):
        short_sale_price_restricted(0.0, 1.0)


def test_the_disclosure_thresholds_are_0_2_and_0_5_percent():
    assert SHORT_REPORT_THRESHOLD == 0.002
    assert SHORT_DISCLOSE_THRESHOLD == 0.005
    assert SHORT_DISCLOSE_THRESHOLD > SHORT_REPORT_THRESHOLD


def test_the_restructuring_date_is_carried():
    assert TSE_RESTRUCTURED_ON == date(2022, 4, 4)


# --------------------------------------------------------------------------- the demo
def test_run_main_prints_the_rule_and_the_headline_numbers(run_main):
    out = run_main("fin_skills.asia.japan")
    assert "RULE: the TSE session is 330 minutes with a 60-minute hole" in out
    assert "330 bars" in out and "390" in out
    assert "+18.18%" in out
    assert "Hourly bars need a partial-bar convention" in out
    assert "+8.71%" in out and "-8.04%" in out
    assert "24.0782" in out
    assert out.isascii(), "skill scripts print ASCII only"
