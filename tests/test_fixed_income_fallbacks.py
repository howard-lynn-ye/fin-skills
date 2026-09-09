"""fin_skills.fixed_income.fallbacks - Regulation ZZ LIBOR fallbacks.

The documented properties: the five tenor spread adjustments are the exact statutory values in
12 CFR 253.4(c), the one-week and two-month tenors are outside the statute, the SAME spread
attaches to five DIFFERENT base rates depending on contract type, a backward-looking average
base lags the in-arrears one in both directions (up to 19.56 bp on one reset here), and the
consumer spread is a linear ramp over one year rather than a step.
"""
from __future__ import annotations

from datetime import date

import pytest

from fin_skills.fixed_income.fallbacks import (BENCHMARK_ADMINISTRATORS, BOARD_SELECTED_BASES,
                                               CONSUMER_TRANSITION_END, CONTRACT_TYPES,
                                               COVERED_TENORS, EXCLUDED_TENORS,
                                               LIBOR_REPLACEMENT_DATE, TENOR_SPREADS,
                                               business_days, compounded_sofr,
                                               consumer_transition_spread, fallback_comparison,
                                               fallback_rate, quarterly_periods,
                                               synthetic_sofr, tenor_spread)

HOLIDAYS = [date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
            date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
            date(2026, 10, 12), date(2026, 11, 11), date(2026, 11, 26), date(2026, 12, 25),
            date(2025, 11, 11), date(2025, 11, 27), date(2025, 12, 25)]
STEPS = ((date(2026, 6, 18), 0.0025), (date(2026, 9, 17), -0.0025))
PATH = synthetic_sofr(date(2025, 9, 1), date(2027, 1, 5), HOLIDAYS, seed=3, steps=STEPS)
NOTIONAL = 10_000_000.0
Q3 = (date(2026, 7, 1), date(2026, 10, 1))


# ------------------------------------------------------------------ the statute
def test_the_five_tenor_spread_adjustments_are_the_exact_statutory_values():
    assert TENOR_SPREADS == {"overnight": 0.0000644, "1M": 0.0011448, "3M": 0.0026161,
                             "6M": 0.0042826, "12M": 0.0071513}
    assert tenor_spread("3M") * 1e4 == pytest.approx(26.161, abs=1e-9)
    assert tenor_spread("12M") * 1e4 == pytest.approx(71.513, abs=1e-9)
    # monotone in tenor, and the 3M spread is 6,540.25 per quarter on 10mm
    vals = [tenor_spread(t) for t in COVERED_TENORS]
    assert vals == sorted(vals)
    assert NOTIONAL * tenor_spread("3M") * 0.25 == pytest.approx(6540.25, abs=0.01)


def test_the_one_week_and_two_month_tenors_are_outside_the_statute():
    assert EXCLUDED_TENORS == ("1W", "2M")
    for t in EXCLUDED_TENORS:
        assert t not in TENOR_SPREADS
        with pytest.raises(ValueError, match="outside 12 CFR 253"):
            tenor_spread(t)
    with pytest.raises(ValueError, match="unknown tenor"):
        tenor_spread("9M")
    assert LIBOR_REPLACEMENT_DATE == date(2023, 7, 3)
    assert CONSUMER_TRANSITION_END == date(2024, 7, 3)


def test_five_contract_types_five_base_rates_four_administrators():
    assert len(BOARD_SELECTED_BASES) == 5
    assert CONTRACT_TYPES == ("derivative", "non_consumer_cash", "consumer_loan",
                              "fhfa_entity", "ffelp_abs")
    bases = {b for _, b, _, _ in BOARD_SELECTED_BASES}
    assert len(bases) == 4                       # consumer and non-consumer share CME Term SOFR
    assert "30-day Average SOFR" in bases and "90-day Average SOFR" in bases
    assert len(BENCHMARK_ADMINISTRATORS) == 4
    for _, _, _, cite in BOARD_SELECTED_BASES:
        assert cite.startswith("253.4")


# ------------------------------------------------------------------ the base rates
def test_the_same_spread_attaches_to_windows_that_are_not_the_same_window():
    term = compounded_sofr(PATH, *Q3, HOLIDAYS)
    rates = {r["contract_type"]: r for r in
             fallback_comparison("3M", PATH, *Q3, HOLIDAYS, NOTIONAL, term)}
    # every branch carries the identical statutory spread
    for r in rates.values():
        assert r["spread"] == pytest.approx(TENOR_SPREADS["3M"], abs=1e-15)
    # and four different base rates
    assert rates["derivative"]["base"] == pytest.approx(0.04540911, abs=5e-9)
    assert rates["non_consumer_cash"]["base"] == pytest.approx(rates["derivative"]["base"])
    assert rates["fhfa_entity"]["bp_vs_derivative"] == pytest.approx(-9.26, abs=0.01)
    assert rates["ffelp_abs"]["bp_vs_derivative"] == pytest.approx(-16.18, abs=0.01)
    assert rates["ffelp_abs"]["cash_vs_derivative"] == pytest.approx(-4134.13, abs=0.02)
    # the longer the backward window, the further behind a rising path it is
    assert rates["ffelp_abs"]["base"] < rates["fhfa_entity"]["base"] < \
        rates["derivative"]["base"]


def test_the_backward_looking_lag_changes_sign_with_the_policy_path():
    bps = []
    for ps, pe in quarterly_periods(2026, HOLIDAYS):
        term = compounded_sofr(PATH, ps, pe, HOLIDAYS)
        rows = {r["contract_type"]: r for r in
                fallback_comparison("3M", PATH, ps, pe, HOLIDAYS, NOTIONAL, term)}
        bps.append(rows["ffelp_abs"]["bp_vs_derivative"])
    # after the +25 bp step the backward average is BELOW; after the -25 bp cut it is ABOVE
    assert min(bps) == pytest.approx(-16.18, abs=0.01)
    assert max(bps) == pytest.approx(+19.56, abs=0.01)
    assert min(bps) < 0.0 < max(bps)
    # a single reset can be as far off as most of the whole statutory spread
    assert max(abs(b) for b in bps) > 0.7 * tenor_spread("3M") * 1e4
    # ... while the year nets much smaller, so an annual hedge misses every quarter
    assert abs(sum(bps)) < 0.5 * max(abs(b) for b in bps)


def test_fallback_rate_refuses_what_the_regulation_does_not_define():
    with pytest.raises(ValueError, match="unknown contract type"):
        fallback_rate("municipal", "3M", PATH, *Q3, HOLIDAYS)
    with pytest.raises(ValueError, match="licensed"):
        fallback_rate("non_consumer_cash", "3M", PATH, *Q3, HOLIDAYS)
    with pytest.raises(ValueError, match="outside 12 CFR 253"):
        fallback_rate("derivative", "2M", PATH, *Q3, HOLIDAYS)
    # a consumer loan can carry an explicit transitioning spread instead of the statutory one
    r = fallback_rate("consumer_loan", "3M", PATH, *Q3, HOLIDAYS, term_sofr=0.045,
                      consumer_spread=0.0015)
    assert r["spread"] == 0.0015
    assert r["all_in"] == pytest.approx(0.0465, abs=1e-12)


def test_the_ffelp_six_month_contract_falls_back_to_a_thirty_day_average():
    # 253.4(b)(4): 90-day Average SOFR only for the THREE-month tenor; 30-day for 1M/6M/12M
    r3 = fallback_rate("ffelp_abs", "3M", PATH, *Q3, HOLIDAYS)
    r6 = fallback_rate("ffelp_abs", "6M", PATH, *Q3, HOLIDAYS)
    assert "90 days" in r3["window"] and "30 days" in r6["window"]
    assert r3["base"] != r6["base"]
    # ... while the SIX-month spread still applies to that thirty-day base
    assert r6["spread"] == pytest.approx(TENOR_SPREADS["6M"], abs=1e-15)


# ------------------------------------------------------------------ the consumer ramp
def test_the_consumer_spread_is_a_linear_ramp_that_clamps_at_both_ends():
    statutory, initial, n = tenor_spread("3M"), 0.0015, 252
    assert consumer_transition_spread(0, n, initial, statutory) == pytest.approx(initial)
    assert consumer_transition_spread(n, n, initial, statutory) == pytest.approx(statutory)
    assert consumer_transition_spread(n * 3, n, initial, statutory) == pytest.approx(statutory)
    assert consumer_transition_spread(-5, n, initial, statutory) == pytest.approx(initial)
    half = consumer_transition_spread(n // 2, n, initial, statutory)
    assert half == pytest.approx(0.5 * (initial + statutory), abs=1e-9)
    # equal steps every business day
    steps = [consumer_transition_spread(k + 1, n, initial, statutory)
             - consumer_transition_spread(k, n, initial, statutory) for k in range(0, n - 1)]
    assert max(steps) == pytest.approx(min(steps), abs=1e-15)
    with pytest.raises(ValueError, match="transition_days must be positive"):
        consumer_transition_spread(1, 0, initial, statutory)


# ------------------------------------------------------------------ the path
def test_the_seeded_path_is_deterministic_and_carries_both_policy_steps():
    again = synthetic_sofr(date(2025, 9, 1), date(2027, 1, 5), HOLIDAYS, seed=3, steps=STEPS)
    assert again == PATH
    assert synthetic_sofr(date(2025, 9, 1), date(2027, 1, 5), HOLIDAYS, seed=4,
                          steps=STEPS) != PATH
    assert PATH[date(2026, 6, 18)] - PATH[date(2026, 6, 17)] == pytest.approx(0.0025, abs=6e-4)
    assert PATH[date(2026, 9, 17)] - PATH[date(2026, 9, 16)] == pytest.approx(-0.0025, abs=6e-4)
    assert date(2026, 4, 3) not in PATH            # holidays carry no fixing
    assert len(business_days(*Q3, HOLIDAYS)) == 64      # 92 calendar days, 3 Jul is a holiday
    with pytest.raises(ValueError, match="no fixings"):
        compounded_sofr(PATH, date(2026, 7, 4), date(2026, 7, 5), HOLIDAYS)


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.fixed_income.fallbacks")
    assert "the BASE RATE is not" in out
    assert "0.26161" in out or "26.161" in out
    assert "253.4(b)(3)(i)(B)" in out
    assert out.isascii()
