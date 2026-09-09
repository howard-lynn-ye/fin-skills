"""fin_skills.tax_accounting.section_1256 - 60/40, the year-end mark, and where the sources stop."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.tax_accounting.section_1256 import (CLASSIFICATION, blended_rate, classify,
                                                    compare_regimes, is_long_term,
                                                    last_business_day, make_option_book,
                                                    pretax_pnl, recognise_1256,
                                                    recognise_equity_option, sixty_forty,
                                                    tax_schedule)


def pos(rows):
    return pd.DataFrame(rows, columns=["open_date", "close_date", "qty", "open_price",
                                       "close_price"])


def test_sixty_forty_is_exact_and_ignores_the_holding_period():
    assert sixty_forty(10_000.0) == {"long": 6_000.0, "short": 4_000.0}
    assert sixty_forty(-10_000.0) == {"long": -6_000.0, "short": -4_000.0}
    assert blended_rate(0.37, 0.20) == pytest.approx(0.268)


@pytest.mark.parametrize("year,expected,weekday", [
    (2022, "2022-12-30", "Friday"),    # 2022-12-31 was a Saturday
    (2023, "2023-12-29", "Friday"),    # 2023-12-31 was a Sunday
    (2024, "2024-12-31", "Tuesday"),
    (2025, "2025-12-31", "Wednesday"),
])
def test_last_business_day_is_not_december_31(year, expected, weekday):
    d = last_business_day(year)
    assert d == pd.Timestamp(expected) and d.day_name() == weekday


# IRS Publication 550 (2025), "60/40 rule", p. 57, reproduced exactly: a regulated futures
# contract bought 2024-06-03 for $50,000, worth $57,000 on 2024-12-31 (the last business day
# of the tax year) -> $7,000 gain on the 2024 return, 60/40. Sold 2025-02-03 for $56,000 ->
# a $1,000 loss on the 2025 return, also 60/40.
def test_pub_550_worked_example_for_a_regulated_futures_contract():
    p = pos([("2024-06-03", "2025-02-03", 1.0, 50_000.0, 56_000.0)])
    marks = pd.DataFrame([{"position": 0, "year": 2024, "price": 57_000.0}])
    sched = recognise_1256(p, marks, multiplier=1.0)
    assert float(sched.loc[2024, "recognised"]) == pytest.approx(7_000.0)
    assert float(sched.loc[2024, "long_term"]) == pytest.approx(4_200.0)
    assert float(sched.loc[2024, "short_term"]) == pytest.approx(2_800.0)
    assert float(sched.loc[2025, "recognised"]) == pytest.approx(-1_000.0)
    assert float(sched.loc[2025, "long_term"]) == pytest.approx(-600.0)
    # And the chain does not double count: the two years sum to the whole move.
    assert float(sched["recognised"].sum()) == pytest.approx(6_000.0)


def test_an_equity_option_recognises_nothing_until_it_is_closed():
    p = pos([("2024-06-03", pd.NaT, 1.0, 50_000.0, np.nan)])
    marks = pd.DataFrame([{"position": 0, "year": 2024, "price": 57_000.0}])
    assert float(recognise_1256(p, marks, 1.0)["recognised"].sum()) == pytest.approx(7_000.0)
    assert recognise_equity_option(p, 1.0)["recognised"].sum() == 0.0


def test_the_year_end_mark_chain_equals_the_total_move():
    p = pos([("2022-02-01", "2025-03-01", 2.0, 10.0, 25.0)])
    marks = pd.DataFrame([{"position": 0, "year": y, "price": v}
                          for y, v in ((2022, 12.0), (2023, 9.0), (2024, 30.0))])
    sched = recognise_1256(p, marks, multiplier=100.0)
    assert list(sched.index) == [2022, 2023, 2024, 2025]
    assert float(sched["recognised"].sum()) == pytest.approx((25.0 - 10.0) * 2 * 100)


def test_the_equity_option_path_splits_by_holding_period():
    p = pos([("2023-01-04", "2024-06-03", 1.0, 100.0, 200.0),      # 516 days -> long
             ("2024-01-04", "2024-06-03", 1.0, 100.0, 150.0)])     # 151 days -> short
    sched = recognise_equity_option(p, multiplier=100.0)
    assert float(sched.loc[2024, "long_term"]) == pytest.approx(10_000.0)
    assert float(sched.loc[2024, "short_term"]) == pytest.approx(5_000.0)
    assert is_long_term("2023-01-04", "2024-01-04") is False
    assert is_long_term("2023-01-04", "2024-01-05") is True


def test_sixty_forty_costs_more_than_long_term_on_a_position_held_over_a_year():
    p = pos([("2023-01-04", "2024-06-03", 1.0, 100.0, 200.0)])
    marks = pd.DataFrame([{"position": 0, "year": 2023, "price": 100.0}])
    t1256 = tax_schedule(recognise_1256(p, marks, 100.0), 0.37, 0.20)["tax"].sum()
    tequity = tax_schedule(recognise_equity_option(p, 100.0), 0.37, 0.20)["tax"].sum()
    assert float(t1256) == pytest.approx(2_680.0)
    assert float(tequity) == pytest.approx(2_000.0)
    assert t1256 > tequity


def test_classify_carries_a_source_and_refuses_to_guess():
    spx = classify("broad_based_index_option")
    assert spx["regime"] == "section_1256"
    assert "Standard and Poor's 500 index" in spx["authority"]
    etf = classify("etf_option")
    assert etf["regime"] == "unclear" and "no published IRS ruling" in etf["authority"]
    assert classify("single_stock_option")["regime"] == "equity_option"
    assert classify("swap")["regime"] == "not_1256"
    assert classify("securities_futures_contract")["regime"] == "not_1256"
    with pytest.raises(ValueError, match="unknown instrument"):
        classify("spy_option")
    for name, (regime, authority) in CLASSIFICATION.items():
        assert regime in ("section_1256", "equity_option", "not_1256", "unclear")
        assert len(authority) > 40, name


def test_the_seeded_book_reproduces_the_documented_numbers():
    positions, marks = make_option_book()
    assert len(positions) == 38 and int(positions["close_date"].isna().sum()) == 1
    res = compare_regimes(positions, marks, short_rate=0.37, long_rate=0.20)
    assert res["pretax_pnl"] == pytest.approx(41_280.10, abs=0.01)
    rec_1256 = float(res["section_1256"]["recognised"].sum())
    rec_eq = float(res["equity_option"]["recognised"].sum())
    assert rec_1256 == pytest.approx(41_332.16, abs=0.01)
    assert rec_eq == pytest.approx(28_651.05, abs=0.01)
    assert res["tax_1256"] == pytest.approx(11_077.02, abs=0.01)
    assert res["tax_equity"] == pytest.approx(10_600.89, abs=0.01)
    # The effective rates ARE the statutory ones, which is the implementation's own check.
    assert res["tax_1256"] / rec_1256 == pytest.approx(0.268, abs=1e-9)
    assert res["tax_equity"] / rec_eq == pytest.approx(0.37, abs=1e-9)
    # The gap is exactly the never-closed position's year-end marks.
    ev = res["section_1256"].attrs["events"]
    open_id = positions.index[positions["close_date"].isna()][0]
    never = float(ev.loc[ev["position"] == open_id, "gain"].sum())
    assert never == pytest.approx(12_681.11, abs=0.01)
    assert rec_1256 - rec_eq == pytest.approx(never, abs=1e-6)


def test_pretax_pnl_is_identical_under_both_regimes_by_construction():
    positions, marks = make_option_book()
    a = pretax_pnl(positions, positions.attrs["final_price"])
    # An equity-option book and a 1256 book are the SAME trades; only the tax differs.
    assert a == pytest.approx(41_280.10, abs=0.01)


def test_seeded_runs_are_deterministic():
    p1, m1 = make_option_book()
    p2, m2 = make_option_book()
    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_frame_equal(m1, m2)


def test_inputs_are_validated():
    with pytest.raises(ValueError, match="missing column"):
        recognise_1256(pd.DataFrame({"open_date": ["2024-01-01"]}), pd.DataFrame(
            columns=["position", "year", "price"]))
    with pytest.raises(ValueError, match="close_price"):
        recognise_equity_option(pos([("2024-01-01", "2024-06-01", 1.0, 10.0, np.nan)]))
    with pytest.raises(ValueError, match="close_date is before"):
        recognise_equity_option(pos([("2024-06-01", "2024-01-01", 1.0, 10.0, 12.0)]))
    with pytest.raises(ValueError, match="short_rate"):
        tax_schedule(recognise_equity_option(pos([("2024-01-01", "2024-06-01", 1.0, 10.0,
                                                   12.0)])), 1.4, 0.2)


def test_demo_prints_the_rule_and_the_open_position(run_main):
    out = run_main("fin_skills.tax_accounting.section_1256")
    assert "etf_option                 -> unclear" in out
    assert "the position never closed books" in out
    assert "Rev. Rul. 2026-16" in out
    assert "RULE: 1256 marks open positions" in out
