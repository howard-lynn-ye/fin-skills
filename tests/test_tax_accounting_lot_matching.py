"""fin_skills.tax_accounting.lot_matching - four lot rules, four P&Ls, one invariant total."""
from __future__ import annotations

import pandas as pd
import pytest

from fin_skills.tax_accounting.lot_matching import (average_basis, compare_rules, is_long_term,
                                                    make_blotter, match_lots, realised_gain,
                                                    spread_fraction, tax_due, term_split,
                                                    unrealised)


def blot(rows):
    return pd.DataFrame(rows, columns=["date", "side", "qty", "price"])


# Hand-computable: two lots at 10 and 20, sell 100 at 30.
#   fifo -> sells the 10 lot -> 100 * 20 = 2000
#   lifo -> sells the 20 lot -> 100 * 10 = 1000
#   hifo -> sells the 20 lot -> 1000
#   lofo -> sells the 10 lot -> 2000
TOY = [("2020-01-06", "B", 100, 10.0),
       ("2020-06-01", "B", 100, 20.0),
       ("2020-09-01", "S", 100, 30.0)]


@pytest.mark.parametrize("rule,expected", [("fifo", 2000.0), ("lifo", 1000.0),
                                           ("hifo", 1000.0), ("lofo", 2000.0)])
def test_the_same_trades_give_different_hand_checkable_gains(rule, expected):
    assert realised_gain(match_lots(blot(TOY), rule)) == pytest.approx(expected)


def test_the_rules_disagree_on_which_lot_stays_open():
    assert float(match_lots(blot(TOY), "fifo")["open_lots"]["price"].iloc[0]) == 20.0
    assert float(match_lots(blot(TOY), "lifo")["open_lots"]["price"].iloc[0]) == 10.0


def test_realised_plus_unrealised_is_the_same_under_every_rule():
    table = compare_rules(blot(TOY), mark=25.0)
    # 200 shares, cost 3000, value 100*30 sold + 100*25 open = 5500 -> 2500 total, always.
    assert table["total"].round(9).nunique() == 1
    assert float(table["total"].iloc[0]) == pytest.approx(2500.0)
    assert float(table["realised"].max() - table["realised"].min()) == pytest.approx(1000.0)


def test_seeded_blotter_reproduces_the_documented_numbers():
    b = make_blotter()
    assert len(b) == 72 and int((b["side"] == "B").sum()) == 43
    table = compare_rules(b, mark=b.attrs["final_price"])
    assert float(table.loc["fifo", "realised"]) == pytest.approx(5218.13, abs=0.01)
    assert float(table.loc["lifo", "realised"]) == pytest.approx(16871.94, abs=0.01)
    assert float(table.loc["hifo", "realised"]) == pytest.approx(-752.43, abs=0.01)
    assert float(table.loc["lofo", "realised"]) == pytest.approx(19341.30, abs=0.01)
    assert float(table.loc["lifo", "long_term"]) == pytest.approx(0.0, abs=1e-9)
    s = spread_fraction(table)
    assert s["spread"] == pytest.approx(20093.73, abs=0.01)
    assert s["total_pnl"] == pytest.approx(32595.80, abs=0.01)
    assert s["fraction_of_pnl"] == pytest.approx(0.6165, abs=1e-3)


def test_seeded_run_is_deterministic():
    a, b = make_blotter(), make_blotter()
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(compare_rules(a, 100.0), compare_rules(b, 100.0))


# IRC 1222 / Pub 550: LONGER than one year, counted in calendar years, not 365 days.
@pytest.mark.parametrize("open_d,close_d,expected", [
    ("2024-02-29", "2025-02-28", False),   # not yet a year
    ("2024-02-29", "2025-03-01", True),    # one calendar year and a day
    ("2023-01-10", "2024-01-10", False),   # exactly one year is still short-term
    ("2023-01-10", "2024-01-11", True),
])
def test_holding_period_is_calendar_years_not_365_days(open_d, close_d, expected):
    assert is_long_term(pd.Timestamp(open_d), pd.Timestamp(close_d)) is expected


def test_term_split_sums_to_the_realised_gain():
    res = match_lots(make_blotter(), "fifo")
    split = term_split(res)
    assert split["short"] + split["long"] == pytest.approx(realised_gain(res))


def test_average_basis_is_refused_for_ordinary_stock_and_allowed_for_a_fund():
    with pytest.raises(ValueError, match="Publication 550"):
        average_basis(blot(TOY), share_class="stock")
    out = average_basis(blot(TOY), share_class="mutual_fund")
    # average cost is 15; 100 shares sold at 30 -> 1500, between the fifo and lifo answers.
    assert float(out["disposals"]["gain"].sum()) == pytest.approx(1500.0)
    assert float(out["disposals"]["avg_basis_per_share"].iloc[0]) == pytest.approx(15.0)
    assert out["remaining_qty"] == pytest.approx(100.0)


def test_tax_due_uses_the_rates_it_is_given_and_validates_them():
    assert tax_due({"short": 1000.0, "long": 1000.0}, 0.37, 0.20) == pytest.approx(570.0)
    with pytest.raises(ValueError, match="short_rate"):
        tax_due({"short": 0.0, "long": 0.0}, 1.5, 0.2)


def test_a_sale_that_exceeds_inventory_is_refused_not_netted():
    with pytest.raises(ValueError, match="short"):
        match_lots(blot([("2020-01-06", "B", 100, 10.0),
                         ("2020-02-06", "S", 150, 12.0)]), "fifo")


def test_inputs_are_validated():
    with pytest.raises(ValueError, match="missing column"):
        match_lots(pd.DataFrame({"date": ["2020-01-01"], "side": ["B"]}), "fifo")
    with pytest.raises(ValueError, match="side must be"):
        match_lots(blot([("2020-01-06", "X", 100, 10.0)]), "fifo")
    with pytest.raises(ValueError, match="qty must be positive"):
        match_lots(blot([("2020-01-06", "B", -100, 10.0)]), "fifo")
    with pytest.raises(ValueError, match="unknown rule"):
        match_lots(blot(TOY), "average")


def test_unrealised_is_zero_once_the_book_is_flat():
    flat = blot(TOY + [("2020-10-01", "S", 100, 31.0)])
    res = match_lots(flat, "hifo")
    assert res["open_lots"].empty
    assert unrealised(res, 99.0) == 0.0


def test_demo_prints_the_rule_and_the_invariant(run_main):
    out = run_main("fin_skills.tax_accounting.lot_matching")
    assert "distinct values of realised+unrealised across rules = 1" in out
    assert "RULE: lot choice moves P&L" in out
    assert "DEFAULT when the shares are not adequately identified" in out
