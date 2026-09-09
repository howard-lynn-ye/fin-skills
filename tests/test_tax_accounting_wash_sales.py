"""fin_skills.tax_accounting.wash_sales - a wash sale defers a loss, it does not delete it."""
from __future__ import annotations

import pandas as pd
import pytest

from fin_skills.tax_accounting.wash_sales import (WINDOW_DAYS, WINDOW_TOTAL_DAYS,
                                                  apply_wash_sales, economic_pnl, in_window,
                                                  make_harvest_blotter, make_rebalance_blotter,
                                                  naive_error, summarise)


def blot(rows):
    cols = ["date", "side", "qty", "price", "account"]
    return pd.DataFrame([r if len(r) == 5 else (*r, "taxable") for r in rows], columns=cols)


def test_the_window_is_thirty_days_on_both_sides():
    assert WINDOW_DAYS == 30 and WINDOW_TOTAL_DAYS == 61
    assert in_window("2025-03-01", "2025-01-29") is False   # 31 days before
    assert in_window("2025-03-01", "2025-01-30") is True    # exactly 30 days before
    assert in_window("2025-03-01", "2025-03-31") is True    # exactly 30 days after
    assert in_window("2025-03-01", "2025-04-01") is False   # 31 days after
    # 30 + the sale day + 30 = the 61 days the statute describes.
    assert sum(in_window("2025-03-01", d) for d in pd.date_range("2025-01-25", "2025-04-05")
               ) == WINDOW_TOTAL_DAYS


# IRS Publication 550 (2025), "Wash Sales", Example 1, p. 86, reproduced exactly:
# buy 100 X for $1,000; sell for $750; buy 100 for $800 within 30 days ->
# $250 loss disallowed, basis of the new stock $1,050.
def test_pub_550_wash_sale_example_1():
    res = apply_wash_sales(blot([("2025-01-06", "B", 100, 10.00),
                                 ("2025-03-10", "S", 100, 7.50),
                                 ("2025-03-20", "B", 100, 8.00)]))
    d = res["disposals"]
    assert float(d["gain"].iloc[0]) == pytest.approx(-250.0)
    assert float(d["disallowed"].iloc[0]) == pytest.approx(-250.0)
    assert float(d["allowed"].iloc[0]) == pytest.approx(0.0)
    lots = res["open_lots"]
    assert float((lots["qty"] * lots["basis_ps"]).sum()) == pytest.approx(1050.0)


# Pub 550 (2025) p. 87, "More or less stock bought than sold", Example 1:
# 100 @ $5,000 (2024-09-20); 50 @ $2,750 (2024-12-13); 25 @ $1,125 (2024-12-20);
# sell the September 100 for $4,000 on 2025-01-03 -> $1,000 loss, $750 disallowed on 75
# shares, $250 deductible; new bases $3,250 and $1,375.
def test_pub_550_more_or_less_stock_bought_than_sold_example_1():
    res = apply_wash_sales(blot([("2024-09-20", "B", 100, 50.00),
                                 ("2024-12-13", "B", 50, 55.00),
                                 ("2024-12-20", "B", 25, 45.00),
                                 ("2025-01-03", "S", 100, 40.00)]))
    d = res["disposals"]
    assert float(d["gain"].sum()) == pytest.approx(-1000.0)
    assert float(d["disallowed"].sum()) == pytest.approx(-750.0)
    assert float(d["allowed"].sum()) == pytest.approx(-250.0)
    lots = res["open_lots"].set_index("date")
    basis = (lots["qty"] * lots["basis_ps"]).groupby(level=0).sum()
    assert float(basis[pd.Timestamp("2024-12-13")]) == pytest.approx(3250.0)
    assert float(basis[pd.Timestamp("2024-12-20")]) == pytest.approx(1375.0)


# Pub 550 (2025) p. 87, Example 2: one 100-share loss, then 50 shares bought on each of four
# consecutive days -> the whole loss is disallowed and lands on the FIRST TWO lots only.
def test_pub_550_more_or_less_stock_bought_than_sold_example_2():
    res = apply_wash_sales(blot([("2024-09-16", "B", 100, 20.00),
                                 ("2025-01-29", "S", 100, 10.00)]
                                + [(f"2025-02-0{d}", "B", 50, 12.00) for d in (3, 4, 5, 6)]))
    d = res["disposals"]
    assert float(d["gain"].sum()) == pytest.approx(-1000.0)
    assert float(d["disallowed"].sum()) == pytest.approx(-1000.0)
    lots = res["open_lots"].set_index("date")
    basis = (lots["qty"] * lots["basis_ps"]).groupby(level=0).sum()
    assert float(basis[pd.Timestamp("2025-02-03")]) == pytest.approx(1100.0)   # 600 + 500
    assert float(basis[pd.Timestamp("2025-02-04")]) == pytest.approx(1100.0)   # 600 + 500
    assert float(basis[pd.Timestamp("2025-02-05")]) == pytest.approx(600.0)    # untouched
    assert float(basis[pd.Timestamp("2025-02-06")]) == pytest.approx(600.0)


def test_the_forward_half_of_the_window_is_searched_not_just_the_past():
    # Sell today at a loss, buy back tomorrow: the ONLY replacement is in the future.
    res = apply_wash_sales(blot([("2025-01-06", "B", 100, 10.0),
                                 ("2025-06-02", "S", 100, 6.0),
                                 ("2025-06-03", "B", 100, 6.0)]))
    assert float(res["disposals"]["disallowed"].iloc[0]) == pytest.approx(-400.0)
    lots = res["open_lots"]
    assert float(lots["basis_ps"].iloc[0]) == pytest.approx(10.0)   # 6 + 4 deferred


def test_holding_period_carries_over_to_the_replacement():
    res = apply_wash_sales(blot([("2023-01-06", "B", 100, 10.0),
                                 ("2025-06-02", "S", 100, 6.0),
                                 ("2025-06-03", "B", 100, 6.0)]))
    # IRC 1223(3): the replacement inherits 2023-01-06, not 2025-06-03.
    assert res["open_lots"]["hp_start"].iloc[0] == pd.Timestamp("2023-01-06")


def test_a_repurchase_outside_the_window_is_not_a_wash_sale():
    res = apply_wash_sales(blot([("2025-01-06", "B", 100, 10.0),
                                 ("2025-06-02", "S", 100, 6.0),
                                 ("2025-07-03", "B", 100, 6.0)]))   # 31 days later
    assert float(res["disposals"]["disallowed"].iloc[0]) == 0.0
    assert float(res["disposals"]["allowed"].iloc[0]) == pytest.approx(-400.0)


def test_an_ira_repurchase_disallows_permanently_with_no_basis_step_up():
    res = apply_wash_sales(blot([("2025-01-06", "B", 100, 10.0, "taxable"),
                                 ("2025-06-02", "S", 100, 6.0, "taxable"),
                                 ("2025-06-03", "B", 100, 6.0, "ira")]))
    s = summarise(res)
    assert s["disallowed_loss"] == pytest.approx(-400.0)
    assert s["permanent_loss"] == pytest.approx(-400.0)
    assert bool(res["disposals"]["permanent"].iloc[0]) is True
    # Rev. Rul. 2008-5: no step-up. The IRA lot is still at its purchase price.
    assert float(res["open_lots"]["basis_ps"].iloc[0]) == pytest.approx(6.0)


def test_deferral_not_destruction_once_everything_is_liquidated():
    b = blot([("2025-01-06", "B", 100, 10.0),
              ("2025-06-02", "S", 100, 6.0),
              ("2025-06-03", "B", 100, 6.0),
              ("2026-11-02", "S", 100, 7.0)])          # far outside every window
    res = apply_wash_sales(b)
    assert summarise(res)["reported_realised"] == pytest.approx(economic_pnl(b, 7.0))
    assert summarise(res)["reported_realised"] == pytest.approx(-300.0)


def test_an_ira_replacement_breaks_the_deferral_identity_by_exactly_the_permanent_loss():
    rows = [("2025-01-06", "B", 100, 10.0, "taxable"),
            ("2025-06-02", "S", 100, 6.0, "taxable"),
            ("2025-06-03", "B", 100, 6.0, "ira"),
            ("2026-11-02", "S", 100, 7.0, "taxable")]
    res = apply_wash_sales(blot(rows))
    s = summarise(res)
    assert s["reported_realised"] - economic_pnl(blot(rows), 7.0) == pytest.approx(
        -s["permanent_loss"])


def test_seeded_rebalance_blotter_reproduces_the_documented_numbers():
    b = make_rebalance_blotter()
    assert len(b) == 37
    gaps = b["date"].diff().dt.days.dropna().astype(int)
    assert (gaps.min(), gaps.max()) == (29, 31)
    assert int((gaps <= WINDOW_DAYS).sum()) == 29
    s = summarise(apply_wash_sales(b), mark=b.attrs["final_price"])
    assert s["n_loss_sales"] == 21 and s["n_wash_sales"] == 12
    assert s["gross_loss"] == pytest.approx(-41164.30, abs=0.01)
    assert s["disallowed_loss"] == pytest.approx(-22941.47, abs=0.01)
    assert s["fraction_of_loss_deferred"] == pytest.approx(0.5573, abs=1e-3)
    assert s["reported_realised"] == pytest.approx(-15863.59, abs=0.01)
    assert naive_error(apply_wash_sales(b), 0.37)["overstated_tax_saving"] == pytest.approx(
        8488.34, abs=0.01)


def test_the_harvest_blotter_chains_and_leaves_no_deduction():
    h = make_harvest_blotter()
    s = summarise(apply_wash_sales(h), mark=h.attrs["final_price"])
    assert s["n_loss_sales"] == s["n_wash_sales"] == 10
    assert s["fraction_of_loss_deferred"] == pytest.approx(1.0)
    assert s["reported_realised"] == pytest.approx(0.0, abs=1e-6)
    assert s["max_chain_depth"] == 9
    assert s["gross_loss"] == pytest.approx(-292200.11, abs=0.01)
    # The gross-loss line ratchets because each disallowed loss re-enters the basis.
    assert abs(s["gross_loss"]) > 6 * abs(economic_pnl(h, h.attrs["final_price"]))


def test_the_lot_method_decides_whether_there_are_wash_sales_at_all():
    b = make_rebalance_blotter()
    lifo = summarise(apply_wash_sales(b, "lifo"), mark=b.attrs["final_price"])
    assert lifo["n_loss_sales"] == 0 and lifo["n_wash_sales"] == 0
    assert lifo["reported_realised"] == pytest.approx(8373.07, abs=0.01)
    # Both rules still describe the same book.
    fifo = summarise(apply_wash_sales(b, "fifo"), mark=b.attrs["final_price"])
    assert fifo["reported_total"] == pytest.approx(lifo["reported_total"])


def test_seeded_runs_are_deterministic():
    pd.testing.assert_frame_equal(make_rebalance_blotter(), make_rebalance_blotter())
    a = summarise(apply_wash_sales(make_harvest_blotter()))
    b = summarise(apply_wash_sales(make_harvest_blotter()))
    assert a == b


def test_inputs_are_validated():
    with pytest.raises(ValueError, match="missing column"):
        apply_wash_sales(pd.DataFrame({"date": ["2020-01-01"], "side": ["B"]}))
    with pytest.raises(ValueError, match="fifo"):
        apply_wash_sales(blot([("2025-01-06", "B", 100, 10.0)]), rule="hifo")
    with pytest.raises(ValueError, match="short sales"):
        apply_wash_sales(blot([("2025-01-06", "B", 100, 10.0),
                               ("2025-02-06", "S", 150, 9.0)]))


def test_demo_prints_the_rule_and_the_invariant(run_main):
    out = run_main("fin_skills.tax_accounting.wash_sales")
    assert "deferral, not destruction" in out
    assert "no basis step-up, ever" in out
    assert "RULE: a wash sale defers a loss" in out
