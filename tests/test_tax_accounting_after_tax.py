"""fin_skills.tax_accounting.after_tax - the guard, the netting, and the turnover penalty."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.tax_accounting.after_tax import (REQUIRED_ASSUMPTIONS, MissingAssumptions,
                                                 TaxAssumptions, after_tax_backtest,
                                                 churn_blotter, demo_prices, net_and_tax,
                                                 realised_by_year, report,
                                                 require_assumptions, sharpe, turnover_study)


@pytest.fixture
def assumptions():
    return TaxAssumptions(jurisdiction="US federal, individual, 2025 rules",
                          short_rate=0.37, long_rate=0.20, lot_method="fifo")


@pytest.fixture(scope="module")
def prices():
    return demo_prices()


# ------------------------------------------------------------------ the guard
def test_the_guard_refuses_a_result_with_no_assumptions(prices, assumptions):
    with pytest.raises(MissingAssumptions, match="not comparable without assumptions"):
        after_tax_backtest(prices, 21, assumptions=None)
    with pytest.raises(MissingAssumptions, match="not comparable"):
        report({"assumptions": None})


@pytest.mark.parametrize("missing", REQUIRED_ASSUMPTIONS)
def test_the_guard_names_every_field_that_is_absent(missing):
    full = {"jurisdiction": "US", "short_rate": 0.37, "long_rate": 0.20, "lot_method": "fifo"}
    partial = {k: v for k, v in full.items() if k != missing}
    with pytest.raises(MissingAssumptions, match=missing):
        require_assumptions(partial)
    assert isinstance(require_assumptions(full), TaxAssumptions)


def test_there_are_no_default_rates():
    with pytest.raises(TypeError):                       # all four fields are required
        TaxAssumptions(jurisdiction="US")                # type: ignore[call-arg]
    with pytest.raises(MissingAssumptions, match="short_rate"):
        TaxAssumptions(jurisdiction="US", short_rate=1.4, long_rate=0.2, lot_method="fifo")
    with pytest.raises(MissingAssumptions, match="jurisdiction"):
        TaxAssumptions(jurisdiction="  ", short_rate=0.37, long_rate=0.2, lot_method="fifo")
    with pytest.raises(MissingAssumptions, match="lot_method"):
        TaxAssumptions(jurisdiction="US", short_rate=0.37, long_rate=0.2, lot_method="")


def test_report_prints_the_assumption_line(prices, assumptions):
    text = report(after_tax_backtest(prices, 21, assumptions))
    assert "hypothetical" in text
    assert "short=37.00%" in text and "long=20.00%" in text
    assert "lots='fifo'" in text and "US federal" in text


# ------------------------------------------------------------------ netting
def test_netting_is_within_character_then_across_it():
    a = TaxAssumptions("US", 0.40, 0.20, "fifo")
    r = pd.DataFrame({"short": [1000.0], "long": [500.0]}, index=[2024])
    assert float(net_and_tax(r, a).loc[2024, "tax"]) == pytest.approx(500.0)
    # A long-term loss meets a short-term gain: 1000 short - 400 long -> 600 short taxable.
    r2 = pd.DataFrame({"short": [1000.0], "long": [-400.0]}, index=[2024])
    assert float(net_and_tax(r2, a).loc[2024, "tax"]) == pytest.approx(240.0)


def test_a_net_loss_pays_no_tax_and_carries_forward_with_its_character():
    a = TaxAssumptions("US", 0.40, 0.20, "fifo")
    r = pd.DataFrame({"short": [-1000.0, 1500.0], "long": [0.0, 0.0]}, index=[2024, 2025])
    out = net_and_tax(r, a)
    assert float(out.loc[2024, "tax"]) == 0.0                      # no refund
    assert float(out.loc[2024, "carry_short"]) == pytest.approx(-1000.0)
    assert float(out.loc[2025, "tax"]) == pytest.approx(500.0 * 0.40)


def test_the_carryforward_can_be_switched_off():
    a = TaxAssumptions("US", 0.40, 0.20, "fifo", loss_carryforward=False)
    r = pd.DataFrame({"short": [-1000.0, 1500.0], "long": [0.0, 0.0]}, index=[2024, 2025])
    assert float(net_and_tax(r, a).loc[2025, "tax"]) == pytest.approx(600.0)


def test_the_ordinary_offset_is_zero_unless_you_set_it():
    r = pd.DataFrame({"short": [-5000.0, 0.0], "long": [0.0, 0.0]}, index=[2024, 2025])
    plain = net_and_tax(r, TaxAssumptions("US", 0.4, 0.2, "fifo"))
    assert float(plain.loc[2024, "ordinary_offset"]) == 0.0
    assert float(plain.loc[2025, "carry_short"]) == pytest.approx(-5000.0)
    us = net_and_tax(r, TaxAssumptions("US", 0.4, 0.2, "fifo", ordinary_offset_per_year=3000.0))
    assert float(us.loc[2024, "ordinary_offset"]) == pytest.approx(3000.0)
    assert float(us.loc[2024, "carry_short"]) == pytest.approx(-2000.0)
    # ...and the remaining 2,000 is absorbed by next year's offset, leaving nothing to carry.
    assert float(us.loc[2025, "ordinary_offset"]) == pytest.approx(2000.0)
    assert float(us.loc[2025, "carry_short"]) == pytest.approx(0.0)


# ------------------------------------------------------------------ the study
def test_the_churn_blotter_keeps_exposure_constant(prices):
    b = churn_blotter(prices, 21, shares=1000.0)
    assert float(b.loc[b["side"] == "B", "qty"].sum()
                 - b.loc[b["side"] == "S", "qty"].sum()) == pytest.approx(1000.0)
    assert churn_blotter(prices, 0).shape[0] == 1        # buy and hold: one trade


def test_gross_is_identical_across_turnover_and_only_after_tax_moves(prices, assumptions):
    table = turnover_study(prices, (1, 5, 21, 63, 252, 0), assumptions)
    assert table["gross_sharpe"].round(12).nunique() == 1
    assert table["pretax_pnl"].round(6).nunique() == 1
    assert float(table["pretax_pnl"].iloc[0]) == pytest.approx(36_722.53, abs=0.01)
    assert float(table["gross_sharpe"].iloc[0]) == pytest.approx(0.488, abs=5e-4)
    # Buy and hold realises nothing, so after tax IS gross.
    assert float(table.loc[np.inf, "tax_paid"]) == 0.0
    assert float(table.loc[np.inf, "retained"]) == pytest.approx(1.0)
    # More turnover, more tax, lower after-tax Sharpe.
    assert float(table.loc[1.0, "after_tax_sharpe"]) == pytest.approx(0.312, abs=5e-4)
    assert float(table.loc[1.0, "tax_paid"]) == pytest.approx(17_108.20, abs=0.01)
    assert table.loc[1.0, "after_tax_sharpe"] < table.loc[63.0, "after_tax_sharpe"] \
        < table.loc[np.inf, "after_tax_sharpe"]
    # ...and it saturates: monthly, weekly and daily churn pay within 6% of each other.
    fast = table.loc[[1.0, 5.0, 21.0], "tax_paid"]
    assert (fast.max() - fast.min()) / fast.max() < 0.06


def test_ignoring_wash_sales_reports_the_churner_as_tax_free(prices, assumptions):
    true = after_tax_backtest(prices, 1, assumptions)
    naive = after_tax_backtest(prices, 1, TaxAssumptions(
        assumptions.jurisdiction, assumptions.short_rate, assumptions.long_rate,
        assumptions.lot_method, wash_sales=False))
    assert naive["total_tax"] == pytest.approx(0.0)
    assert naive["after_tax_sharpe"] == pytest.approx(naive["gross_sharpe"])
    assert true["total_tax"] == pytest.approx(17_108.20, abs=0.01)
    assert true["after_tax_sharpe"] < 0.7 * naive["after_tax_sharpe"]


def test_buy_and_hold_after_tax_equals_pre_tax_exactly(prices, assumptions):
    r = after_tax_backtest(prices, 0, assumptions)
    assert r["total_tax"] == 0.0
    assert r["after_tax_sharpe"] == pytest.approx(r["gross_sharpe"])
    pd.testing.assert_series_equal(r["gross_returns"], r["net_returns"])


def test_realised_by_year_splits_on_the_holding_period():
    d = pd.DataFrame([
        {"sell_date": pd.Timestamp("2024-06-01"), "open_date": pd.Timestamp("2024-01-01"),
         "allowed": 100.0},
        {"sell_date": pd.Timestamp("2024-06-01"), "open_date": pd.Timestamp("2022-01-01"),
         "allowed": 250.0}])
    out = realised_by_year(d)
    assert float(out.loc[2024, "short"]) == pytest.approx(100.0)
    assert float(out.loc[2024, "long"]) == pytest.approx(250.0)
    assert realised_by_year(pd.DataFrame()).empty


def test_sharpe_is_nan_on_a_flat_series():
    assert np.isnan(sharpe(pd.Series([0.0] * 10)))


def test_seeded_runs_are_deterministic(assumptions):
    pd.testing.assert_series_equal(demo_prices(), demo_prices())
    a = after_tax_backtest(demo_prices(), 21, assumptions)
    b = after_tax_backtest(demo_prices(), 21, assumptions)
    assert a["total_tax"] == b["total_tax"]
    assert a["after_tax_sharpe"] == b["after_tax_sharpe"]


def test_demo_prints_the_guard_and_the_rule(run_main):
    out = run_main("fin_skills.tax_accounting.after_tax")
    assert "guard: after-tax result is missing" in out
    assert "gross Sharpe is IDENTICAL across every row" in out
    assert "RULE: an after-tax result without a rate" in out
