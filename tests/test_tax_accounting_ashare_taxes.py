"""fin_skills.tax_accounting.ashare_taxes - seller-side duty, and a dividend tax on turnover."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.tax_accounting.ashare_taxes import (DIVIDEND_EFFECTIVE_RATE,
                                                    STAMP_DUTY_HISTORY, compare_cost_models,
                                                    dividend_drag, dividend_tax,
                                                    dividend_tax_rate, flat_symmetric_cost,
                                                    holding_bucket, make_ashare_path,
                                                    make_turnover_blotter, rate_history_table,
                                                    real_stamp_duty, stamp_duty,
                                                    stamp_duty_rate)


# ------------------------------------------------------------------ stamp duty
def test_a_buy_pays_nothing_and_a_sell_pays_the_rate():
    # Stamp Tax Law art. 3: levied on the transferor, not on the transferee.
    assert stamp_duty(1_000_000, "B", "2025-01-02") == 0.0
    assert stamp_duty(1_000_000, "S", "2025-01-02") == pytest.approx(500.0)
    with pytest.raises(ValueError, match="side must be"):
        stamp_duty(1_000, "X", "2025-01-02")


@pytest.mark.parametrize("date,rate", [
    ("2023-08-25", 0.0010),      # the Friday before
    ("2023-08-27", 0.0010),      # the announcement's own date
    ("2023-08-28", 0.0005),      # effective from
    ("2026-09-09", 0.0005),
    ("2010-06-01", 0.0010),
])
def test_the_rate_halved_on_2023_08_28_and_not_a_day_earlier(date, rate):
    assert stamp_duty_rate(date) == pytest.approx(rate)


def test_dates_before_the_verified_window_are_refused_not_extrapolated():
    with pytest.raises(ValueError, match="2008-09-19"):
        stamp_duty_rate("2008-09-18")
    with pytest.raises(ValueError, match="BOTH sides"):
        stamp_duty(1_000, "S", "2007-06-01")


def test_funds_are_outside_the_tax_entirely():
    # Art. 3 reaches shares and share-based depositary receipts, not fund units.
    assert stamp_duty(1_000_000, "S", "2025-01-02", instrument="etf") == 0.0
    assert stamp_duty(1_000_000, "S", "2025-01-02", instrument="cdr") == pytest.approx(500.0)


def test_the_rate_table_is_dated_sourced_and_seller_side_throughout():
    t = rate_history_table()
    assert list(t["from"]) == sorted(t["from"])
    assert set(t["sides"]) == {"S"}
    assert all(len(a) > 20 for a in t["authority"])
    assert len(STAMP_DUTY_HISTORY) == 3


# ------------------------------------------------------------------ dividends
@pytest.mark.parametrize("buy,sell,bucket,rate", [
    ("2024-01-10", "2024-01-10", "<=1m", 0.20),
    ("2024-01-10", "2024-02-10", "<=1m", 0.20),    # exactly one month: still 20%
    ("2024-01-10", "2024-02-11", "1m-1y", 0.10),
    ("2024-01-10", "2025-01-10", "1m-1y", 0.10),   # exactly one year: still 10%
    ("2024-01-10", "2025-01-11", ">1y", 0.00),
    ("2024-01-31", "2024-02-29", "<=1m", 0.20),    # calendar months, not 30 days
])
def test_the_dividend_step_function_and_its_inclusive_boundaries(buy, sell, bucket, rate):
    assert holding_bucket(buy, sell) == bucket
    assert dividend_tax_rate(buy, sell) == pytest.approx(rate)
    assert dividend_tax(10_000.0, buy, sell) == pytest.approx(10_000.0 * rate)


def test_the_effective_rates_are_the_statutory_rate_times_the_inclusion():
    assert DIVIDEND_EFFECTIVE_RATE == {"<=1m": 0.20, "1m-1y": 0.10, ">1y": 0.00}


def test_record_dates_before_the_2015_circular_are_refused():
    with pytest.raises(ValueError, match="2015-09-08"):
        dividend_tax_rate("2015-01-02", "2015-03-02", record_date="2015-06-01")
    assert dividend_tax_rate("2015-01-02", "2015-03-02",
                             record_date="2015-09-08") == pytest.approx(0.10)


def test_a_sale_before_the_purchase_is_refused():
    with pytest.raises(ValueError, match="before buy_date"):
        holding_bucket("2024-06-01", "2024-01-01")


# ------------------------------------------------------------------ the measured study
@pytest.fixture(scope="module")
def path():
    return make_ashare_path()


def test_the_seeded_path_spans_the_rate_cut(path):
    assert path.index[0] == pd.Timestamp("2021-01-04")
    assert path.index[0] < pd.Timestamp("2023-08-28") < path.index[-1]


def test_a_flat_symmetric_model_overstates_the_duty_and_charges_the_wrong_side(path):
    b = make_turnover_blotter(path, 21)
    c = compare_cost_models(b)
    assert c["n_trades"] == 117
    assert c["flat_symmetric"] == pytest.approx(203_666.73, abs=0.01)
    assert c["real_stamp_duty"] == pytest.approx(77_042.06, abs=0.5)
    assert c["ratio"] == pytest.approx(2.64, abs=0.01)
    assert c["real_bps_of_notional"] == pytest.approx(3.78, abs=0.01)
    # The buy leg carries the whole overstatement plus the halving.
    buys = b.loc[b["side"] == "B"]
    assert real_stamp_duty(buys) == 0.0
    assert flat_symmetric_cost(buys) > 100_000.0


def test_the_same_blotter_pays_10bps_before_the_cut_and_5bps_after(path):
    b = make_turnover_blotter(path, 21)
    cut = pd.Timestamp("2023-08-28")
    for part, expected in ((b.loc[b["date"] < cut], 10.0), (b.loc[b["date"] >= cut], 5.0)):
        c = compare_cost_models(part)
        assert 1e4 * c["real_stamp_duty"] / c["sell_notional"] == pytest.approx(expected,
                                                                                abs=1e-6)


def test_buy_and_hold_pays_no_duty_at_all(path):
    c = compare_cost_models(make_turnover_blotter(path, 0))
    assert c["n_trades"] == 1 and c["real_stamp_duty"] == 0.0
    assert not np.isfinite(c["ratio"])


@pytest.mark.parametrize("hold,rate,drag", [(5, 0.20, 48.1), (21, 0.20, 48.1),
                                            (63, 0.10, 24.1), (244, 0.10, 24.1),
                                            (0, 0.00, 0.0)])
def test_the_dividend_turnover_penalty(path, hold, rate, drag):
    d = dividend_drag(path, hold)
    assert d["gross_dividend"] == pytest.approx(234_380.0, abs=1.0)
    assert d["effective_rate"] == pytest.approx(rate, abs=1e-9)
    assert d["drag_bps_pa"] == pytest.approx(drag, abs=0.1)


def test_the_dividend_is_the_same_cash_at_every_turnover_level(path):
    gross = {dividend_drag(path, h)["gross_dividend"] for h in (5, 21, 63, 244, 0)}
    assert len(gross) == 1          # same shares, same dividends; only the tax differs


def test_seeded_runs_are_deterministic():
    pd.testing.assert_series_equal(make_ashare_path(), make_ashare_path())
    a = compare_cost_models(make_turnover_blotter(make_ashare_path(), 21))
    b = compare_cost_models(make_turnover_blotter(make_ashare_path(), 21))
    assert a == b


def test_demo_prints_the_rule_and_the_boundary(run_main):
    out = run_main("fin_skills.tax_accounting.ashare_taxes")
    assert "a BUY pays 0.00" in out
    assert "the boundary is one day wide" in out
    assert "RULE: A-share stamp duty is seller-side only" in out
