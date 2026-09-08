"""fin_skills.futures_fx.fx_conventions - quote direction, pip size and carry."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fin_skills.futures_fx.fx_conventions import (_synthetic_carry_pair, carry_return,
                                                  check_convention, is_inverted,
                                                  notional_for_pip_risk, parse_pair, pip_size,
                                                  pip_value, render_carry, total_return)


def test_parse_pair_accepts_the_common_spellings():
    for spelling in ("EURUSD", "EUR/USD", "eurusd", "EUR-USD", "eur_usd", " EURUSD "):
        assert parse_pair(spelling) == ("EUR", "USD")
    for bad in ("EURUS", "GBP/US", "EUR USD X", "EUR123"):
        with pytest.raises(ValueError, match="cannot parse"):
            parse_pair(bad)


def test_check_convention_catches_a_backwards_feed():
    with pytest.raises(ValueError, match="quoted backwards"):
        check_convention("JPYUSD")
    with pytest.raises(ValueError, match="USDEUR is quoted backwards"):
        check_convention("USDEUR")
    for ok in ("EURUSD", "USDJPY", "GBPJPY", "EURGBP", "USDXYZ"):
        assert check_convention(ok) is None


def test_is_inverted_means_usd_is_the_base():
    assert is_inverted("USDJPY") and is_inverted("USDCHF") and is_inverted("USDCAD")
    assert not is_inverted("EURUSD") and not is_inverted("AUDUSD")
    with pytest.raises(ValueError, match="no USD leg"):
        is_inverted("EURGBP")


def test_pip_size_is_0_01_on_jpy_quotes_and_0_0001_elsewhere():
    assert pip_size("USDJPY") == 0.01
    assert pip_size("EUR/JPY") == 0.01
    assert pip_size("EURUSD") == 0.0001
    assert pip_size("JPYUSD") == 0.0001          # the quote currency decides, not membership


def test_pip_value_converts_the_foreign_pip_back_to_dollars():
    jpy = pip_value("USDJPY", 100_000, 150.25)
    assert jpy.pip_size == 0.01 and jpy.value_quote == 1_000.0 and jpy.quote_ccy == "JPY"
    assert jpy.value_usd == pytest.approx(100_000 * 0.01 / 150.25)      # 6.66, not 0.067
    eur = pip_value("EURUSD", 100_000, 1.0850)
    assert eur.value_usd == 10.0 and eur.value_quote == 10.0
    assert math.isnan(pip_value("EURGBP", 100_000, 0.85).value_usd)   # a cross needs a third leg
    with pytest.raises(ValueError, match="positive"):
        pip_value("EURUSD", 1e5, 0.0)


def test_notional_for_pip_risk_round_trips_and_the_4dp_bug_is_100x():
    n = notional_for_pip_risk("USDJPY", 10.0, 150.25)
    assert pip_value("USDJPY", n, 150.25).value_usd == pytest.approx(10.0)
    n_bug = 10.0 / (0.0001 / 150.25)              # sizing with the default pip on yen
    assert n_bug / n == pytest.approx(100.0)
    with pytest.raises(ValueError, match="cross"):
        notional_for_pip_risk("EURGBP", 10.0, 0.85)


def test_carry_return_and_forward_points_have_opposite_signs():
    c = carry_return(0.98, 0.0475, 0.0025, 365, pair="AUDUSD")
    assert c.ret > 0 and c.points < 0 and c.pips == pytest.approx(c.points / 1e-4)
    assert c.forward == pytest.approx(0.98 * (1 + 0.0025 * 365 / 360) / (1 + 0.0475))
    low_yield_base = carry_return(150.0, 0.0, 0.05, 365, pair="USDJPY")
    assert low_yield_base.ret < 0 and low_yield_base.points > 0
    assert low_yield_base.pips == pytest.approx(low_yield_base.points / 0.01)
    with pytest.raises(ValueError):
        carry_return(1.0, 0.05, 0.0, -1)


def test_day_count_basis_follows_the_deposit_currency():
    assert carry_return(1.0, 0.05, 0.0, 365, pair="AUDUSD").ret == pytest.approx(0.05)       # ACT/365
    assert carry_return(1.0, 0.05, 0.0, 365).ret == pytest.approx(0.05 * 365 / 360)          # ACT/360 default
    assert carry_return(1.0, 0.05, 0.0, 360, pair="EURUSD").ret == pytest.approx(0.05)


def test_total_return_accrues_carry_on_calendar_days():
    idx = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])  # Thu Fri Mon Tue
    tr = total_return(pd.Series(1.0, index=idx), 0.05, 0.01, pair="AUDUSD")
    assert tr["calendar_days"].tolist() == [1.0, 3.0, 1.0]
    assert tr["spot_ret"].tolist() == [0.0, 0.0, 0.0]
    for days, got in zip((1, 3, 1), tr["carry_ret"]):
        assert got == pytest.approx(carry_return(1.0, 0.05, 0.01, days, pair="AUDUSD").ret)
    pd.testing.assert_series_equal(tr["total_ret"], tr["carry_ret"], check_names=False)
    moving = total_return(pd.Series([1.0, 1.01], index=idx[:2]), 0.05, 0.01)
    assert moving["total_ret"].iloc[0] == pytest.approx(1.01 * (1 + moving["carry_ret"].iloc[0]) - 1)


def test_total_return_input_checks():
    with pytest.raises(TypeError, match="DatetimeIndex"):
        total_return(pd.Series([1.0, 1.0]), 0.05, 0.01)
    idx = pd.bdate_range("2024-01-01", periods=3)
    with pytest.raises(ValueError, match="sorted"):
        total_return(pd.Series([1.0, 1.0, 1.0], index=idx[::-1]), 0.05, 0.01)
    with pytest.raises(ValueError, match="positive"):
        total_return(pd.Series([1.0, 0.0, 1.0], index=idx), 0.05, 0.01)


def test_carry_trade_is_negative_on_spot_and_positive_in_total():
    spot = _synthetic_carry_pair()
    pd.testing.assert_series_equal(spot, _synthetic_carry_pair())        # seeded
    tr = total_return(spot, r_base=0.0475, r_quote=0.0025, pair="AUDUSD")
    assert (1 + tr["spot_ret"]).prod() < 1.0 < (1 + tr["total_ret"]).prod()
    text = render_carry(tr, "AUDUSD", 0.0475, 0.0025)
    assert "LOSER" in text and "winner" in text and "n/m" in text        # carry Sharpe refused


def test_demo_prints_the_rules(run_main):
    out = run_main("fin_skills.futures_fx.fx_conventions")
    assert "NEVER backtest FX on spot alone" in out and "size is 100x" in out
