"""fin_skills.core.cost_curve - a backtest as a cost-sensitivity curve, and its breakeven."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.core.cost_curve import (breakeven_bps, cost_curve, max_drawdown, net_returns,
                                        render)


@pytest.fixture
def noisy():
    rng = np.random.default_rng(11)
    n = 252 * 2
    gross = pd.Series(rng.normal(0.00065, 0.006, n), index=pd.bdate_range("2020-01-01", periods=n))
    turn = pd.Series(np.clip(rng.lognormal(np.log(0.35), 0.30, n), 0.0, 2.0), index=gross.index)
    return gross, turn


def test_net_returns_apply_the_documented_round_trip_convention():
    r = pd.Series([0.01, -0.02])
    assert net_returns(r, 0.5, 10).tolist() == pytest.approx([0.01 - 0.0005, -0.02 - 0.0005])
    assert net_returns(r, [1.0, 0.0], 20).tolist() == pytest.approx([0.01 - 0.002, -0.02])


def test_curve_is_monotone_in_cost_and_starts_at_the_gross_stats(noisy):
    gross, turn = noisy
    curve = cost_curve(gross, turn, bps=(0, 5, 10, 20, 50))
    assert list(curve.index) == [0.0, 5.0, 10.0, 20.0, 50.0]
    assert curve["ann_return"].is_monotonic_decreasing
    assert curve["sharpe"].is_monotonic_decreasing
    assert curve["total_return"].is_monotonic_decreasing
    assert curve.loc[0.0, "total_return"] == pytest.approx(float((1 + gross).prod() - 1))
    assert curve.attrs["avg_turnover"] == pytest.approx(float(turn.mean()))
    assert curve.attrs["n_periods"] == len(gross)


def test_breakeven_solves_the_constant_case_exactly():
    # r - t * c / 1e4 = 0  ->  c = 20 bps for r = 10 bps/period and 50% one-way turnover
    r = pd.Series(0.001, index=range(500))
    t = pd.Series(0.5, index=range(500))
    assert breakeven_bps(r, t) == pytest.approx(20.0, abs=1e-3)
    assert breakeven_bps(r, t, benchmark=0.0) == pytest.approx(20.0, abs=1e-3)


def test_breakeven_edge_cases(noisy):
    gross, turn = noisy
    assert breakeven_bps(-gross.abs(), turn) == 0.0          # never clears the hurdle
    assert breakeven_bps(pd.Series([0.5] * 10), 1e-9) == 10_000.0   # survives past the bound
    hurdle = breakeven_bps(gross, turn, benchmark=0.05)
    assert hurdle < breakeven_bps(gross, turn)                 # a higher bar breaks even sooner
    bh = breakeven_bps(gross, turn, benchmark=gross * 0.0)     # a flat per-period series
    assert bh == pytest.approx(breakeven_bps(gross, turn), abs=1e-3)


def test_max_drawdown_is_on_the_compounded_curve():
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert max_drawdown(pd.Series([0.01, 0.02])) == 0.0


def test_inputs_are_validated(noisy):
    gross, turn = noisy
    with pytest.raises(ValueError, match="non-negative"):
        cost_curve(gross, -turn)
    with pytest.raises(ValueError, match="NaN"):
        cost_curve(gross.where(gross > -1e9, np.nan).mask(gross.index == gross.index[3]), turn)
    with pytest.raises(ValueError, match="rows"):
        cost_curve(gross, turn.iloc[:-1].to_numpy())


def test_render_reports_the_breakeven(noisy):
    gross, turn = noisy
    text = render(cost_curve(gross, turn), breakeven_bps(gross, turn), title="T")
    assert text.startswith("T") and "BREAKEVEN" in text and "avg one-way turnover" in text


def test_demo_prints_the_curve_and_the_hurdle(run_main):
    out = run_main("fin_skills.core.cost_curve")
    assert "Report the curve, not the peak" in out and "5% cash hurdle" in out
