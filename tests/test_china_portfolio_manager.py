"""Tests for fin_skills.china.portfolio_manager (PortfolioState, DeadbandRebalancer)."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from fin_skills.china.portfolio_manager import (
    HoldingRecord,
    PortfolioState,
    calculate_trade_fee,
    plan_portfolio_rebalance,
)


def test_holding_record_valuation():
    h = HoldingRecord(code="510300", name="沪深300ETF", shares=1000, avg_cost=4.0, market_price=4.5)
    h.update_valuation(current_price=4.5, total_nav=45000.0)
    assert h.market_value == 4500.0
    assert pytest.approx(h.weight) == 0.10
    assert h.unrealized_pnl == 500.0
    assert pytest.approx(h.unrealized_pnl_pct) == 0.125


def test_portfolio_state_serialization(tmp_path: Path):
    state = PortfolioState(cash=5000.0, last_rebalance_date="2026-06-01", rebalance_count=2)
    state.holdings["518880"] = HoldingRecord(
        code="518880", name="黄金ETF", shares=500, avg_cost=6.0, market_price=6.5
    )
    save_file = tmp_path / "test_portfolio.json"
    state.save_to_file(save_file)

    loaded = PortfolioState.load_from_file(save_file)
    assert loaded.cash == 5000.0
    assert loaded.rebalance_count == 2
    assert "518880" in loaded.holdings
    assert loaded.holdings["518880"].shares == 500


def test_trade_fee_calculation():
    # Min commission 2 RMB
    assert calculate_trade_fee(1000.0) == 2.0
    # 0.02% (2 bps) above min
    assert pytest.approx(calculate_trade_fee(50000.0)) == 10.0


def test_deadband_holding():
    state = PortfolioState(cash=200.0)
    # Weights closely aligned to 20% and 80%
    state.holdings["510300"] = HoldingRecord(code="510300", name="HS300", shares=2500, avg_cost=4.0, market_price=4.0) # 10,000 (19.6%)
    state.holdings["511010"] = HoldingRecord(code="511010", name="Bond", shares=300, avg_cost=136.0, market_price=136.0) # 40,800 (80.0%)
    state.recalculate()

    targets = {"510300": 0.20, "511010": 0.80}
    prices = {"510300": 4.0, "511010": 136.0}

    plan = plan_portfolio_rebalance(state, targets, prices, new_cash_inflow=0.0)
    assert plan.decision == "HOLD_WITHIN_DEADBAND"
    assert len(plan.trade_tickets) == 0


def test_inflow_only_balancing():
    state = PortfolioState(cash=0.0)
    state.holdings["510300"] = HoldingRecord(code="510300", name="HS300", shares=5000, avg_cost=4.0, market_price=4.0) # 20,000 RMB
    state.holdings["518880"] = HoldingRecord(code="518880", name="Gold", shares=1000, avg_cost=6.0, market_price=6.0) # 6,000 RMB
    state.recalculate()

    # Target 50/50: gold is significantly underweight (6k vs 20k)
    targets = {"510300": 0.50, "518880": 0.50}
    prices = {"510300": 4.0, "518880": 6.0}

    # Deposit 14,000 RMB salary savings
    plan = plan_portfolio_rebalance(state, targets, prices, new_cash_inflow=14000.0)
    assert plan.decision == "INFLOW_ONLY_BALANCING"
    # Every trade ticket must be BUY, zero SELL!
    for t in plan.trade_tickets:
        assert t.side == "BUY"
    assert any(t.code == "518880" for t in plan.trade_tickets)
