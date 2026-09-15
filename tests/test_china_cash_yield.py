"""Tests for fin_skills.china.cash_yield_optimizer and CashDragGuard."""
from __future__ import annotations

import pytest

from fin_skills.api import get
from fin_skills.china.cash_yield_optimizer import (
    calculate_cash_drag,
    get_repo_interest_days,
    plan_cash_placement,
)


def test_repo_interest_days_calendar():
    # 2026-09-17 is a Thursday -> 3 days interest!
    assert get_repo_interest_days("2026-09-17") == 3
    # 2026-09-15 is a Tuesday -> 1 day
    assert get_repo_interest_days("2026-09-15") == 1
    # 2026-09-18 is a Friday -> 1 day
    assert get_repo_interest_days("2026-09-18") == 1


def test_calculate_cash_drag():
    # 100,000 RMB idle for a year at 0.30% vs 2.10% benchmark -> 1,800 RMB drag
    drag = calculate_cash_drag(100000.0, demand_rate=0.0030, benchmark_yield=0.0210, holding_days=365)
    assert pytest.approx(drag, 0.01) == 1800.0

    # 0 cash -> 0 drag
    assert calculate_cash_drag(0.0) == 0.0


def test_plan_cash_placement_thresholds():
    # Tiny cash < 100 RMB -> keep in account
    p_tiny = plan_cash_placement(50.0)
    assert p_tiny.recommended_vehicle == "NONE"

    # Between 100 and 1,000 RMB -> Money Market ETF (511880)
    p_mid = plan_cash_placement(600.0)
    assert "511880" in p_mid.recommended_vehicle

    # Large cash >= 1,000 RMB on Thursday -> GC001 with 3-day multiplier
    p_large = plan_cash_placement(20000.0, current_date="2026-09-17")
    assert "204001" in p_large.recommended_vehicle
    assert p_large.interest_days == 3
    assert "Thursday" in p_large.notes


def test_cash_drag_guard_api():
    g = get("cash_drag")
    # Moderate cash ratio (10%) -> passed with warning
    res_warn = g.run(idle_cash=10000.0, total_capital=100000.0, trade_date="2026-09-17")
    assert res_warn.passed
    assert any(f.severity == "warning" for f in res_warn.findings)
    assert "GC001" in res_warn.findings[0].message or "reverse repo" in res_warn.findings[0].message

    # Severe cash ratio (30%) -> fails with error
    res_err = g.run(idle_cash=30000.0, total_capital=100000.0, trade_date="2026-09-17")
    assert not res_err.passed
    assert any(f.severity == "error" for f in res_err.findings)

    # Low cash ratio (2%) -> pass with info
    res_pass = g.run(idle_cash=2000.0, total_capital=100000.0)
    assert res_pass.passed
    assert not any(f.severity in ("warning", "error") for f in res_pass.findings)
