"""Tests for fin_skills.china.core_satellite_advisor."""
from __future__ import annotations

import pytest
from fin_skills.china.core_satellite_advisor import (
    CoreSatellitePlan,
    SatelliteAlphaTicket,
    generate_core_satellite_plan,
)


def test_core_satellite_advisor_module():
    core_base = {"510300": 0.50, "511010": 0.50}
    plan = generate_core_satellite_plan(
        total_capital=100000.0,
        core_base_weights=core_base,
        candidate_stock_signals=[
            {"symbol": "SZ300760", "name": "迈瑞医疗", "kol_weighted_sentiment": 0.88},
            {"symbol": "02015", "name": "理想汽车-W", "kol_weighted_sentiment": 0.85},
        ],
    )
    assert isinstance(plan, CoreSatellitePlan)
    assert plan.total_weight_sum == pytest.approx(1.0, abs=1e-6)
    assert "SZ300760" in plan.satellite_stock_weights
    assert "02015" not in plan.satellite_stock_weights
    assert any(b["symbol"] == "02015" for b in plan.blocked_noisy_stocks)
