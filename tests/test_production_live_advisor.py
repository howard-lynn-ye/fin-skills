"""Tests for research.production.live_advisor_bot."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.production.live_advisor_bot import (
    compute_target_weights,
    run_live_advisor,
)


def test_compute_target_weights_profiles():
    fake_market = {
        "510300": {"vol_60d": 0.18},
        "510500": {"vol_60d": 0.22},
        "510880": {"vol_60d": 0.12},
        "518880": {"vol_60d": 0.14},
        "511010": {"vol_60d": 0.04},  # Low vol -> should get high weight in risk parity
        "513100": {"vol_60d": 0.24},
        "513500": {"vol_60d": 0.16},
        "510900": {"vol_60d": 0.20},
    }
    # Conservative (Risk Parity)
    w_cons = compute_target_weights("conservative", fake_market)
    assert pytest.approx(sum(w_cons.values()), 0.001) == 1.0
    assert w_cons["511010"] >= 0.38  # Bond allocation heavily prioritized

    # Balanced (All Weather)
    w_bal = compute_target_weights("balanced", fake_market)
    assert pytest.approx(sum(w_bal.values()), 0.001) == 1.0
    assert w_bal["511010"] == 0.40
    assert w_bal["518880"] == 0.15


def test_run_live_advisor_end_to_end(tmp_path: Path):
    h_file = tmp_path / "test_advisor_holdings.json"
    result = run_live_advisor(
        holdings_path=h_file,
        profile="conservative",
        inflow=0.0,
        force_rebalance=False,
        webhook_url=None,
        confirm_execution=True,  # Test applying trades
        initial_capital_if_empty=80000.0,
    )
    assert "state" in result
    assert "plan" in result
    assert "core_satellite_plan" in result
    assert "cash_plan" in result
    assert "markdown" in result
    assert result["state"].total_nav > 0
    assert "## 🌐 全球大类资产自适应全天候" in result["markdown"]
    assert "🎯 卫星增强仓 (20% Tier-S 个股大V事件 Alpha 狙击单 - T+5 策略)" in result["markdown"]
    assert h_file.exists()
