"""Unit tests for fin_skills.china.core_satellite_advisor (Core-Satellite 80/20 Unified Cockpit)."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fin_skills.china.core_satellite_advisor import (
    CoreSatellitePlan,
    SatelliteAlphaTicket,
    generate_core_satellite_plan,
)


def test_core_satellite_80_20_budget_allocation_and_sum_to_one_invariant():
    """Test 1: Verify 80/20 budget allocation and strict sum-to-1.0 invariant across all scenarios."""
    core_base = {
        "510300": 0.15,
        "510880": 0.15,
        "511010": 0.40,
        "518880": 0.15,
        "513500": 0.075,
        "513100": 0.075,
    }

    # Scenario A: Full 4 Tier-S stocks (4 * 5% = 20% satellite budget utilized)
    candidates_full = [
        {"symbol": "SZ300760", "name": "迈瑞医疗", "kol_weighted_sentiment": 0.88},
        {"symbol": "SH600036", "name": "招商银行", "kol_weighted_sentiment": 0.85},
        {"symbol": "SZ000963", "name": "华东医药", "kol_weighted_sentiment": 0.82},
        {"symbol": "BEKE", "name": "贝壳", "kol_weighted_sentiment": 0.86},
    ]
    plan_full = generate_core_satellite_plan(
        total_capital=200000.0,
        core_base_weights=core_base,
        candidate_stock_signals=candidates_full,
    )

    assert plan_full.core_weight_budget == 0.80
    assert plan_full.satellite_weight_budget == 0.20
    assert pytest.approx(sum(plan_full.core_etf_weights.values()), abs=1e-6) == 0.80
    assert pytest.approx(sum(plan_full.satellite_stock_weights.values()), abs=1e-6) == 0.20
    # Strict sum-to-1.0 invariant
    total_w = sum(plan_full.core_etf_weights.values()) + sum(plan_full.satellite_stock_weights.values())
    assert total_w == pytest.approx(1.0, abs=1e-6)
    assert plan_full.total_weight_sum == pytest.approx(1.0, abs=1e-6)

    # Every single stock weight must be strictly capped at <= 5% (0.05)
    for sym, w in plan_full.satellite_stock_weights.items():
        assert w <= 0.05, f"Stock {sym} weight {w} exceeds 5% concentration cap"


def test_tier_s_stock_amplification_and_t5_ticket_generation():
    """Test 2: Verify Tier S stock amplification (1.2x vs Tier A 1.0x) and T+5 ticket generation."""
    core_base = {"511010": 0.50, "510300": 0.50}
    # Pass one Tier-S stock (SZ300760) and one Tier-A stock (NVDA) with identical base_weight=0.04
    candidates = [
        {
            "symbol": "SZ300760",
            "name": "迈瑞医疗",
            "base_weight": 0.04,
            "posts": [
                {
                    "author": "阿尔法工场",
                    "text": "迈瑞医疗PE仅22倍处于近10年15%分位数，订单超预期增长25%，安全边际极高，建议买入。",
                    "polarity": 0.90,
                    "verified": True,
                },
                {
                    "author": "钟华守正出奇",
                    "text": "医疗集采彻底没戏了，赶紧割肉清仓止损！",
                    "polarity": -0.85,
                    "verified": True,
                },
            ],
        },
        {
            "symbol": "NVDA",
            "name": "英伟达 (NVIDIA)",
            "base_weight": 0.04,
            "kol_weighted_sentiment": 0.80,
            "kol_trigger_summary": "海外大V看多共振",
        },
    ]

    plan = generate_core_satellite_plan(
        total_capital=100000.0,
        core_base_weights=core_base,
        candidate_stock_signals=candidates,
        market_prices={"SZ300760": 250.0, "NVDA": 100.0},
    )

    # Tier S (1.2x multiplier) should be amplified relative to Tier A (1.0x multiplier)
    w_tier_s = plan.satellite_stock_weights["SZ300760"]
    w_tier_a = plan.satellite_stock_weights["NVDA"]
    assert w_tier_s == pytest.approx(0.04 * 1.20, abs=1e-4)  # 0.048 (4.8%)
    assert w_tier_a == pytest.approx(0.04 * 1.00, abs=1e-4)  # 0.040 (4.0%)
    assert w_tier_s > w_tier_a

    # Verify SatelliteAlphaTicket fields and T+5 holding strategy
    ticket_map = {t.symbol: t for t in plan.satellite_tickets}
    assert "SZ300760" in ticket_map
    mindray_ticket = ticket_map["SZ300760"]
    assert isinstance(mindray_ticket, SatelliteAlphaTicket)
    assert mindray_ticket.tier == "TIER_S_HIGH_PREDICTABILITY"
    assert mindray_ticket.target_holding_days == 5
    assert mindray_ticket.action == "BUY"
    assert mindray_ticket.shares_to_buy > 0
    assert mindray_ticket.entry_price == 250.0
    assert mindray_ticket.stop_loss_pct < 0
    assert mindray_ticket.take_profit_pct > 0
    assert "阿尔法工场" in mindray_ticket.kol_trigger_summary or mindray_ticket.kol_weighted_sentiment > 0.5


def test_unused_satellite_budget_fallback_to_bond_etf_511010():
    """Test 3: Verify unused satellite budget automatically parks in Treasury Bond ETF 511010."""
    core_base = {
        "510300": 0.50,
        "511010": 0.50,
    }
    # Base core weights when scaled to 80%: 510300 = 0.40, 511010 = 0.40

    # Case 1: Only 1 Tier-S stock taking 5% weight -> 15% unused satellite budget returns to 511010
    plan_partial = generate_core_satellite_plan(
        total_capital=100000.0,
        core_base_weights=core_base,
        candidate_stock_signals=[
            {"symbol": "SH600036", "name": "招商银行", "kol_weighted_sentiment": 0.85}
        ],
    )
    assert plan_partial.satellite_stock_weights["SH600036"] == pytest.approx(0.05, abs=1e-4)
    assert plan_partial.unused_satellite_budget == pytest.approx(0.15, abs=1e-4)
    # 511010 should receive its 0.40 base core share + 0.15 unused satellite budget = 0.55
    assert plan_partial.core_etf_weights["511010"] == pytest.approx(0.40 + 0.15, abs=1e-4)
    assert plan_partial.core_etf_weights["510300"] == pytest.approx(0.40, abs=1e-4)
    assert plan_partial.total_weight_sum == pytest.approx(1.0, abs=1e-6)

    # Case 2: Empty candidate list -> 0% satellite used -> full 20% returns to 511010
    plan_empty = generate_core_satellite_plan(
        total_capital=100000.0,
        core_base_weights=core_base,
        candidate_stock_signals=[],
    )
    assert len(plan_empty.satellite_tickets) == 0
    assert sum(plan_empty.satellite_stock_weights.values()) == 0.0
    assert plan_empty.unused_satellite_budget == pytest.approx(0.20, abs=1e-4)
    assert plan_empty.core_etf_weights["511010"] == pytest.approx(0.40 + 0.20, abs=1e-4)
    assert plan_empty.total_weight_sum == pytest.approx(1.0, abs=1e-6)


def test_tier_c_noisy_stock_interception_and_etf_substitution():
    """Test 4: Verify Tier C noisy stocks (e.g. 02015 Li Auto, 09868 XPeng, SZ300014) are intercepted and rerouted."""
    core_base = {"511010": 0.50, "510300": 0.50}
    candidates = [
        {"symbol": "02015", "name": "理想汽车-W", "raw_signal": 0.90, "kol_weighted_sentiment": 0.90},
        {"symbol": "09868", "name": "小鹏汽车-W", "raw_signal": 0.85, "kol_weighted_sentiment": 0.85},
        {"symbol": "SZ300014", "name": "亿纬锂能", "raw_signal": 0.80, "kol_weighted_sentiment": 0.80},
        {"symbol": "SZ300760", "name": "迈瑞医疗", "raw_signal": 0.88, "kol_weighted_sentiment": 0.88},
    ]

    plan = generate_core_satellite_plan(
        total_capital=100000.0,
        core_base_weights=core_base,
        candidate_stock_signals=candidates,
    )

    # Tier C stocks must NEVER enter satellite_stock_weights or satellite_tickets
    assert "02015" not in plan.satellite_stock_weights
    assert "09868" not in plan.satellite_stock_weights
    assert "SZ300014" not in plan.satellite_stock_weights
    assert "SZ300760" in plan.satellite_stock_weights

    # Verify blocked_noisy_stocks records and ETF substitution targets
    blocked_map = {b["symbol"]: b for b in plan.blocked_noisy_stocks}
    assert "02015" in blocked_map
    assert "09868" in blocked_map
    assert "SZ300014" in blocked_map

    # HK/ADR noisy EV stocks rerouted to Hang Seng ETF 510900
    assert blocked_map["02015"]["etf_substitute"] == "510900"
    assert blocked_map["09868"]["etf_substitute"] == "510900"
    # ChiNext / A-share noisy stock rerouted to CSI 500 ETF 510500
    assert blocked_map["SZ300014"]["etf_substitute"] == "510500"

    # Markdown rendering should display both active tickets and blocked noisy stocks
    md = plan.to_markdown()
    assert "🎯 卫星增强仓 (20% Tier-S 个股大V事件 Alpha 狙击单 - T+5 策略)" in md
    assert "迈瑞医疗" in md
    assert "02015" in md
    assert "510900" in md
