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


def test_satellite_lifecycle_manager_holding_and_exits(tmp_path):
    """Test 5: Verify SatelliteLifecycleManager tracks positions, audits T+5 horizon, take-profit, stop-loss, and closes trades."""
    from fin_skills.china.core_satellite_advisor import (
        SatelliteLifecycleManager,
        SatellitePositionRecord,
    )

    mgr = SatelliteLifecycleManager(ledger_path=tmp_path / "test_satellite_ledger.json")

    # 1. Add positions
    t1 = SatelliteAlphaTicket(
        symbol="SZ300760",
        name="迈瑞医疗",
        tier="TIER_S_HIGH_PREDICTABILITY",
        kol_trigger_summary="阿尔法工场看多",
        kol_weighted_sentiment=0.88,
        action="BUY",
        shares_to_buy=200,
        entry_price=250.0,
        target_holding_days=5,
        stop_loss_pct=-0.05,
        take_profit_pct=0.08,
        rationale="低估值蓝筹反弹",
    )
    t2 = SatelliteAlphaTicket(
        symbol="SH600036",
        name="招商银行",
        tier="TIER_S_HIGH_PREDICTABILITY",
        kol_trigger_summary="大V共振看多",
        kol_weighted_sentiment=0.85,
        action="BUY",
        shares_to_buy=1000,
        entry_price=35.0,
        target_holding_days=5,
        stop_loss_pct=-0.05,
        take_profit_pct=0.08,
        rationale="高股息价值中枢",
    )
    t3 = SatelliteAlphaTicket(
        symbol="SZ000963",
        name="华东医药",
        tier="TIER_S_HIGH_PREDICTABILITY",
        kol_trigger_summary="医药大V研报催化",
        kol_weighted_sentiment=0.82,
        action="BUY",
        shares_to_buy=500,
        entry_price=30.0,
        target_holding_days=5,
        stop_loss_pct=-0.05,
        take_profit_pct=0.08,
        rationale="创新药放量",
    )

    mgr.add_ticket(t1, entry_date="2026-09-08")
    mgr.add_ticket(t2, entry_date="2026-09-10")
    mgr.add_ticket(t3, entry_date="2026-09-12")

    assert len(mgr.positions) == 3

    # 2. Audit scenario on 2026-09-16:
    # - SZ300760 (entry 2026-09-08): holding days = 6 >= 5 -> trigger EXIT_TARGET_HORIZON_REACHED
    # - SH600036 (entry 2026-09-10, price 38.0): return = (38-35)/35 = +8.57% >= +8% -> trigger EXIT_TAKE_PROFIT
    # - SZ000963 (entry 2026-09-12, price 28.0): return = (28-30)/30 = -6.67% <= -5% -> trigger EXIT_STOP_LOSS
    prices = {
        "SZ300760": 252.0,  # +0.8%, but holding days = 6 >= 5
        "SH600036": 38.0,   # +8.57% (Take profit)
        "SZ000963": 28.0,   # -6.67% (Stop loss)
    }

    active, exits = mgr.audit_positions(prices, today_date="2026-09-16")
    assert len(active) == 0
    assert len(exits) == 3

    exit_reasons = {e.symbol: e.lifecycle_status for e in exits}
    assert exit_reasons["SZ300760"] == "EXIT_TARGET_HORIZON_REACHED"
    assert exit_reasons["SH600036"] == "EXIT_TAKE_PROFIT"
    assert exit_reasons["SZ000963"] == "EXIT_STOP_LOSS"

    # Close positions in ledger
    for e in exits:
        mgr.close_position(
            symbol=e.symbol,
            exit_date="2026-09-16",
            exit_price=prices[e.symbol],
            reason=e.exit_reason,
        )

    assert len(mgr.positions) == 0
    assert len(mgr.history) == 3
    # Check realized PnL
    hist_map = {h["symbol"]: h for h in mgr.history}
    assert hist_map["SH600036"]["realized_pnl_pct"] == pytest.approx(0.0857, abs=1e-3)
    assert hist_map["SZ000963"]["realized_pnl_pct"] == pytest.approx(-0.0667, abs=1e-3)

    # 3. Test In-Progress Holding Status (Holding < 5 days, within [-5%, +8%])
    mgr.add_ticket(t1, entry_date="2026-09-15")
    active_now, exits_now = mgr.audit_positions({"SZ300760": 255.0}, today_date="2026-09-16")
    assert len(active_now) == 1
    assert len(exits_now) == 0
    assert active_now[0].lifecycle_status == "HOLD_IN_PROGRESS"
    assert active_now[0].action == "HOLD"
    assert active_now[0].current_holding_days == 1
    assert active_now[0].unrealized_pnl_pct == pytest.approx(0.02, abs=1e-4)

    # 4. Test Serialization / Deserialization
    state_dict = mgr.to_dict()
    mgr_restored = SatelliteLifecycleManager.from_dict(state_dict, ledger_path=tmp_path / "test_restored.json")
    assert len(mgr_restored.positions) == 1
    assert "SZ300760" in mgr_restored.positions
    assert len(mgr_restored.history) == 3



def test_core_satellite_volatility_regime_shield_integration():
    """Test 6: Verify VolatilityRegimeShield shifts Core-Satellite budget dynamically."""
    core_base = {
        "510300": 0.20,
        "510880": 0.20,
        "511010": 0.40,
        "518880": 0.20,
    }
    candidates = [
        {"symbol": "SZ300760", "name": "迈瑞医疗", "kol_weighted_sentiment": 0.88},
        {"symbol": "SH600036", "name": "招商银行", "kol_weighted_sentiment": 0.85},
        {"symbol": "SZ000963", "name": "华东医药", "kol_weighted_sentiment": 0.82},
    ]

    # 1. Elevated Volatility Stress (> 80th percentile) -> Core 90% / Satellite 10%
    plan_stress = generate_core_satellite_plan(
        total_capital=200000.0,
        core_base_weights=core_base,
        candidate_stock_signals=candidates,
        market_volatility_metrics={"vol_60d": 0.26, "vol_percentile": 0.85},
    )
    assert plan_stress.volatility_regime == "ELEVATED_VOL_STRESS"
    assert plan_stress.core_weight_budget == pytest.approx(0.90, abs=1e-4)
    assert plan_stress.satellite_weight_budget == pytest.approx(0.10, abs=1e-4)
    assert sum(plan_stress.satellite_stock_weights.values()) == pytest.approx(0.10, abs=1e-4)
    assert plan_stress.total_weight_sum == pytest.approx(1.0, abs=1e-6)

    # 2. Extreme Panic Freeze (> 95th percentile or liquidity freeze) -> Core 95% / Satellite 0% (5% cash)
    plan_panic = generate_core_satellite_plan(
        total_capital=200000.0,
        core_base_weights=core_base,
        candidate_stock_signals=candidates,
        market_volatility_metrics={"vol_60d": 0.38, "vol_percentile": 0.98},
    )
    assert plan_panic.volatility_regime == "EXTREME_PANIC_FREEZE"
    assert plan_panic.core_weight_budget == pytest.approx(0.95, abs=1e-4)
    assert plan_panic.satellite_weight_budget == pytest.approx(0.00, abs=1e-4)
    assert plan_panic.cash_defensive_budget == pytest.approx(0.05, abs=1e-4)
    assert len(plan_panic.satellite_tickets) == 0
    assert len(plan_panic.satellite_stock_weights) == 0
    assert plan_panic.total_weight_sum == pytest.approx(1.0, abs=1e-6)


def test_core_satellite_qdii_smart_router_integration():
    """Test 7: Verify QDIISmartRouter intercepts bubbly QDII ETFs inside generate_core_satellite_plan."""
    core_base = {
        "510300": 0.20,
        "511010": 0.30,
        "513100": 0.15,
        "513500": 0.15,
    }
    # 513100 is at 3.5% premium -> should substitute to 159509 (0.2%)
    # 513500 is at 4.0% premium, all substitutes bubbly -> should fallback to 511010
    qdii_premiums = {
        "513100": 0.035,
        "159509": 0.002,
        "513870": 0.012,
        "513500": 0.040,
        "513520": 0.025,
        "511010": 0.000,
    }
    plan = generate_core_satellite_plan(
        total_capital=100000.0,
        core_base_weights=core_base,
        candidate_stock_signals=[],
        qdii_current_premiums=qdii_premiums,
    )

    # 513100 should be routed to 159509
    assert "513100" not in plan.core_etf_weights
    assert "159509" in plan.core_etf_weights
    # 513500 should be routed to 511010
    assert "513500" not in plan.core_etf_weights
    assert len(plan.qdii_substitution_events) == 2
    assert plan.total_weight_sum == pytest.approx(1.0, abs=1e-6)

    # Verify Markdown rendering includes both new status cards
    md = plan.to_markdown()
    assert "🛡️ 宏观波动率自适应防爆盾状态" in md
    assert "🔄 QDII 智能溢价平替与防泡沫调度" in md
    assert "159509" in md
    assert "避免摩擦损耗" in md or "避免追高" in md
