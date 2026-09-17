"""Unit tests for Multi-Channel Webhook CardKit v2 and GC001 Auto-Sweep."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fin_skills.china.core_satellite_advisor import (
    SatelliteAlphaTicket,
    SatellitePositionRecord,
)
from fin_skills.china.portfolio_manager import TradeTicket
from research.production.run_intraday_pipeline import (
    build_feishu_card_v2,
    build_wecom_markdown,
    calculate_gc001_auto_sweep,
    determine_header_theme,
    run_intraday_pipeline,
    send_intraday_webhook,
)
from research.production.webhook_cards import GC001SweepInstruction


def test_calculate_gc001_auto_sweep_large_capital():
    """Test 14:50 GC001 sweep with cash above 1,000 RMB threshold."""
    # 52,345 RMB: should sweep 52,000 RMB (52 lots), remaining 345 RMB parked in 511010
    sweep = calculate_gc001_auto_sweep(
        post_trade_cash=52345.0,
        trade_date="2026-09-16",  # Wednesday -> 1 day
    )

    assert sweep.post_trade_cash == 52345.0
    assert sweep.sweep_amount == 52000.0
    assert sweep.sweep_lots == 52
    assert sweep.residual_cash == 345.0
    assert sweep.interest_days == 1
    assert sweep.is_thursday_multiplier is False
    assert sweep.secondary_vehicle == "511010.SH (国债ETF)"
    assert sweep.secondary_amount == 345.0
    assert sweep.status == "GC001_SWEEP_SCHEDULED"
    assert "204001" in sweep.primary_vehicle


def test_calculate_gc001_auto_sweep_thursday_multiplier():
    """Test Thursday 3-day interest multiplier detection."""
    # 2026-09-17 is a Thursday
    sweep = calculate_gc001_auto_sweep(
        post_trade_cash=30000.0,
        trade_date="2026-09-17",
    )
    assert sweep.is_thursday_multiplier is True
    assert sweep.interest_days == 3
    assert sweep.sweep_lots == 30
    assert "周四3天计息特权" in sweep.notes


def test_calculate_gc001_auto_sweep_sub_thousand_cash():
    """Test sub-1000 cash parks directly in ETF."""
    sweep = calculate_gc001_auto_sweep(
        post_trade_cash=650.0,
        trade_date="2026-09-16",
    )
    assert sweep.status == "ETF_PARKED"
    assert sweep.sweep_amount == 650.0
    assert sweep.sweep_lots == 0
    assert "511010" in sweep.primary_vehicle or "511880" in sweep.primary_vehicle


def test_calculate_gc001_auto_sweep_small_reserve():
    """Test tiny cash balance below 100 RMB stays as broker reserve."""
    sweep = calculate_gc001_auto_sweep(
        post_trade_cash=45.5,
        trade_date="2026-09-16",
    )
    assert sweep.status == "RESERVE_HELD"
    assert sweep.sweep_amount == 0.0
    assert sweep.residual_cash == 45.5
    assert sweep.primary_vehicle == "CASH_RESERVE"


def test_determine_header_theme_dynamic_colors():
    """Verify color selection: Red for Stop Loss, Orange for Defensive, Green for Bullish/Hold."""
    # 1. Stop loss -> Red
    color_sl, badge_sl, _ = determine_header_theme(has_stop_loss=True)
    assert color_sl == "red"
    assert "止损" in badge_sl

    # 2. Defensive risk off -> Orange
    color_def, badge_def, _ = determine_header_theme(
        has_stop_loss=False, is_defensive_risk_off=True
    )
    assert color_def == "orange"
    assert "防御" in badge_def

    # 3. Inflow only balancing -> Green
    color_inf, badge_inf, _ = determine_header_theme(
        has_stop_loss=False, is_defensive_risk_off=False, decision="INFLOW_ONLY_BALANCING"
    )
    assert color_inf == "green"
    assert "定投" in badge_inf

    # 4. Safe hold within deadband -> Green
    color_hold, badge_hold, _ = determine_header_theme(
        has_stop_loss=False, is_defensive_risk_off=False, decision="HOLD_WITHIN_DEADBAND"
    )
    assert color_hold == "green"
    assert "持仓" in badge_hold


def test_feishu_card_v2_payload_structure_and_serialization():
    """Test Feishu CardKit v2 payload conforms to schema and serializes safely to JSON."""
    sweep = calculate_gc001_auto_sweep(post_trade_cash=15000.0, trade_date="2026-09-16")

    # Mock trades
    trade_ticket = TradeTicket(
        code="510300",
        name="沪深300ETF",
        side="BUY",
        lots=12,
        shares=1200,
        price=4.515,
        amount=5418.0,
        est_fee=2.0,
        rationale="核心仓补平低配",
    )

    exit_ticket = SatellitePositionRecord(
        symbol="SZ300760",
        name="迈瑞医疗",
        shares=200,
        entry_price=260.0,
        entry_date="2026-09-01",
        current_price=245.0,
        current_holding_days=15,
        unrealized_pnl_pct=-0.057,
        lifecycle_status="EXIT_STOP_LOSS",
        action="SELL_EXIT",
        exit_reason="触发硬止损，截断个股尾部风险",
    )

    blocked = [
        {
            "symbol": "02015",
            "name": "理想汽车-W",
            "tier": "TIER_C_NOISY_RANDOM_WALK",
            "explanation": "噪声标的禁止个股追涨",
            "etf_substitute": "510900",
            "etf_name": "恒生ETF",
        }
    ]

    card = build_feishu_card_v2(
        today_str="2026-09-16",
        total_nav=100000.0,
        cash=15000.0,
        inflow=5000.0,
        plan_decision="REBALANCE_TRIGGERED",
        decision_reason="偏离度突破阈值",
        core_weight_pct=80.0,
        satellite_weight_pct=15.0,
        cash_weight_pct=5.0,
        has_stop_loss=True,
        is_defensive_risk_off=False,
        is_bullish_rebalance=False,
        exit_tickets=[exit_ticket],
        trade_tickets=[trade_ticket],
        satellite_tickets=[],
        blocked_noisy_stocks=blocked,
        sweep_instruction=sweep,
    )

    # 1. Structure assertions
    assert card["msg_type"] == "interactive"
    card_body = card["card"]
    assert card_body["schema"] == "2.0"
    assert card_body["header"]["template"] == "red"  # Stop loss triggered!
    assert "止损" in card_body["header"]["title"]["content"]

    elements = card_body["body"]["elements"]
    # Check pills
    pills_elem = elements[0]["text"]["content"]
    assert "核心全天候 ETF (80.0%)" in pills_elem
    assert "卫星 T+5 狙击 (15.0%)" in pills_elem
    assert "GC001扫尾/现金 (5.0%)" in pills_elem

    # Check blocked stocks
    blocked_elem = next(e for e in elements if "噪声个股拦截" in e.get("text", {}).get("content", ""))
    assert "02015" in blocked_elem["text"]["content"]
    assert "510900" in blocked_elem["text"]["content"]

    # Check trade table
    trade_elem = next(e for e in elements if "今日建议交易执行清单" in e.get("text", {}).get("content", ""))
    table_content = trade_elem["text"]["content"]
    assert "迈瑞医疗" in table_content
    assert "2手" in table_content
    assert "沪深300ETF" in table_content
    assert "12手" in table_content
    assert "GC001逆回购" in table_content
    assert "15手" in table_content

    # 2. Test JSON serialization
    serialized = json.dumps(card, ensure_ascii=False)
    deserialized = json.loads(serialized)
    assert deserialized["msg_type"] == "interactive"
    assert deserialized["card"]["header"]["template"] == "red"


def test_wecom_markdown_payload_formatting():
    """Test WeChat Work markdown payload formatting, status emojis, and font tags."""
    sweep = calculate_gc001_auto_sweep(post_trade_cash=20000.0, trade_date="2026-09-17")  # Thursday

    wecom = build_wecom_markdown(
        today_str="2026-09-17",
        total_nav=100000.0,
        cash=20000.0,
        inflow=0.0,
        plan_decision="HOLD_WITHIN_DEADBAND",
        decision_reason="资产偏离处于安全死区以内",
        core_weight_pct=80.0,
        satellite_weight_pct=0.0,
        cash_weight_pct=20.0,
        has_stop_loss=False,
        is_defensive_risk_off=False,
        is_bullish_rebalance=False,
        exit_tickets=[],
        trade_tickets=[],
        satellite_tickets=[],
        blocked_noisy_stocks=[],
        sweep_instruction=sweep,
    )

    assert wecom["msg_type"] == "markdown"
    content = wecom["markdown"]["content"]

    assert "🎯 14:25 每日实盘操作指南" in content
    assert "<font color=\"info\">【核心底仓 80.0%】</font>" in content
    assert "GC001 逆回购自动扫尾" in content
    assert "20,000 RMB" in content
    assert "20手" in content
    assert "3天" in content  # Thursday bonus


def test_send_intraday_webhook_dry_run_and_mock():
    """Verify dry_run skips network and mock receives valid JSON payload."""
    feishu = {"msg_type": "interactive", "card": {"header": {"template": "green"}}}
    wecom = {"msg_type": "markdown", "markdown": {"content": "test"}}

    # 1. Test dry-run bypass
    res_dry = send_intraday_webhook(
        webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/dummy",
        feishu_payload=feishu,
        wecom_payload=wecom,
        webhook_type="feishu",
        dry_run=True,
    )
    assert res_dry["sent"] is False
    assert res_dry["dry_run"] is True
    assert res_dry["status"] == "dry_run_simulated"

    # 2. Test live mock dispatch
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        # Feishu
        res_feishu = send_intraday_webhook(
            webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/dummy",
            feishu_payload=feishu,
            wecom_payload=wecom,
            webhook_type="feishu",
            dry_run=False,
        )
        assert res_feishu["sent"] is True
        assert res_feishu["status"] == "success"

        # Verify request body sent
        args, kwargs = mock_urlopen.call_args
        req = args[0]
        sent_data = json.loads(req.data.decode("utf-8"))
        assert sent_data["msg_type"] == "interactive"

        # WeChat Work
        res_wecom = send_intraday_webhook(
            webhook_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=dummy",
            feishu_payload=feishu,
            wecom_payload=wecom,
            webhook_type="wecom",
            dry_run=False,
        )
        assert res_wecom["sent"] is True


def test_run_intraday_pipeline_dry_run_with_auto_sweep(tmp_path: Path):
    """End-to-end dry-run test with --auto-sweep-gc001 generating cards and no network calls."""
    h_file = tmp_path / "holdings.json"
    l_file = tmp_path / "ledger.json"
    card_file = tmp_path / "1430_DAILY_ACTION_CARD.md"

    # Ensure no network requests are made during pipeline run
    with patch("urllib.request.urlopen") as mock_urlopen:
        def mock_fetch(req, *args, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "eastmoney" in url:
                m = MagicMock()
                m.read.return_value = json.dumps({"data": {"klines": ["2026-09-15,1,2,3,4,100,200", "2026-09-16,1,2.05,3,4,100,200"]}}).encode()
                m.status = 200
                ctx = MagicMock()
                ctx.__enter__.return_value = m
                return ctx
            elif "sina" in url:
                m = MagicMock()
                m.read.return_value = json.dumps({"result": {"data": {"feed": {"list": []}}}}).encode()
                m.status = 200
                ctx = MagicMock()
                ctx.__enter__.return_value = m
                return ctx
            raise RuntimeError(f"Unexpected network request in dry-run to {url}")

        mock_urlopen.side_effect = mock_fetch

        res = run_intraday_pipeline(
            holdings_path=h_file,
            ledger_path=l_file,
            profile="conservative",
            inflow=10000.0,
            dry_run=True,
            auto_sweep_gc001=True,
            action_card_path=card_file,
            initial_capital_if_empty=100000.0,
            webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/test",
            webhook_type="all",
        )

        assert res["dry_run"] is True
        assert res["auto_sweep_gc001"] is True
        assert res["webhook_result"]["sent"] is False
        assert card_file.exists()

        feishu_file = tmp_path / "feishu_card.json"
        wecom_file = tmp_path / "wecom_card.json"
        assert feishu_file.exists()
        assert wecom_file.exists()

        # Check action card markdown
        card_text = card_file.read_text(encoding="utf-8")
        assert "自动扫尾模式" in card_text
        assert "--auto-sweep-gc001" in card_text
        assert "204001" in card_text or "GC001" in card_text
        assert "【配置分层看板】" in card_text
        assert "核心全天候 ETF 再平衡交易清单" in card_text
        assert "噪声个股拦截与宽基 ETF 替代明细" in card_text
