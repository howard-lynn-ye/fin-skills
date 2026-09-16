"""Comprehensive Unit Tests for Unified Portfolio Risk Guards Audit CLI & Engine."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.production.audit_portfolio_guards import (
    PortfolioRiskAuditor,
    audit_portfolio,
    main,
)


@pytest.fixture
def clean_balanced_portfolio() -> dict:
    """A clean, well-diversified All-Weather Core ETF portfolio with low cash drag."""
    return {
        "total_nav": 100000.0,
        "cash": 3000.0,  # 3.0% <= 5.0% tolerance
        "last_rebalance_date": "2026-09-01",
        "rebalance_count": 5,
        "holdings": [
            {"code": "510300", "name": "沪深300ETF", "shares": 5000, "avg_cost": 4.10, "market_price": 4.10},   # 20,500
            {"code": "510500", "name": "中证500ETF", "shares": 1700, "avg_cost": 5.90, "market_price": 5.90},   # 10,030
            {"code": "510880", "name": "红利ETF", "shares": 3200, "avg_cost": 3.10, "market_price": 3.10},      # 9,920
            {"code": "518880", "name": "黄金ETF", "shares": 2200, "avg_cost": 6.80, "market_price": 6.80},      # 14,960
            {"code": "511010", "name": "国债ETF", "shares": 200, "avg_cost": 135.0, "market_price": 135.0},      # 27,000
            {"code": "513100", "name": "纳指ETF", "shares": 4200, "avg_cost": 1.70, "market_price": 1.70},      # 7,140
            {"code": "513500", "name": "标普500ETF", "shares": 4100, "avg_cost": 1.80, "market_price": 1.80},   # 7,380
        ],
    }


def test_clean_balanced_portfolio_passes(clean_balanced_portfolio):
    """Test clean balanced portfolio achieves Health Score >= 90 and verdict PASSED."""
    auditor = PortfolioRiskAuditor()
    report = auditor.audit(
        holdings_input=clean_balanced_portfolio,
        live=False,
        iopv_map={"513100": 1.70, "513500": 1.80},  # 0% premium
        today_date="2026-09-16",
    )

    assert report.health_score >= 90.0, f"Expected health score >= 90, got {report.health_score}"
    assert report.verdict == "PASSED", f"Expected verdict PASSED, got {report.verdict}"
    assert report.guard_results["qdii_premium"].status == "PASS"
    assert report.guard_results["ashare_rules"].status == "PASS"
    assert report.guard_results["board_lot_feasibility"].status == "PASS"
    assert report.guard_results["cash_drag"].status == "PASS"
    assert report.guard_results["weight_traps"].status == "PASS"
    assert report.guard_results["data_quality"].status == "PASS"


def test_portfolio_high_qdii_premium_triggers_warning(clean_balanced_portfolio):
    """Test portfolio with high QDII premium (1.5%~3.0% derating zone) triggers WARNING."""
    auditor = PortfolioRiskAuditor()
    # Secondary price is 1.745 while IOPV is 1.70 -> premium = +2.65% (in DERATE_50 zone)
    report = auditor.audit(
        holdings_input=clean_balanced_portfolio,
        live=False,
        price_map={"513100": 1.745, "513500": 1.80},
        iopv_map={"513100": 1.70, "513500": 1.80},
        today_date="2026-09-16",
    )

    qdii_res = report.guard_results["qdii_premium"]
    assert qdii_res.status == "WARN", f"Expected QDII guard WARN, got {qdii_res.status}"
    assert report.verdict == "WARNING_REQUIRES_ATTENTION"
    assert any("溢价预警" in f or "DERATE_50" in f for f in qdii_res.findings)
    assert report.health_score < 90.0


def test_portfolio_excessive_cash_drag_triggers_warning():
    """Test portfolio with excessive cash drag (>20% idle uninvested cash) triggers WARNING."""
    portfolio_high_cash = {
        "total_nav": 100000.0,
        "cash": 28000.0,  # 28.0% > 20.0%
        "last_rebalance_date": "2026-09-01",
        "holdings": [
            {"code": "510300", "name": "沪深300ETF", "shares": 10000, "avg_cost": 4.10, "market_price": 4.10},  # 41,000
            {"code": "511010", "name": "国债ETF", "shares": 200, "avg_cost": 135.0, "market_price": 135.0},      # 27,000
            {"code": "518880", "name": "黄金ETF", "shares": 600, "avg_cost": 6.80, "market_price": 6.80},        # 4,080
        ],
    }

    auditor = PortfolioRiskAuditor()
    report = auditor.audit(
        holdings_input=portfolio_high_cash,
        live=False,
        today_date="2026-09-16",
    )

    cash_res = report.guard_results["cash_drag"]
    assert cash_res.status == "WARN", f"Expected cash drag WARN, got {cash_res.status}"
    assert report.verdict == "WARNING_REQUIRES_ATTENTION"
    assert report.cash_weight > 0.20
    assert any("严重现金拖累" in f for f in cash_res.findings)
    assert "annual_drag_rmb" in cash_res.metrics


def test_portfolio_board_lot_violation_triggers_warning_or_rejected(clean_balanced_portfolio):
    """Test portfolio with board lot violation (fractional A-shares, e.g. 150 shares) triggers WARNING/REJECTED."""
    # Introduce fractional / non-100 share lot: 510300 has 5150 shares (not multiple of 100)
    dirty_portfolio = dict(clean_balanced_portfolio)
    dirty_portfolio["holdings"] = [dict(h) for h in clean_balanced_portfolio["holdings"]]
    dirty_portfolio["holdings"][0]["shares"] = 5150  # Fractional violation (51手 + 50股)

    auditor = PortfolioRiskAuditor()
    report = auditor.audit(
        holdings_input=dirty_portfolio,
        live=False,
        today_date="2026-09-16",
    )

    lot_res = report.guard_results["board_lot_feasibility"]
    assert lot_res.status in ("WARN", "FAIL"), f"Expected lot guard WARN or FAIL, got {lot_res.status}"
    assert report.verdict in ("WARNING_REQUIRES_ATTENTION", "REJECTED_DANGEROUS")
    assert any("零股违规" in f for f in lot_res.findings)


def test_portfolio_single_stock_concentration_trap_triggers_warning():
    """Test portfolio with single-stock concentration trap (>10% in one stock) triggers WARNING."""
    portfolio_concentrated = {
        "total_nav": 100000.0,
        "cash": 4000.0,
        "last_rebalance_date": "2026-09-01",
        "holdings": [
            # Satellite stock with 14.6% weight (>10% institutional hard cap)
            {"code": "SH600036", "name": "招商银行", "shares": 400, "avg_cost": 36.50, "market_price": 36.50},   # 14,600 (14.6%)
            {"code": "510300", "name": "沪深300ETF", "shares": 8000, "avg_cost": 4.10, "market_price": 4.10},    # 32,800
            {"code": "511010", "name": "国债ETF", "shares": 300, "avg_cost": 135.0, "market_price": 135.0},       # 40,500
            {"code": "518880", "name": "黄金ETF", "shares": 1200, "avg_cost": 6.80, "market_price": 6.80},       # 8,160
        ],
    }

    auditor = PortfolioRiskAuditor()
    report = auditor.audit(
        holdings_input=portfolio_concentrated,
        live=False,
        today_date="2026-09-16",
    )

    weight_res = report.guard_results["weight_traps"]
    assert weight_res.status == "WARN", f"Expected weight traps WARN, got {weight_res.status}"
    assert report.verdict == "WARNING_REQUIRES_ATTENTION"
    assert any("个股过度集中陷阱" in f for f in weight_res.findings)
    assert report.health_score < 90.0


def test_data_quality_missing_prices_triggers_rejection(clean_balanced_portfolio):
    """Test data quality guard flags missing or non-positive price as fatal rejection."""
    corrupt_portfolio = dict(clean_balanced_portfolio)
    corrupt_portfolio["holdings"] = [dict(h) for h in clean_balanced_portfolio["holdings"]]
    corrupt_portfolio["holdings"][0]["market_price"] = 0.0  # Invalid price

    auditor = PortfolioRiskAuditor()
    report = auditor.audit(
        holdings_input=corrupt_portfolio,
        price_map={"510300": 0.0},
        live=False,
        today_date="2026-09-16",
    )

    dq_res = report.guard_results["data_quality"]
    assert dq_res.status == "FAIL"
    assert report.verdict == "REJECTED_DANGEROUS"


def test_cli_standalone_execution(tmp_path: Path):
    """Test standalone CLI execution generates valid Markdown and JSON reports."""
    holdings_file = tmp_path / "test_holdings.json"
    holdings_file.write_text(
        json.dumps(
            {
                "total_nav": 100000.0,
                "cash": 3000.0,
                "last_rebalance_date": "2026-09-01",
                "holdings": [
                    {"code": "510300", "name": "沪深300ETF", "shares": 10000, "market_price": 4.10, "avg_cost": 4.10},
                    {"code": "511010", "name": "国债ETF", "shares": 400, "market_price": 135.0, "avg_cost": 135.0},
                    {"code": "518880", "name": "黄金ETF", "shares": 300, "market_price": 6.80, "avg_cost": 6.80},
                ],
            }
        ),
        encoding="utf-8",
    )

    md_output = tmp_path / "TEST_AUDIT_REPORT.md"
    json_output = tmp_path / "TEST_AUDIT_REPORT.json"

    exit_code = main(
        [
            "--holdings",
            str(holdings_file),
            "--output-md",
            str(md_output),
            "--output-json",
            str(json_output),
            "--no-live",
        ]
    )

    assert exit_code == 0
    assert md_output.exists()
    assert json_output.exists()

    md_text = md_output.read_text(encoding="utf-8")
    assert "# 🛡️ 机构级投资组合全维度风控守卫审计综合报告" in md_text
    assert "六大风控守卫矩阵体检总览" in md_text
    assert "资产组合核心指标看板" in md_text

    with open(json_output, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert "health_score" in json_data
    assert "verdict" in json_data
    assert "guard_results" in json_data
    assert len(json_data["guard_results"]) == 6


def test_qdii_hard_circuit_triggers_rejection(clean_balanced_portfolio):
    """Test QDII premium > 3.0% triggers HARD_CIRCUIT and REJECTED_DANGEROUS."""
    auditor = PortfolioRiskAuditor()
    # 513100 trading at 1.85 while IOPV is 1.70 -> premium = +8.82% (>3.0%)
    report = auditor.audit(
        holdings_input=clean_balanced_portfolio,
        live=False,
        price_map={"513100": 1.85, "513500": 1.80},
        iopv_map={"513100": 1.70, "513500": 1.80},
        today_date="2026-09-16",
    )

    qdii_res = report.guard_results["qdii_premium"]
    assert qdii_res.status == "FAIL"
    assert report.verdict == "REJECTED_DANGEROUS"
    assert any("硬熔断" in f for f in qdii_res.findings)


def test_negative_weight_triggers_rejection(clean_balanced_portfolio):
    """Test negative asset weight or negative cash triggers REJECTED_DANGEROUS."""
    corrupt_portfolio = dict(clean_balanced_portfolio)
    corrupt_portfolio["holdings"] = [dict(h) for h in clean_balanced_portfolio["holdings"]]
    corrupt_portfolio["holdings"][0]["shares"] = -500  # Negative shares

    auditor = PortfolioRiskAuditor()
    report = auditor.audit(
        holdings_input=corrupt_portfolio,
        live=False,
        today_date="2026-09-16",
    )

    wt_res = report.guard_results["weight_traps"]
    assert wt_res.status == "FAIL"
    assert report.verdict == "REJECTED_DANGEROUS"
    assert any("裸做空" in f or "负权重" in f for f in wt_res.findings)


def test_data_quality_non_monotonic_timestamps_triggers_failure(clean_balanced_portfolio):
    """Test price history with non-monotonic timestamps triggers data quality FAIL."""
    auditor = PortfolioRiskAuditor()
    report = auditor.audit(
        holdings_input=clean_balanced_portfolio,
        live=False,
        price_history={
            "510300": ["2026-09-01", "2026-09-03", "2026-09-02"],  # Inverted timestamp!
        },
        today_date="2026-09-16",
    )

    dq_res = report.guard_results["data_quality"]
    assert dq_res.status == "FAIL"
    assert report.verdict == "REJECTED_DANGEROUS"
    assert any("非单调" in f for f in dq_res.findings)


def test_audit_standalone_against_default_my_holdings(tmp_path: Path):
    """Verify audit_portfolio runs against default research/production/my_holdings.json."""
    holdings_path = REPO_ROOT / "research/production/my_holdings.json"
    assert holdings_path.exists()

    output_md = tmp_path / "PORTFOLIO_RISK_GUARDS_AUDIT_REPORT.md"
    report = audit_portfolio(
        holdings_path=holdings_path,
        live=False,
        output_md=output_md,
    )

    assert report.total_nav == 100000.0
    assert report.cash == 100000.0
    assert report.verdict == "WARNING_REQUIRES_ATTENTION"
    assert output_md.exists()
