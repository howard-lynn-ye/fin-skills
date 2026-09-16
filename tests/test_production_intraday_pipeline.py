"""Tests for research.production.run_intraday_pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fin_skills.china.core_satellite_advisor import SatelliteLifecycleManager
from research.production.run_intraday_pipeline import run_intraday_pipeline


def test_run_intraday_pipeline_dry_run_generates_action_card(tmp_path: Path):
    h_file = tmp_path / "test_holdings.json"
    l_file = tmp_path / "test_ledger.json"
    card_file = tmp_path / "test_action_card.md"

    res = run_intraday_pipeline(
        holdings_path=h_file,
        ledger_path=l_file,
        profile="conservative",
        inflow=0.0,
        confirm_execution=False,
        action_card_path=card_file,
        initial_capital_if_empty=100000.0,
    )

    assert card_file.exists()
    card_text = card_file.read_text(encoding="utf-8")
    assert "# 🎯 14:25 每日实盘操作行动指南 (Daily Action Card)" in card_text
    assert "核心全天候 ETF 再平衡交易清单" in card_text
    assert "噪声个股拦截与宽基 ETF 替代明细" in card_text
    assert "14:50 - 15:30 现金增益管理执行" in card_text
    assert res["action_card_path"] == str(card_file)


def test_run_intraday_pipeline_confirm_execution_and_lifecycle_updates(tmp_path: Path):
    h_file = tmp_path / "test_holdings.json"
    l_file = tmp_path / "test_ledger.json"
    card_file = tmp_path / "test_action_card.md"

    # 1. Day 1: Confirm entry execution
    res1 = run_intraday_pipeline(
        holdings_path=h_file,
        ledger_path=l_file,
        profile="conservative",
        inflow=0.0,
        confirm_execution=True,
        action_card_path=card_file,
        initial_capital_if_empty=100000.0,
        candidate_signals=[
            {"symbol": "SZ300760", "name": "迈瑞医疗", "kol_weighted_sentiment": 0.88, "base_weight": 0.04},
        ],
    )

    assert h_file.exists()
    assert l_file.exists()

    mgr1 = SatelliteLifecycleManager(ledger_path=l_file)
    assert "SZ300760" in mgr1.positions
    assert mgr1.positions["SZ300760"].shares > 0

    # 2. Day 2: Advance holding date to simulate T+6 maturation exit
    mgr1.positions["SZ300760"].entry_date = "2026-09-01"  # 15 days ago -> > 5 days
    mgr1.save()

    res2 = run_intraday_pipeline(
        holdings_path=h_file,
        ledger_path=l_file,
        profile="conservative",
        inflow=0.0,
        confirm_execution=True,
        action_card_path=card_file,
        initial_capital_if_empty=100000.0,
        candidate_signals=[],  # No new buys
    )

    mgr2 = SatelliteLifecycleManager(ledger_path=l_file)
    # The matured position should now be closed and recorded in closed_history
    assert "SZ300760" not in mgr2.positions
    assert len(mgr2.history) == 1
    assert mgr2.history[0]["symbol"] == "SZ300760"
    assert "持有期届满" in mgr2.history[0]["reason"] or "T+5" in mgr2.history[0]["reason"]
