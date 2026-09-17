"""Tests for Market Intelligence & Sentiment Flow Collector (fin_skills.china.sentiment_flow_collector)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fin_skills.china.sentiment_flow_collector import (
    AssetSentimentMetric, CapitalFlowSnapshot, MacroNewsItem, MarketIntelReport,
    apply_sentiment_overlay, collect_daily_market_intel,
    compute_asset_sentiments, fetch_live_capital_flows, fetch_live_macro_news)
from research.production.live_advisor_bot import run_live_advisor


def test_macro_news_item_dataclass():
    item = MacroNewsItem(
        timestamp="2026-09-15 14:15:00",
        headline="央行宣布公开市场降准25个基点，保持流动性合理充裕",
        category="MONETARY_POLICY",
        bias="EXPANSION_BIAS",
    )
    d = item.to_dict()
    assert d["timestamp"] == "2026-09-15 14:15:00"
    assert d["bias"] == "EXPANSION_BIAS"
    assert d["category"] == "MONETARY_POLICY"


def test_fetch_live_macro_news_offline_fallback():
    # Test fallback behavior on connection failure
    with patch("urllib.request.urlopen", side_effect=Exception("Network down")):
        items = fetch_live_macro_news(max_items=3, timeout=1.0)
        assert len(items) >= 1
        assert any("逆回购" in it.headline or "QDII" in it.headline for it in items)


def test_capital_flow_snapshot_classification():
    # Test risk regime classification logic
    mock_data = {
        "data": {
            "s2n": ["9:30,0,5200000,0,5200000,0", "14:30,0,5200000,0,5200000,450000"],  # +45亿
            "n2s": ["9:30,0,4200000,0,4200000,0", "14:30,0,4200000,0,4200000,120000"],  # +12亿
        }
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        snapshot = fetch_live_capital_flows(cutoff_time="14:30:00")
        assert snapshot.risk_regime == "RISK_ON"
        assert snapshot.northbound_net_inflow_bn == 45.0
        assert snapshot.southbound_net_inflow_bn == 12.0


def test_compute_asset_sentiments_heuristics():
    market_snapshot = {
        "513100": {"price": 2.20, "daily_pct": 0.035, "vol_60d": 0.25, "trend_ma60": True},  # Extreme FOMO
        "510300": {"price": 4.10, "daily_pct": -0.025, "vol_60d": 0.16, "trend_ma60": False}, # Capitulation Panic
        "518880": {"price": 8.80, "daily_pct": 0.005, "vol_60d": 0.12, "trend_ma60": True},   # Safe haven
        "511010": {"price": 141.0, "daily_pct": 0.001, "vol_60d": 0.04, "trend_ma60": True},  # Anchor
    }
    sentiments = compute_asset_sentiments(market_snapshot=market_snapshot)

    # 513100 should have negative tilt to avoid chasing euphoria
    assert sentiments["513100"].temperature_label == "极度贪婪(FOMO)"
    assert sentiments["513100"].recommended_tilt < 0.0
    assert sentiments["513100"].sentiment_zscore > 2.0

    # 510300 should have contrarian accumulation tilt
    assert sentiments["510300"].temperature_label == "极度恐慌(冰点)"
    assert sentiments["510300"].recommended_tilt > 0.0
    assert sentiments["510300"].sentiment_zscore < -2.0


def test_apply_sentiment_overlay_strict_guarantees():
    base_weights = {
        "511010": 0.45,
        "510300": 0.15,
        "510880": 0.15,
        "518880": 0.10,
        "513100": 0.08,
        "513500": 0.07,
    }
    sentiments = {
        "513100": AssetSentimentMetric("513100", "纳指", 0.0, 2.5, 2.5, "极度贪婪", -0.02, ""),
        "510300": AssetSentimentMetric("510300", "300", 0.0, -2.5, 2.0, "极度恐慌", +0.02, ""),
        "518880": AssetSentimentMetric("518880", "黄金", 0.0, 1.0, 1.2, "温和活跃", +0.01, ""),
    }

    tilted = apply_sentiment_overlay(base_weights, sentiments, max_tilt=0.015, min_bond_weight=0.35)

    # Sum must strictly equal 1.0
    assert round(sum(tilted.values()), 4) == 1.0

    # All weights non-negative
    assert all(w >= 0.0 for w in tilted.values())

    # Bond floor preserved
    assert tilted["511010"] >= 0.35

    # Tilts respected
    assert tilted["513100"] < base_weights["513100"]
    assert tilted["510300"] > base_weights["510300"]


def test_collect_daily_market_intel_markdown():
    report = collect_daily_market_intel(cutoff_time="14:30:00")
    md = report.to_markdown()

    assert "### 📰 市场情报与情绪前瞻" in md
    assert "审计截止时间" in md
    assert "8大核心资产情绪温度计" in md


def test_live_advisor_bot_with_sentiment(tmp_path: Path):
    holdings_file = tmp_path / "test_holdings.json"
    res = run_live_advisor(
        holdings_path=holdings_file,
        profile="conservative",
        initial_capital_if_empty=50000.0,
        sentiment_tilt=True,
    )
    assert res is not None
    assert "intel_report" in res
    assert res["intel_report"].cutoff_time == "14:30:00"
    assert "市场情报与情绪前瞻" in res["markdown"]
