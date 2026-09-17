"""Tests for StockPredictionBridge in fin_skills.bridges.stock_prediction."""
from __future__ import annotations

import pytest

from fin_skills.bridges.stock_prediction import (StockPredictionBridge,
                                                 find_stock_prediction_dir)


def _has_stock_prediction() -> bool:
    try:
        find_stock_prediction_dir()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_stock_prediction(),
    reason="stock_prediction repository (or STOCK_PREDICTION_DIR) not available in this environment",
)


def test_find_stock_prediction_dir():
    d = find_stock_prediction_dir()
    assert d.is_dir()
    assert (d / "data" / "fetched").exists()


def test_bridge_load_macro_indicators():
    bridge = StockPredictionBridge()
    df = bridge.load_macro_indicators(start_date="2022-01-01", end_date="2023-01-01")
    assert not df.empty
    assert "date" in df.columns
    assert "vix" in df.columns
    assert "csi300_close" in df.columns
    assert "macro_regime" in df.columns


def test_bridge_load_smart_money():
    bridge = StockPredictionBridge()
    df = bridge.load_smart_money_flows(start_date="2022-01-01", end_date="2023-01-01", sample_symbols=5)
    assert not df.empty
    assert "northbound_flow_agg" in df.columns
    assert "smart_money_regime" in df.columns


def test_bridge_build_multimodal_feature_panel():
    bridge = StockPredictionBridge()
    panel = bridge.build_multimodal_feature_panel(start_date="2022-01-01", end_date="2022-06-30")
    assert not panel.empty
    # Must contain lagged PIT features
    assert "vix_spike_lag1" in panel.columns
    assert "smart_money_regime_lag1" in panel.columns
    assert "social_heat_z30_lag1" in panel.columns
