"""Tests for fin_skills.china.stock_predictability_stratifier."""
from __future__ import annotations

from fin_skills.china.stock_predictability_stratifier import StockPredictabilityStratifier


def test_stock_predictability_stratifier_basic():
    strat = StockPredictabilityStratifier()
    res_s = strat.evaluate_stock_signal("SZ300760", 0.80)
    assert res_s["tier"] == "TIER_S_HIGH_PREDICTABILITY"
    res_c = strat.evaluate_stock_signal("02015", 0.80)
    assert res_c["action"] == "REJECT_NOISY_STOCK_USE_ETF"
