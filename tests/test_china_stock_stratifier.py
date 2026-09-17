"""Unit tests for Stock Predictability Stratifier and Execution Regime Filter."""

from fin_skills.china.stock_predictability_stratifier import StockPredictabilityStratifier


def test_tier_s_high_predictability_amplification() -> None:
    """Verify Tier S equities receive 1.2x conviction sizing."""
    strat = StockPredictabilityStratifier()
    res = strat.evaluate_stock_signal("SZ300760", 0.80, is_substantive_event=False)
    assert res["tier"] == "TIER_S_HIGH_PREDICTABILITY"
    assert res["action"] == "EXECUTE_SIGNAL"
    assert res["effective_signal"] == 0.96  # 0.80 * 1.20


def test_tier_c_noisy_stock_etf_substitution() -> None:
    """Verify Tier C random-walk equities are blocked from single-stock trading and routed to ETF."""
    strat = StockPredictabilityStratifier()
    res = strat.evaluate_stock_signal("02015", 0.90, is_substantive_event=True)
    assert res["tier"] == "TIER_C_LOW_NOISY_RANDOM_WALK"
    assert res["action"] == "REJECT_NOISY_STOCK_USE_ETF"
    assert res["effective_signal"] == 0.0
    assert res["recommended_instrument"] in ("510900", "513100")


def test_tier_b_event_driven_gating() -> None:
    """Verify Tier B equities ignore quiet daily noise but execute on substantive events."""
    strat = StockPredictabilityStratifier()
    # Unknown or moderate ticker defaults to Tier B
    quiet_res = strat.evaluate_stock_signal("SH600519_TEST", 0.75, is_substantive_event=False)
    assert quiet_res["action"] == "HOLD_WAIT_FOR_SUBSTANTIVE_EVENT"
    assert quiet_res["effective_signal"] == 0.0

    event_res = strat.evaluate_stock_signal("SH600519_TEST", 0.75, is_substantive_event=True)
    assert event_res["action"] == "EXECUTE_SIGNAL"
    assert event_res["effective_signal"] == 0.45  # 0.75 * 0.60
