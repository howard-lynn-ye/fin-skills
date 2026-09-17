"""Unit tests for VolatilityRegimeShield (Macro Volatility Adaptive Target Shield)."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fin_skills.china.volatility_regime_shield import (
    REGIME_CALM_EXPANSION,
    REGIME_ELEVATED_VOL_STRESS,
    REGIME_EXTREME_PANIC_FREEZE,
    REGIME_NORMAL_OSCILLATION,
    MarketVolatilityMetrics,
    VolatilityRegimeShield,
    VolatilityShieldResult,
)


def test_volatility_regime_classification_all_four_regimes():
    """Verify all 4 distinct regimes are correctly identified and adapted."""
    shield = VolatilityRegimeShield()

    # 1. CALM_EXPANSION: Low Vol (<35th percentile or < 14%), Broad Advance (>60% breadth)
    m_calm = MarketVolatilityMetrics(
        realized_vol_60d=0.12,
        vol_percentile=0.25,
        market_breadth=0.75,
    )
    r_calm = shield.adapt_budget(core_base_budget=0.80, satellite_base_budget=0.20, market_volatility_metrics=m_calm)
    assert r_calm.regime == REGIME_CALM_EXPANSION
    assert r_calm.core_budget == pytest.approx(0.80, abs=1e-4)
    assert r_calm.satellite_budget == pytest.approx(0.20, abs=1e-4)
    assert r_calm.cash_defensive_budget == 0.0

    # 2. NORMAL_OSCILLATION: Balanced oscillation
    m_norm = MarketVolatilityMetrics(
        realized_vol_60d=0.16,
        vol_percentile=0.50,
        market_breadth=0.50,
    )
    r_norm = shield.adapt_budget(core_base_budget=0.80, satellite_base_budget=0.20, market_volatility_metrics=m_norm)
    assert r_norm.regime == REGIME_NORMAL_OSCILLATION
    assert r_norm.core_budget == pytest.approx(0.80, abs=1e-4)
    assert r_norm.satellite_budget == pytest.approx(0.20, abs=1e-4)
    assert r_norm.cash_defensive_budget == 0.0

    # 3. ELEVATED_VOL_STRESS: Vol > 80th percentile -> Core 90% / Satellite 10% (risky bets dialed down 50%)
    m_stress = MarketVolatilityMetrics(
        realized_vol_60d=0.26,
        vol_percentile=0.85,
        market_breadth=0.35,
    )
    r_stress = shield.adapt_budget(core_base_budget=0.80, satellite_base_budget=0.20, market_volatility_metrics=m_stress)
    assert r_stress.regime == REGIME_ELEVATED_VOL_STRESS
    assert r_stress.core_budget == pytest.approx(0.90, abs=1e-4)
    assert r_stress.satellite_budget == pytest.approx(0.10, abs=1e-4)
    assert r_stress.cash_defensive_budget == 0.0
    assert "压缩50%" in r_stress.reason or "10%" in r_stress.reason

    # 4. EXTREME_PANIC_FREEZE: Vol > 95th percentile -> Core 95% / Satellite 0% (5% cash to GC001)
    m_freeze = MarketVolatilityMetrics(
        realized_vol_60d=0.38,
        vol_percentile=0.98,
        market_breadth=0.10,
        liquidity_freeze=False,
    )
    r_freeze = shield.adapt_budget(core_base_budget=0.80, satellite_base_budget=0.20, market_volatility_metrics=m_freeze)
    assert r_freeze.regime == REGIME_EXTREME_PANIC_FREEZE
    assert r_freeze.core_budget == pytest.approx(0.95, abs=1e-4)
    assert r_freeze.satellite_budget == pytest.approx(0.00, abs=1e-4)
    assert r_freeze.cash_defensive_budget == pytest.approx(0.05, abs=1e-4)
    assert "降至0%" in r_freeze.reason or "熔断" in r_freeze.reason


def test_volatility_liquidity_freeze_override():
    """Verify market liquidity freeze triggers EXTREME_PANIC_FREEZE even at lower volatility."""
    shield = VolatilityRegimeShield()
    m_liq = MarketVolatilityMetrics(
        realized_vol_60d=0.18,
        vol_percentile=0.55,
        liquidity_freeze=True,
    )
    res = shield.adapt_budget(market_volatility_metrics=m_liq)
    assert res.regime == REGIME_EXTREME_PANIC_FREEZE
    assert res.core_budget == 0.95
    assert res.satellite_budget == 0.00
    assert res.cash_defensive_budget == 0.05


def test_volatility_shield_result_tuple_compatibility():
    """Verify VolatilityShieldResult supports tuple unpacking, indexing, and dict conversion."""
    shield = VolatilityRegimeShield()
    res = shield.adapt_budget(market_volatility_metrics={"vol_60d": 0.25, "vol_percentile": 0.82})

    assert isinstance(res, tuple)
    assert len(res) == 3
    core_b, sat_b, regime = res
    assert core_b == 0.90
    assert sat_b == 0.10
    assert regime == REGIME_ELEVATED_VOL_STRESS

    # Property access
    assert res.core_budget == 0.90
    assert res.satellite_budget == 0.10
    assert res.regime == REGIME_ELEVATED_VOL_STRESS
    assert res.cash_defensive_budget == 0.0

    d = res.to_dict()
    assert d["core_budget"] == 0.90
    assert d["regime"] == REGIME_ELEVATED_VOL_STRESS
    assert "metrics" in d


def test_volatility_calculation_and_percentile_mapping():
    """Verify realized volatility calculation on returns and percentile calibration."""
    shield = VolatilityRegimeShield()

    # Synthetic daily price series with ~1% daily moves
    prices = [100.0 * (1.01 ** i) for i in range(70)]
    vol = shield.calculate_realized_volatility(prices, window=60)
    assert vol > 0.0

    # Percentile mapping
    assert shield.estimate_vol_percentile(0.08) < 0.20
    assert shield.estimate_vol_percentile(0.16) == pytest.approx(0.425, abs=0.1)
    assert shield.estimate_vol_percentile(0.24) >= 0.80
    assert shield.estimate_vol_percentile(0.36) >= 0.95


def test_volatility_evaluate_from_snapshot():
    """Verify parsing Eastmoney-style live market snapshot."""
    shield = VolatilityRegimeShield()
    fake_snapshot = {
        "510300": {"vol_60d": 0.18, "trend_ma60": True, "daily_pct": 0.01},
        "510500": {"vol_60d": 0.20, "trend_ma60": True, "daily_pct": 0.02},
        "510880": {"vol_60d": 0.12, "trend_ma60": True, "daily_pct": 0.005},
        "511010": {"vol_60d": 0.04, "trend_ma60": True, "daily_pct": 0.0005},
    }
    metrics = shield.evaluate_metrics_from_snapshot(fake_snapshot)
    assert metrics.realized_vol_60d > 0.10
    assert metrics.market_breadth == 1.0
    assert not metrics.liquidity_freeze

    regime, classified_m = shield.classify_regime(metrics)
    assert regime in (REGIME_CALM_EXPANSION, REGIME_NORMAL_OSCILLATION)


def test_volatility_format_shield_report():
    """Verify Markdown report generation."""
    shield = VolatilityRegimeShield()
    r = shield.adapt_budget(market_volatility_metrics={"vol_60d": 0.38, "vol_percentile": 0.98})
    md = shield.format_shield_report(r)
    assert "#### 🛡️ 宏观波动率自适应防爆盾状态" in md
    assert "EXTREME_PANIC_FREEZE" in md
    assert "95%" in md
    assert "0%" in md
    assert "5%" in md
