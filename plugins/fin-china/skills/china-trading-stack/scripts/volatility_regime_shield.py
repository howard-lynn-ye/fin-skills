"""Volatility Adaptive Target Shield (VolatilityRegimeShield).

Dynamically shifts the Core-Satellite asset allocation budget (from 80/20 to 90/10 or cash-defensive mode)
based on real-time rolling realized volatility (20-day and 60-day windows) and market breadth stress.

Defined Regimes:
1. CALM_EXPANSION: Low Vol (<35th percentile), Broad Advance (>60% breadth) -> Core 80% / Satellite 20%
2. NORMAL_OSCILLATION: Balanced market oscillation -> Core 80% / Satellite 20%
3. ELEVATED_VOL_STRESS: Vol > 80th percentile -> Core 90% / Satellite 10% (risky stock bets dialed down 50%)
4. EXTREME_PANIC_FREEZE: Vol > 95th percentile or Liquidity Freeze -> Core 95% / Satellite 0%
   (satellite alpha fully disabled, 5% cash defensive to GC001, Core bonds scaled up)
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("volatility_regime_shield")

REGIME_CALM_EXPANSION = "CALM_EXPANSION"
REGIME_NORMAL_OSCILLATION = "NORMAL_OSCILLATION"
REGIME_ELEVATED_VOL_STRESS = "ELEVATED_VOL_STRESS"
REGIME_EXTREME_PANIC_FREEZE = "EXTREME_PANIC_FREEZE"

REGIME_DESCRIPTIONS: dict[str, str] = {
    REGIME_CALM_EXPANSION: "低波平稳扩张 (赚钱效应良性扩散，维持 80/20 满额配置)",
    REGIME_NORMAL_OSCILLATION: "正常区间震荡 (宏观波动在安全死区，维持 80/20 标准配置)",
    REGIME_ELEVATED_VOL_STRESS: "高波压力预警 (波动率突破80%分位，防爆盾启动：个股仓压缩50%至10%，核心仓提升至90%)",
    REGIME_EXTREME_PANIC_FREEZE: "极度恐慌冰冻 (波动率突破95%分位或流动性冻结，关闭个股Alpha至0%，核心仓95%+5%现金GC001避险)",
}


@dataclass
class MarketVolatilityMetrics:
    """Quantitative inputs representing rolling volatility and market breadth."""

    realized_vol_20d: float = 0.15
    realized_vol_60d: float = 0.16
    vol_percentile: float = 0.50  # 0.0 to 1.0 (percentile rank vs historical A-share cycles)
    market_breadth: float = 0.50  # 0.0 to 1.0 (fraction of advancing assets or above 60MA)
    liquidity_freeze: bool = False  # True if limit-down lock / liquidity seizure detected
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VolatilityShieldResult(tuple):
    """Tuple subclass (core_budget, satellite_budget, regime) with rich metadata."""

    def __new__(
        cls,
        core_budget: float,
        satellite_budget: float,
        regime: str,
        cash_defensive_budget: float = 0.0,
        metrics: MarketVolatilityMetrics | None = None,
        reason: str = "",
    ):
        inst = super().__new__(
            cls,
            (round(float(core_budget), 4), round(float(satellite_budget), 4), str(regime)),
        )
        inst.core_budget = round(float(core_budget), 4)
        inst.satellite_budget = round(float(satellite_budget), 4)
        inst.regime = str(regime)
        inst.cash_defensive_budget = round(float(cash_defensive_budget), 4)
        inst.metrics = metrics or MarketVolatilityMetrics()
        inst.reason = reason
        return inst

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_budget": self.core_budget,
            "satellite_budget": self.satellite_budget,
            "regime": self.regime,
            "cash_defensive_budget": self.cash_defensive_budget,
            "reason": self.reason,
            "metrics": self.metrics.to_dict() if self.metrics else {},
        }


class VolatilityRegimeShield:
    """Calculates realized volatility, market breadth, and adapts Core-Satellite risk budgets."""

    def __init__(
        self,
        vol_80th_percentile: float = 0.24,  # ~24% annualized realized vol is ~80th percentile in A-shares
        vol_95th_percentile: float = 0.35,  # ~35% annualized realized vol is ~95th percentile
        low_vol_threshold: float = 0.14,  # <14% annualized realized vol
        broad_advance_breadth: float = 0.60,  # >60% breadth
    ):
        self.vol_80th_percentile = float(vol_80th_percentile)
        self.vol_95th_percentile = float(vol_95th_percentile)
        self.low_vol_threshold = float(low_vol_threshold)
        self.broad_advance_breadth = float(broad_advance_breadth)

    @staticmethod
    def calculate_realized_volatility(
        prices: Sequence[float] | pd.Series | np.ndarray,
        window: int = 60,
        annualize: bool = True,
    ) -> float:
        """Compute rolling annualized realized volatility from price sequence."""
        if len(prices) < 3:
            return 0.15
        s = pd.Series(prices).pct_change().dropna()
        if len(s) > window:
            s = s.iloc[-window:]
        if len(s) < 2:
            return 0.15
        std = float(s.std())
        if annualize:
            return float(std * np.sqrt(252))
        return std

    @staticmethod
    def estimate_vol_percentile(annualized_vol: float) -> float:
        """Map annualized realized volatility to historical A-share percentile rank (0.0 to 1.0)."""
        v = max(0.01, float(annualized_vol))
        # Piecewise linear approximation calibrated to 10-year A-share ETF history:
        # <= 10%: 5th percentile
        # 14%: 30th percentile
        # 18%: 55th percentile
        # 24%: 80th percentile
        # 30%: 90th percentile
        # 35%: 95th percentile
        # >= 45%: 99th percentile
        if v <= 0.10:
            return max(0.02, v / 0.10 * 0.15)
        elif v <= 0.14:
            return 0.15 + (v - 0.10) / (0.14 - 0.10) * 0.15  # 0.15 -> 0.30
        elif v <= 0.18:
            return 0.30 + (v - 0.14) / (0.18 - 0.14) * 0.25  # 0.30 -> 0.55
        elif v <= 0.24:
            return 0.55 + (v - 0.18) / (0.24 - 0.18) * 0.25  # 0.55 -> 0.80
        elif v <= 0.35:
            return 0.80 + (v - 0.24) / (0.35 - 0.24) * 0.15  # 0.80 -> 0.95
        elif v <= 0.50:
            return 0.95 + (v - 0.35) / (0.50 - 0.35) * 0.04  # 0.95 -> 0.99
        else:
            return 0.999

    def evaluate_metrics_from_snapshot(
        self,
        market_snapshot: Mapping[str, Mapping[str, Any]],
    ) -> MarketVolatilityMetrics:
        """Parse live market snapshot dict into MarketVolatilityMetrics."""
        if not market_snapshot:
            return MarketVolatilityMetrics()

        vols_60d = []
        breadth_counts = 0
        total_assets = 0
        limit_down_count = 0

        # Focus primarily on equity assets (e.g. 510300, 510500, 510880, 513100, 513500)
        for code, meta in market_snapshot.items():
            total_assets += 1
            vol = float(meta.get("vol_60d", 0.16))
            if vol > 0.01:
                vols_60d.append(vol)
            if meta.get("trend_ma60", False) or meta.get("daily_pct", 0.0) >= 0:
                breadth_counts += 1
            if meta.get("daily_pct", 0.0) <= -0.095:  # Near limit-down in A-shares
                limit_down_count += 1

        avg_vol_60d = float(np.mean(vols_60d)) if vols_60d else 0.16
        # Realized 20d volatility estimate from snapshot
        avg_vol_20d = float(avg_vol_60d * 1.05)
        breadth = breadth_counts / total_assets if total_assets > 0 else 0.50
        vol_pctile = self.estimate_vol_percentile(avg_vol_60d)
        is_freeze = limit_down_count >= 3  # Multiple broad ETFs hitting limit-down

        return MarketVolatilityMetrics(
            realized_vol_20d=round(avg_vol_20d, 4),
            realized_vol_60d=round(avg_vol_60d, 4),
            vol_percentile=round(vol_pctile, 4),
            market_breadth=round(breadth, 4),
            liquidity_freeze=is_freeze,
            notes=f"Snapshot audited {total_assets} instruments; Mean 60D Vol: {avg_vol_60d * 100:.1f}%",
        )

    def classify_regime(
        self,
        metrics: MarketVolatilityMetrics | Mapping[str, Any] | float | None,
    ) -> tuple[str, MarketVolatilityMetrics]:
        """Classify current market state into one of the 4 regimes."""
        if metrics is None:
            m = MarketVolatilityMetrics()
        elif isinstance(metrics, MarketVolatilityMetrics):
            m = metrics
        elif isinstance(metrics, (int, float)):
            vol = float(metrics)
            m = MarketVolatilityMetrics(
                realized_vol_60d=vol,
                realized_vol_20d=vol,
                vol_percentile=self.estimate_vol_percentile(vol),
                market_breadth=0.50,
            )
        elif isinstance(metrics, Mapping):
            if any(k in metrics for k in ("510300", "510500", "511010")):
                m = self.evaluate_metrics_from_snapshot(metrics)
            else:
                m = MarketVolatilityMetrics(
                    realized_vol_20d=float(metrics.get("realized_vol_20d", metrics.get("vol_20d", 0.15))),
                    realized_vol_60d=float(metrics.get("realized_vol_60d", metrics.get("vol_60d", 0.16))),
                    vol_percentile=float(metrics.get("vol_percentile", 0.50)),
                    market_breadth=float(metrics.get("market_breadth", 0.50)),
                    liquidity_freeze=bool(metrics.get("liquidity_freeze", False)),
                    notes=str(metrics.get("notes", "")),
                )
                if "vol_percentile" not in metrics and "vol_60d" in metrics:
                    m.vol_percentile = self.estimate_vol_percentile(m.realized_vol_60d)
        else:
            m = MarketVolatilityMetrics()

        # Regime 4: EXTREME_PANIC_FREEZE (Vol > 95th percentile or Liquidity Freeze)
        if (
            m.liquidity_freeze
            or m.vol_percentile > 0.95
            or m.realized_vol_60d > self.vol_95th_percentile
            or m.realized_vol_20d > 0.45
        ):
            return REGIME_EXTREME_PANIC_FREEZE, m

        # Regime 3: ELEVATED_VOL_STRESS (Vol > 80th percentile)
        if (
            m.vol_percentile > 0.80
            or m.realized_vol_60d > self.vol_80th_percentile
            or m.realized_vol_20d > 0.30
        ):
            return REGIME_ELEVATED_VOL_STRESS, m

        # Regime 1: CALM_EXPANSION (Low Vol < 35th percentile, Broad Advance > 60% breadth)
        if (
            (m.vol_percentile < 0.35 or m.realized_vol_60d < self.low_vol_threshold)
            and m.market_breadth >= self.broad_advance_breadth
        ):
            return REGIME_CALM_EXPANSION, m

        # Regime 2: NORMAL_OSCILLATION (Default balanced regime)
        return REGIME_NORMAL_OSCILLATION, m

    def adapt_budget(
        self,
        core_base_budget: float = 0.80,
        satellite_base_budget: float = 0.20,
        market_volatility_metrics: Any = None,
    ) -> VolatilityShieldResult:
        """Dynamically adapt the Core-Satellite asset allocation budget based on volatility regime.

        Args:
            core_base_budget: Default core ETF allocation (0.80).
            satellite_base_budget: Default satellite stock alpha allocation (0.20).
            market_volatility_metrics: Realized volatility metrics, snapshot dict, float vol, or None.

        Returns:
            VolatilityShieldResult (can be unpacked as tuple: core_budget, satellite_budget, regime).
        """
        regime, metrics = self.classify_regime(market_volatility_metrics)

        if regime == REGIME_EXTREME_PANIC_FREEZE:
            # Core 95% / Satellite 0% (satellite alpha fully disabled, 5% cash to GC001, Core bonds scaled up)
            core_budget = 0.95
            satellite_budget = 0.00
            cash_defensive_budget = 0.05
            reason = (
                f"市场触发【极度恐慌/流动性挤兑冰冻】模式 (波动率 {metrics.realized_vol_60d * 100:.1f}% 位于历史 {metrics.vol_percentile * 100:.0f}% 分位，"
                f"流动性冻结={metrics.liquidity_freeze})。防爆盾强行熔断个股Alpha仓 (降至0%)，"
                f"核心仓扩容至 95% (国债避险ETF规模拉升)，预留 5% 闲置现金执行尾盘 GC001 逆回购锁定流动性避险。"
            )
        elif regime == REGIME_ELEVATED_VOL_STRESS:
            # Core 90% / Satellite 10% (dials down risky stock bets by 50%)
            core_budget = 0.90
            satellite_budget = 0.10
            cash_defensive_budget = 0.00
            reason = (
                f"市场处于【高波动率承压应激】状态 (波动率 {metrics.realized_vol_60d * 100:.1f}% 突破80%历史分位数)。"
                f"防爆盾主动压缩个股卫星风险暴露 50% (由 20% 收敛至 10%)，核心压舱石扩容至 90%，防范高波尾部单票踩踏。"
            )
        elif regime == REGIME_CALM_EXPANSION:
            # Low Vol, Broad Advance -> Core 80% / Satellite 20%
            core_budget = float(core_base_budget)
            satellite_budget = float(satellite_base_budget)
            cash_defensive_budget = 0.00
            reason = (
                f"市场处于【低波良性平稳扩张】环境 (波动率 {metrics.realized_vol_60d * 100:.1f}% 处于安全低位，市场进攻宽度 {metrics.market_breadth * 100:.1f}%)。"
                f"维持标准配置架构：80% 核心全天候稳健底仓 + 20% 卫星个股事件 Alpha 满额攻击仓。"
            )
        else:  # NORMAL_OSCILLATION
            core_budget = float(core_base_budget)
            satellite_budget = float(satellite_base_budget)
            cash_defensive_budget = 0.00
            reason = (
                f"市场处于【正常区间平衡震荡】状态 (波动率 {metrics.realized_vol_60d * 100:.1f}%，分位数 {metrics.vol_percentile * 100:.0f}%)。"
                f"基准配置运行：80% 核心全天候稳健底仓 + 20% 卫星个股事件 Alpha 预算。"
            )

        return VolatilityShieldResult(
            core_budget=core_budget,
            satellite_budget=satellite_budget,
            regime=regime,
            cash_defensive_budget=cash_defensive_budget,
            metrics=metrics,
            reason=reason,
        )

    def format_shield_report(self, result: VolatilityShieldResult) -> str:
        """Format an executive Markdown status card of the Volatility Regime Shield."""
        regime = result.regime
        desc = REGIME_DESCRIPTIONS.get(regime, regime)
        m = result.metrics

        status_emoji = {
            REGIME_CALM_EXPANSION: "🟢",
            REGIME_NORMAL_OSCILLATION: "🔵",
            REGIME_ELEVATED_VOL_STRESS: "🟠",
            REGIME_EXTREME_PANIC_FREEZE: "🔴",
        }.get(regime, "🛡️")

        lines: list[str] = []
        lines.append("#### 🛡️ 宏观波动率自适应防爆盾状态")
        lines.append(
            f"- **当前宏观市场体制**: {status_emoji} **`{regime}`** ({desc})"
        )
        lines.append(
            f"- **自适应预算分配**: 核心全天候 ETF **{result.core_budget * 100:.0f}%** | "
            f"卫星事件 Alpha **{result.satellite_budget * 100:.0f}%**"
            + (
                f" | 防御现金 GC001 **{result.cash_defensive_budget * 100:.0f}%**"
                if result.cash_defensive_budget > 0
                else ""
            )
        )
        lines.append(
            f"- **宏观波动与宽度监控**: 60日年化波动率 **{m.realized_vol_60d * 100:.1f}%** "
            f"(历史分位数 **{m.vol_percentile * 100:.1f}%**) | 市场进攻宽度 **{m.market_breadth * 100:.1f}%**"
        )
        lines.append(f"- **防爆盾风控决策理由**: {result.reason}")
        return "\n".join(lines)
