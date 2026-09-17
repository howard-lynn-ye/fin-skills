"""Multi-Source Financial Signal Conflict Resolution & Hierarchical Consensus Engine.

Why this exists:
In real-world financial markets, different information channels frequently contradict each other:
- Central bank / exchange warns of risk (Bearish), while retail social forums scream "Buy the dip!" (Bullish).
- Northbound smart money quietly accumulates (Bullish) and PE/dividend yield is at a 10-year bottom (Bullish),
  while mainstream news and retail forums panic over short-term headlines (Bearish).
- US StockTwits is euphoric on Tech (Bullish), while domestic A-share QDII ETF trades at a 3% premium (Toxic).

A naive linear average across conflicting channels is fatal:
1. Averaging Smart Money selling (-0.8) with Retail FOMO (+1.0) yields +0.1 (neutral/buy), walking straight
   into a classic Distribution Trap (主力派发、散户接盘).
2. Averaging Valuation/Smart Money buying (+0.8) with Retail Capitulation (-1.0) yields -0.1, missing the
   Golden Contrarian Accumulation bottom (黄金坑背离).

This module implements a 5-Tier Hierarchical Precedence & Conflict Matrix:
- Tier 1: Regulatory & Structural Veto (Level 1 Policy / QDII Premium / VIX Shock) -> Hard Veto Override
- Tier 2: Smart Money Flows (Northbound / Margin / Institutional Skin-in-the-Game) -> Primary Directional Anchor
- Tier 3: Fundamental Valuation & Events (PE/PB/Dividend Yield / Verified Filings) -> Value Safety Margin
- Tier 4: Mainstream Financial Telegraph News (7x24 Macro Feeds) -> Contextual Catalyst
- Tier 5: Retail Social Sentiment (Xueqiu / StockTwits / Guba) -> Contrarian Exhaustion Indicator
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


# Channel Hierarchy Weights (prior to conflict regime adjustment)
CHANNEL_BASE_WEIGHTS = {
    "REGULATORY_POLICY": 1.00,      # Gatekeeper / Hard Veto
    "SMART_MONEY_FLOW": 0.40,       # Primary Institutional Anchor
    "FUNDAMENTAL_VALUATION": 0.35,  # Safety Margin Anchor
    "MAINSTREAM_NEWS": 0.10,        # Catalyst Context
    "RETAIL_SOCIAL_CN": 0.15,       # Contrarian Indicator (A-share/HK)
    "RETAIL_SOCIAL_US": 0.15,       # Contrarian Indicator (US Tech/StockTwits)
}


@dataclass(frozen=True)
class ChannelSignal:
    channel: str  # Key in CHANNEL_BASE_WEIGHTS
    score: float  # -1.0 (extreme bearish) to +1.0 (extreme bullish)
    confidence: float = 1.0  # 0.0 to 1.0 (e.g. from DataQualityAuditor)
    veto_flag: bool = False  # True if hard circuit breaker triggered
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconciliationResult:
    asset_code: str
    final_score: float  # -1.0 to +1.0
    recommended_tilt: float  # -0.02 to +0.02
    conflict_type: str
    disagreement_index: float  # 0.0 (unanimous) to 1.0 (polar opposite)
    confidence_multiplier: float  # 0.2 to 1.0
    dominant_channel: str
    resolution_rationale: str
    channel_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SignalReconciler:
    """Hierarchical conflict resolver for multi-modal financial signals."""

    def __init__(self, max_tilt: float = 0.015, dispersion_damping: float = 1.2):
        self.max_tilt = max_tilt
        self.dispersion_damping = dispersion_damping

    def reconcile_asset_signals(
        self,
        asset_code: str,
        signals: list[ChannelSignal],
        qdii_premium_pct: float = 0.0,
    ) -> ReconciliationResult:
        """Resolve contradictory signals across channels into a single actionable decision."""
        if not signals:
            return ReconciliationResult(
                asset_code=asset_code,
                final_score=0.0,
                recommended_tilt=0.0,
                conflict_type="NO_SIGNAL_NEUTRAL",
                disagreement_index=0.0,
                confidence_multiplier=1.0,
                dominant_channel="NONE",
                resolution_rationale="无多源冲突信号，维持风险平价基准配置",
                channel_breakdown={},
            )

        sig_map: dict[str, ChannelSignal] = {s.channel: s for s in signals}
        breakdown = {s.channel: round(s.score, 3) for s in signals}

        # Extract key tier scores (default 0.0 if channel absent)
        reg_sig = sig_map.get("REGULATORY_POLICY")
        flow_score = sig_map["SMART_MONEY_FLOW"].score if "SMART_MONEY_FLOW" in sig_map else 0.0
        val_score = sig_map["FUNDAMENTAL_VALUATION"].score if "FUNDAMENTAL_VALUATION" in sig_map else 0.0
        news_score = sig_map["MAINSTREAM_NEWS"].score if "MAINSTREAM_NEWS" in sig_map else 0.0

        # Combine CN and US retail social if both exist
        social_scores = [
            s.score * s.confidence
            for k, s in sig_map.items()
            if k in ("RETAIL_SOCIAL_CN", "RETAIL_SOCIAL_US")
        ]
        social_score = sum(social_scores) / len(social_scores) if social_scores else 0.0

        # Compute Cross-Channel Disagreement Index (standard deviation of active scores)
        active_scores = [s.score for s in signals if abs(s.score) > 0.05]
        if len(active_scores) >= 2:
            mean_s = sum(active_scores) / len(active_scores)
            variance = sum((x - mean_s) ** 2 for x in active_scores) / len(active_scores)
            disagreement = min(1.0, round(math.sqrt(variance), 4))
        else:
            disagreement = 0.0

        # =====================================================================
        # ARCHETYPE 1: Hard Veto / Regulatory & Structural Circuit Breaker
        # =====================================================================
        if (reg_sig and (reg_sig.veto_flag or reg_sig.score <= -0.8)) or qdii_premium_pct >= 2.5:
            reason = (
                f"触发一票否决风控(QDII溢价率 {qdii_premium_pct:.2f}% 过高或监管警示)，"
                f"无视散户看多情绪({social_score:+.2f})，强制执行防守熔断"
            )
            return ReconciliationResult(
                asset_code=asset_code,
                final_score=-1.0,
                recommended_tilt=-self.max_tilt,
                conflict_type="CONFLICT_VETO_OVERRIDE",
                disagreement_index=disagreement,
                confidence_multiplier=1.0,
                dominant_channel="REGULATORY_POLICY",
                resolution_rationale=reason,
                channel_breakdown=breakdown,
            )

        # =====================================================================
        # ARCHETYPE 2: Cross-Border QDII Premium Disconnect (US Bullish vs CN Premium)
        # =====================================================================
        if qdii_premium_pct >= 1.5 and social_score > 0.3:
            reason = (
                f"海外标的讨论偏热({social_score:+.2f})但国内场内溢价达 {qdii_premium_pct:.2f}%，"
                f"工具结构性折溢价风险优先于底层涨幅，禁止追高并适度减配"
            )
            return ReconciliationResult(
                asset_code=asset_code,
                final_score=-0.6,
                recommended_tilt=-round(self.max_tilt * 0.8, 4),
                conflict_type="CONFLICT_CROSS_BORDER_PREMIUM",
                disagreement_index=disagreement,
                confidence_multiplier=0.9,
                dominant_channel="REGULATORY_POLICY",
                resolution_rationale=reason,
                channel_breakdown=breakdown,
            )

        # =====================================================================
        # ARCHETYPE 3: Smart Money Distribution vs Retail FOMO Trap (主力派发 vs 散户接盘)
        # =====================================================================
        if flow_score <= -0.35 and social_score >= 0.45:
            # Smart money selling heavily while retail is euphoric
            resolved = max(-1.0, flow_score - 0.35 * social_score)
            tilt = -self.max_tilt
            reason = (
                f"识别到【诱多派发背离】: 主力资金显著流出({flow_score:+.2f})而散户社区狂热看多({social_score:+.2f})，"
                f"坚决跟随主力真金白银方向并反向惩罚FOMO情绪，执行防守减仓"
            )
            return ReconciliationResult(
                asset_code=asset_code,
                final_score=round(resolved, 4),
                recommended_tilt=tilt,
                conflict_type="CONFLICT_DISTRIBUTION_TRAP",
                disagreement_index=disagreement,
                confidence_multiplier=0.95,
                dominant_channel="SMART_MONEY_FLOW",
                resolution_rationale=reason,
                channel_breakdown=breakdown,
            )

        # =====================================================================
        # ARCHETYPE 4: Contrarian Bottom Accumulation (主力吸筹+低估 vs 散户恐慌割肉)
        # =====================================================================
        institutional_anchor = 0.55 * flow_score + 0.45 * val_score
        if institutional_anchor >= 0.25 and social_score <= -0.45:
            # Smart money & valuation positive, retail in deep capitulation panic
            resolved = min(1.0, institutional_anchor - 0.25 * social_score)  # negative social boosts score
            tilt = +self.max_tilt
            reason = (
                f"识别到【黄金坑底部背离】: 基本面与主力资金稳步吸筹(锚定值{institutional_anchor:+.2f})，"
                f"而散户舆情恐慌割肉({social_score:+.2f})构成极佳逆向买点，执行右侧/定投增配"
            )
            return ReconciliationResult(
                asset_code=asset_code,
                final_score=round(resolved, 4),
                recommended_tilt=tilt,
                conflict_type="CONFLICT_CONTRARIAN_BOTTOM",
                disagreement_index=disagreement,
                confidence_multiplier=0.95,
                dominant_channel="FUNDAMENTAL_VALUATION" if val_score > flow_score else "SMART_MONEY_FLOW",
                resolution_rationale=reason,
                channel_breakdown=breakdown,
            )

        # =====================================================================
        # ARCHETYPE 5: Standard Hierarchical Synthesis with Disagreement Damping
        # =====================================================================
        # Notice retail social is weighted with a negative contrarian sign when extreme (>0.6),
        # or mild momentum when moderate.
        social_contribution = -0.15 * social_score if abs(social_score) > 0.6 else 0.05 * social_score
        raw_composite = (
            0.45 * flow_score
            + 0.35 * val_score
            + 0.10 * news_score
            + social_contribution
        )

        # Uncertainty Attenuation: high disagreement shrinks bet size toward zero
        conf_mult = round(math.exp(-self.dispersion_damping * (disagreement ** 2)), 4)
        conf_mult = max(0.25, min(1.0, conf_mult))

        final_score = round(max(-1.0, min(1.0, raw_composite * conf_mult)), 4)

        # Convert final_score into bounded weight tilt
        if abs(final_score) < 0.15:
            tilt = 0.0
        else:
            tilt = round(max(-self.max_tilt, min(self.max_tilt, final_score * self.max_tilt * 1.5)), 4)

        # Determine dominant channel
        dominant = "SMART_MONEY_FLOW"
        if abs(val_score) > abs(flow_score) and abs(val_score) > 0.2:
            dominant = "FUNDAMENTAL_VALUATION"

        if disagreement >= 0.45:
            conflict_label = "CONFLICT_HIGH_DISPERSION_DAMPED"
            reason = (
                f"多源信号存在分歧(分歧度 {disagreement:.2f})，启动不确定性阻尼(置信折算 {conf_mult:.0%})，"
                f"以主力与估值锚({dominant})为主导审慎微调"
            )
        else:
            conflict_label = "NO_CONFLICT_CONSENSUS"
            reason = f"多源情报方向一致(分歧度低 {disagreement:.2f})，综合评分 {final_score:+.2f}"

        return ReconciliationResult(
            asset_code=asset_code,
            final_score=final_score,
            recommended_tilt=tilt,
            conflict_type=conflict_label,
            disagreement_index=disagreement,
            confidence_multiplier=conf_mult,
            dominant_channel=dominant,
            resolution_rationale=reason,
            channel_breakdown=breakdown,
        )


ReconciledSignal = ReconciliationResult


def compute_belief_entropy(probabilities: list[float]) -> float:
    """Compute Shannon belief entropy across multi-channel probability weights: -sum(p * log2(p)).

    Higher entropy reflects severe information divergence / disagreement across channels.
    """
    valid = [p for p in probabilities if p > 0.0]
    if not valid:
        return 0.0
    total = sum(valid)
    norm = [p / total for p in valid]
    return round(-sum(p * math.log2(p) for p in norm if p > 0), 4)


def reconcile_views(
    asset_code: str,
    signals: list[ChannelSignal],
    qdii_premium_pct: float = 0.0,
    reconciler: SignalReconciler | None = None,
) -> ReconciliationResult:
    """Resolve conflicting signals across macro, valuation, smart money, and sentiment channels.

    Applies the 5-tier hierarchical conflict resolution rules:
    1. Macro & Regulatory Veto (Hard circuit breaker for policy risk or extreme QDII premium)
    2. Cross-Border Premium Disconnect (US tech hype vs domestic premium risk)
    3. Distribution Trap (Smart Money Selling vs Retail Euphoria -> defensive tilt)
    4. Contrarian Bottom (Institutional Accumulation + Valuation vs Retail Panic -> contrarian buy)
    5. Hierarchical Consensus with Disagreement Uncertainty Damping

    Args:
        asset_code: Security ticker (e.g. "510300", "513100").
        signals: List of ChannelSignal observations from various market channels.
        qdii_premium_pct: Domestic ETF market premium percentage over IOPV/NAV.
        reconciler: Optional pre-configured SignalReconciler instance.

    Returns:
        ReconciliationResult (or ReconciledSignal) with resolved score, tilt, and rationale.
    """
    if reconciler is None:
        reconciler = SignalReconciler()
    return reconciler.reconcile_asset_signals(asset_code, signals, qdii_premium_pct=qdii_premium_pct)


__all__ = [
    "CHANNEL_BASE_WEIGHTS",
    "ChannelSignal",
    "ReconciliationResult",
    "ReconciledSignal",
    "SignalReconciler",
    "compute_belief_entropy",
    "reconcile_views",
]


if __name__ == "__main__":
    reconciler = SignalReconciler(max_tilt=0.015)

    print("Testing Multi-Source Conflict Resolution Engine:\n")

    # Case 1: Distribution Trap (Smart Money Selling vs Retail Euphoria)
    case1 = [
        ChannelSignal("SMART_MONEY_FLOW", score=-0.75, evidence="Northbound outflow -6.2B RMB"),
        ChannelSignal("FUNDAMENTAL_VALUATION", score=-0.20, evidence="PE at 75th percentile"),
        ChannelSignal("RETAIL_SOCIAL_CN", score=+0.85, evidence="Xueqiu retail screaming 'To the moon!'"),
    ]
    res1 = reconciler.reconcile_asset_signals("510300", case1)
    print(f"[Case 1: 510300 Distribution Trap]")
    print(f"  Conflict Type: {res1.conflict_type} | Disagreement: {res1.disagreement_index:.2f}")
    print(f"  Final Score: {res1.final_score:+.2f} | Recommended Tilt: {res1.recommended_tilt*100:+.2f}%")
    print(f"  Rationale: {res1.resolution_rationale}\n")

    # Case 2: Contrarian Bottom (Smart Money + Valuation Buying vs Retail Capitulation Panic)
    case2 = [
        ChannelSignal("SMART_MONEY_FLOW", score=+0.60, evidence="Northbound inflow +4.5B RMB"),
        ChannelSignal("FUNDAMENTAL_VALUATION", score=+0.80, evidence="Dividend yield 4.8%, 10y bottom"),
        ChannelSignal("RETAIL_SOCIAL_CN", score=-0.85, evidence="Retail capitulation panic selling"),
    ]
    res2 = reconciler.reconcile_asset_signals("510880", case2)
    print(f"[Case 2: 510880 Contrarian Bottom]")
    print(f"  Conflict Type: {res2.conflict_type} | Disagreement: {res2.disagreement_index:.2f}")
    print(f"  Final Score: {res2.final_score:+.2f} | Recommended Tilt: {res2.recommended_tilt*100:+.2f}%")
    print(f"  Rationale: {res2.resolution_rationale}\n")

    # Case 3: Cross-Border Disconnect (US StockTwits Bullish vs Domestic QDII High Premium)
    case3 = [
        ChannelSignal("RETAIL_SOCIAL_US", score=+0.75, evidence="StockTwits $NVDA/$QQQ 85% Bullish"),
        ChannelSignal("SMART_MONEY_FLOW", score=+0.20, evidence="Moderate inflow"),
    ]
    res3 = reconciler.reconcile_asset_signals("513100", case3, qdii_premium_pct=2.80)
    print(f"[Case 3: 513100 Cross-Border QDII Veto]")
    print(f"  Conflict Type: {res3.conflict_type} | Disagreement: {res3.disagreement_index:.2f}")
    print(f"  Final Score: {res3.final_score:+.2f} | Recommended Tilt: {res3.recommended_tilt*100:+.2f}%")
    print(f"  Rationale: {res3.resolution_rationale}\n")
