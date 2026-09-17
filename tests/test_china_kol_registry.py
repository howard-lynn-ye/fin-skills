"""Unit tests for KOL Credibility Registry and Bayesian Track-Record Weighting Engine."""

import os
from pathlib import Path

from fin_skills.china.kol_credibility_registry import (
    KOLCredibilityRegistry,
    KOLProfile,
    KOLWeightedSentimentResult,
    TIER_WEIGHT_MAP,
)


def test_kol_registry_loads_profiles() -> None:
    """Verify that the KOL registry initializes and contains core profiles."""
    registry = KOLCredibilityRegistry()
    assert len(registry.profiles) > 0
    assert "cn_elite_01" in registry.profiles
    assert "cn_contrarian_01" in registry.profiles
    assert "cn_media_04" in registry.profiles


def test_elite_kol_amplification() -> None:
    """Verify that Tier 0/1 alpha KOLs receive positive directional amplification."""
    registry = KOLCredibilityRegistry()
    profile = registry.get_author_profile("cn_elite_01")
    assert isinstance(profile, KOLProfile)
    assert profile.tier in ("TIER_0_ELITE_KOL", "TIER_1_CORE_ALPHA")
    assert profile.win_rate_5d > 0.65
    assert profile.directional_weight >= 2.5

    res = registry.evaluate_weighted_sentiment([{"author": "cn_elite_01", "polarity": -0.80}])
    assert isinstance(res, KOLWeightedSentimentResult)
    assert res.elite_alpha_count == 1
    assert res.kol_weighted_polarity < -0.70


def test_contrarian_indicator_inversion() -> None:
    """Verify that Contrarian Indicators ('反向明灯') have their directional signals inverted."""
    registry = KOLCredibilityRegistry()
    profile = registry.get_author_profile("cn_contrarian_01")
    assert profile.tier == "TIER_CONTRARIAN_INDICATOR"
    assert profile.is_contrarian is True
    assert profile.fans > 100000
    assert profile.win_rate_5d <= 0.45
    assert profile.directional_weight < 0.0

    # Bullish FOMO call (+0.90) from a contrarian indicator must invert to bearish (< 0)
    res = registry.evaluate_weighted_sentiment([{"author": "cn_contrarian_01", "polarity": 0.90}])
    assert res.contrarian_inverted_count == 1
    assert res.raw_unweighted_polarity > 0.80
    assert res.kol_weighted_polarity < -0.50


def test_media_aggregator_neutralization() -> None:
    """Verify that financial media accounts have zero directional weight."""
    registry = KOLCredibilityRegistry()
    profile = registry.get_author_profile("cn_media_04")
    assert profile.tier == "TIER_MEDIA_AGGREGATOR"
    assert profile.is_media is True
    assert profile.directional_weight == 0.0

    res = registry.evaluate_weighted_sentiment([{"author": "cn_media_04", "polarity": 0.85}])
    assert res.media_neutralized_count == 1
    assert res.kol_weighted_polarity == 0.0


def test_evaluate_weighted_sentiment_divergence_resolution() -> None:
    """Verify that KOL credibility weighting resolves retail/contrarian traps."""
    registry = KOLCredibilityRegistry()
    posts = [
        # Two mega-follower contrarians shouting bullish
        {"author": "cn_contrarian_01", "polarity": 0.90},
        {"author": "cn_contrarian_08", "polarity": 0.85},
        # Media aggregator reporting bullish headline
        {"author": "cn_media_04", "polarity": 0.60},
        # Two high-win-rate alpha researchers warning of downside
        {"author": "cn_elite_01", "polarity": -0.75},
        {"author": "cn_core_01", "polarity": -0.65},
    ]

    result = registry.evaluate_weighted_sentiment(posts)
    # Naive unweighted sentiment is fooled into a bullish trap (+0.19)
    assert result.raw_unweighted_polarity > 0.15
    # Credibility-weighted sentiment flips to strong bearish conviction (< -0.50)
    assert result.kol_weighted_polarity < -0.50
    assert result.elite_alpha_count == 2
    assert result.contrarian_inverted_count == 2
    assert result.media_neutralized_count == 1


def test_sentiment_flow_collector_kol_integration() -> None:
    """Verify that compute_asset_sentiments applies KOL credibility weighting when authors are provided."""
    from fin_skills.china.sentiment_flow_collector import compute_asset_sentiments

    raw_posts = [
        {
            "author": "cn_contrarian_01",
            "text": "沪深300(510300)已经全面突破，全仓满融梭哈冲！牛市暴涨就在眼前！",
            "verified": True,
        },
        {
            "author": "cn_contrarian_08",
            "text": "沪深300(510300)大盘即将起飞，千金难买牛回头，赶紧追高买入！",
            "verified": True,
        },
        {
            "author": "cn_elite_01",
            "text": "沪深300(510300)当前短期情绪过热，PE分位数修复过快，警惕短期获利回吐风险，建议逢高减仓防守。",
            "verified": True,
        },
        {
            "author": "cn_core_01",
            "text": "沪深300(510300)估值性价比短期下降，基本面盈利增速尚未跟上，建议降低权益仓位。",
            "verified": True,
        },
    ]

    sentiments = compute_asset_sentiments(raw_social_posts=raw_posts)
    assert "510300" in sentiments
    metric = sentiments["510300"]
    # Even though 2 mega-influencer posts are bullish and 2 research posts are bearish,
    # KOL credibility weighting flips the contrarians negative and amplifies the alpha KOLs!
    assert metric.sentiment_score < -0.30
    assert "大V信誉加权" in metric.rationale


def test_overseas_global_kol_credibility() -> None:
    """Verify that overseas (StockTwits / FinTwit) KOLs are loaded and properly weighted."""
    registry = KOLCredibilityRegistry()
    assert len(registry.profiles) >= 40
    has_full_catalog = any(
        p.exists()
        for p in [
            Path(os.environ.get("FIN_SKILLS_BENCHMARK_DIR", "../stock_prediction/data/benchmark")) / "GLOBAL_KOL_ALPHA_PROFILES.csv",
            Path(__file__).resolve().parents[1] / "data/benchmark/GLOBAL_KOL_ALPHA_PROFILES.csv",
            Path(__file__).resolve().parents[2] / "data/benchmark/GLOBAL_KOL_ALPHA_PROFILES.csv",
        ]
    )
    if has_full_catalog:
        assert len(registry.profiles) >= 2000

    # 1. Overseas Elite KOL amplification
    elite = registry.get_author_profile("@us_elite_01")
    assert elite.tier == "TIER_0_ELITE_KOL"
    assert elite.directional_weight == 3.0
    assert elite.win_rate_5d > 0.85

    # 2. Overseas Contrarian Indicator inversion (e.g. Jim Cramer & us_contrarian_02)
    cramer = registry.get_author_profile("us_contrarian_01")
    assert cramer.tier == "TIER_CONTRARIAN_INDICATOR"
    assert cramer.is_contrarian is True
    assert cramer.directional_weight < 0.0

    jfdi = registry.get_author_profile("@us_contrarian_02")
    assert jfdi.tier == "TIER_CONTRARIAN_INDICATOR"
    assert jfdi.directional_weight < 0.0

    # 3. Overseas Breaking News aggregator neutralization
    delta = registry.get_author_profile("@us_media_01")
    assert delta.tier == "TIER_MEDIA_AGGREGATOR"
    assert delta.directional_weight == 0.0

    # Test evaluation on an overseas QDII / US Tech basket (e.g. NVDA / QQQ)
    overseas_posts = [
        {"author": "@us_contrarian_01", "polarity": 0.95},        # Bullish call from Cramer -> inverted to bearish!
        {"author": "@us_contrarian_02", "polarity": 0.85},             # Bullish call from us_contrarian_02 -> inverted to bearish!
        {"author": "@us_media_01", "polarity": 0.70},         # Breaking news -> direction stripped (0.0x)
        {"author": "@us_elite_01", "polarity": -0.80},       # Bearish warning from 89.8% win-rate researcher -> amplified 3.0x!
    ]
    res = registry.evaluate_weighted_sentiment(overseas_posts)
    assert res.raw_unweighted_polarity > 0.40  # Naive average is fooled into +0.425 bullish
    assert res.kol_weighted_polarity < -0.60   # KOL credibility weighting flips to -0.78 strong bearish!
