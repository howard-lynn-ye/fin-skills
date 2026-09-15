"""Unit tests for KOL Credibility Registry and Bayesian Track-Record Weighting Engine."""

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
    assert "阿尔法工场" in registry.profiles
    assert "钟华守正出奇" in registry.profiles
    assert "财联社" in registry.profiles


def test_elite_kol_amplification() -> None:
    """Verify that Tier 0/1 alpha KOLs receive positive directional amplification."""
    registry = KOLCredibilityRegistry()
    profile = registry.get_author_profile("阿尔法工场")
    assert isinstance(profile, KOLProfile)
    assert profile.tier in ("TIER_0_ELITE_KOL", "TIER_1_CORE_ALPHA")
    assert profile.win_rate_5d > 0.65
    assert profile.directional_weight >= 2.5

    res = registry.evaluate_weighted_sentiment([{"author": "阿尔法工场", "polarity": -0.80}])
    assert isinstance(res, KOLWeightedSentimentResult)
    assert res.elite_alpha_count == 1
    assert res.kol_weighted_polarity < -0.70


def test_contrarian_indicator_inversion() -> None:
    """Verify that Contrarian Indicators ('反向明灯') have their directional signals inverted."""
    registry = KOLCredibilityRegistry()
    profile = registry.get_author_profile("钟华守正出奇")
    assert profile.tier == "TIER_CONTRARIAN_INDICATOR"
    assert profile.is_contrarian is True
    assert profile.fans > 100000
    assert profile.win_rate_5d <= 0.45
    assert profile.directional_weight < 0.0

    # Bullish FOMO call (+0.90) from a contrarian indicator must invert to bearish (< 0)
    res = registry.evaluate_weighted_sentiment([{"author": "钟华守正出奇", "polarity": 0.90}])
    assert res.contrarian_inverted_count == 1
    assert res.raw_unweighted_polarity > 0.80
    assert res.kol_weighted_polarity < -0.50


def test_media_aggregator_neutralization() -> None:
    """Verify that financial media accounts have zero directional weight."""
    registry = KOLCredibilityRegistry()
    profile = registry.get_author_profile("财联社")
    assert profile.tier == "TIER_MEDIA_AGGREGATOR"
    assert profile.is_media is True
    assert profile.directional_weight == 0.0

    res = registry.evaluate_weighted_sentiment([{"author": "财联社", "polarity": 0.85}])
    assert res.media_neutralized_count == 1
    assert res.kol_weighted_polarity == 0.0


def test_evaluate_weighted_sentiment_divergence_resolution() -> None:
    """Verify that KOL credibility weighting resolves retail/contrarian traps."""
    registry = KOLCredibilityRegistry()
    posts = [
        # Two mega-follower contrarians shouting bullish
        {"author": "钟华守正出奇", "polarity": 0.90},
        {"author": "东先生", "polarity": 0.85},
        # Media aggregator reporting bullish headline
        {"author": "财联社", "polarity": 0.60},
        # Two high-win-rate alpha researchers warning of downside
        {"author": "阿尔法工场", "polarity": -0.75},
        {"author": "价投傻鱼", "polarity": -0.65},
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
            "author": "钟华守正出奇",
            "text": "沪深300(510300)已经全面突破，全仓满融梭哈冲！牛市暴涨就在眼前！",
            "verified": True,
        },
        {
            "author": "东先生",
            "text": "沪深300(510300)大盘即将起飞，千金难买牛回头，赶紧追高买入！",
            "verified": True,
        },
        {
            "author": "阿尔法工场",
            "text": "沪深300(510300)当前短期情绪过热，PE分位数修复过快，警惕短期获利回吐风险，建议逢高减仓防守。",
            "verified": True,
        },
        {
            "author": "价投傻鱼",
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
