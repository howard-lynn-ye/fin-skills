"""Unit tests for institutional-grade data cleaning & anti-misinformation filter."""
from __future__ import annotations

import pytest
from fin_skills.china.data_cleaning_filter import (
    DataQualityAuditor,
    TextAuditResult,
    filter_high_signal_posts,
)
from fin_skills.china.sentiment_flow_collector import (
    collect_daily_market_intel,
    compute_asset_sentiments,
)


@pytest.fixture
def auditor() -> DataQualityAuditor:
    return DataQualityAuditor(min_clean_length=10, min_quality_threshold=0.35)


def test_sanitization(auditor: DataQualityAuditor):
    raw = "<p>回复 @股神123: 看好 $贵州茅台(SH600519)$ [赞][哈哈] 链接: https://example.com/xyz</p>"
    cleaned = auditor.sanitize(raw)
    assert "https://" not in cleaned
    assert "<p>" not in cleaned
    assert "[赞]" not in cleaned
    assert "回复 @" not in cleaned
    assert "贵州茅台" in cleaned


def test_spam_rejection(auditor: DataQualityAuditor):
    spam_samples = [
        "加V领明天开盘必涨停金股，内部指导微信: vip888999",
        "带单进群，资深名师指导翻倍黑马战法，扫码私信获取！",
        "专业代客理财，免费领内部票，扣群: 12345678",
    ]
    for s in spam_samples:
        res = auditor.audit_text(s)
        assert not res.is_valid
        assert res.rejection_reason == "SPAM_SOLICITATION"
        assert res.quality_score == 0.0


def test_clickbait_rejection(auditor: DataQualityAuditor):
    clickbait_samples = [
        "【重磅内幕】某万亿大白马即将突发爆雷！内部速看！",
        "震惊！刚刚突发大动作，惊天利空来袭，惊掉下巴！",
        "绝密小作文疯传，惊天利好即将引爆A股！",
    ]
    for s in clickbait_samples:
        res = auditor.audit_text(s)
        assert not res.is_valid
        assert res.rejection_reason == "CLICKBAIT_RUMOR"
        assert "CLICKBAIT_SENSATIONALISM" in res.flags


def test_sarcasm_detection_and_inversion(auditor: DataQualityAuditor):
    sarcasm_post = "主力可真是大善人啊，连续跌停让我抄底，跌得太棒了，多谢套牢！"
    res = auditor.audit_text(sarcasm_post)
    assert "CYNICAL_SARCASM" in res.flags
    # Apparent praise should be inverted into negative sentiment
    assert res.sentiment_polarity < 0.0


def test_information_density_and_thesis_pass(auditor: DataQualityAuditor):
    high_signal_sample = (
        "沪深300当前动态PE为11.8倍，股息率达到3.2%，处于近10年20%分位数，"
        "安全边际充裕，建议分批加仓定投配置。"
    )
    res = auditor.audit_text(high_signal_sample)
    assert res.is_valid
    assert res.rejection_reason == "CLEAN_AND_VALID"
    assert res.has_substantive_thesis
    assert res.quality_score >= 0.70
    assert res.sentiment_polarity > 0.0
    assert len(res.detected_metrics) >= 3


def test_too_short_noise_rejection(auditor: DataQualityAuditor):
    short_posts = ["冲！", "牛逼", "吃面了", "赶紧跑"]
    for s in short_posts:
        res = auditor.audit_text(s)
        assert not res.is_valid
        assert res.rejection_reason == "TOO_SHORT_NOISE"


def test_filter_high_signal_posts_batch(auditor: DataQualityAuditor):
    posts = [
        {"text": "加微领大牛股 13800000000", "verified": False},
        {"text": "【震惊】惊天大爆料小作文速看！", "verified": False},
        {"text": "央行公开市场开展2000亿元逆回购操作，保持流动性充裕，银行间利率平稳。", "verified": True},
        {"text": "红利ETF股息率达到4.5%，具备类固收防御属性，低估值下逢低配置加仓。", "verified": True},
        {"text": "太牛了！", "verified": False},
    ]
    clean_records, stats = filter_high_signal_posts(posts, auditor)
    assert len(clean_records) == 2
    assert stats["total_evaluated"] == 5
    assert stats["clean_valid_passed"] == 2
    assert stats["spam_rejected"] == 1
    assert stats["clickbait_rejected"] == 1
    assert stats["too_short_rejected"] == 1


def test_sentiment_flow_collector_integration(auditor: DataQualityAuditor):
    posts = [
        "红利ETF当前股息率超4.2%，安全边际显著，持续定投分批买入配置。",
        "纳指100高位冲高，连续涨停溢价过热，极度贪婪追高风险大！",
        "加V领明日必涨牛股微信: topstock123",  # Spam
        "【震惊】某资产即将暴跌，小作文内部爆料！",  # Clickbait
    ]
    report = collect_daily_market_intel(
        cutoff_time="14:30:00",
        raw_social_posts=posts,
        auditor=auditor,
    )
    assert report.cleaning_stats["total_evaluated"] >= 4
    assert report.cleaning_stats["spam_rejected"] >= 1
    assert report.cleaning_stats["clickbait_rejected"] >= 1
    assert report.cleaning_stats["clean_valid_passed"] >= 1

    # Check Markdown report formatting
    md = report.to_markdown()
    assert "数据源清洗与防误导审计" in md
    assert "拦截导流广告" in md
    assert "拦截虚假爆料" in md
