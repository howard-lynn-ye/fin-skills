"""Tests for fin_skills.china.data_cleaning_filter."""
from __future__ import annotations

from fin_skills.china.data_cleaning_filter import DataQualityAuditor


def test_data_quality_auditor_basic():
    auditor = DataQualityAuditor()
    res = auditor.audit_text("沪深300当前PE为11.8倍，股息率达到3.2%，估值处于近10年低位，建议买入配置。")
    assert res.is_valid
    assert res.quality_score >= 0.35
