"""Tests for fin_skills.china.kol_credibility_registry."""
from __future__ import annotations

from fin_skills.china.kol_credibility_registry import KOLCredibilityRegistry


def test_kol_credibility_registry_basic():
    registry = KOLCredibilityRegistry()
    prof = registry.get_author_profile("cn_elite_01")
    assert prof.tier in ("TIER_0_ELITE_KOL", "TIER_1_CORE_ALPHA")
    assert prof.directional_weight >= 2.5
