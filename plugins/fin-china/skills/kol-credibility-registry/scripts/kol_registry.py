"""fin_skills.china.kol_registry - re-export of KOL credibility registry symbols.

Dual-import compatibility wrapper providing:
- KOLCredibilityRegistry
- KOLProfile
- KOLWeightedSentimentResult
- TIER_WEIGHT_MAP
- calibrate_sentiment
- compute_brier_score
- update_bayesian_track_record
"""
from __future__ import annotations

try:
    from .kol_credibility_registry import (
        EMBEDDED_KOL_PROFILES,
        KOLCredibilityRegistry,
        KOLProfile,
        KOLWeightedSentimentResult,
        TIER_WEIGHT_MAP,
        calibrate_sentiment,
        compute_brier_score,
        update_bayesian_track_record,
    )
except ImportError:
    from kol_credibility_registry import (
        EMBEDDED_KOL_PROFILES,
        KOLCredibilityRegistry,
        KOLProfile,
        KOLWeightedSentimentResult,
        TIER_WEIGHT_MAP,
        calibrate_sentiment,
        compute_brier_score,
        update_bayesian_track_record,
    )

__all__ = [
    "EMBEDDED_KOL_PROFILES",
    "KOLCredibilityRegistry",
    "KOLProfile",
    "KOLWeightedSentimentResult",
    "TIER_WEIGHT_MAP",
    "calibrate_sentiment",
    "compute_brier_score",
    "update_bayesian_track_record",
]
