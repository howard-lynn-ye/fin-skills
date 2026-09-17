"""Tests for fin_skills.china.sentiment_flow_collector."""
from __future__ import annotations

from fin_skills.china.sentiment_flow_collector import apply_sentiment_overlay


def test_sentiment_flow_collector_overlay():
    base = {"510300": 0.50, "511010": 0.50}
    tilted = apply_sentiment_overlay(base, {}, max_tilt=0.015)
    assert sum(tilted.values()) == 1.0
