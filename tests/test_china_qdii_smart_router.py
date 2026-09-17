"""Unit tests for QDIISmartRouter (QDII Smart Substitution Router)."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fin_skills.china.qdii_smart_router import (
    DEFAULT_QDII_REPLACEMENT_GRAPH,
    QDIIRouteResult,
    QDIISmartRouter,
    QDIISubstitutionEvent,
    QDII_Substitution_Event,
)


def test_qdii_smart_router_graph_and_normalization():
    """Verify default replacement graph contains all required QDII primary ETFs."""
    router = QDIISmartRouter()
    assert "513100" in router.replacement_graph
    assert "513500" in router.replacement_graph
    assert "510900" in router.replacement_graph
    assert "159920" in router.replacement_graph

    # Verify 513100 candidate list
    assert "159509" in router.replacement_graph["513100"]
    assert "513870" in router.replacement_graph["513100"]
    assert "513500" in router.replacement_graph["513100"]
    assert "000043" in router.replacement_graph["513100"]

    # Verify 513500 candidate list
    assert "513520" in router.replacement_graph["513500"]
    assert "513100" in router.replacement_graph["513500"]
    assert "006075" in router.replacement_graph["513500"]

    # Check normalization
    assert router._normalize_symbol("513100.SH") == "513100"
    assert router._normalize_symbol("159509.SZ") == "159509"
    assert router._normalize_premium(1.8, "513100") == pytest.approx(0.018, abs=1e-6)
    assert router._normalize_premium(0.018, "513100") == pytest.approx(0.018, abs=1e-6)


def test_qdii_pass_through_when_premium_below_threshold():
    """Verify normal premiums (<= 1.5%) pass through original symbol unchanged."""
    router = QDIISmartRouter(premium_threshold_pct=0.015)

    # 1. Premium 1.2% <= 1.5%
    res = router.evaluate_and_route("513100", 0.075, {"513100": 0.012})
    assert not res.is_substituted
    assert res.routed_symbol == "513100"
    assert res.routed_weight == 0.075
    assert res.event is None

    # 2. Non-monitored domestic asset (e.g. 510300) passes through
    res_dom = router.evaluate_and_route("510300", 0.20, {"510300": 0.025})
    assert not res_dom.is_substituted
    assert res_dom.routed_symbol == "510300"
    assert res_dom.routed_weight == 0.20
    assert res_dom.event is None

    # 3. Zero weight passes through
    res_zero = router.evaluate_and_route("513100", 0.0, {"513100": 0.050})
    assert not res_zero.is_substituted
    assert res_zero.routed_weight == 0.0


def test_qdii_substitution_triggers_and_lowest_premium_selection():
    """Verify premium > 1.5% triggers substitution and selects lowest-premium candidate."""
    router = QDIISmartRouter(premium_threshold_pct=0.015, low_premium_threshold_pct=0.005)

    # 513100 has 3.5% premium.
    # Candidates: 159509 (0.2%), 513870 (1.1%), 513500 (2.5%)
    premiums = {
        "513100": 0.035,
        "159509": 0.002,  # <= 0.5% (ultra-low)
        "513870": 0.011,  # <= 1.5%
        "513500": 0.025,  # > 1.5% (bubbly)
    }
    res = router.evaluate_and_route("513100", 0.075, premiums)
    assert res.is_substituted
    assert res.routed_symbol == "159509"
    assert res.routed_weight == 0.075
    assert res.event is not None

    ev = res.event
    assert ev.original_symbol == "513100"
    assert ev.substitute_symbol == "159509"
    assert ev.original_premium_pct == pytest.approx(0.035, abs=1e-4)
    assert ev.substitute_premium_pct == pytest.approx(0.002, abs=1e-4)
    # Avoided friction: (0.035 - 0.002) * 10000 = 330 bps
    assert ev.avoided_friction_bps == pytest.approx(330.0, abs=1e-2)
    assert ev.status == "SUBSTITUTED"
    assert "突破安全阀值" in ev.reason

    # Tuple unpacking test
    routed_sym, routed_w, event_obj = res
    assert routed_sym == "159509"
    assert routed_w == 0.075
    assert event_obj is ev


def test_qdii_otc_feeder_fund_substitution():
    """Verify OTC feeder funds (0% premium) can be chosen as safe lowest-premium equivalents."""
    router = QDIISmartRouter(premium_threshold_pct=0.015)

    # 513500 has 3.6% premium. 513520 (2.2%), 513100 (2.5%), 006075 OTC Feeder (0.0%)
    premiums = {
        "513500": 0.036,
        "513520": 0.022,
        "513100": 0.025,
        "006075": 0.000,
    }
    res = router.evaluate_and_route("513500", 0.075, premiums)
    assert res.is_substituted
    assert res.routed_symbol == "006075"
    assert res.event.avoided_friction_bps == pytest.approx(360.0, abs=1e-2)
    assert res.event.substitute_premium_pct == 0.0


def test_qdii_fallback_to_bond_etf_511010_when_all_substitutes_bubble():
    """Verify fallback to Treasury Bond ETF 511010 when all substitutes exceed 1.5% premium."""
    router = QDIISmartRouter(premium_threshold_pct=0.015, fallback_safety_symbol="511010")

    # 513100 is at 4.5% premium. All candidate equivalents are also > 1.5% bubble
    premiums = {
        "513100": 0.045,
        "159509": 0.028,
        "513870": 0.032,
        "513500": 0.025,
    }
    res = router.evaluate_and_route("513100", 0.075, premiums)
    assert res.is_substituted
    assert res.routed_symbol == "511010"
    assert res.routed_weight == 0.075
    assert res.event is not None
    assert res.event.status == "FALLBACK_SAFETY_BOND"
    # Avoided friction vs 511010 (0% premium): 450 bps
    assert res.event.avoided_friction_bps == pytest.approx(450.0, abs=1e-2)
    assert "触发硬核熔断拦截" in res.event.reason


def test_qdii_route_portfolio_weights_aggregation_and_sum_invariant():
    """Verify route_portfolio_weights aggregates weights and strictly preserves sum invariant."""
    router = QDIISmartRouter(premium_threshold_pct=0.015)
    base_weights = {
        "510300": 0.20,
        "511010": 0.40,
        "513100": 0.075,
        "513500": 0.075,
    }
    # 513100 substitutes to 159509 (0.2%)
    # 513500 falls back to 511010 (all substitutes bubbly)
    premiums = {
        "513100": 0.035,
        "159509": 0.002,
        "513500": 0.040,
        "513520": 0.025,
        "511010": 0.0,
    }
    routed_w, events = router.route_portfolio_weights(base_weights, premiums)

    assert "513100" not in routed_w
    assert "513500" not in routed_w
    assert routed_w["159509"] == pytest.approx(0.075, abs=1e-6)
    # 511010 received its original 0.40 + 0.075 redirected from bubbly 513500 = 0.475
    assert routed_w["511010"] == pytest.approx(0.475, abs=1e-6)
    assert routed_w["510300"] == pytest.approx(0.20, abs=1e-6)
    assert sum(routed_w.values()) == pytest.approx(sum(base_weights.values()), abs=1e-6)
    assert len(events) == 2


def test_qdii_format_substitution_report():
    """Verify Markdown report generation."""
    router = QDIISmartRouter()

    # Empty events
    md_empty = router.format_substitution_report([])
    assert "#### 🔄 QDII 智能溢价平替与防泡沫调度" in md_empty
    assert "安全阀值" in md_empty

    # With events
    event = QDIISubstitutionEvent(
        original_symbol="513100",
        substitute_symbol="159509",
        original_name="纳指100ETF",
        substitute_name="纳斯达克科技ETF",
        original_premium_pct=0.035,
        substitute_premium_pct=0.002,
        avoided_friction_bps=330.0,
        routed_weight=0.075,
        status="SUBSTITUTED",
        reason="测试平替",
    )
    md_active = router.format_substitution_report([event])
    assert "#### 🔄 QDII 智能溢价平替与防泡沫调度" in md_active
    assert "513100" in md_active
    assert "159509" in md_active
    assert "+330 bps" in md_active
    assert "防泡沫调度成效" in md_active
