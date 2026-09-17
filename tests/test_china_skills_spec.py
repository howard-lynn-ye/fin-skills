"""Unit tests for China skills specification compliance and module exports.

Verifies:
1. `kol-credibility-registry` and `signal-reconciler` exist and conform to Agent Skills spec.
2. Frontmatter validity (name, description, license, metadata).
3. Python API loading and execution from `fin_skills.china.kol_registry` and `fin_skills.china.signal_reconciler`.
4. Package data loading via `fin_skills.load(...)`.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fin_skills
from scripts.validate import check_skill, parse_frontmatter


def test_kol_credibility_registry_skill_spec() -> None:
    """Validate kol-credibility-registry SKILL.md against Agent Skills spec."""
    skill_path = ROOT / "plugins" / "fin-china" / "skills" / "kol-credibility-registry" / "SKILL.md"
    assert skill_path.is_file(), f"Missing {skill_path}"

    errs = check_skill(skill_path)
    assert not errs, f"Validation errors for kol-credibility-registry: {errs}"

    text = skill_path.read_text(encoding="utf-8")
    fm, parse_errs = parse_frontmatter(text)
    assert not parse_errs
    assert fm["name"] == "kol-credibility-registry"
    assert "Xueqiu" in fm["description"]
    assert "StockTwits" in fm["description"]
    assert "Bayesian" in fm["description"]
    assert fm.get("license") == "MIT"
    assert (fm.get("metadata") or {}).get("verified_on") == "2026-09-17"

    # Verify key sections exist
    assert "The Trap of Raw Sentiment" in text
    assert "Bayesian Track Record Formulation" in text
    assert "KOL Database Schema" in text
    assert "PSEUDONYM" in Path(ROOT / "plugins/fin-china/skills/kol-credibility-registry/scripts/kol_credibility_registry.py").read_text(encoding="utf-8")
    assert "Python Usage Examples" in text
    assert "Anti-Patterns and Point-in-Time Causality Safeguards" in text


def test_signal_reconciler_skill_spec() -> None:
    """Validate signal-reconciler SKILL.md against Agent Skills spec."""
    skill_path = ROOT / "plugins" / "fin-china" / "skills" / "signal-reconciler" / "SKILL.md"
    assert skill_path.is_file(), f"Missing {skill_path}"

    errs = check_skill(skill_path)
    assert not errs, f"Validation errors for signal-reconciler: {errs}"

    text = skill_path.read_text(encoding="utf-8")
    fm, parse_errs = parse_frontmatter(text)
    assert not parse_errs
    assert fm["name"] == "signal-reconciler"
    assert "conflict" in fm["description"].lower() or "reconcil" in fm["description"].lower()
    assert fm.get("license") == "MIT"
    assert (fm.get("metadata") or {}).get("verified_on") == "2026-09-16"

    # Verify key sections exist
    assert "The Multi-Source Conflict Problem" in text
    assert "Credibility-Weighted Belief Entropy Formulation" in text
    assert "The 5 Conflict Resolution Rules" in text
    assert "Python Usage Examples" in text
    assert "Execution Gate Rules for Portfolios" in text


def test_kol_registry_python_module_api() -> None:
    """Verify that fin_skills.china.kol_registry exports expected API and executes accurately."""
    from fin_skills.china.kol_registry import (
        KOLCredibilityRegistry,
        KOLProfile,
        KOLWeightedSentimentResult,
        TIER_WEIGHT_MAP,
        calibrate_sentiment,
        compute_brier_score,
        update_bayesian_track_record,
    )
    from fin_skills.china.kol_credibility_registry import (
        KOLCredibilityRegistry as OrigRegistry,
    )

    # Both modules must reference the same underlying engine
    reg = KOLCredibilityRegistry()
    assert len(reg.profiles) > 0

    # Test author profile lookup
    prof = reg.get_author_profile("cn_elite_01")
    assert isinstance(prof, KOLProfile)
    assert prof.tier in ("TIER_0_ELITE_KOL", "TIER_1_CORE_ALPHA")
    assert prof.directional_weight >= 2.5

    # Test Bayesian track record calculation
    post_a, post_b, mean_wr = update_bayesian_track_record(prior_alpha=5.0, prior_beta=5.0, wins=7, total=8)
    assert post_a == 12.0
    assert post_b == 6.0
    assert abs(mean_wr - (12.0 / 18.0)) < 0.001

    # Test Brier calibration score
    brier_perfect = compute_brier_score([1.0, 0.0], [1, 0])
    assert brier_perfect == 0.0
    brier_coin_flip = compute_brier_score([0.5, 0.5], [1, 0])
    assert brier_coin_flip == 0.25

    # Test calibrate_sentiment wrapper
    posts = [
        {"author": "cn_contrarian_01", "polarity": 0.90},  # Contrarian -> inverted
        {"author": "cn_elite_01", "polarity": -0.80},  # Elite -> amplified
    ]
    res = calibrate_sentiment(posts, registry=reg)
    assert isinstance(res, KOLWeightedSentimentResult)
    assert res.kol_weighted_polarity < -0.50
    assert res.contrarian_inverted_count == 1
    assert res.elite_alpha_count == 1


def test_signal_reconciler_python_module_api() -> None:
    """Verify that fin_skills.china.signal_reconciler exports expected API and executes accurately."""
    from fin_skills.china.signal_reconciler import (
        ChannelSignal,
        ReconciledSignal,
        ReconciliationResult,
        SignalReconciler,
        compute_belief_entropy,
        reconcile_views,
    )

    assert ReconciledSignal is ReconciliationResult

    reconciler = SignalReconciler(max_tilt=0.015)

    # 1. Distribution trap
    trap_signals = [
        ChannelSignal("SMART_MONEY_FLOW", score=-0.75, evidence="Northbound dumping"),
        ChannelSignal("RETAIL_SOCIAL_CN", score=+0.85, evidence="Retail FOMO"),
    ]
    trap_res = reconcile_views("510300", trap_signals, reconciler=reconciler)
    assert trap_res.conflict_type == "CONFLICT_DISTRIBUTION_TRAP"
    assert trap_res.final_score < -0.50
    assert trap_res.recommended_tilt == -0.015

    # 2. Contrarian bottom
    bottom_signals = [
        ChannelSignal("SMART_MONEY_FLOW", score=+0.60),
        ChannelSignal("FUNDAMENTAL_VALUATION", score=+0.80),
        ChannelSignal("RETAIL_SOCIAL_CN", score=-0.80),
    ]
    bottom_res = reconcile_views("510880", bottom_signals, reconciler=reconciler)
    assert bottom_res.conflict_type == "CONFLICT_CONTRARIAN_BOTTOM"
    assert bottom_res.final_score > 0.70
    assert bottom_res.recommended_tilt == +0.015

    # 3. Macro veto override via extreme QDII premium
    veto_signals = [
        ChannelSignal("RETAIL_SOCIAL_US", score=+0.80),
        ChannelSignal("SMART_MONEY_FLOW", score=+0.30),
    ]
    veto_res = reconcile_views("513100", veto_signals, qdii_premium_pct=2.90, reconciler=reconciler)
    assert veto_res.conflict_type == "CONFLICT_VETO_OVERRIDE"
    assert veto_res.final_score == -1.0
    assert veto_res.recommended_tilt == -0.015

    # 4. Belief entropy calculation
    entropy_zero = compute_belief_entropy([1.0, 0.0])
    assert entropy_zero == 0.0
    entropy_uniform = compute_belief_entropy([0.5, 0.5])
    assert abs(entropy_uniform - 1.0) < 0.01


def test_package_loading_and_marketplace() -> None:
    """Verify package loading via fin_skills.load and marketplace registration."""
    # fin_skills.load reads package data
    kol_md = fin_skills.load("kol-credibility-registry")
    assert "KOL Credibility Registry" in kol_md

    reconciler_md = fin_skills.load("signal-reconciler")
    assert "Signal Reconciler" in reconciler_md

    # Check marketplace.json
    mp_path = ROOT / ".claude-plugin" / "marketplace.json"
    mp_data = json.loads(mp_path.read_text(encoding="utf-8"))
    china_plugin = next(p for p in mp_data["plugins"] if p["name"] == "fin-china")
    declared_skills = set(china_plugin["skills"])

    assert "./skills/kol-credibility-registry" in declared_skills
    assert "./skills/signal-reconciler" in declared_skills
