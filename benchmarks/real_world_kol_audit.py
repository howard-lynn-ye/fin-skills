#!/usr/bin/env python3
"""Experiment E4: summary-fixture inspection, NOT prediction reproduction.

Recomputes tier aggregates from stored author profiles. Predictive IC, model scores,
and linguistic metrics are imported historical summaries, not recomputed from posts,
predictions or outcomes. Use prediction_audit.py with row-level data for that check.

Verifies:
1. Tier-level distribution, follower paradox (Contrarian accounts having 3.96x more
   followers than Tier-1 Core Alpha researchers), and 5-day Information Coefficients.
2. Means of stored posterior win rates (not a refit of the Bayesian model).
3. Reporting of saved MMAN / RoBERTa metrics with their provenance limits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIXTURE_PATH = ROOT / "data" / "kol_credibility_linguistic_fixture.json"
OUTPUT_PATH = ROOT / "REAL_WORLD_KOL_AUDIT_RESULTS.json"


def verify_and_reproduce_kol_audit() -> dict[str, Any]:
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(f"Missing self-contained fixture: {FIXTURE_PATH}")

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    profiles = pd.DataFrame(fixture["anonymized_kol_profiles"])

    # Verify strict anonymization (no raw author names)
    assert "author" not in profiles.columns, "Raw author handle must not be present in fixture"
    assert all(str(aid).startswith("KOL_") for aid in profiles["author_id"]), "Invalid anonymized ID"

    # Recompute tier-level empirical summary directly from the 1,445 records
    recomputed_tiers: list[dict[str, Any]] = []
    for tier_name, grp in profiles.groupby("tier"):
        recomputed_tiers.append({
            "tier": str(tier_name),
            "kol_count": int(len(grp)),
            "share_pct": round(100.0 * len(grp) / len(profiles), 2),
            "mean_fans": round(float(grp["fans"].mean()), 1),
            "mean_win_rate_5d": round(float(grp["win_rate_5d"].mean()), 4),
            "mean_bayesian_win_rate_5d": round(float(grp["bayesian_win_rate_5d"].mean()), 4),
            "mean_realized_return_5d": round(float(grp["mean_realized_return_5d"].mean()), 5),
            "mean_ic_5d": round(float(grp["information_coefficient_5d"].mean()), 4),
            "median_payoff_ratio": round(float(grp["payoff_ratio"].median()), 3),
        })

    tier_map = {t["tier"]: t for t in recomputed_tiers}
    alpha_fans = tier_map["TIER_1_CORE_ALPHA"]["mean_fans"]
    contrarian_fans = tier_map["TIER_CONTRARIAN_INDICATOR"]["mean_fans"]
    follower_paradox_ratio = round(contrarian_fans / max(alpha_fans, 1.0), 2)

    mman_ablation = fixture["mman_credibility_ablation"]
    unweighted_ic = float(mman_ablation["fact_surge_mman"]["rank_ic"])
    gated_ic = float(mman_ablation["credibility_gated_mman"]["rank_ic"])
    ic_reversal_delta = round(gated_ic - unweighted_ic, 5)

    rw_gains = fixture["raw_vs_rewritten_text_gains"]
    roberta_raw_ic = float(rw_gains["roberta_evaluation"]["raw_text_model"]["rank_ic"])
    roberta_dist_ic = float(rw_gains["roberta_evaluation"]["rewritten_text_model"]["rank_ic"])

    result = {
        "experiment_id": "E4_real_world_bilingual_kol_and_linguistic_audit",
        "reproducibility_status": "SUMMARY_FIXTURE_ONLY",
        "prediction_reproduction": "NOT_RUN_MISSING_ROW_LEVEL_PREDICTIONS",
        "recomputed": ["tier_counts", "tier_means", "follower_ratio"],
        "imported_not_recomputed": ["model_IC", "model_auc", "linguistic_metrics",
                                    "posterior_win_rates"],
        "limitations": ["No row-level predictions/outcomes or time-split provenance in fixture.",
                        "Tier definitions may use the same outcomes summarized within tiers.",
                        "Pooled Rank IC is not mean daily cross-sectional Rank IC."],
        "total_kols_verified": int(len(profiles)),
        "key_findings": {
            "follower_paradox_ratio_contrarian_vs_alpha": follower_paradox_ratio,
            "tier1_core_alpha_5d_ic": tier_map["TIER_1_CORE_ALPHA"]["mean_ic_5d"],
            "contrarian_indicator_5d_ic": tier_map["TIER_CONTRARIAN_INDICATOR"]["mean_ic_5d"],
            "mman_unweighted_rank_ic": round(unweighted_ic, 5),
            "mman_credibility_gated_rank_ic": round(gated_ic, 5),
            "mman_credibility_gated_mean_daily_rank_ic": float(
                mman_ablation["credibility_gated_mman"]["mean_daily_rank_ic"]),
            "mman_rank_ic_net_reversal": ic_reversal_delta,
            "roberta_raw_text_rank_ic": round(roberta_raw_ic, 5),
            "roberta_distilled_text_rank_ic": round(roberta_dist_ic, 5),
        },
        "recomputed_tier_breakdown": recomputed_tiers,
        "linguistic_contrast": fixture["linguistic_contrast"],
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = verify_and_reproduce_kol_audit()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
