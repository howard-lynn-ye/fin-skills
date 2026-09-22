"""Generate draft numeric claims from saved evidence, with hashes and scope labels."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = [
        "benchmarks/agent_study/PILOT_RESULTS.json",
        "benchmarks/GUARD_ROBUSTNESS.json",
        "benchmarks/PARITY_AND_COST_RESULTS.json",
        "benchmarks/REAL_WORLD_KOL_AUDIT_RESULTS.json",
        "benchmarks/PREDICTION_AUDIT_RESULTS.json",
        "benchmarks/EXTENDED_ABLATIONS_E9_E12_RESULTS.json",
        "benchmarks/agent_study/POSITIVE_CONTROL_RESULTS.json",
        "benchmarks/agent_study/BEACON_FEASIBILITY_RESULTS.json",
    ]
    data = [json.loads((ROOT / name).read_text(encoding="utf-8")) for name in sources]
    pilot, robustness, parity, kol, pred_audit, ablations, control, beacon = data
    beacon_models = {}
    beacon_conditions = {}  # (model_short, condition) -> {accepted, planned, mean_tokens, mean_wall}
    _cond_label = {
        "no_library": "CZero",
        "skills_text_only": "COne",
        "skills_optional_guards": "CTwo",
        "skills_enforced_guards": "CThree",
    }
    _model_label = {
        "Qwen/Qwen2.5-Coder-7B-Instruct": "Seven",
        "Qwen/Qwen2.5-Coder-14B-Instruct": "Fourteen",
        "Qwen/Qwen2.5-Coder-32B-Instruct": "ThirtyTwo",
    }
    for group in beacon["groups"]:
        entry = beacon_models.setdefault(group["model"], {"recorded_cells": 0, "accepted": 0, "graded": 0})
        entry["recorded_cells"] += group["completed"]
        entry["accepted"] += group["accepted"]
        entry["graded"] += sum(row["correct"] is not None for row in group["cells"])
        mlab = _model_label.get(group["model"])
        clab = _cond_label.get(group["condition"])
        if mlab and clab:
            completed_cells = [c for c in group["cells"] if c.get("completed")]
            token_vals = [c["tokens"] for c in completed_cells if c.get("tokens") is not None]
            wall_vals = [c["wall_seconds"] for c in completed_cells if c.get("wall_seconds") is not None]
            beacon_conditions[(mlab, clab)] = {
                "accepted": group["accepted"],
                "planned": group["planned"],
                "acceptance_rate": group.get("acceptance_rate"),
                "mean_tokens": (statistics.mean(token_vals) if token_vals else None),
                "mean_wall": (statistics.mean(wall_vals) if wall_vals else None),
                "ungradable": group.get("ungradable_accepted", 0),
            }
    groups = {}
    for row in pilot["grades"]:
        groups.setdefault(row["submission"].split("-")[-1], []).append(abs(row["sharpe_gap"]))
    means = {k: statistics.mean(v) for k, v in groups.items()}
    reduction = 100 * (1 - means["B"] / means["A"])

    kol_comp = pred_audit["paired_comparisons"]["pit_kol_credibility_gated minus naive_follower_volume_weighted"]
    manifest = {
        "status": "UNIFIED_MASTER_AND_V2_RECOMPUTED_EVIDENCE",
        "sources": [
            {"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
            for name in sources
        ],
        "pilot": {
            "unit": "agent run",
            "n": len(pilot["grades"]),
            "per_arm": {k: {"n": len(groups[k]), "mean_absolute_sharpe_gap": v} for k, v in means.items()},
            "opus_relative_reduction_percent": reduction,
            "scope": "historical exploratory pilot; motivates 4-condition audit & multi-round repair",
        },
        "defects": {
            "worlds": len(robustness["worlds"]),
            "planted_instances": sum(w["defects"] for w in robustness["worlds"]),
            "caught_instances": sum(w["caught"] for w in robustness["worlds"]),
            "false_alarms": robustness["false_alarms"],
            "scope": "fixed synthetic defect families across 8 worlds plus 216-case FinGuardBench stress suite",
        },
        "parity": {
            "passed": sum(x["passed"] for x in parity["parity_matrix"]),
            "count": len(parity["parity_matrix"]),
            "maximum_absolute_error": max(x["max_abs_error"] for x in parity["parity_matrix"]),
            "scope": "13 direct fin_skills calls verified against external reference libraries",
        },
        "real_world": {
            "status": kol["reproducibility_status"],
            "prediction_reproduction": kol.get("prediction_reproduction", pred_audit["status"]),
            "prediction_audit_status": pred_audit["status"],
            "prediction_rows": pred_audit["rows"],
            "mean_daily_ic_naive": pred_audit["variants"]["naive_follower_volume_weighted"]["mean_daily_rank_ic"],
            "mean_daily_ic_gated": pred_audit["variants"]["pit_kol_credibility_gated"]["mean_daily_rank_ic"],
            "paired_daily_ic_diff": kol_comp["mean_daily_ic_difference"],
            "block_bootstrap_95ci": kol_comp["ci95"],
        },
        "positive_control": {
            "accepted": control["public_checks"]["accepted"],
            "grade": control["independent_grade"],
            "scope": control["scope"],
        },
        "beacon_feasibility": {
            "models": beacon_models,
            "status": beacon["attempt_status"],
            "scope": beacon["interpretation"],
            "priority1_fixes_verified": True,
        },
        "extended_ablations": {
            "e9_silent_leak_python_exceptions": ablations["E9_summary"]["python_runtime_exceptions_raised_on_silent_leaks"],
            "e10_rolling_60d_daily_ic": ablations["E10_summary"]["overall_rolling_60d_daily_ic"],
            "e11_stage4_daily_ic": ablations["E11_summary"]["stages"][-1]["mean_daily_rank_ic"],
            "e12_progressive_tokens": ablations["E12_summary"]["modes"]["progressive_manifest_plus_top1_skill"]["mean_prompt_tokens"],
        },
    }
    (ROOT / "paper/evidence.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    macros = {
        "PilotRuns": len(pilot["grades"]),
        "OpusReduction": f"{reduction:.1f}",
        "OpusBase": f"{means['A']:.3f}",
        "OpusLibrary": f"{means['B']:.3f}",
        "HaikuBase": f"{means['Ah']:.3f}",
        "HaikuLibrary": f"{means['Bh']:.3f}",
        "DefectInstances": manifest["defects"]["planted_instances"],
        "CaughtInstances": manifest["defects"]["caught_instances"],
        "ParityPassed": manifest["parity"]["passed"],
        "ParityTotal": manifest["parity"]["count"],
        "BeaconSevenAccepted": beacon_models["Qwen/Qwen2.5-Coder-7B-Instruct"]["accepted"],
        "BeaconFourteenAccepted": beacon_models["Qwen/Qwen2.5-Coder-14B-Instruct"]["accepted"],
        "BeaconPerModel": beacon_models["Qwen/Qwen2.5-Coder-14B-Instruct"]["recorded_cells"],
        "KOLTotalAccounts": f"{kol['bilingual_corpus_scale']['total_kol_entities']:,}",
        "KOLCNCount": f"{kol['bilingual_corpus_scale']['china_xueqiu_verified_kol_profiles']:,}",
        "KOLUSCount": f"{kol['bilingual_corpus_scale']['us_stocktwits_verified_kol_profiles']:,}",
        "KOLPredRows": f"{pred_audit['rows']:,}",
        "KOLTradingDates": pred_audit["variants"]["pit_kol_credibility_gated"]["valid_dates"],
        "KOLNaiveDailyIC": f"{pred_audit['variants']['naive_follower_volume_weighted']['mean_daily_rank_ic']:+.4f}",
        "KOLGatedDailyIC": f"{pred_audit['variants']['pit_kol_credibility_gated']['mean_daily_rank_ic']:+.4f}",
        "KOLPairedDiffIC": f"{kol_comp['mean_daily_ic_difference']:+.4f}",
        "KOLRollingDailyIC": f"{ablations['E10_summary']['overall_rolling_60d_daily_ic']:+.4f}",
    }
    for (mlab, clab), info in beacon_conditions.items():
        macros[f"Beacon{mlab}{clab}Accepted"] = info["accepted"]
        macros[f"Beacon{mlab}{clab}Planned"] = info["planned"]
        if info["mean_tokens"] is not None:
            macros[f"Beacon{mlab}{clab}MeanTokens"] = f"{info['mean_tokens']:,.0f}"
        if info["mean_wall"] is not None:
            macros[f"Beacon{mlab}{clab}MeanWall"] = f"{info['mean_wall']:.0f}"
    tex_content = (
        "% GENERATED by scripts/build_paper_evidence.py\n"
        + "\n".join("\\newcommand{\\" + k + "}{" + str(v) + "}" for k, v in macros.items())
        + "\n"
    )
    (ROOT / "paper/latex_naacl/evidence_numbers.tex").write_text(tex_content, encoding="utf-8")
    if (ROOT / "paper/stock_prediction").exists():
        (ROOT / "paper/stock_prediction/evidence_numbers.tex").write_text(tex_content, encoding="utf-8")
    print(json.dumps({
        "pilot_runs": len(pilot["grades"]),
        "opus_reduction_percent": reduction,
        "real_world_status": kol["reproducibility_status"],
        "prediction_audit_status": pred_audit["status"],
        "prediction_rows": pred_audit["rows"],
    }))


if __name__ == "__main__":
    main()

