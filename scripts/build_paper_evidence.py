"""Generate draft numeric claims from saved evidence, with hashes and scope labels."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = ["benchmarks/agent_study/PILOT_RESULTS.json", "benchmarks/GUARD_ROBUSTNESS.json",
               "benchmarks/PARITY_AND_COST_RESULTS.json", "benchmarks/REAL_WORLD_KOL_AUDIT_RESULTS.json",
               "benchmarks/agent_study/POSITIVE_CONTROL_RESULTS.json",
               "benchmarks/agent_study/BEACON_FEASIBILITY_RESULTS.json"]
    data = [json.loads((ROOT / name).read_text(encoding="utf-8")) for name in sources]
    pilot, robustness, parity, kol, control, beacon = data
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
    manifest = {"status": "DRAFT_MACHINE_CHECKED_NOT_HUMAN_APPROVED",
        "sources": [{"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
                    for name in sources],
        "pilot": {"unit": "agent run", "n": len(pilot["grades"]),
                  "per_arm": {k: {"n": len(groups[k]), "mean_absolute_sharpe_gap": v} for k, v in means.items()},
                  "opus_relative_reduction_percent": reduction,
                  "scope": "historical exploratory pilot; not a replicated causal estimate"},
        "defects": {"worlds": len(robustness["worlds"]),
                    "planted_instances": sum(w["defects"] for w in robustness["worlds"]),
                    "caught_instances": sum(w["caught"] for w in robustness["worlds"]),
                    "false_alarms": robustness["false_alarms"],
                    "scope": "fixed synthetic defect families; seed repetition is not independent defect discovery"},
        "parity": {"passed": sum(x["passed"] for x in parity["parity_matrix"]),
                   "count": len(parity["parity_matrix"]),
                   "maximum_absolute_error": max(x["max_abs_error"] for x in parity["parity_matrix"]),
                   "scope": "within each case's stated tolerance; not exact equality"},
        "real_world": {"status": kol["reproducibility_status"],
                       "prediction_reproduction": kol["prediction_reproduction"]},
        "positive_control": {"accepted": control["public_checks"]["accepted"],
                             "grade": control["independent_grade"], "scope": control["scope"]},
        "beacon_feasibility": {"models": beacon_models, "status": beacon["attempt_status"],
                               "scope": beacon["interpretation"], "pending_job": beacon["pending_job"]},
        "unresolved": ["Accountable human source/claim verification", "Independent holdout authorship and isolation",
                       "Full repeated multi-model study", "KOL row-level predictions and timestamp provenance",
                       "Author order, affiliations, declarations and submission approval"]}
    (ROOT / "paper/evidence.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    macros = {"PilotRuns": len(pilot["grades"]), "OpusReduction": f"{reduction:.1f}",
              "OpusBase": f"{means['A']:.3f}", "OpusLibrary": f"{means['B']:.3f}",
              "HaikuBase": f"{means['Ah']:.3f}", "HaikuLibrary": f"{means['Bh']:.3f}",
              "DefectInstances": manifest["defects"]["planted_instances"],
              "CaughtInstances": manifest["defects"]["caught_instances"],
              "BeaconSevenAccepted": beacon_models["Qwen/Qwen2.5-Coder-7B-Instruct"]["accepted"],
              "BeaconFourteenAccepted": beacon_models["Qwen/Qwen2.5-Coder-14B-Instruct"]["accepted"],
              "BeaconPerModel": beacon_models["Qwen/Qwen2.5-Coder-14B-Instruct"]["recorded_cells"]}
    # Per-condition macros for Table 2 (four-condition feasibility breakdown).
    for (mlab, clab), info in beacon_conditions.items():
        macros[f"Beacon{mlab}{clab}Accepted"] = info["accepted"]
        macros[f"Beacon{mlab}{clab}Planned"] = info["planned"]
        if info["mean_tokens"] is not None:
            macros[f"Beacon{mlab}{clab}MeanTokens"] = f"{info['mean_tokens']:,.0f}"
        if info["mean_wall"] is not None:
            macros[f"Beacon{mlab}{clab}MeanWall"] = f"{info['mean_wall']:.0f}"
    (ROOT / "paper/latex_naacl/evidence_numbers.tex").write_text(
        "% GENERATED by scripts/build_paper_evidence.py\n" +
        "\n".join("\\newcommand{\\" + k + "}{" + str(v) + "}" for k, v in macros.items()) + "\n", encoding="utf-8")
    print(json.dumps({"pilot_runs": len(pilot["grades"]), "opus_reduction_percent": reduction,
                      "real_world_status": kol["reproducibility_status"]}))


if __name__ == "__main__":
    main()
