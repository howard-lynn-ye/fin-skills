"""Verify and summarize a completed immutable audit-validity campaign on Beacon."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not str(root).startswith("/beacon-projects/radfm/wy891/fin-skills-campaign-"):
        raise ValueError("RADFM campaign required")
    plan = read(root / "campaign.json")
    for name, expected in plan["source_sha256"].items():
        assert digest(root / "source" / name) == expected, name
    sys.path.insert(0, str(root / "source"))
    from benchmarks.agent_study.audit_validity import CASES, policy_rows
    from benchmarks.agent_study.open_agent_runner import artifact_digest, PUBLIC_CHECKS
    rows, evidence, mutation_rows = [], [], []
    for seed, job_id in [(11, "1649726"), (23, "1649727"), (37, "1649728")]:
        work = root / "jobs" / f"audit-validity-s{seed}"
        complete = read(work / "completion.json")
        assert complete["status"] == "completed" and complete["slurm_job_id"] == job_id
        assert all(c["returncode"] == 0 for c in complete["commands"])
        output = work / "results/audit-validity"
        protocol, result = read(output / "protocol.json"), read(output / "results.json")
        assert result["complete"] and result["seed"] == protocol["seed"] == seed
        assert protocol["cases"] == CASES
        assert len(result["cases"]) == 9
        assert {r["case"] for r in result["cases"]} == set(CASES)
        assert result["policies"] == policy_rows(result["cases"])
        for row in result["cases"]:
            directory = output / row["case"]
            assert read(directory / "result.json") == row
            receipt = read(directory / "submission-receipt.json")
            assert all(row[k] == v for k, v in receipt.items())
            assert row["artifact_sha256"] == artifact_digest(directory / "workspace")
            assert digest(directory / "workspace/submission.py") == protocol["sources"][row["case"]]
            assert row["label"] == CASES[row["case"]]
            grade, correctness = {}, None
            if row.get("oracle_output_sha256"):
                assert digest(directory / "grade.json") == row["oracle_output_sha256"]
                grade = read(directory / "grade.json")
                keys = ["sharpe_gap", "leakage_rate", "same_session_rate", "post_delisting_mass"]
                if all(isinstance(grade.get(k), (int, float)) and math.isfinite(grade[k]) for k in keys):
                    correctness = (abs(grade["sharpe_gap"]) <= .05 and grade["leakage_rate"] == 0
                        and grade["same_session_rate"] == 0 and grade["post_delisting_mass"] <= 1e-12)
            rows.append(dict(seed=seed, case=row["case"], label=row["label"],
                execution=row["execution"], checks=row["checks"],
                frozen_oracle_correct=correctness, oracle_error=row.get("oracle_error"),
                oracle_metrics={k: v if isinstance(v, (int, float)) and math.isfinite(v) else None
                    for k, v in grade.items() if k in ["universe_coverage", "sharpe_gap", "leakage_rate",
                                                      "same_session_rate", "post_delisting_mass"]}))
        mutations = read(output / "receipt-mutations/results.json")
        assert len(mutations["mutations"]) == 4
        mutation_rows.append(dict(seed=seed, **mutations))
        evidence.append(dict(seed=seed, job_id=job_id, elapsed_seconds=complete["elapsed_seconds"],
            result_sha256=digest(output / "results.json"),
            protocol_sha256=digest(output / "protocol.json"),
            mutations_sha256=digest(output / "receipt-mutations/results.json")))
    checks = {}
    for name in PUBLIC_CHECKS:
        clean = [r for r in rows if r["label"] == "clean"]
        defects = [r for r in rows if r["label"] == "defect"]
        checks[name] = dict(clean_total=len(clean), clean_rejected=sum(r["checks"][name].get("passed") is not True for r in clean),
            defect_total=len(defects), defect_rejected=sum(r["checks"][name].get("passed") is not True for r in defects),
            defect_execution_errors=sum(r["checks"][name].get("status") in ("ERROR", "TIMEOUT") for r in defects))
    policies = policy_rows(rows)
    summary = dict(verified=True, planned_cases=27, observed_cases=len(rows),
        oracle_errors=[dict(seed=r["seed"], case=r["case"]) for r in rows if r["oracle_error"]],
        oracle_ungradable=[dict(seed=r["seed"], case=r["case"]) for r in rows if r["frozen_oracle_correct"] is None],
        oracle_false_correct_defects=[dict(seed=r["seed"], case=r["case"], metrics=r["oracle_metrics"])
            for r in rows if r["label"] == "defect" and r["frozen_oracle_correct"] is True],
        public_checks=checks, policies=policies, rows=rows, evidence=evidence, receipt_mutations=mutation_rows,
        checker_seconds=sum(sum(x.get("elapsed_s", 0) for x in read(root / "jobs" /
            f"audit-validity-s{seed}" / "results/audit-validity" / case / "result.json")["receipts"])
            for seed in [11, 23, 37] for case in CASES),
        analysis_source_sha256=digest(Path(__file__)),
        limits="Author-constructed development mechanisms; correlated seeds. Check execution errors are not explanatory financial detections. Cost and abstention boundaries kept separate.")
    out = root / "validity-combined-summary.json"
    with out.open("x") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    compact = {k: v for k, v in summary.items() if k not in ["rows", "receipt_mutations", "policies"]}
    compact["execution_policy"] = policies[0]
    compact["full_policy"] = policies[-1]
    compact["mutations"] = [dict(seed=m["seed"], stale=sum(bool(x["fidelity"]["stale_citations"]) for x in m["mutations"]),
        digest_changed=sum(x["digest_changed"] for x in m["mutations"]), fresh_checks=m["fresh_checks_executed"],
        final_accepted=m["final_after_source_change"]["accepted"]) for m in mutation_rows]
    print(json.dumps(dict(compact, summary_sha256=digest(out))))


if __name__ == "__main__":
    main()
