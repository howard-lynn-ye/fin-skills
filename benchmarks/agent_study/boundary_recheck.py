"""Recheck a demonstrated empty-universe transport defect; no model inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from benchmarks.agent_study.audit_validity import write
from benchmarks.agent_study.open_agent_runner import AgentWorkspaceSession, artifact_digest, PUBLIC_CHECKS

PRIOR = Path("/beacon-projects/radfm/wy891/fin-skills-campaign-validity-20260923-v1")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    write(a.output / "protocol.json", dict(seeds=[11, 23, 37], cases=["zero_positions", "clean_news"],
        prior_root=str(PRIOR), change="empty active universe returns UNASSESSABLE instead of NaN serialization error",
        policy="No numerical score or original result changes. Accounting zero-variance error remains explicit.",
        scope="execution-defect boundary recheck; no new model or correctness result"))
    for seed in [11, 23, 37]:
        old = PRIOR / "jobs" / f"audit-validity-s{seed}" / "results/audit-validity"
        for case in ["zero_positions", "clean_news"]:
            prior = json.loads((old / case / "result.json").read_text())
            workspace = a.output / f"s{seed}-{case}/workspace"
            shutil.copytree(old / case / "workspace", workspace)
            assert artifact_digest(workspace) == prior["artifact_sha256"]
            session = AgentWorkspaceSession(workspace, "skills_enforced_guards")
            checks = {c: session._worker(c) for c in PUBLIC_CHECKS}
            if case == "zero_positions":
                assert checks["survivorship_audit"]["status"] == "UNASSESSABLE"
                assert checks["survivorship_audit"]["passed"] is False
            else:
                assert all(c.get("passed") is True for c in checks.values())
            rows.append(dict(seed=seed, case=case, artifact_sha256=artifact_digest(workspace),
                prior_result_sha256=hashlib.sha256((old / case / "result.json").read_bytes()).hexdigest(),
                prior_survivorship=prior["checks"]["survivorship_audit"], checks=checks, receipts=session.receipts))
            write(workspace.parent / "result.json", rows[-1])
            print(json.dumps(dict(seed=seed, case=case, checked=True)), flush=True)
    write(a.output / "results.json", dict(complete=True, rows=rows))


if __name__ == "__main__":
    main()
