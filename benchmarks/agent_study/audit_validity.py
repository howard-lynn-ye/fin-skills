"""Development audit-validity matrix, policy ablations and stale-evidence checks.

Cases are explicitly author constructed. Test labels are not independent holdout labels.
The frozen independent scorer is itself evaluated for blind spots; it is not assumed to
define every defect. No model inference or hyperparameter search is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from benchmarks.agent_study.build_task import export
from benchmarks.agent_study.open_agent_runner import AgentWorkspaceSession, PUBLIC_CHECKS, artifact_digest

CASES = {
    "clean_news": "clean", "clean_price": "clean", "same_session": "defect",
    "future_price": "defect", "survivor_only": "defect", "wrong_report": "defect",
    "stated_cost_5": "cost_convention_boundary", "zero_positions": "abstention_boundary",
    "invalid_schema": "invalid",
}
TEMPLATE = '''from pathlib import Path
import pandas as pd

def build_positions(data_dir):
    d = Path(data_dir)
    close = pd.read_csv(d / "close_quoted.csv", index_col="date", parse_dates=["date"])
    news = pd.read_csv(d / "news_feed.csv", parse_dates=["feed_ts"])
    news = news.pivot(index="feed_ts", columns="ticker", values="score").reindex_like(close)
    mode = __MODE__
    if mode == "future_price":
        signal = close.pct_change(fill_method=None).shift(-1)
    elif mode == "clean_price":
        signal = close.pct_change(fill_method=None).shift(1)
    elif mode == "same_session":
        signal = news
    else:
        signal = news.shift(1)
    signal = signal.rank(axis=1, pct=True).sub(.5).fillna(0.0)
    listing = pd.read_csv(d / "listings.csv", parse_dates=["listing_date", "delisting_date"])
    eligible = close.shift(1).notna()
    for row in listing.itertuples():
        eligible.loc[eligible.index <= row.listing_date, row.ticker] = False
        if pd.notna(row.delisting_date):
            eligible.loc[eligible.index > row.delisting_date, row.ticker] = False
            if mode == "survivor_only":
                eligible.loc[:, row.ticker] = False
    signal = signal.where(eligible)
    signal = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
    gross = signal.abs().sum(axis=1)
    weights = signal.div(gross.where(gross > 0, 1.0), axis=0)
    if mode == "zero_positions":
        weights *= 0
    if mode == "invalid_schema":
        weights.columns = ["UNKNOWN_" + str(i) for i in range(len(weights.columns))]
    return weights
'''


def source(case):
    if case not in CASES:
        raise ValueError("unknown case")
    return TEMPLATE.replace("__MODE__", repr(case))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=str, allow_nan=False)


def reference_report(workspace, cost):
    """Separate accounting implementation used only to construct test reports."""
    import numpy as np
    import pandas as pd
    from benchmarks.agent_study.submission_audit import positions
    w = positions(workspace)
    px = pd.read_csv(workspace / "data/close_quoted.csv", index_col="date", parse_dates=["date"])
    events = pd.read_csv(workspace / "data/corporate_actions.csv", parse_dates=["date"])
    for row in events.itertuples():
        px.loc[px.index < row.date, row.ticker] /= row.ratio
    returns = (px / px.shift(1) - 1).fillna(0.0).to_numpy()
    weights = w.reindex(index=px.index, columns=px.columns).fillna(0).to_numpy()
    turnover = np.abs(np.diff(weights, axis=0, prepend=weights[:1])).sum(axis=1)
    net = (weights * returns).sum(axis=1) - cost * turnover / 10000
    m = json.loads((workspace / "manifest.json").read_text())
    keep = (px.index >= m["eval_start"]) & (px.index <= m["eval_end"])
    segment = net[keep]
    return float(segment.mean()/segment.std(ddof=1)*np.sqrt(252)) if segment.std(ddof=1) else 0.0


def policy_rows(records):
    rows = []
    for count in range(len(PUBLIC_CHECKS)+1):
        for checks in itertools.combinations(PUBLIC_CHECKS, count):
            outcomes = []
            for row in records:
                passed = row["execution"].get("passed") is True and all(
                    row["checks"][c].get("passed") is True for c in checks)
                outcomes.append(dict(case=row["case"], label=row["label"], accepted=passed))
            clean = [r for r in outcomes if r["label"] == "clean"]
            bad = [r for r in outcomes if r["label"] in ("defect", "invalid")]
            rows.append(dict(required=list(checks), clean_total=len(clean),
                clean_rejected=sum(not r["accepted"] for r in clean), defect_total=len(bad),
                defect_accepted=sum(r["accepted"] for r in bad), outcomes=outcomes))
    return rows


def receipt_mutations(workspace):
    session = AgentWorkspaceSession(workspace, "skills_enforced_guards")
    for check in PUBLIC_CHECKS:
        session._worker(check)
    report_path = workspace / "report.json"
    report = json.loads(report_path.read_text())
    report["guards_cited"] = ["assert_causal", "survivorship_audit"]
    report_path.write_text(json.dumps(report))
    baseline = artifact_digest(workspace)
    mutations = []
    for name, path, replacement in (
        ("source", workspace / "submission.py", source("same_session").encode()),
        ("numerical_report", report_path, json.dumps(dict(report, reported_sharpe=report["reported_sharpe"]+1)).encode()),
        ("market_data", workspace / "data/close_quoted.csv", (workspace / "data/close_quoted.csv").read_bytes()+b"\n"),
        ("manifest", workspace / "manifest.json", (workspace / "manifest.json").read_bytes()+b"\n"),
    ):
        original = path.read_bytes()
        try:
            path.write_bytes(replacement)
            mutations.append(dict(mutation=name, digest_changed=artifact_digest(workspace)!=baseline,
                                  fidelity=session.audit_compliance_fidelity()))
        finally:
            path.write_bytes(original)
    # Real enforcement must re-run against revised code, rather than replay prior PASS.
    (workspace / "submission.py").write_text(source("same_session"))
    before = len(session.receipts)
    revised = session.finish()
    return dict(mutations=mutations, final_after_source_change=revised,
                fresh_checks_executed=len(session.receipts)-before,
                scope="in-process receipt freshness; not a cryptographic or hostile-host guarantee")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", dict(seed=args.seed, cases=CASES,
        sources={c: hashlib.sha256(source(c).encode()).hexdigest() for c in CASES},
        checks=PUBLIC_CHECKS, policy="all subsets plus execution precondition; descriptive, not policy selection",
        scope="author-created instrument tests; no independent holdout or model-capability claim",
        report_cost="10 bps except explicit 5-bps convention boundary", oracle_timeout_seconds=300))
    original = args.output / "base"
    export(args.seed, original)
    records = []
    for case, label in CASES.items():
        directory = args.output / case
        workspace = directory / "workspace"
        shutil.copytree(original, workspace)
        (workspace / "submission.py").write_text(source(case))
        cost = 5.0 if case == "stated_cost_5" else 10.0
        report = reference_report(workspace, cost) if case != "invalid_schema" else 0.0
        if case == "wrong_report":
            report += 1.0
        write(workspace / "report.json", dict(reported_sharpe=report, cost_bps_per_side=cost, guards_cited=[]))
        session = AgentWorkspaceSession(workspace, "skills_enforced_guards")
        execution = session._worker("execution")
        checks = {name: session._worker(name) for name in PUBLIC_CHECKS}
        row = dict(case=case, label=label, execution=execution, checks=checks,
                   artifact_sha256=artifact_digest(workspace), receipts=session.receipts)
        write(directory / "submission-receipt.json", row)
        try:
            proc = subprocess.run([sys.executable, "-m", "benchmarks.agent_study.oracle",
                "--task", str(workspace), "--submission", str(workspace),
                "--out", str(directory / "grade.json")], capture_output=True, text=True, timeout=300)
            row["oracle_exit"] = proc.returncode
            if proc.returncode:
                row["oracle_error"] = proc.stderr[-2500:]
            elif (directory / "grade.json").is_file():
                grade = json.loads((directory / "grade.json").read_text())
                # Preserve raw grader output, including undefined numerical metrics, separately.
                row["oracle_grade_path"] = "grade.json"
                row["oracle_output_sha256"] = hashlib.sha256((directory / "grade.json").read_bytes()).hexdigest()
        except subprocess.TimeoutExpired:
            row["oracle_error"] = "timeout"
        records.append(row)
        write(directory / "result.json", row)
        print(json.dumps(dict(case=case, checks={k:v.get("passed") for k,v in checks.items()})), flush=True)
    write(args.output / "results.json", dict(complete=True, seed=args.seed, cases=records,
                                             policies=policy_rows(records)))
    copied = args.output / "receipt-mutations/workspace"
    shutil.copytree(args.output / "clean_news/workspace", copied)
    write(args.output / "receipt-mutations/results.json", receipt_mutations(copied))


if __name__ == "__main__":
    main()
