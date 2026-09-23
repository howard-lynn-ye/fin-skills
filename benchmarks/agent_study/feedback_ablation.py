"""Public development repair study separating feedback from final enforcement.

This is a seeded repair task, not a replacement for failed end-to-end runs. All arms
start from the same intentionally flawed executable submission and the same guidance.
The independent oracle is run only after the final artifact receipt is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from benchmarks.agent_study.build_task import export
from benchmarks.agent_study.open_agent_runner import (
    AgentWorkspaceSession, PUBLIC_CHECKS, run_agent,
)
from benchmarks.agent_study.prepare_followup import MODEL, REVISION
from benchmarks.agent_study.summarize_matrix import summarize

ARMS = ("execution_only", "accounting_feedback", "guard_feedback", "guard_feedback_gate")
STARTER = '''from pathlib import Path
import pandas as pd

def build_positions(data_dir):
    data = Path(data_dir)
    close = pd.read_csv(data / "close_quoted.csv", index_col="date", parse_dates=["date"])
    news = pd.read_csv(data / "news_feed.csv", parse_dates=["feed_ts"])
    signal = news.pivot(index="feed_ts", columns="ticker", values="score")
    signal = signal.reindex(index=close.index, columns=close.columns).fillna(0.0)
    listings = pd.read_csv(data / "listings.csv")
    survivors = listings.loc[listings.delisting_date.isna(), "ticker"]
    signal.loc[:, ~signal.columns.isin(survivors)] = 0.0
    signal = signal.sub(signal.mean(axis=1), axis=0)
    gross = signal.abs().sum(axis=1)
    return signal.div(gross.where(gross > 0, 1.0), axis=0).fillna(0.0)
'''


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=str, allow_nan=False)


class RepairSession(AgentWorkspaceSession):
    def __init__(self, workspace, arm):
        if arm not in ARMS:
            raise ValueError("unknown repair arm")
        super().__init__(workspace, "skills_text_only")
        self.arm = arm
        self.initial_feedback = self.inspect()

    def inspect(self):
        names = ("execution",) if self.arm == "execution_only" else (
            ("accounting",) if self.arm == "accounting_feedback" else PUBLIC_CHECKS)
        return {name: self._worker(name) for name in names}

    def system_prompt(self):
        prompt = super().system_prompt().replace(
            "inspect_submission recomputes accounting at your stated costs.",
            "inspect_submission returns the checks enabled for this condition.")
        prompt += ("\nThis is a REPAIR task. submission.py and report.json already exist. "
                   "They are a fallible starting point, not an approved solution. Repair them "
                   "to satisfy TASK.md. Every arm has the same starting files and guidance. "
                   "Enabled feedback: " + ", ".join(self.initial_feedback) + ".")
        if self.arm == "guard_feedback_gate":
            prompt += " finish() repeats all enabled checks and accepts only if every check passes."
        else:
            prompt += " finish() freezes your submission without enforcing these checks."
        return prompt + "\nInitial check results: " + json.dumps(self.initial_feedback, default=str)

    def dispatch(self, name, arguments):
        if name == "inspect_submission":
            self.tool_call_count += 1
            return self.inspect()
        return super().dispatch(name, arguments)

    def finish(self):
        basic = super().finish()
        if not basic.get("accepted") or self.arm != "guard_feedback_gate":
            return basic
        checks = self.inspect()
        passed = all(v.get("passed") is True for v in checks.values())
        return {**basic, "accepted": passed, "status": "PASS" if passed else "FAIL",
                "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    args.output.mkdir(parents=True, exist_ok=False)
    cells = [dict(market_seed=args.seed, repetition=0, condition=arm) for arm in ARMS]
    random.Random(20260923 + args.seed).shuffle(cells)
    write(args.output / "protocol.json", dict(model=MODEL, revision=REVISION, cells=cells,
        max_turns=16, max_tokens=2048, initial_feedback_outside_turn_budget=True,
        task="repair identical author-generated flawed starter",
        exposure="public development generator; no independent holdout claim",
        starter_sha256=hashlib.sha256(STARTER.encode()).hexdigest(),
        comparisons="execution vs accounting vs full feedback; full feedback vs final gate",
        limitations="fixed starter; sampled public probes; CPU check costs differ by arm"))
    from benchmarks.agent_study.transformers_chat import TransformersChat
    chat = TransformersChat(MODEL, REVISION, max_tokens=2048)
    for cell in cells:
        run = args.output / f"s{args.seed}-r0-{cell['condition']}"
        workspace = run / "workspace"
        export(args.seed, workspace)
        (workspace / "submission.py").write_text(STARTER, encoding="utf-8")
        write(workspace / "report.json", dict(reported_sharpe=1.0, cost_bps_per_side=5.0,
                                              guards_cited=[]))
        write(run / "input_receipt.json", {str(p.relative_to(workspace)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in workspace.rglob("*") if p.is_file()})
        chat.seed, chat.calls = args.seed * 1000, 0
        session = RepairSession(workspace, cell["condition"])
        result = run_agent(session, chat, max_turns=16)
        result.update(cell, model=MODEL, revision=REVISION, initial_feedback=session.initial_feedback)
        write(run / "result.json", result)
        write(run / "submission_receipt.json", dict(artifact_sha256=result["artifact_sha256"],
            result_sha256=hashlib.sha256((run / "result.json").read_bytes()).hexdigest(),
            slurm_job_id=os.environ["SLURM_JOB_ID"]))
        try:
            proc = subprocess.run([sys.executable, "-m", "benchmarks.agent_study.oracle",
                "--task", str(workspace), "--submission", str(workspace),
                "--out", str(run / "grade.json")], capture_output=True, text=True, timeout=300)
            if proc.returncode:
                write(run / "grading_error.json", dict(error=proc.stderr[-3000:]))
        except subprocess.TimeoutExpired:
            write(run / "grading_error.json", dict(error="oracle timeout"))
        print(json.dumps(dict(cell=cell, accepted=result["accepted"])), flush=True)
    write(args.output / "summary.json", summarize([args.output]))


if __name__ == "__main__":
    main()
