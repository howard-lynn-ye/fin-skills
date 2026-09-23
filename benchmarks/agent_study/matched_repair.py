"""Matched maximum-budget repair controls, separate from earlier end-to-end studies.

All arms receive identical domain text and accounting assistance. The estimand is the
incremental effect of generic checks, required self-review, domain probes and enforcement.
This author-generated development study does not establish independent task transfer.
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
import time

from benchmarks.agent_study.build_task import export
from benchmarks.agent_study.feedback_ablation import STARTER, write
from benchmarks.agent_study.open_agent_runner import (
    AgentWorkspaceSession, PUBLIC_CHECKS, artifact_digest, run_agent,
)
from benchmarks.agent_study.prepare_followup import MODEL, REVISION
from benchmarks.agent_study.summarize_matrix import summarize

ARMS = ("accounting_only", "self_review", "generic_checks", "domain_feedback", "domain_gate")


class MatchedSession(AgentWorkspaceSession):
    def __init__(self, workspace, arm):
        if arm not in ARMS:
            raise ValueError("unknown matched arm")
        super().__init__(workspace, "skills_text_only")
        self.arm, self.review_digest = arm, None
        self.initial_feedback = self.inspect()

    def inspect(self):
        checks = ("execution", "accounting")
        if self.arm == "generic_checks":
            checks += ("generic",)
        elif self.arm in ("domain_feedback", "domain_gate"):
            checks += tuple(c for c in PUBLIC_CHECKS if c != "accounting")
        return {name: self._worker(name) for name in checks}

    def system_prompt(self):
        prompt = super().system_prompt().replace(
            "inspect_submission recomputes accounting at your stated costs.",
            "inspect_submission returns the enabled checks, including accounting for every arm.")
        prompt += ("\nREPAIR the existing fallible submission.py and report.json. "
            "This protocol fixes cost_bps_per_side at 10.0 for every condition; TASK.md's "
            "illustrative 0.0 is not the cost for this experiment. All arms receive the same "
            "task, manifest, starter, domain guidance, accounting utility and maximum budget. "
            "You may use write_artifacts(source, report) to write both artifacts in one call: "
            "source is the full Python string, report is a JSON object. Also available: "
            "review_submission(review), where review is your own nonempty critique. "
            "Enabled checks: " + ", ".join(self.initial_feedback) + ".")
        if self.arm == "self_review":
            prompt += (" Before finish, call review_submission with your assessment of the CURRENT "
                "code and report for errors against the task. This call consumes the same turn "
                "budget. If you edit either artifact, review the revised artifacts again.")
        if self.arm == "domain_gate":
            prompt += " finish re-executes enabled checks on final artifacts and requires all to pass."
        else:
            prompt += " finish freezes the artifacts without requiring the enabled checks to pass."
        for name in ("TASK.md", "manifest.json", "submission.py", "report.json"):
            prompt += "\n" + name + ":\n" + (self.workspace / name).read_text(encoding="utf-8")
        return prompt + "\nInitial feedback:\n" + json.dumps(self.initial_feedback, default=str)

    def dispatch(self, name, arguments):
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")
        if name == "inspect_submission":
            self.tool_call_count += 1
            return self.inspect()
        if name == "review_submission":
            self.tool_call_count += 1
            review = arguments.get("review")
            if not isinstance(review, str) or not review.strip() or len(review) > 20000:
                raise ValueError("review must be a nonempty string at most 20000 characters")
            self.review_digest = artifact_digest(self.workspace)
            return dict(status="REVIEW_RECORDED", artifact_sha256=self.review_digest,
                        scope="model self-review; no external correctness check")
        if name == "write_artifacts":
            self.tool_call_count += 1
            source, report = arguments.get("source"), arguments.get("report")
            if not isinstance(source, str) or len(source) > 200000 or not isinstance(report, dict):
                raise ValueError("source must be a string and report must be an object")
            encoded = json.dumps(report, allow_nan=False)
            (self.workspace / "submission.py").write_text(source, encoding="utf-8")
            (self.workspace / "report.json").write_text(encoded, encoding="utf-8")
            return dict(written=["submission.py", "report.json"])
        return super().dispatch(name, arguments)

    def finish(self):
        basic = super().finish()
        if not basic.get("accepted"):
            return basic
        report = json.loads((self.workspace / "report.json").read_text())
        if report["cost_bps_per_side"] != 10.0:
            return dict(accepted=False, status="COST_PROTOCOL", required_cost=10.0)
        if self.arm == "self_review" and self.review_digest != artifact_digest(self.workspace):
            return dict(accepted=False, status="REVIEW_REQUIRED",
                        instruction="Review current code and report with review_submission before finish.")
        if self.arm != "domain_gate":
            return basic
        checks = self.inspect()
        passed = all(v.get("passed") is True for v in checks.values())
        return dict(basic, accepted=passed, status="PASS" if passed else "FAIL", checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    from benchmarks.agent_study.model_transfer import MODEL as SECOND_MODEL, REVISION as SECOND_REVISION
    if (args.model, args.revision) not in ((MODEL, REVISION), (SECOND_MODEL, SECOND_REVISION)):
        raise ValueError("model and immutable revision must be a prespecified pair")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    args.output.mkdir(parents=True, exist_ok=False)
    cells = [dict(market_seed=args.seed, repetition=0, condition=arm) for arm in ARMS]
    random.Random(2026092302 + args.seed).shuffle(cells)
    write(args.output / "protocol.json", dict(model=args.model, revision=args.revision, cells=cells,
        max_turns=16, max_tokens=2048, initial_feedback_outside_turn_budget=True,
        cost_bps_per_side=10.0, starter_sha256=hashlib.sha256(STARTER.encode()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="author-generated development repair; paired seeds, not independent tasks",
        budget="same maximum calls/output tokens; actual input tokens and check times differ",
        common="task, manifest, starter, domain text, execution/accounting, artifact-writing tools",
        limits="Distinct protocol from prior runs: preload task/files, batch write tool, fixed costs. "
               "Self-review is an elicited textual review, not proof of reasoning quality. "
               "Frozen oracle has separately measured coverage limits; no unseen-defect claim."))
    from benchmarks.agent_study.transformers_chat import TransformersChat
    chat = TransformersChat(args.model, args.revision, max_tokens=2048)
    for cell in cells:
        run = args.output / f"s{args.seed}-r0-{cell['condition']}"
        workspace = run / "workspace"
        export(args.seed, workspace)
        (workspace / "submission.py").write_text(STARTER, encoding="utf-8")
        write(workspace / "report.json", dict(reported_sharpe=1.0,
            cost_bps_per_side=10.0, guards_cited=[]))
        write(run / "input_receipt.json", {str(p.relative_to(workspace)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in workspace.rglob("*") if p.is_file()})
        chat.seed, chat.calls = args.seed * 1000, 0
        started = time.monotonic()
        session = MatchedSession(workspace, cell["condition"])
        result = run_agent(session, chat, max_turns=16)
        result.update(cell, model=args.model, revision=args.revision,
                      initial_feedback=session.initial_feedback,
                      total_wall_including_initial_checks=time.monotonic()-started)
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
