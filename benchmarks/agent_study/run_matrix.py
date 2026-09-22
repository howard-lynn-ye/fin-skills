"""Freeze a balanced run manifest before inference; score only after submission freeze.

Public-generator development experiment. It is not an unseen capability benchmark.
Incomplete cells and provider/tool failures remain in the denominator.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.agent_study.build_task import export
from benchmarks.agent_study.open_agent_runner import CONDITIONS, AgentWorkspaceSession, run_agent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", default="11,23,37")
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cells = [{"market_seed": int(seed), "repetition": rep, "condition": condition}
             for seed in args.seeds.split(",") for rep in range(args.repetitions) for condition in CONDITIONS]
    random.Random(20260921).shuffle(cells)
    source_files = sorted({*(ROOT / "benchmarks/agent_study").glob("*.py"),
                           *(ROOT / "benchmarks/agent_study").glob("*.sh"),
                           *(ROOT / "fin_skills").rglob("*.py"), ROOT / "pyproject.toml"})
    protocol = {"schema_version": 1, "task_interface_version": 2,
                "model": args.model, "revision": args.revision,
                "max_turns": args.max_turns, "max_tokens": args.max_tokens, "cells": cells,
                "exposure": "public generator and public development seeds; not a private holdout",
                "source_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}}
    frozen = args.output / "protocol.json"
    text = json.dumps(protocol, indent=2)
    if frozen.exists() and frozen.read_text() != text:
        raise ValueError("frozen protocol differs; create a new experiment directory")
    frozen.write_text(text)
    if args.plan_only:
        print(json.dumps({"planned_cells": len(cells), "protocol": str(frozen)}))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("launch GPU inference through Slurm")
    from benchmarks.agent_study.transformers_chat import TransformersChat
    chat = TransformersChat(args.model, args.revision, max_tokens=args.max_tokens)
    for cell in cells:
        cell_id = f"s{cell['market_seed']}-r{cell['repetition']}-{cell['condition']}"
        run = args.output / cell_id
        if (run / "result.json").exists():
            continue
        if run.exists():
            raise RuntimeError(f"interrupted cell {cell_id}; retain and classify before resume")
        run.mkdir()
        workspace = run / "workspace"
        export(cell["market_seed"], workspace)
        inputs = [workspace / "TASK.md", workspace / "manifest.json",
                  *sorted((workspace / "data").glob("*.csv"))]
        (run / "input_receipt.json").write_text(json.dumps({
            "task_interface_version": 2,
            "sha256": {str(f.relative_to(workspace)): hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in inputs}}, indent=2), encoding="utf-8")
        # Same inference seed across paired conditions; no shared dialogue or KV state.
        chat.seed, chat.calls = cell["market_seed"] * 1000 + cell["repetition"], 0
        session = AgentWorkspaceSession(workspace, cell["condition"])
        result = run_agent(session, chat, max_turns=args.max_turns)
        result.update(cell, model=args.model, revision=args.revision)
        frozen_result = run / "result.json"
        frozen_result.write_text(json.dumps(result, indent=2, default=str, allow_nan=False))
        (run / "submission_receipt.json").write_text(json.dumps({
            "result_sha256": hashlib.sha256(frozen_result.read_bytes()).hexdigest(),
            "artifact_sha256": result["artifact_sha256"], "slurm_job_id": os.environ["SLURM_JOB_ID"]}))
        # Grade rejected submissions too, without feeding grades back to the model.
        try:
            proc = subprocess.run([sys.executable, "-m", "benchmarks.agent_study.oracle",
                "--task", str(workspace), "--submission", str(workspace), "--out", str(run / "grade.json")],
                cwd=ROOT, capture_output=True, text=True, timeout=300)
            if proc.returncode:
                (run / "grading_error.json").write_text(json.dumps({"error": proc.stderr[-3000:]}))
        except subprocess.TimeoutExpired:
            (run / "grading_error.json").write_text(json.dumps({"error": "oracle timeout"}))
        print(json.dumps({"cell": cell_id, "accepted": result["accepted"]}), flush=True)


if __name__ == "__main__":
    main()
