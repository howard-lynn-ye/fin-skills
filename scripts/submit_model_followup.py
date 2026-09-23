"""Submit staged Jev/fly studies with absolute RADFM scheduler paths.

Run on Beacon after staging source. Default prints a plan without touching files
or invoking Slurm. --submit records one attempt; uncertain submissions are not retried.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

RADFM = PurePosixPath("/beacon-projects/radfm/wy891")


def remote_root(value: str, prefix: str) -> str:
    path = PurePosixPath(value)
    if (str(path) != value or path.parent != RADFM
            or not re.fullmatch(re.escape(prefix) + r"[a-zA-Z0-9_-]+", path.name)):
        raise ValueError("require a canonical direct RADFM study directory")
    return value


def submission_command(study: str, root: str, prior_root: str | None = None) -> list[str]:
    if study == "rag-jev":
        root = remote_root(root, "fin-skills-rag-jev-")
        if prior_root is not None:
            raise ValueError("prior_root is only valid for fly-gate")
        script, cpus, memory, duration = "rag_jev/beacon_run.sh", "1", "8G", "00:10:00"
    elif study == "fly-gate":
        root = remote_root(root, "fin-skills-fly-")
        prior_root = remote_root(prior_root or "", "fin-skills-fly-")
        if root == prior_root:
            raise ValueError("the output root must differ from the prior study")
        script, cpus, memory, duration = "fly_reuse/beacon_gate_control.sh", "4", "32G", "01:00:00"
    else:
        raise ValueError("unknown study")
    command = ["sbatch", "--parsable", "--account=angliece", "--partition=beacon", "--qos=medium",
               "--nodes=1", "--ntasks=1", f"--cpus-per-task={cpus}", f"--mem={memory}",
               f"--time={duration}", f"--job-name=fin-{study}", f"--chdir={root}",
               f"--output={root}/logs/slurm-%j.out", f"--error={root}/logs/slurm-%j.err",
               f"{root}/source/benchmarks/{script}", root]
    return command + ([prior_root] if prior_root else [])


def write_once(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def _dispatch(root: Path, command: list[str]) -> dict:
    # Persist before sbatch; interruption after acceptance must not permit a duplicate.
    write_once(root / "submission-started.json", {
        "status": "submission_in_progress", "command": command,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    })
    try:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                timeout=60, check=False)
        job = result.stdout.strip()
        if result.returncode:
            receipt = {"status": "submission_failed", "exit_code": result.returncode,
                       "scheduler_error": result.stderr[-2000:]}
        elif re.fullmatch(r"[0-9]+(?:;[a-zA-Z0-9_.-]+)?", job):
            receipt = {"status": "submitted", "job_id": job.split(";", 1)[0]}
        else:
            receipt = {"status": "submission_unknown", "reason": "unrecognized_scheduler_response"}
    except (OSError, subprocess.TimeoutExpired) as exc:
        receipt = {"status": "submission_unknown", "error_type": type(exc).__name__}
    receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                   interpretation="Submission is not evidence of job execution or completion.")
    write_once(root / "submission-receipt.json", receipt)
    return receipt


def execute(study: str, root: str, *, prior_root: str | None = None,
            submit: bool = False) -> dict:
    command = submission_command(study, root, prior_root)
    if not submit:
        return {"status": "prepared_not_submitted", "command": command}
    directory = Path(root)
    source = directory / "source"
    script = source / "benchmarks" / ("rag_jev/beacon_run.sh" if study == "rag-jev"
                                        else "fly_reuse/beacon_gate_control.sh")
    for path in (directory, source, source / "benchmarks", script.parent, script):
        if not path.exists() or path.resolve() != path:
            raise ValueError("staged RADFM paths must exist without symlink redirection")
    if prior_root is not None:
        prior = Path(prior_root)
        if not prior.is_dir() or prior.resolve() != prior:
            raise ValueError("prior RADFM directory must exist without redirection")
    if study == "rag-jev" and not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise ValueError("TYPESAFE_API_KEY must be configured in the submission environment")
    if any((directory / name).exists() for name in
           ("submission-started.json", "submission-receipt.json")):
        raise FileExistsError("an attempt is already recorded; inspect Slurm before any resubmission")
    logs = directory / "logs"
    if logs.is_symlink() or logs.resolve() != logs:
        raise ValueError("scheduler logs must stay inside the RADFM study directory")
    logs.mkdir(exist_ok=True)
    return _dispatch(directory, command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("rag-jev", "fly-gate"))
    parser.add_argument("root")
    parser.add_argument("--prior-root")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    receipt = execute(**vars(args))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] in ("prepared_not_submitted", "submitted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
