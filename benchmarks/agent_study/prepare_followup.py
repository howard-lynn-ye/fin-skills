"""Prepare an offline, hashed Beacon experiment bundle. No login, upload or submission.

Extract source.tar.gz under a fresh RADFM root, keep bundle-manifest.json beside it,
then use the generated submit.sh. Credentials and old experiment results are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RADFM = PurePosixPath("/beacon-projects/radfm/wy891")
MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
REVISION = "aedcc2d42b622764e023cf882b6652e646b95671"


def validate_remote_root(value: str) -> str:
    path = PurePosixPath(value)
    if (path.parent != RADFM or str(path) != value
            or not re.fullmatch(r"fin-skills-followup-[a-zA-Z0-9_-]+", path.name)):
        raise ValueError("require a canonical, direct fin-skills-followup-* RADFM child")
    return str(path)


def source_names() -> list[str]:
    listed = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    names = set()
    for name in listed:
        p = PurePosixPath(name)
        include = (name.startswith("fin_skills/") and p.suffix in (".py", ".json", ".md"))
        include |= name.startswith("benchmarks/agent_study/") and p.suffix in (".py", ".sh")
        include |= p.parent == PurePosixPath("benchmarks") and p.suffix == ".py"
        include |= p.parent == PurePosixPath("tests") and p.name.startswith("test_agent_study")
        include |= name in ("pyproject.toml", "README.md", "AGENTS.md", "tests/conftest.py")
        if include and (ROOT / name).is_file():
            if (ROOT / name).is_symlink() or not (ROOT / name).resolve().is_relative_to(ROOT):
                raise ValueError(f"source file outside the repository: {name}")
            names.add(name)
    return sorted(names)


def prepare(output: Path, remote_root: str) -> dict:
    remote_root = validate_remote_root(remote_root)
    # Resolve first so all returned local artifact paths are usable from any directory.
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace an existing bundle: {output}")
    sources = {name: (ROOT / name).read_bytes() for name in source_names()}
    required = ("benchmarks/agent_study/beacon_followup.sh",
                "benchmarks/agent_study/run_matrix.py", "fin_skills/__init__.py")
    if any(name not in sources for name in required):
        raise ValueError("required source missing from bundle")
    manifest = {
        "schema_version": 1, "status": "prepared_not_submitted", "remote_root": remote_root,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT).decode(),
        "model": MODEL, "revision": REVISION, "task_interface_version": 2,
        "seeds": [11, 23, 37], "repetitions": 1, "planned_cells": 12,
        "max_turns": 16, "max_tokens": 2048,
        "scope": "public development feasibility; no private holdout claims",
        "source_sha256": {n: hashlib.sha256(data).hexdigest() for n, data in sources.items()},
    }
    output.mkdir(parents=True)
    archive = output / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in sources.items():
            info = tarfile.TarInfo("source/" + name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    quoted = shlex.quote(remote_root)
    submit = f'''#!/usr/bin/env bash
set -euo pipefail
root={quoted}
test "$(realpath -e "$root")" = "$root"
cd "$root"
test -f bundle-manifest.json
test -f source/benchmarks/agent_study/beacon_followup.sh
test ! -L logs
mkdir -p logs
test ! -e submission-job-id.txt
# --chdir, --output and --error are absolute, including scheduler-created logs.
job_id=$(sbatch --parsable --account=angliece --partition=beacon --qos=medium \\
  --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=64G --time=02:00:00 \\
  --job-name=fin-followup-14b --chdir="$root" \\
  --output="$root/logs/slurm-%j.out" --error="$root/logs/slurm-%j.err" \\
  "$root/source/benchmarks/agent_study/beacon_followup.sh" "$root")
printf '%s\\n' "$job_id" > submission-job-id.txt
printf '%s\\n' "$job_id"
'''
    (output / "submit.sh").write_text(submit, encoding="utf-8", newline="\n")
    return {"status": manifest["status"], "output": str(output),
            "remote_root": remote_root, "files": len(sources), "planned_cells": 12,
            "archive_sha256": manifest["archive_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.remote_root), indent=2))


if __name__ == "__main__":
    main()
