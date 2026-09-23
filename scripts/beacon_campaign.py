"""Freeze, deploy and inspect independent Slurm jobs under one RADFM campaign.

No node names are requested. A missing prerequisite blocks only the affected job.
An uncertain submission is never retried automatically. Credentials stay local.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import getpass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import socket
import subprocess
import sys
import tarfile

from benchmarks.agent_study.prepare_followup import MODEL, REVISION, source_names
from scripts.deploy_beacon_followup import HOST, FINGERPRINT, remote_command
from scripts.submit_model_followup import _dispatch

REPO = Path(__file__).resolve().parents[1]
BASE = "/beacon-projects/radfm/wy891"
RUNTIME = BASE + "/fin-skills-audit-20260921"
FLY = BASE + "/fin-skills-fly-reuse-20260921"


def validate_root(value):
    path = PurePosixPath(value)
    if (str(path) != value or str(path.parent) != BASE or not
            re.fullmatch(r"fin-skills-campaign-[A-Za-z0-9_-]+", path.name)):
        raise ValueError("campaign must be a canonical direct RADFM child")
    return value


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def jobs(suite="initial"):
    if suite in ("decision-qualification", "decision-loop"):
        from benchmarks.agent_study.model_transfer import MODEL as second_model, REVISION as second_revision
        groups = ("qualification",) if suite == "decision-qualification" else ("tools", "memory", "context")
        planned = [dict(id=f"decision-{family}-{group}", study="decision-loop", group=group,
                     seed=11, gpu=True, cpus=4, memory="64G", time="02:00:00",
                     model=model, revision=revision, model_cache=cache)
                for family,model,revision,cache in (
                    ("qwen",MODEL,REVISION,RUNTIME + "/cache/hf/hub"),
                    ("mistral",second_model,second_revision,
                     BASE + "/fin-skills-campaign-model-transfer-20260923-v1/model-cache/hub"))
                for group in groups]
        if suite == "decision-qualification":
            # One allocation qualifies both models sequentially, avoiding two
            # submissions when the association has only one remaining slot.
            planned[0]["qualification_models"] = [dict(model=j["model"], revision=j["revision"],
                model_cache=j["model_cache"], family=j["id"].split('-')[1]) for j in planned]
            planned[0]["id"] = "decision-qualification"
            return planned[:1]
        return planned
    if suite == "boundary-recheck":
        return [dict(id="boundary-recheck", study="boundary-recheck", gpu=False,
                     cpus=2, memory="8G", time="00:20:00")]
    if suite == "model-transfer":
        from benchmarks.agent_study.model_transfer import MODEL as second_model, REVISION as second_revision
        return [dict(id="model-staging", study="model-staging", gpu=False,
                     cpus=4, memory="16G", time="01:00:00"),
                *[dict(j, depends_on="model-staging", model=second_model, revision=second_revision)
                  for j in jobs("matched-repair")]]
    if suite == "matched-repair":
        return [dict(id=f"matched-repair-s{s}", study="matched-repair", seed=s, gpu=True,
                     cpus=4, memory="64G", time="02:00:00") for s in (11, 23, 37)]
    if suite == "components":
        return [dict(id="component-comparison", study="components", gpu=False,
                     cpus=2, memory="8G", time="01:00:00")]
    if suite == "audit-validity":
        return [dict(id=f"audit-validity-s{s}", study="audit-validity", seed=s, gpu=False,
                     cpus=2, memory="8G", time="01:00:00") for s in (11, 23, 37)]
    if suite == "routing":
        return [dict(id="routing", study="routing", gpu=False,
                     cpus=4, memory="16G", time="00:30:00")]
    if suite == "retrieval":
        return [dict(id="retrieval-pilot", study="retrieval", gpu=True,
                     cpus=4, memory="64G", time="01:00:00")]
    if suite == "feedback":
        return [dict(id=f"feedback-s{s}", study="feedback", seed=s, gpu=True,
                     cpus=4, memory="64G", time="02:00:00") for s in (11, 23, 37)]
    if suite == "recovery":
        return [dict(id="validation", study="validation", gpu=False, cpus=1,
                     memory="8G", time="00:15:00"),
                *[dict(j, depends_on="validation") for j in jobs("pipeline")],
                dict(id="fly-gate", study="fly-gate", gpu=False, cpus=4, memory="32G",
                     time="01:00:00", depends_on="validation")]
    if suite == "pipeline":
        return [dict(id=f"pipeline-s{s}", study="pipeline", seed=s, gpu=True,
                     cpus=4, memory="64G", time="02:00:00") for s in (11, 23, 37)]
    if suite != "initial":
        raise ValueError("unknown suite")
    return [*[dict(id=f"agent-s{s}", study="agent", seed=s, gpu=True,
                   cpus=4, memory="64G", time="02:00:00") for s in (11, 23, 37)],
        dict(id="temporal", study="temporal", gpu=False, cpus=1, memory="8G", time="00:15:00"),
        dict(id="fly-gate", study="fly-gate", gpu=False, cpus=4, memory="32G", time="01:00:00"),
        dict(id="rag-jev", study="rag-jev", gpu=False, cpus=1, memory="8G", time="00:15:00")]


def prepare(output, remote_root, suite="initial"):
    remote_root = validate_root(remote_root)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = set(source_names())
    for prefix in ("benchmarks/fly_reuse", "benchmarks/rag_jev", "benchmarks/library_workflows", "benchmarks/fly_paper"):
        names.update(p.relative_to(REPO).as_posix() for p in (REPO / prefix).glob("*")
                     if p.suffix in (".py", ".json", ".sh") and p.is_file())
    names.update(("scripts/beacon_campaign.py", "scripts/beacon_campaign_worker.py",
        "scripts/deploy_beacon_followup.py", "scripts/submit_model_followup.py",
        "examples/collected_rag.py", "tests/test_fly_gate_control.py",
        "tests/test_rag_jev_evaluation.py", "tests/test_beacon_campaign.py",
        "tests/test_temporal_benchmark.py", "tests/test_pipeline_benchmark.py",
        "tests/test_fly_snapshot.py", "tests/test_feedback_ablation.py",
        "tests/test_retrieval_downstream.py", "tests/test_audit_validity.py",
        "tests/test_component_comparison.py", "tests/test_matched_repair.py", "tests/test_decision_loop.py",
        "paper/DECISION_LOOP_PROTOCOL_20260923.md", "evals/queries.jsonl"))
    sources = {}
    for name in sorted(names):
        path = REPO / name
        if path.is_symlink() or not path.resolve().is_relative_to(REPO):
            raise ValueError(f"source outside repository: {name}")
        sources[name] = path.read_bytes()
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for name, data in sources.items():
            entry = tarfile.TarInfo("source/" + name)
            entry.size, entry.mode, entry.mtime = len(data), 0o644, 0
            archive.addfile(entry, io.BytesIO(data))
    plan = dict(schema_version=1, status="prepared_not_submitted", remote_root=remote_root,
        created_utc=datetime.now(timezone.utc).isoformat(), model=MODEL, revision=REVISION,
        runtime_root=RUNTIME, fly_prior_root=FLY, suite=suite, jobs=jobs(suite),
        exposure="public development data; no independent holdout claim",
        pending_studies=[
            dict(id="retrieval-expansion", status="development_downstream_pilot_prepared",
                 reason="Static context, BM25 and pinned E5 with downstream choice scoring exist; "
                 "the six-query development pilot cannot replace broader independent answer evaluation."),
            dict(id="pipeline-efficiency", status="coding_agent_protocol_prepared",
                 reason="Public development build/change/restore coding-agent protocol exists; "
                 "a human developer study and externally authored tasks still require independent input."),
            dict(id="independent-defects", status="needs_independent_cases",
                 reason="Author-created defects cannot be reported as independently authored holdout cases."),
            dict(id="feedback-ablation", status="development_repair_protocol_prepared",
                 reason="Execution, accounting, full feedback and final gate use identical "
                 "flawed starters; this is separate from the original end-to-end study.")],
        source_sha256={n: hashlib.sha256(b).hexdigest() for n, b in sources.items()},
        archive_sha256=hashlib.sha256((output / "source.tar.gz").read_bytes()).hexdigest())
    write_once(output / "campaign.json", plan)
    return {k: plan[k] for k in ("status", "remote_root", "jobs", "pending_studies")}


def validate_bundle(bundle):
    bundle = Path(bundle)
    plan = json.loads((bundle / "campaign.json").read_text())
    validate_root(plan["remote_root"])
    data = (bundle / "source.tar.gz").read_bytes()
    if hashlib.sha256(data).hexdigest() != plan["archive_sha256"]:
        raise ValueError("archive hash mismatch")
    found = set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            name = member.name.removeprefix("source/")
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or not member.name.startswith("source/") or name in found
                    or name not in plan["source_sha256"]):
                raise ValueError("unsafe/unlisted archive member")
            found.add(name)
            if hashlib.sha256(archive.extractfile(member).read()).hexdigest() != plan["source_sha256"][name]:
                raise ValueError("source hash mismatch")
    if found != set(plan["source_sha256"]):
        raise ValueError("incomplete archive")
    return plan


def scheduler_command(plan, job):
    root = validate_root(plan["remote_root"])
    if not re.fullmatch(r"[a-z0-9-]+", job["id"]):
        raise ValueError("unsafe job id")
    work = f"{root}/jobs/{job['id']}"
    command = ["sbatch", "--parsable", "--account=angliece", "--partition=beacon",
        "--qos=medium", "--nodes=1", "--ntasks=1", f"--cpus-per-task={job['cpus']}",
        f"--mem={job['memory']}", f"--time={job['time']}", f"--job-name=fin-{job['id']}",
        f"--chdir={work}", f"--output={work}/logs/slurm-%j.out", f"--error={work}/logs/slurm-%j.err"]
    if job["gpu"]:
        command += ["--gres=gpu:1"]
    return command + [f"{work}/job.sh"]


def submit_remote(root):
    root = Path(validate_root(root))
    if root.resolve() != root:
        raise ValueError("campaign directory resolves elsewhere")
    plan = validate_bundle(root)
    for name, expected in plan["source_sha256"].items():
        path = root / "source" / name
        if path.resolve() != path or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("staged source hash/path mismatch")
    snapshots = {}
    for label, args in (("queue", ["squeue", "-u", "wy891", "-o", "%i|%j|%T|%R"]),
                        ("resources", ["sinfo", "-o", "%P|%a|%l|%D|%t|%G"])):
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        snapshots[label] = dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode:
            raise RuntimeError(f"scheduler preflight failed: {label}")
    write_once(root / "scheduler-before.json", snapshots)
    receipts = []
    for job in plan["jobs"]:
        work = root / "jobs" / job["id"]
        work.mkdir(parents=True, exist_ok=False)
        (work / "logs").mkdir()
        reason = None
        if (job["study"] == "rag-jev" and not os.environ.get("TYPESAFE_API_KEY")
                and not (root / "private/jev-key").is_file()):
            reason = "TYPESAFE_API_KEY is absent in the submission environment"
        if job["study"] == "fly-gate":
            prior = Path(plan["fly_prior_root"])
            if not all((prior / item).exists() for item in
                       ("env/bin/python", "data/kraken-daily.json", "upstream/fruit-fly-fund", "upstream/stonkfly-lab")):
                reason = "frozen fly runtime, source or market snapshot is absent"
        runtime = Path(plan["runtime_root"]) / "env/bin/python"
        if not runtime.is_file():
            reason = "common RADFM Python runtime is absent"
        if reason:
            receipt = dict(status="blocked_prerequisite", reason=reason)
            write_once(work / "submission-receipt.json", receipt)
        else:
            py = (f"{plan['fly_prior_root']}/env/bin/python" if job["study"] == "fly-gate"
                  else str(runtime))
            command = [py, str(root / "source/scripts/beacon_campaign_worker.py"),
                       str(root), job["id"]]
            shell = "#!/usr/bin/env bash\nset -euo pipefail\nunset HISTFILE\n"
            shell += "export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1\n"
            shell += "exec " + shlex.join(command) + "\n"
            (work / "job.sh").write_text(shell, encoding="utf-8", newline="\n")
            command = scheduler_command(plan, job)
            if job.get("depends_on"):
                dependency = next(r for r in receipts if r["id"] == job["depends_on"])
                if dependency["status"] != "submitted":
                    receipt = dict(status="blocked_prerequisite", reason="validation was not submitted")
                    write_once(work / "submission-receipt.json", receipt)
                    receipts.append(dict(id=job["id"], **receipt))
                    continue
                command.insert(1, f"--dependency=afterok:{dependency['job_id']}")
            receipt = _dispatch(work, command)
            if job["study"] == "rag-jev" and receipt["status"] == "submission_failed":
                (root / "private/jev-key").unlink(missing_ok=True)
        receipts.append(dict(id=job["id"], **receipt))
    write_once(root / "campaign-submission.json", dict(jobs=receipts))
    return dict(jobs=receipts, pending_studies=plan["pending_studies"])


def connect(credential_file=None):
    import paramiko
    transport = paramiko.Transport(socket.create_connection((HOST, 22), timeout=15))
    try:
        transport.start_client(timeout=30)
        key = transport.get_remote_server_key()
        fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
        if key.get_name() != "ssh-ed25519" or fingerprint != FINGERPRINT:
            raise RuntimeError("Beacon host key mismatch")
        if credential_file:
            password = Path(credential_file).read_text(encoding="utf-8").splitlines()[0]
        elif sys.stdin.isatty():
            password = getpass.getpass("Beacon password (wy891): ")
        else:
            raise RuntimeError("supply the authorized credential file or run in a private terminal")
        try:
            transport.auth_password("wy891", password, fallback=False)
        finally:
            del password
        return transport
    except BaseException:
        transport.close()
        raise


def deploy(bundle, credential_file=None):
    import paramiko
    bundle = Path(bundle)
    plan = validate_bundle(bundle)
    if (bundle / "deployment-started.json").exists():
        raise FileExistsError("inspect remote status before retrying an existing deployment")
    root = plan["remote_root"]
    with connect(credential_file) as transport:
        # Check the account and any existing campaign before mutating remote files.
        preflight = remote_command(transport, "test -d " + shlex.quote(BASE)
            + " && test ! -e " + shlex.quote(root)
            + " && squeue -u wy891 -o '%i|%j|%T|%R'")
        write_once(bundle / "deployment-started.json", dict(remote_root=root,
            status="staging", preflight=preflight, archive_sha256=plan["archive_sha256"]))
        with paramiko.SFTPClient.from_transport(transport) as client:
            if client.normalize(BASE) != BASE:
                raise RuntimeError("RADFM parent resolves elsewhere")
            client.mkdir(root, mode=0o700)
            for name in ("source.tar.gz", "campaign.json"):
                blob = (bundle / name).read_bytes()
                with client.open(f"{root}/{name}", "wx") as out:
                    out.write(blob)
                with client.open(f"{root}/{name}", "rb") as inp:
                    if hashlib.sha256(inp.read()).digest() != hashlib.sha256(blob).digest():
                        raise RuntimeError("upload hash mismatch")
            # Only the Jev job reads this key; it deletes the file before its first request.
            # This file is excluded from source archives, receipts and downloaded results.
            api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
            if api_key:
                client.mkdir(root + "/private", mode=0o700)
                with client.open(root + "/private/jev-key", "wx") as secret:
                    secret.write(api_key.encode("utf-8"))
                client.chmod(root + "/private/jev-key", 0o600)
                del api_key
        remote_command(transport, "cd " + shlex.quote(root) + " && tar -xzf source.tar.gz")
        command = [f"{RUNTIME}/env/bin/python", "-B", "-m", "scripts.beacon_campaign",
                   "submit-remote", root]
        # Each job creates its own durable submission marker before invoking sbatch.
        result = remote_command(transport, "cd " + shlex.quote(root + "/source")
            + " && " + shlex.join(command))
        receipt = json.loads(result)
        write_once(bundle / "deployment-receipt.json", receipt)
        return receipt


def status(bundle, credential_file=None):
    plan = validate_bundle(bundle)
    root = plan["remote_root"]
    with connect(credential_file) as transport:
        script = ("import json,pathlib; r=pathlib.Path(" + repr(root) + "); "
            "print(json.dumps({str(p.relative_to(r)):json.loads(p.read_text()) "
            "for pattern in ('jobs/*/submission-receipt.json','jobs/*/completion.json') "
            "for p in r.glob(pattern)},indent=2))")
        command = shlex.join([f"{RUNTIME}/env/bin/python", "-B", "-c", script])
        return remote_command(transport, command) + "\n" + remote_command(transport,
            "squeue -u wy891 -o '%i|%j|%T|%R'")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "deploy", "status", "submit-remote"))
    parser.add_argument("path")
    parser.add_argument("--remote-root")
    parser.add_argument("--suite", choices=("initial", "pipeline", "recovery", "feedback", "retrieval", "routing", "audit-validity", "components", "matched-repair", "model-transfer", "boundary-recheck", "decision-qualification", "decision-loop"), default="initial")
    parser.add_argument("--credential-file", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.path, args.remote_root, args.suite)
    elif args.action == "validate":
        p = validate_bundle(args.path)
        result = dict(status="validated_not_submitted", jobs=len(p["jobs"]), archive_sha256=p["archive_sha256"])
    elif args.action == "submit-remote":
        result = submit_remote(args.path)
    else:
        result = globals()[args.action](args.path, args.credential_file)
    print(result if isinstance(result, str) else json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
