"""Validate and stage a frozen followup bundle, then submit one RADFM-only job.

Run from the repository with ``python -m scripts.deploy_beacon_followup BUNDLE``.
The default login prompts privately in the local terminal. It never searches for
credentials. ``--dry-run`` validates locally and does not connect or write files.
An existing destination or deployment receipt prevents accidental resubmission.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import getpass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import socket
import sys
import tarfile

from benchmarks.agent_study.prepare_followup import validate_remote_root

HOST = "beaconsub01.ihc.umd.edu"
FINGERPRINT = "SHA256:F0/SPeAvRCv/rg28Tx5Z+uKmtcU5DSwW8T5iwbQzKmo"


def validate_bundle(bundle: Path) -> tuple[dict, dict[str, bytes]]:
    """Check the exact bytes to upload, including duplicate/traversing tar names."""
    payload = {name: (bundle / name).read_bytes()
               for name in ("source.tar.gz", "bundle-manifest.json", "submit.sh")}
    manifest = json.loads(payload["bundle-manifest.json"])
    validate_remote_root(manifest["remote_root"])
    if hashlib.sha256(payload["source.tar.gz"]).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("archive hash differs from the frozen manifest")
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(payload["source.tar.gz"]), mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or str(path) != member.name or not member.name.startswith("source/")):
                raise ValueError(f"unsafe archive member: {member.name}")
            name = member.name.removeprefix("source/")
            if name in seen or name not in manifest["source_sha256"]:
                raise ValueError(f"duplicate or unlisted archive member: {name}")
            seen.add(name)
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != manifest["source_sha256"][name]:
                raise ValueError(f"source hash mismatch: {name}")
    if seen != set(manifest["source_sha256"]):
        raise ValueError("archive is missing frozen source files")
    if "benchmarks/agent_study/beacon_followup.sh" not in seen:
        raise ValueError("bundle has no Beacon runtime entry point")
    return manifest, payload


def submission_command(remote: str) -> str:
    """Every scheduler-created file and the job's working directory use RADFM."""
    remote = validate_remote_root(remote)
    args = ["sbatch", "--parsable", "--account=angliece", "--partition=beacon",
            "--qos=medium", "--nodes=1", "--ntasks=1", "--cpus-per-task=4",
            "--gres=gpu:1", "--mem=64G", "--time=02:00:00",
            "--job-name=fin-followup-14b", f"--chdir={remote}",
            f"--output={remote}/logs/slurm-%j.out", f"--error={remote}/logs/slurm-%j.err",
            f"{remote}/source/benchmarks/agent_study/beacon_followup.sh", remote]
    return ("set -eu\n" + f"cd {shlex.quote(remote)}\n"
            'test "$(pwd -P)" = ' + shlex.quote(remote) + "\n"
            "test ! -e submission-job-id.txt\n"
            "job_id=$(" + shlex.join(args) + ")\n"
            "printf '%s\\n' \"$job_id\" > submission-job-id.txt\n"
            "printf '%s\\n' \"$job_id\"\n")


def remote_command(transport, command: str) -> str:
    channel = transport.open_session(timeout=30)
    try:
        channel.settimeout(60)
        channel.exec_command(command)
        out = channel.makefile().read().decode("utf-8", "replace")
        err = channel.makefile_stderr().read().decode("utf-8", "replace")
        code = channel.recv_exit_status()
        if code:
            raise RuntimeError(f"remote command exited {code}: {err.strip()} {out.strip()}")
        return out.strip()
    finally:
        channel.close()


def deploy(manifest: dict, payload: dict[str, bytes], receipt_path: Path,
           credential_file: Path | None = None) -> dict:
    # Paramiko is only needed for a live deployment; validation remains stdlib-only.
    import paramiko

    if receipt_path.exists():
        raise FileExistsError(f"inspect the existing deployment receipt: {receipt_path}")
    if credential_file is None and not sys.stdin.isatty():
        raise RuntimeError("run in a local terminal for the private Beacon password prompt")
    remote = validate_remote_root(manifest["remote_root"])
    transport = paramiko.Transport(socket.create_connection((HOST, 22), timeout=15))
    try:
        transport.start_client(timeout=30)
        key = transport.get_remote_server_key()
        actual = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
        if key.get_name() != "ssh-ed25519" or actual != FINGERPRINT:
            raise RuntimeError("Beacon host key does not match the pinned key")
        password = (credential_file.read_text(encoding="utf-8").splitlines()[0]
                    if credential_file else getpass.getpass("Beacon password (wy891): "))
        try:
            transport.auth_password("wy891", password, fallback=False)
        finally:
            del password
        receipt = {"status": "authenticated_not_staged", "remote_root": remote,
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "archive_sha256": manifest["archive_sha256"],
                   "uploaded_sha256": {n: hashlib.sha256(b).hexdigest()
                                       for n, b in payload.items()},
                   "planned_cells": manifest["planned_cells"]}
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with receipt_path.open("x", encoding="utf-8") as out:
            json.dump(receipt, out, indent=2)

        def save(status: str) -> None:
            receipt["status"] = status
            temporary = receipt_path.with_name(receipt_path.name + ".tmp")
            temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            temporary.replace(receipt_path)

        with paramiko.SFTPClient.from_transport(transport) as sftp:
            parent = str(PurePosixPath(remote).parent)
            if sftp.normalize(parent) != parent:
                raise RuntimeError("RADFM parent resolves to a different directory")
            sftp.mkdir(remote, mode=0o700)  # Deliberately refuses an existing run.
            if sftp.normalize(remote) != remote:
                raise RuntimeError("experiment directory is not canonical")
            for name in ("logs", "tmp", "cache", "results"):
                sftp.mkdir(f"{remote}/{name}", mode=0o700)
            for name, data in payload.items():
                with sftp.open(f"{remote}/{name}", "wx") as out:
                    out.write(data)
                with sftp.open(f"{remote}/{name}", "rb") as uploaded:
                    if hashlib.sha256(uploaded.read()).hexdigest() != receipt["uploaded_sha256"][name]:
                        raise RuntimeError(f"remote upload hash mismatch: {name}")
        save("uploaded_not_submitted")
        remote_command(transport, f"cd {shlex.quote(remote)} && tar -xzf source.tar.gz")
        # Persist this state before sbatch: a disconnect must never trigger a blind retry.
        save("submission_in_progress")
        job = remote_command(transport, submission_command(remote))
        if not re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", job):
            raise RuntimeError("unexpected sbatch response; inspect the remote submission receipt")
        receipt["job_id"] = job.split(";", 1)[0]
        save("submitted")
        receipt["scheduler"] = remote_command(
            transport, f"squeue -h -j {receipt['job_id']} -o '%i|%j|%T|%R'")
        save("submitted")
        return receipt
    finally:
        transport.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--credential-file", type=Path,
                        help="optional explicitly supplied local file; never uploaded or logged")
    args = parser.parse_args()
    manifest, payload = validate_bundle(args.bundle.resolve())
    if args.dry_run:
        print(json.dumps({"status": "validated_not_submitted", "remote_root": manifest["remote_root"],
                          "files": len(manifest["source_sha256"]),
                          "archive_sha256": manifest["archive_sha256"],
                          "planned_cells": manifest["planned_cells"]}, indent=2))
        return
    result = deploy(manifest, payload, args.bundle.resolve() / "deployment-receipt.json",
                    args.credential_file)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
