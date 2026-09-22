"""Pinned-host transport for this project's RADFM workspace; never print credentials.

Only source sync and commands explicitly supplied by the caller. No HOME writes.
The credential file stays local and is never included in the archive.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import socket
import subprocess
import tarfile
import time
import sys

import paramiko

HOST = "beaconsub01.ihc.umd.edu"
FINGERPRINT = "SHA256:F0/SPeAvRCv/rg28Tx5Z+uKmtcU5DSwW8T5iwbQzKmo"
BASE = "/beacon-projects/radfm/wy891"
ROOT = Path(__file__).resolve().parents[1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["sync", "run", "fetch"])
    p.add_argument("--credential-file", type=Path, required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--command")
    p.add_argument("--remote-file", help="relative file inside the RADFM workspace")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    target = PurePosixPath(args.remote_root)
    if target.parent != PurePosixPath(BASE) or not target.name.startswith("fin-skills-"):
        p.error("remote root must be a direct fin-skills-* child of the user's RADFM directory")
    password = args.credential_file.read_text(encoding="utf-8").splitlines()[0]
    transport = paramiko.Transport(socket.create_connection((HOST, 22), timeout=15))
    transport.start_client(timeout=30)
    key = transport.get_remote_server_key()
    fp = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
    if key.get_name() != "ssh-ed25519" or fp != FINGERPRINT:
        raise RuntimeError("host key mismatch")
    transport.auth_password("wy891", password, fallback=False)
    del password
    try:
        if args.action == "fetch":
            relative = PurePosixPath(args.remote_file or "")
            if (not args.remote_file or relative.is_absolute() or ".." in relative.parts
                    or args.output is None):
                p.error("fetch requires a workspace-relative --remote-file and local --output")
            if args.output.exists():
                p.error("fetch refuses to overwrite an existing local file")
            client = paramiko.SFTPClient.from_transport(transport)
            try:
                remote = client.normalize(str(target / relative))
                if not PurePosixPath(remote).is_relative_to(target):
                    raise ValueError("remote file resolves outside the RADFM workspace")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                client.get(remote, str(args.output))
            finally:
                client.close()
            print(json.dumps({"output": str(args.output), "sha256":
                              hashlib.sha256(args.output.read_bytes()).hexdigest()}))
            return
        if args.action == "sync":
            names = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
            allowed = ("fin_skills/", "plugins/", "scripts/", "benchmarks/", "catalog/", "shared/",
                       "tests/", "examples/", ".claude-plugin/", "ci/")
            names = sorted({n for n in names if n and (n.startswith(allowed) or
                            (n.startswith("research/") and n.endswith(".py")) or n in
                            ("pyproject.toml", "README.md", "AGENTS.md",
                             "research/production/my_holdings.json")) and (ROOT / n).is_file()})
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w:gz") as tar:
                for name in names:
                    tar.add(ROOT / name, arcname=name, recursive=False)
            blob = archive.getvalue()
            client = paramiko.SFTPClient.from_transport(transport)
            try:
                client.mkdir(str(target))
            except OSError:
                client.stat(str(target))
            with client.open(str(target / "source.tar.gz"), "wb") as f:
                f.write(blob)
            client.close()
            command = "mkdir -p source tmp cache logs && tar -xzf source.tar.gz -C source"
            print(json.dumps({"files": len(names), "archive_sha256": hashlib.sha256(blob).hexdigest()}))
        else:
            if not args.command:
                p.error("run requires --command")
            command = args.command
        ch = transport.open_session()
        ch.exec_command("cd " + shlex.quote(str(target)) + " && " + command)
        while not ch.exit_status_ready() or ch.recv_ready() or ch.recv_stderr_ready():
            if ch.recv_ready():
                print(ch.recv(65536).decode("utf-8", "replace"), end="", flush=True)
            if ch.recv_stderr_ready():
                print(ch.recv_stderr(65536).decode("utf-8", "replace"), end="", flush=True)
            time.sleep(.05)
        raise SystemExit(ch.recv_exit_status())
    finally:
        transport.close()


if __name__ == "__main__":
    main()
