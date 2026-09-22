"""Upload an immutable adapter snapshot and submit paper-only jobs to RADFM.

Uses the project's existing pinned-host transport constants. A credential file is
read locally for Beacon authentication and is never archived or printed.
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
import sys
import tarfile

import paramiko

HOST = "beaconsub01.ihc.umd.edu"
FINGERPRINT = "SHA256:F0/SPeAvRCv/rg28Tx5Z+uKmtcU5DSwW8T5iwbQzKmo"
BASE = PurePosixPath("/beacon-projects/radfm/wy891")
ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--credential-file", type=Path, required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    remote = PurePosixPath(args.remote_root)
    if remote.parent != BASE or not remote.name.startswith("fin-skills-fly-reuse-"):
        p.error("Require a direct RADFM fin-skills-fly-reuse-* directory")
    if args.receipt.exists():
        p.error("Receipt exists; inspect submitted jobs before retrying")
    blob = io.BytesIO()
    paths = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).parent.glob("*.sh"))
    paths += [ROOT/n for n in subprocess.check_output(
        ["git", "ls-files", "fin_skills"], cwd=ROOT, text=True).splitlines() if n.endswith(".py")]
    hashes = {str(f.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in paths}
    with tarfile.open(fileobj=blob, mode="w:gz") as tar:
        for f in paths:
            tar.add(f, arcname="source-v1/"+f.relative_to(ROOT).as_posix(), recursive=False)
    t = paramiko.Transport(socket.create_connection((HOST, 22), timeout=15))
    t.start_client(timeout=30)
    key = t.get_remote_server_key()
    fp = "SHA256:"+base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
    if key.get_name() != "ssh-ed25519" or fp != FINGERPRINT:
        raise RuntimeError("Unexpected Beacon host key")
    t.auth_password("wy891", args.credential_file.read_text().splitlines()[0], fallback=False)
    receipt = {"remote_root": str(remote), "source_hashes": hashes, "jobs": {}}
    def save():
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
    def command(cmd):
        c = t.open_session()
        c.exec_command(cmd)
        out, err = c.makefile().read().decode(), c.makefile_stderr().read().decode()
        code = c.recv_exit_status()
        if code:
            raise RuntimeError(f"Remote command failed: {err} {out}")
        return out.strip()
    try:
        s = paramiko.SFTPClient.from_transport(t)
        if s.normalize(str(remote)) != str(remote):
            raise RuntimeError("Unexpected canonical directory")
        s.stat(str(remote/"PREPARED"))
        s.mkdir(str(remote/"source-v1"))  # Immutable: fails instead of overwriting.
        with s.open(str(remote/"source-v1.tar.gz"), "wb") as f:
            f.write(blob.getvalue())
        s.close()
        command(f"cd {shlex.quote(str(remote))} && tar -xzf source-v1.tar.gz")
        save()
        for mode in ("compact", "full"):
            dependency = "" if mode == "compact" else f" --dependency=afterok:{receipt['jobs']['compact']}"
            cmd = (f"sbatch --parsable --account=angliece --partition=scavenger --qos=scavenger "
                   f"--job-name=fly-reuse-{mode} --cpus-per-task=2 --mem=8G --time=02:00:00 "
                   f"--chdir={remote} --output={remote}/logs/{mode}-%j.out "
                   f"--error={remote}/logs/{mode}-%j.err{dependency} "
                   f"--export=ALL,FLY_REUSE_ROOT={remote},FLY_REUSE_SOURCE={remote}/source-v1 "
                   f"{remote}/source-v1/benchmarks/fly_reuse/beacon_run.sh {mode}")
            receipt["jobs"][mode] = command(cmd)
            save()
        print(json.dumps({k:v for k,v in receipt.items() if k != "source_hashes"}))
    finally:
        t.close()


if __name__ == "__main__":
    main()
