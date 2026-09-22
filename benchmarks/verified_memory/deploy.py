"""Stage an explicit source allowlist to a fresh RADFM directory and submit a CPU job."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import socket
import sys
import tarfile
import time

import paramiko

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.beacon_workspace import HOST, FINGERPRINT, BASE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--credential-file", type=Path, required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--experiment", choices=("v1", "fly-v2", "fly-v3", "architecture-audit"), default="v1")
    args = p.parse_args()
    remote = PurePosixPath(args.remote_root)
    if remote.parent != PurePosixPath(BASE) or not remote.name.startswith("fin-skills-memory-"):
        p.error("require a direct fin-skills-memory-* child of the RADFM user directory")
    if args.receipt.exists():
        p.error("local receipt already exists")
    files = list((ROOT / "fin_skills").rglob("*.py"))
    files += [f for f in (ROOT / "benchmarks/verified_memory").iterdir() if f.is_file()]
    files += [ROOT / "benchmarks/library_utility/evaluate.py",
              ROOT / "tests/test_verified_memory.py", ROOT / "scripts/beacon_workspace.py",
              ROOT / "benchmarks/data/ecb_fx.csv",
              ROOT / "benchmarks/data/ecb_fx.provenance.json"]
    if args.experiment in ("fly-v2", "fly-v3"):
        files.append(ROOT / "tests/test_fly_v2.py")
    if args.experiment == "fly-v3":
        files.append(ROOT / "tests/test_fly_v3.py")
    files = sorted(set(files))
    archive = io.BytesIO()
    hashes = {}
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for file in files:
            relative = file.relative_to(ROOT).as_posix()
            hashes[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
            tar.add(file, arcname=relative, recursive=False)
    blob = archive.getvalue()
    transport = paramiko.Transport(socket.create_connection((HOST, 22), timeout=15))
    transport.start_client(timeout=30)
    key = transport.get_remote_server_key()
    fp = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
    if key.get_name() != "ssh-ed25519" or fp != FINGERPRINT:
        transport.close()
        raise RuntimeError("host key mismatch")
    secret = args.credential_file.read_text(encoding="utf-8").splitlines()[0]
    transport.auth_password("wy891", secret, fallback=False)
    del secret
    receipt = {"remote_root": str(remote), "archive_sha256": hashlib.sha256(blob).hexdigest(),
               "source_hashes": hashes}
    try:
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            # Validate canonical parent before mkdir; never reuse a source tree in flight.
            if sftp.normalize(BASE) != BASE:
                raise RuntimeError("RADFM parent resolves to an unexpected path")
            sftp.mkdir(str(remote))
            for name in ("source", "tmp", "cache", "logs"):
                sftp.mkdir(str(remote / name))
            with sftp.open(str(remote / "source.tar.gz"), "wb") as stream:
                stream.write(blob)
            with sftp.open(str(remote / "deployment.json"), "w") as stream:
                stream.write(json.dumps(receipt, indent=2))
        finally:
            sftp.close()
        root = shlex.quote(str(remote))
        runner = {"v1": "beacon_run.sh", "fly-v2": "beacon_fly_v2.sh",
                  "fly-v3": "beacon_fly_v3.sh",
                  "architecture-audit": "beacon_architecture_audit.sh"}[args.experiment]
        command = (f"cd {root} && tar -xzf source.tar.gz -C source && "
            f"sbatch --parsable --account=angliece --partition=beacon --qos=medium "
            f"--job-name=fin-memory --cpus-per-task=2 --mem=4G --time=01:00:00 "
            f"--chdir={root} --output={root}/logs/slurm-%j.out "
            f"--error={root}/logs/slurm-%j.err "
            f"--export=ALL,FIN_MEMORY_ROOT={root} source/benchmarks/verified_memory/{runner}")
        ch = transport.open_session()
        ch.exec_command(command)
        stdout, stderr = [], []
        while not ch.exit_status_ready() or ch.recv_ready() or ch.recv_stderr_ready():
            if ch.recv_ready():
                stdout.append(ch.recv(65536).decode())
            if ch.recv_stderr_ready():
                stderr.append(ch.recv_stderr(65536).decode())
            time.sleep(0.05)
        receipt.update(stdout="".join(stdout).strip(), stderr="".join(stderr).strip(),
                       exit_code=ch.recv_exit_status())
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in receipt.items() if k != "source_hashes"}))
        if receipt["exit_code"]:
            raise SystemExit(receipt["exit_code"])
    finally:
        transport.close()


if __name__ == "__main__":
    main()
