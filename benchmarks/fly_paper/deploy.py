"""Stage only this reproduction and pinned upstream files to RADFM.

The credential is read locally for the authorized Beacon login, never uploaded.
"""
import argparse
import base64
import hashlib
import io
import json
import re
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "port", "bridge", "control", "market"))
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stage-name", help="fresh directory for a retry, preserving prior output")
    parser.add_argument("--reproduction-stage", default="port")
    args = parser.parse_args()
    stage = args.stage_name or args.phase
    if not re.fullmatch(r"[a-z][a-z0-9-]*",stage) or not re.fullmatch(r"[a-z][a-z0-9-]*",args.reproduction_stage):
        parser.error("invalid stage name")
    remote = PurePosixPath(args.remote_root)
    if remote.parent != PurePosixPath(BASE) or not remote.name.startswith("fin-skills-memory-paper-"):
        parser.error("require a fresh direct RADFM paper-reproduction directory")
    if args.receipt.exists():
        parser.error("receipt already exists")
    transport = paramiko.Transport(socket.create_connection((HOST,22),timeout=15))
    transport.start_client(timeout=30)
    key = transport.get_remote_server_key()
    fp = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
    if key.get_name() != "ssh-ed25519" or fp != FINGERPRINT:
        transport.close()
        raise RuntimeError("host key mismatch")
    secret = args.credential_file.read_text(encoding="utf-8").splitlines()[0]
    transport.auth_password("wy891",secret,fallback=False)
    del secret
    receipt = {"phase":args.phase,"stage":stage,"remote_root":str(remote),
               "reproduction_stage":args.reproduction_stage}
    try:
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            if sftp.normalize(BASE) != BASE:
                raise RuntimeError("unexpected canonical RADFM parent")
            if args.phase == "prepare":
                sftp.mkdir(str(remote))
                for name in ("source","upstream","tmp","cache","logs"):
                    sftp.mkdir(str(remote/name))
                sftp.put(str(Path(__file__).with_name("beacon_prepare.sh")),str(remote/"prepare.sh"))
                runner = "prepare.sh"
            elif args.phase in ("bridge", "port", "control", "market"):
                if sftp.normalize(str(remote)) != str(remote):
                    raise RuntimeError("unexpected experiment path")
                if args.phase in ("bridge", "control", "market"):
                    prerequisite = "bridge" if args.phase in ("control", "market") else args.reproduction_stage
                    with sftp.open(str(remote/prerequisite/"results/results.json")) as stream:
                        reproduction = json.load(stream)
                    if not reproduction["all_checks_passed"]:
                        raise RuntimeError("Native reproduction gate failed")
                else:
                    sftp.stat(str(remote/"native/native_complete.mat"))
                prefix = stage
                sftp.mkdir(str(remote/prefix))
                files = [(p,f"{prefix}/source/benchmarks/fly_paper/"+p.name)
                         for p in Path(__file__).parent.iterdir() if p.is_file()]
                files += [(p,f"{prefix}/source/"+p.relative_to(ROOT).as_posix())
                          for p in (ROOT/"fin_skills").rglob("*.py")]
                files += [(ROOT/"benchmarks/verified_memory/episode_credit.py",
                           f"{prefix}/source/benchmarks/verified_memory/episode_credit.py")]
                if args.phase == "market":
                    extras = ["benchmarks/verified_memory/fly_v2.py",
                              "benchmarks/verified_memory/fly_v3.py",
                              "benchmarks/verified_memory/model.py",
                              "benchmarks/verified_memory/run.py",
                              "benchmarks/library_utility/evaluate.py",
                              "benchmarks/data/ecb_fx.csv",
                              "benchmarks/data/ecb_fx.provenance.json"]
                    files += [(ROOT/name, f"{prefix}/source/{name}") for name in extras]
                blob = io.BytesIO(); hashes = {}
                with tarfile.open(fileobj=blob,mode="w:gz") as tar:
                    for path,name in files:
                        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
                        tar.add(path,arcname=name,recursive=False)
                receipt["source_hashes"]=hashes
                with sftp.open(str(remote/prefix/"source.tar.gz"),"wb") as stream:
                    stream.write(blob.getvalue())
                with sftp.open(str(remote/prefix/"deployment.json"),"w") as stream:
                    stream.write(json.dumps(receipt,indent=2))
                runner=f"{prefix}/source/benchmarks/fly_paper/beacon_{args.phase}.sh"
            else:
                if sftp.normalize(str(remote)) != str(remote):
                    raise RuntimeError("unexpected experiment path")
                sftp.stat(str(remote/"octave.sha256"))
                # Native and results must not exist, so uncertain retries cannot duplicate work.
                for name in ("native","results","source.tar.gz"):
                    try:
                        sftp.stat(str(remote/name))
                    except FileNotFoundError:
                        pass
                    else:
                        raise RuntimeError(f"refusing existing {name}")
                files = [(p,"source/benchmarks/fly_paper/"+p.name)
                         for p in Path(__file__).parent.iterdir()
                         if p.is_file() and p.suffix in (".py",".m",".sh",".md",".json",".txt")]
                files += [(p,"upstream/"+p.relative_to(args.upstream).as_posix())
                          for p in args.upstream.rglob("*") if p.is_file()
                          and ".git" not in p.relative_to(args.upstream).parts
                          and (p.suffix in (".m",".mat",".xlsx") or p.name=="README.md")]
                files += [(ROOT/"scripts/beacon_workspace.py","source/scripts/beacon_workspace.py")]
                blob = io.BytesIO()
                hashes = {}
                with tarfile.open(fileobj=blob,mode="w:gz") as tar:
                    for path,name in files:
                        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                        tar.add(path,arcname=name,recursive=False)
                receipt["source_hashes"] = hashes
                with sftp.open(str(remote/"source.tar.gz"),"wb") as stream:
                    stream.write(blob.getvalue())
                with sftp.open(str(remote/"protocol.json"),"w") as stream:
                    stream.write(json.dumps(receipt,indent=2))
                runner = "source/benchmarks/fly_paper/beacon_run.sh"
        finally:
            sftp.close()
        root = shlex.quote(str(remote))
        extract = ("" if args.phase=="prepare" else "tar -xzf source.tar.gz && "
                   if args.phase=="run" else f"tar -xzf {stage}/source.tar.gz && ")
        command = (f"cd {root} && {extract}sbatch --parsable --account=angliece "
            f"--partition=beacon --qos=medium --job-name=fly-paper-{args.phase} "
            f"--cpus-per-task=2 --mem=8G --time=01:00:00 --chdir={root} "
            f"--output={root}/logs/{args.phase}-%j.out --error={root}/logs/{args.phase}-%j.err "
            f"--export=ALL,FLY_PAPER_ROOT={root},FLY_PAPER_STAGE={stage},"
            f"FLY_REPRODUCTION_STAGE={args.reproduction_stage} {runner}")
        ch = transport.open_session(); ch.exec_command(command)
        out,err = [],[]
        while not ch.exit_status_ready() or ch.recv_ready() or ch.recv_stderr_ready():
            if ch.recv_ready(): out.append(ch.recv(65536).decode())
            if ch.recv_stderr_ready(): err.append(ch.recv_stderr(65536).decode())
            time.sleep(.05)
        receipt.update(stdout="".join(out).strip(),stderr="".join(err).strip(),
                       exit_code=ch.recv_exit_status())
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(receipt,indent=2),encoding="utf8")
        print(json.dumps({k:v for k,v in receipt.items() if k!="source_hashes"}))
        if receipt["exit_code"]: raise SystemExit(receipt["exit_code"])
    finally:
        transport.close()


if __name__ == "__main__":
    main()
