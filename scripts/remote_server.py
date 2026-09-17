#!/usr/bin/env python3
"""Work on fin-skills on the remote server; the local checkout only syncs and pushes.

Status: written 2026-09-17, syntax-checked only; not yet exercised against a
live server. Run with --dry-run first.

Reuses the SSH/plink helpers and server config of claude-codex-bridge
(experiments/run_server.py and an ignored server.local.json). Nothing here stores
credentials: plink reads the password from the env var named in that config.

    python scripts/remote_server.py sync                 # ship all local refs, check out HEAD's branch
    python scripts/remote_server.py run "python -m pytest -q"
    python scripts/remote_server.py pull [BRANCH]        # bring server commits back as origin-less refs
    python scripts/remote_server.py sync --dry-run       # print commands, touch nothing

Environment:
    FIN_SKILLS_BRIDGE_ROOT   default D:/claude-codex-bridge
    FIN_SKILLS_SERVER_JSON   default <bridge>/experiments/server.local.json
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BRIDGE = Path(os.environ.get("FIN_SKILLS_BRIDGE_ROOT", "D:/claude-codex-bridge"))
sys.path.insert(0, str(BRIDGE / "experiments"))
from run_server import (  # noqa: E402
    container_to_host_path, password, printable_command, q, read_json,
    run_remote_script, validate_server,
)

REMOTE_NAME = "fin_skill"


def scp_base(cfg: dict) -> list[str]:
    if cfg.get("transport") == "plink":
        cmd = [str(cfg.get("pscp_path") or "pscp"), "-batch"]
        if cfg.get("hostkey"):
            cmd += ["-hostkey", str(cfg["hostkey"])]
        pw = password(cfg)
        if pw:
            cmd += ["-pw", pw]
        return cmd
    cmd = ["scp"]
    if cfg.get("ssh_port"):
        cmd += ["-P", str(cfg["ssh_port"])]
    if cfg.get("identity_file"):
        cmd += ["-i", str(cfg["identity_file"])]
    return cmd


def paths(cfg: dict) -> tuple[str, str, str]:
    root = str(cfg.get("container_workspace_mount") or "/workspace").rstrip("/")
    repo = f"{root}/{REMOTE_NAME}"
    host_repo = container_to_host_path(cfg, repo)
    host_root = str(cfg.get("host_workspace_mount") or "")
    if host_root and not host_repo.startswith(host_root.rstrip("/") + "/"):
        raise SystemExit(f"refusing path outside host_workspace_mount: {host_repo}")
    return repo, host_repo, f"{root}/{REMOTE_NAME}-venv"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def copy(cfg: dict, src: str, dst: str, dry: bool) -> None:
    cmd = [*scp_base(cfg), src, dst]
    if dry:
        print(printable_command(cmd))
        return
    subprocess.run(cmd, check=True)


def cmd_sync(cfg: dict, a: argparse.Namespace) -> None:
    repo, host_repo, venv = paths(cfg)
    branch = a.ref or git("rev-parse", "--abbrev-ref", "HEAD")
    sha = git("rev-parse", branch)
    bundle_c, bundle_h = f"{repo}.bundle", f"{host_repo}.bundle"
    script = "\n".join([
        "set -euo pipefail",
        f"repo={q(repo)}; bundle={q(bundle_c)}",
        'git config --global --add safe.directory "$repo" >/dev/null 2>&1 || true',
        'if [ ! -d "$repo/.git" ]; then git clone -q "$bundle" "$repo"; fi',
        'cd "$repo"',
        'git fetch -q "$bundle" "+refs/heads/*:refs/remotes/local/*"',
        'if [ -n "$(git status --porcelain)" ]; then echo "server checkout is dirty; commit or stash there first" >&2; exit 3; fi',
        f"git checkout -q -B {q(branch)} {q(sha)}",
        f"test \"$(git rev-parse HEAD)\" = {q(sha)}",
        f"[ -x {q(venv)}/bin/python ] || python3 -m venv {q(venv)}",
        f"{q(venv)}/bin/pip install -q -U pip >/dev/null",
        f"{q(venv)}/bin/pip install -q -e '.[stats,dev]' pytest pytest-cov jsonschema >/dev/null",
        f"echo fin_skill_sync_ok {branch} {sha[:7]}",
    ])
    with tempfile.TemporaryDirectory() as tmp:
        b = Path(tmp) / "fin_skill.bundle"
        if a.dry_run:
            print("git bundle create <tmp.bundle> --branches")
        else:
            subprocess.run(["git", "-C", str(REPO), "bundle", "create", str(b), "--branches"], check=True)
        copy(cfg, str(b), f"{cfg['ssh_target']}:{bundle_h}", a.dry_run)
        run_remote_script(cfg, script, dry_run=a.dry_run)


def cmd_run(cfg: dict, a: argparse.Namespace) -> None:
    repo, _, venv = paths(cfg)
    script = "\n".join([
        "set -eo pipefail", f"cd {q(repo)}", f"source {q(venv)}/bin/activate",
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1", a.command,
    ])
    run_remote_script(cfg, script, dry_run=a.dry_run)


def cmd_pull(cfg: dict, a: argparse.Namespace) -> None:
    repo, host_repo, _ = paths(cfg)
    out_c, out_h = f"{repo}.out.bundle", f"{host_repo}.out.bundle"
    ref = a.branch or "HEAD"
    run_remote_script(cfg, f"set -e\ncd {q(repo)}\nb=$(git rev-parse --abbrev-ref {q(ref)})\n"
                           f"git bundle create {q(out_c)} \"$b\"", dry_run=a.dry_run)
    with tempfile.TemporaryDirectory() as tmp:
        b = Path(tmp) / "out.bundle"
        copy(cfg, f"{cfg['ssh_target']}:{out_h}", str(b), a.dry_run)
        if a.dry_run:
            print("git fetch <out.bundle> '+refs/heads/*:refs/remotes/server/*'")
            return
        subprocess.run(["git", "-C", str(REPO), "fetch", str(b),
                        "+refs/heads/*:refs/remotes/server/*"], check=True)
    print("fetched into refs/remotes/server/*; review, then merge/push locally")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default=os.environ.get(
        "FIN_SKILLS_SERVER_JSON", str(BRIDGE / "experiments" / "server.local.json")))
    p.add_argument("--dry-run", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync"); s.add_argument("--ref")
    r = sub.add_parser("run"); r.add_argument("command")
    pl = sub.add_parser("pull"); pl.add_argument("branch", nargs="?")
    a = p.parse_args()
    cfg = read_json(Path(a.server))
    validate_server(cfg)
    {"sync": cmd_sync, "run": cmd_run, "pull": cmd_pull}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
