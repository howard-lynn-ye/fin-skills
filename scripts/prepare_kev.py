"""Prepare an isolated, pinned Kev environment and public weights on Beacon."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import urllib.request

REVISION = "557598fced1dada75dfbf36ed144dce309ac6ceb"
MODELS = [
    ("jaredpalmer/kev-0.8b", "54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8"),
    ("jaredpalmer/kev-4b", "485ace8703592fcf405488b262449990824cfed1"),
    ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
    ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
]


def write(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2)


def run(root):
    if (not os.environ.get("SLURM_JOB_ID") or root.resolve() != root
            or root.parent != Path("/beacon-projects/radfm/wy891")):
        raise ValueError("requires canonical RADFM Slurm workspace")
    url = f"https://codeload.github.com/jaredpalmer/kev/tar.gz/{REVISION}"
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read(128 * 1024 * 1024 + 1)
    if len(data) > 128 * 1024 * 1024:
        raise ValueError("upstream source archive exceeds preparation limit")
    dest = root / "upstream"
    dest.mkdir()
    hashes = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe archive path")
            relative = PurePosixPath(*path.parts[1:])
            if not (str(relative).startswith("kev/") or str(relative) in
                    {"pyproject.toml", "README.md", "LICENSE"}):
                continue
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("selected archive member is not a regular file")
            blob = archive.extractfile(member).read()
            target = dest / str(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as out:
                out.write(blob)
            hashes[str(relative)] = hashlib.sha256(blob).hexdigest()
    write(root / "source-provenance.json", dict(revision=REVISION, sha256=hashes,
        archive_sha256=hashlib.sha256(data).hexdigest(), url=url))
    subprocess.run([sys.executable, "-m", "venv", str(root / "env")], check=True)
    py = str(root / "env/bin/python")
    with (root / "logs/install.log").open("x") as log:
        subprocess.run([py, "-m", "pip", "install", "--disable-pip-version-check",
            "--report", str(root / "install-report.json"), str(dest) + "[serve]",
            "torch==2.8.0", "transformers==5.17.0", "peft==0.21.0"],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    (root / "requirements-resolved.txt").write_bytes(subprocess.check_output([py, "-m", "pip", "freeze"]))
    subprocess.run([py, "-B", str(Path(__file__)), str(root), "--download"], check=True)
    return dict(status="prepared_not_inferred", code_revision=REVISION)


def download(root):
    from huggingface_hub import snapshot_download
    from kev.checkpoint import Checkpoint, LoadOptions
    rows = []
    for repo, revision in MODELS:
        path = snapshot_download(repo, revision=revision, token=False,
            cache_dir=str(root / "cache/hf/hub"),
            allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.model", "LICENSE*", "README.md"])
        rows.append(dict(repo=repo, revision=revision, path=path))
    write(root / "snapshots.json", dict(models=rows, inference=False))


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    if "--download" in sys.argv:
        download(directory)
    else:
        try:
            result = run(directory)
        except Exception as exc:
            write(directory / "completion.json", dict(status="failed", error_type=type(exc).__name__, error=str(exc)))
            raise
        write(directory / "completion.json", result)
