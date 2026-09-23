"""Verify the prior archive-based deployment without requiring a .git directory.

Archive digest was measured on Beacon on 2026-09-23 before the paired follow-up.
This establishes reuse of that deployment, not a fresh independent upstream checkout.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ARCHIVE_SHA256 = "b147e0494f1b792b5b83e065d60d219ccb7cdb3703181a815b74dad7ac6de880"


def verify_archive(root, expected_sha256):
    root = Path(root).resolve()
    archive = root / "upstream.tar.gz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("prior upstream archive digest differs")
    files = {}
    with tarfile.open(archive) as stream:
        for member in stream:
            if member.isdir():
                continue
            rel = PurePosixPath(member.name)
            if (not member.isfile() or rel.is_absolute() or ".." in rel.parts
                    or not member.name.startswith("upstream/") or member.name in files):
                raise ValueError("unsafe or duplicate archive entry")
            path = root / member.name
            if not path.is_file() or path.resolve() != path:
                raise ValueError("missing or redirected upstream file")
            expected = hashlib.sha256(stream.extractfile(member).read()).hexdigest()
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"upstream file differs from archive: {member.name}")
            files[member.name] = expected
    actual_python = {p.relative_to(root).as_posix() for p in (root / "upstream").rglob("*.py")}
    if actual_python != {p for p in files if p.endswith(".py")}:
        raise ValueError("unlisted Python source in upstream deployment")
    return {"archive_sha256": expected_sha256, "files_verified": len(files), "source_sha256": files}


def verify_prior(root):
    root = Path(root)
    result = verify_archive(root, ARCHIVE_SHA256)
    prior = json.loads((root / "results-compact-v1/protocol.json").read_text())
    from benchmarks.fly_reuse.run import PINS
    if prior["pins"] != PINS:
        raise ValueError("prior protocol identifies different upstream revisions")
    actual = hashlib.sha256((root / "data/kraken-daily.json").read_bytes()).hexdigest()
    if actual != prior["data_sha256"]:
        raise ValueError("market snapshot differs from original comparison")
    result.update(data_sha256=actual, recorded_pins=prior["pins"],
        interpretation="Byte-for-byte prior deployment and market reuse; commit IDs are prior protocol provenance.")
    return result
