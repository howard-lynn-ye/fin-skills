"""Deployment refuses corrupt or unsafe archives before any authentication."""
import hashlib
import io
import json
import tarfile

import pytest

from scripts.deploy_beacon_followup import submission_command, validate_bundle


ROOT = "/beacon-projects/radfm/wy891/fin-skills-followup-test"
ENTRY = "benchmarks/agent_study/beacon_followup.sh"


def bundle(tmp_path, members):
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    manifest = {"remote_root": ROOT, "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_sha256": {ENTRY: hashlib.sha256(b"ok").hexdigest()}, "planned_cells": 12}
    (tmp_path / "bundle-manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "submit.sh").write_text("#!/bin/sh\n")
    return tmp_path


def test_validate_exact_bundle_without_network(tmp_path):
    path = bundle(tmp_path, [("source/" + ENTRY, b"ok")])
    manifest, payload = validate_bundle(path)
    assert manifest["remote_root"] == ROOT
    assert set(payload) == {"source.tar.gz", "bundle-manifest.json", "submit.sh"}


@pytest.mark.parametrize("members", [
    [("source/../outside", b"ok")],
    [("/outside", b"ok")],
    [("source/" + ENTRY, b"ok"), ("source/" + ENTRY, b"ok")],
    [("source/" + ENTRY, b"changed")],
    [],
])
def test_refuse_unsafe_or_incomplete_archive(tmp_path, members):
    with pytest.raises(ValueError):
        validate_bundle(bundle(tmp_path, members))


def test_refuse_modified_archive(tmp_path):
    path = bundle(tmp_path, [("source/" + ENTRY, b"ok")])
    with (path / "source.tar.gz").open("ab") as out:
        out.write(b"changed after freezing")
    with pytest.raises(ValueError, match="archive hash"):
        validate_bundle(path)


def test_scheduler_paths_are_absolute_and_home_is_rejected():
    command = submission_command(ROOT)
    for option, path in (("chdir", ROOT), ("output", ROOT + "/logs/slurm-%j.out"),
                         ("error", ROOT + "/logs/slurm-%j.err")):
        assert f"--{option}={path}" in command
    assert "test ! -e submission-job-id.txt" in command
    with pytest.raises(ValueError, match="RADFM"):
        submission_command("/home/wy891/fin-skills-followup-test")
