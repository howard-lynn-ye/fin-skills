"""Check bundle provenance and reject accidental HOME or noncanonical destinations."""
import hashlib
import json
import tarfile

import pytest

from benchmarks.agent_study.prepare_followup import prepare, validate_remote_root


@pytest.mark.parametrize("root", [
    "/home/wy891/fin-skills-followup-test", "/beacon-projects/radfm/wy891",
    "/beacon-projects/radfm/wy891/fin-skills-followup-test/../elsewhere",
    "/beacon-projects/radfm/wy891/fin-skills-followup-test/nested",
    "/beacon-projects/radfm/wy891/fin-skills-followup-test/",
    "/beacon-projects/radfm/wy891/fin-skills-followup-$(whoami)",
])
def test_refuse_unsafe_experiment_destinations(root):
    with pytest.raises(ValueError, match="RADFM"):
        validate_remote_root(root)


def test_offline_bundle_hashes_exact_source_and_cannot_overwrite(tmp_path):
    output = tmp_path / "bundle"
    remote = "/beacon-projects/radfm/wy891/fin-skills-followup-test"
    receipt = prepare(output, remote)
    assert receipt["status"] == "prepared_not_submitted"
    manifest = json.loads((output / "bundle-manifest.json").read_text())
    with tarfile.open(output / "source.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            assert member.isfile() and member.name.startswith("source/")
            name = member.name.removeprefix("source/")
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == (
                manifest["source_sha256"][name])
            assert ".git-credentials" not in name and not name.startswith("runs/")
    assert manifest["archive_sha256"] == receipt["archive_sha256"]
    with pytest.raises(FileExistsError):
        prepare(output, remote)
