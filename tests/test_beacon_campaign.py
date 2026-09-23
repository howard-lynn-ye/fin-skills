"""Campaign path, immutable payload and independent scheduling checks; no remote access."""
import json

import pytest

from scripts.beacon_campaign import jobs, prepare, scheduler_command, validate_bundle, validate_root

ROOT = "/beacon-projects/radfm/wy891/fin-skills-campaign-test"


@pytest.mark.parametrize("value", ["/home/wy891/run", ROOT + "/../other", ROOT + "/",
                                  ROOT + "/nested", ROOT + ";echo secret"])
def test_root_rejects_escape(value):
    with pytest.raises(ValueError):
        validate_root(value)


def test_jobs_use_radfm_without_pinning_a_node():
    for job in jobs() + jobs("pipeline"):
        command = scheduler_command({"remote_root": ROOT}, job)
        assert not any(x.startswith(("--nodelist", "--exclude", "--reservation")) for x in command)
        assert ("--gres=gpu:1" in command) is job["gpu"]
        for flag in ("--chdir=", "--output=", "--error="):
            assert next(x for x in command if x.startswith(flag)).startswith(flag + ROOT + "/jobs/")


def test_pipeline_batch_does_not_resubmit_initial_studies():
    batch = jobs("pipeline")
    assert len(batch) == 3
    assert all(j["study"] == "pipeline" for j in batch)
    assert {j["seed"] for j in batch} == {11, 23, 37}


def test_campaign_freezes_exact_sources_and_records_unimplemented_studies(tmp_path):
    bundle = tmp_path / "bundle"
    prepare(bundle, ROOT)
    plan = validate_bundle(bundle)
    assert len([j for j in plan["jobs"] if j["study"] == "agent"]) == 3
    assert any(x["id"] == "pipeline-efficiency" for x in plan["pending_studies"])
    assert all(not name.startswith("runs/") and "credential" not in name
               for name in plan["source_sha256"])
    with pytest.raises(FileExistsError):
        prepare(bundle, ROOT)
    manifest = bundle / "campaign.json"
    value = json.loads(manifest.read_text())
    value["archive_sha256"] = "tampered"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash"):
        validate_bundle(bundle)
