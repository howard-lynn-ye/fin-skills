"""Submission-path boundaries and interrupted sbatch attempts; no Slurm access."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import submit_model_followup as submit

ROOT = "/beacon-projects/radfm/wy891/fin-skills-rag-jev-test"


@pytest.mark.parametrize("root", ["/home/wy891/run", ROOT + "/../escape", ROOT + "/nested",
                                  ROOT + "/", ROOT.replace("/radfm/", "/other/")])
def test_output_escape_is_refused(root):
    with pytest.raises(ValueError, match="RADFM"):
        submit.submission_command("rag-jev", root)


def test_plan_stays_offline_and_sets_scheduler_paths(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not execute sbatch")
    monkeypatch.setattr(submit.subprocess, "run", forbidden)
    plan = submit.execute("rag-jev", ROOT)
    assert plan["status"] == "prepared_not_submitted"
    for option in ("chdir", "output", "error"):
        value = next(arg.split("=", 1)[1] for arg in plan["command"] if arg.startswith(f"--{option}="))
        assert value.startswith(ROOT)
    assert not any("gres=gpu" in arg for arg in plan["command"])


def test_fly_requires_distinct_input_snapshot():
    root = "/beacon-projects/radfm/wy891/fin-skills-fly-test"
    with pytest.raises(ValueError, match="differ"):
        submit.submission_command("fly-gate", root, root)
    with pytest.raises(ValueError, match="RADFM"):
        submit.submission_command("fly-gate", root, "/home/wy891/prior")


def test_submission_is_recorded_before_call_and_cannot_repeat(tmp_path, monkeypatch):
    calls = []
    def sbatch(command, **kwargs):
        calls.append(command)
        assert (tmp_path / "submission-started.json").is_file()
        return SimpleNamespace(returncode=0, stdout="123456;beacon\n")
    monkeypatch.setattr(submit.subprocess, "run", sbatch)
    result = submit._dispatch(tmp_path, ["sbatch", "test-script"])
    assert result["status"] == "submitted" and result["job_id"] == "123456"
    with pytest.raises(FileExistsError):
        submit._dispatch(tmp_path, ["sbatch", "test-script"])
    assert len(calls) == 1


def test_timeout_is_unknown_and_persists_no_secret_output(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("sbatch", 60, output="sensitive response")
    monkeypatch.setattr(submit.subprocess, "run", timeout)
    receipt = submit._dispatch(tmp_path, ["sbatch"])
    assert receipt["status"] == "submission_unknown"
    saved = (tmp_path / "submission-receipt.json").read_text()
    assert "sensitive response" not in saved
    assert json.loads(saved)["error_type"] == "TimeoutExpired"
