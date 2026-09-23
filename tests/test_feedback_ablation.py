import json

import pytest

from benchmarks.agent_study.feedback_ablation import RepairSession
from benchmarks.agent_study.open_agent_runner import PUBLIC_CHECKS


@pytest.mark.parametrize("arm,checks", [
    ("execution_only", ("execution",)),
    ("accounting_feedback", ("accounting",)),
    ("guard_feedback", PUBLIC_CHECKS),
    ("guard_feedback_gate", PUBLIC_CHECKS),
])
def test_feedback_is_separated_from_enforcement(tmp_path, monkeypatch, arm, checks):
    (tmp_path / "data").mkdir()
    (tmp_path / "submission.py").write_text("# starter")
    (tmp_path / "report.json").write_text(json.dumps(dict(
        reported_sharpe=1, cost_bps_per_side=5, guards_cited=[])))
    calls = []

    def worker(self, action):
        calls.append(action)
        return dict(status="EXECUTED", passed=False)

    monkeypatch.setattr(RepairSession, "_worker", worker)
    session = RepairSession(tmp_path, arm)
    assert calls == list(checks)
    assert tuple(session.initial_feedback) == checks
    assert "oracle" not in session.system_prompt()
    calls.clear()
    result = session.finish()
    assert result["accepted"] == (arm != "guard_feedback_gate")
    assert calls == (list(checks) if arm == "guard_feedback_gate" else [])
    assert session.dispatch("run_guard", {"name": "assert_causal"})["status"] == "BLOCKED"


def test_missing_artifacts_cannot_pass_final_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(RepairSession, "_worker", lambda *_: dict(passed=True))
    session = RepairSession(tmp_path, "guard_feedback_gate")
    assert session.finish()["accepted"] is False


def test_starter_is_executable_but_detectably_wrong(tmp_path):
    from benchmarks.agent_study.build_task import export
    from benchmarks.agent_study.feedback_ablation import STARTER
    from benchmarks.agent_study.submission_audit import positions, same_session_probe
    export(11, tmp_path)
    (tmp_path / "submission.py").write_text(STARTER)
    assert not positions(tmp_path).empty
    assert same_session_probe(tmp_path)["passed"] is False
