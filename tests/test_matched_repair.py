import json

import pytest

from benchmarks.agent_study.matched_repair import MatchedSession
from benchmarks.agent_study.open_agent_runner import PUBLIC_CHECKS


def workspace(path):
    (path / "data").mkdir()
    for name, content in {"submission.py": "# initial", "TASK.md": "shared task",
                          "manifest.json": "{}", "report.json": json.dumps(dict(
                              reported_sharpe=1, cost_bps_per_side=10, guards_cited=[]))}.items():
        (path / name).write_text(content)


@pytest.mark.parametrize("arm,checks", [
    ("accounting_only", ("execution", "accounting")),
    ("self_review", ("execution", "accounting")),
    ("generic_checks", ("execution", "accounting", "generic")),
    ("domain_feedback", ("execution", "accounting", "assert_causal", "survivorship_audit", "same_session_probe")),
    ("domain_gate", ("execution", "accounting", "assert_causal", "survivorship_audit", "same_session_probe")),
])
def test_conditions_receive_only_assigned_feedback(tmp_path, monkeypatch, arm, checks):
    workspace(tmp_path)
    calls = []
    def worker(self, action):
        calls.append(action)
        return dict(status="EXECUTED", passed=False)
    monkeypatch.setattr(MatchedSession, "_worker", worker)
    session = MatchedSession(tmp_path, arm)
    assert tuple(calls) == checks
    assert "shared task" in session.system_prompt()
    assert session.dispatch("run_guard", dict(name="assert_causal"))["status"] == "BLOCKED"
    if arm == "self_review":
        assert session.finish()["status"] == "REVIEW_REQUIRED"
        session.dispatch("review_submission", dict(review="The report needs independent recomputation."))
    calls.clear()
    assert session.finish()["accepted"] is (arm != "domain_gate")
    assert calls == (list(checks) if arm == "domain_gate" else [])


def test_review_must_follow_latest_artifacts_and_costs_fixed(tmp_path, monkeypatch):
    workspace(tmp_path)
    monkeypatch.setattr(MatchedSession, "_worker", lambda *_: dict(passed=True))
    session = MatchedSession(tmp_path, "self_review")
    session.dispatch("review_submission", dict(review="reviewed"))
    assert session.finish()["accepted"]
    session.dispatch("write_artifacts", dict(source="# changed", report=dict(
        reported_sharpe=2, cost_bps_per_side=10, guards_cited=[])))
    assert session.finish()["status"] == "REVIEW_REQUIRED"
    session.dispatch("review_submission", dict(review="reviewed changed code"))
    assert session.finish()["accepted"]
    session.dispatch("write_file", dict(path="report.json", content=json.dumps(dict(
        reported_sharpe=2, cost_bps_per_side=5, guards_cited=[]))))
    assert session.finish()["status"] == "COST_PROTOCOL"


def test_generic_checks_pass_causal_and_leaking_outputs(tmp_path):
    from benchmarks.agent_study.build_task import export
    from benchmarks.agent_study.audit_validity import source
    from benchmarks.agent_study.submission_audit import generic_checks
    export(11, tmp_path)
    for case in ("clean_news", "same_session"):
        (tmp_path / "submission.py").write_text(source(case))
        assert generic_checks(tmp_path)["passed"]
    (tmp_path / "submission.py").write_text(source("zero_positions"))
    assert generic_checks(tmp_path)["passed"] is False
