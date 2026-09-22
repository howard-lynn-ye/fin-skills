"""Adversarial regression cases for real execution, stale evidence and rejection."""
import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.agent_study.open_agent_runner import AgentWorkspaceSession, parse_action, run_agent
from benchmarks.agent_study.submission_audit import accounting, same_session_probe


@pytest.fixture
def workspace(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    dates = pd.bdate_range("2021-01-01", periods=80)
    rng = np.random.default_rng(87)
    prices = pd.DataFrame(100 * np.exp(rng.normal(0, .01, (80, 3)).cumsum(0)),
                          index=dates, columns=["A", "B", "C"])
    for name in ("close_quoted.csv", "llm_score.csv", "volume.csv"):
        prices.to_csv(data / name)
    pd.DataFrame({"feed_ts": dates, "ticker": "A", "score": rng.normal(size=80)}).to_csv(
        data / "news_feed.csv", index=False)
    pd.DataFrame(columns=["date", "ticker", "ratio"]).to_csv(data / "corporate_actions.csv", index=False)
    pd.DataFrame({"ticker": ["A", "B", "C"], "listing_date": ["2020-01-01"] * 3,
                  "delisting_date": [None] * 3}).to_csv(data / "listings.csv", index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({"eval_start": str(dates[30].date()),
                                                       "eval_end": str(dates[-1].date())}))
    (tmp_path / "TASK.md").write_text("test task")
    (tmp_path / "submission.py").write_text('''import pandas as pd
def build_positions(data_dir):
    p = pd.read_csv(data_dir + '/close_quoted.csv', index_col=0, parse_dates=True)
    x = p.pct_change(fill_method=None).shift(1).fillna(0)
    return x.div(x.abs().sum(axis=1).clip(lower=1), axis=0)
''')
    (tmp_path / "report.json").write_text(json.dumps({"reported_sharpe": 0,
                                                    "cost_bps_per_side": 10, "guards_cited": []}))
    return tmp_path


def test_nonexistent_and_disabled_guards_never_get_receipts(workspace):
    s = AgentWorkspaceSession(workspace, "skills_optional_guards")
    assert s.run_guard("invented")["status"] == "ERROR" and not s.receipts
    s = AgentWorkspaceSession(workspace, "skills_text_only")
    assert s.run_guard("assert_causal")["status"] == "BLOCKED" and not s.receipts


def test_real_guard_fails_leaky_submission_and_invalidates_old_receipt(workspace):
    s = AgentWorkspaceSession(workspace, "skills_optional_guards")
    assert s.run_guard("assert_causal")["passed"] is True
    report = workspace / "report.json"
    obj = json.loads(report.read_text())
    obj["guards_cited"] = ["assert_causal", "invented"]
    report.write_text(json.dumps(obj))
    fidelity = s.audit_compliance_fidelity()
    assert fidelity["fresh_guards"] == ["assert_causal"]
    assert fidelity["hallucinated_citations"] == ["invented"]
    source = workspace / "submission.py"
    source.write_text(source.read_text().replace("shift(1)", "shift(-1)"))
    assert s.audit_compliance_fidelity()["stale_citations"] == ["assert_causal"]
    assert s.run_guard("assert_causal")["passed"] is False


def test_same_session_probe_discriminates(workspace):
    assert same_session_probe(workspace)["passed"] is True
    source = workspace / "submission.py"
    source.write_text(source.read_text().replace("shift(1)", "shift(0)"))
    assert same_session_probe(workspace)["passed"] is False


def test_accounting_recomputes_and_rejects_false_report(workspace):
    value = accounting(workspace)
    obj = json.loads((workspace / "report.json").read_text())
    obj["reported_sharpe"] = value["recomputed_at_stated_cost"]
    (workspace / "report.json").write_text(json.dumps(obj))
    assert accounting(workspace)["passed"] is True
    obj["reported_sharpe"] += 5
    (workspace / "report.json").write_text(json.dumps(obj))
    assert accounting(workspace)["passed"] is False


def test_missing_and_error_are_not_acceptance(workspace):
    s = AgentWorkspaceSession(workspace, "skills_enforced_guards")
    (workspace / "submission.py").write_text("raise RuntimeError('broken')")
    result = s.finish()
    assert not result["accepted"] and result["status"] == "INCOMPLETE"


def test_transport_turn_limit_and_provider_failure_retained(workspace):
    s = AgentWorkspaceSession(workspace, "no_library")
    value = run_agent(s, lambda _: {"choices": [{"message": {"content": "bad JSON"}}]}, max_turns=2)
    assert value["termination"] == "TURN_LIMIT" and value["turns"] == 2
    def failing(_):
        raise ConnectionError("unavailable")
    value = run_agent(s, failing)
    assert value["termination"] == "PROVIDER_ERROR" and not value["accepted"]


def test_tools_cannot_read_parent_or_write_inputs(workspace):
    s = AgentWorkspaceSession(workspace, "no_library")
    with pytest.raises(ValueError):
        s.dispatch("read_file", {"path": "../private.json"})
    with pytest.raises(ValueError):
        s.dispatch("write_file", {"path": "data/close_quoted.csv", "content": "bad"})


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Reading the task.\n```json\n{}\n```"])
def test_provider_json_wrappers_execute_the_same_call(workspace, wrapper):
    action = {"tool": "read_file", "arguments": {"path": "TASK.md"}}
    content = wrapper.format(json.dumps(action))
    assert parse_action(content) == action
    session = AgentWorkspaceSession(workspace, "no_library")
    result = run_agent(session, lambda _: {"choices": [{"message": {"content": content}}]}, max_turns=1)
    assert result["transcript"][0]["feedback"]["text"] == "test task"


def test_ambiguous_or_broken_calls_are_not_repaired():
    with pytest.raises(ValueError, match="multiple"):
        parse_action('```json\n{}\n```\n```json\n{}\n```')
    with pytest.raises(ValueError, match="no tool ran"):
        parse_action('```json\n{"tool": missing_quote}\n```')
