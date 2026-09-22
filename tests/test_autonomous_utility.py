"""Behavioral checks for optional, sequential, past-only model-selected tool use."""
import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.library_utility.agent_tools import catalog_rows, execute
from benchmarks.library_utility.autonomous import one_decision, parse_message
from benchmarks.library_utility.run import context


@pytest.fixture
def prices():
    return pd.DataFrame(np.exp(np.random.default_rng(19).normal(0, .01, (180, 4)).cumsum(0)),
        index=pd.date_range("2024-01-01", periods=180, tz="UTC"),
        columns=[f"asset_{i}" for i in range(4)])


class ScriptedChat:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.seen = []

    def __call__(self, messages):
        self.seen.append(json.loads(json.dumps(messages)))
        return {"choices": [{"message": {"content": json.dumps(next(self.actions))}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


FINAL = {"weights": [.25] * 4, "reason": "diversify"}


def test_json_fences_work_for_tools_reflection_and_final_but_prose_does_not(prices, tmp_path):
    class FencedChat(ScriptedChat):
        def __call__(self, messages):
            response = super().__call__(messages)
            message = response["choices"][0]["message"]
            message["content"] = "```json\n" + message["content"] + "\n```"
            return response
    chat = FencedChat([{"analysis": "inspect available algorithms"},
                       {"tool": "list_algorithms", "arguments": {}}, FINAL])
    rec = one_decision(chat, prices, 150, "autonomous_library", 11, tmp_path, [0.] * 4, [])
    assert rec["status"] == "valid" and rec["usage"]["tool_successes"] == 1
    assert rec["usage"]["format_errors"] == 0
    with pytest.raises(ValueError):
        parse_message('Here is the result: {"analysis":"x"}')
    with pytest.raises(ValueError):
        parse_message('```json\n{"analysis":"x"}\n``` trailing prose')


def test_model_can_finalize_without_any_tool_execution(prices, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a tool executed without a request")
    monkeypatch.setattr("benchmarks.library_utility.autonomous.execute", forbidden)
    rec = one_decision(ScriptedChat([FINAL]), prices, 150, "autonomous_library", 11,
                       tmp_path, [0.] * 4, [])
    assert rec["status"] == "valid" and rec["usage"]["tool_requests"] == 0
    assert rec["usage"]["model_calls"] == 1


def test_model_selects_call_then_reads_actual_result_before_final(prices, tmp_path):
    chat = ScriptedChat([{"tool": "list_algorithms", "arguments": {}},
        {"tool": "run_algorithm", "arguments": {"algorithm": "inverse_volatility"}}, FINAL])
    rec = one_decision(chat, prices, 150, "autonomous_library", 11, tmp_path, [0.] * 4, [])
    assert rec["usage"]["tool_successes"] == 2
    second = json.loads(chat.seen[1][-1]["content"])
    assert any(r["id"] == "inverse_volatility" for r in second["result"])
    third = json.loads(chat.seen[2][-1]["content"])
    expected, _ = execute("run_algorithm", {"algorithm": "inverse_volatility"}, prices.iloc[25:151])
    assert third["result"] == expected
    assert (tmp_path / "response-0.json").exists()
    assert json.loads((tmp_path / "event-1.json").read_text())["receipt"]["sha256"]


def test_error_is_returned_and_counts_against_budget(prices, tmp_path):
    chat = ScriptedChat([{"tool": "live_news", "arguments": {}}, FINAL])
    rec = one_decision(chat, prices, 150, "autonomous_library", 11, tmp_path, [0.] * 4, [])
    assert rec["status"] == "valid" and rec["usage"]["tool_errors"] == 1
    assert "tool_error" in json.loads(chat.seen[1][-1]["content"])


def test_control_cannot_call_library_and_can_recover(prices, tmp_path):
    chat = ScriptedChat([{"tool": "list_algorithms", "arguments": {}}, FINAL])
    rec = one_decision(chat, prices, 150, "no_library", 11, tmp_path, [0.] * 4, [])
    assert rec["usage"]["tool_successes"] == 0
    assert rec["usage"]["tool_errors"] == 1


def test_fifth_turn_cannot_execute_tool_and_no_final_means_hold(prices, tmp_path):
    chat = ScriptedChat([{"tool": "list_algorithms", "arguments": {}}] * 5)
    rec = one_decision(chat, prices, 150, "autonomous_library", 11, tmp_path, [0.] * 4, [])
    assert rec["status"] == "invalid_hold" and rec["action"] is None
    assert rec["usage"]["tool_successes"] == 4 and rec["usage"]["tool_errors"] == 1


def test_future_prices_cannot_change_any_advertised_algorithm(prices):
    changed = prices.copy()
    changed.iloc[151:] *= 30
    _, past = context(prices, 150)
    _, other = context(changed, 150)
    for row in catalog_rows():
        args = {"algorithm": row["id"]}
        if row["task"] != "portfolio":
            args["asset"] = "asset_0"
        assert execute("run_algorithm", args, past) == execute("run_algorithm", args, other)


@pytest.mark.parametrize("tool,args", [
    ("run_algorithm", {"algorithm": "inverse_volatility", "data": [999]}),
    ("read_skill", {"name": "../../private/labels"}),
    ("run_algorithm", {"algorithm": "drift", "asset": "asset_0", "parameters": {"horizon": 1000}}),
    ("recommend_strategy", {"asset": "asset_0", "as_of": "2030-01-01"}),
])
def test_caller_cannot_override_data_clock_or_read_arbitrary_paths(prices, tool, args):
    with pytest.raises(ValueError):
        execute(tool, args, prices.iloc[:126])


def test_strategy_and_skill_tools_do_not_leak_calendar_dates(prices):
    _, past = context(prices, 150)
    result, receipt = execute("recommend_strategy", {"asset": "asset_0"}, past)
    assert "as_of" not in result and "price_as_of" not in result
    assert receipt["observed_through"] == past.index[-1].isoformat()
    result, _ = execute("read_skill", {"name": "portfolio-and-risk"}, past)
    assert result["text"] and len(result["text"]) <= 4000


def test_provider_failure_aborts_instead_of_fabricating_hold(prices, tmp_path):
    class Broken:
        def __call__(self, messages):
            raise RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        one_decision(Broken(), prices, 150, "autonomous_library", 11, tmp_path, [0.] * 4, [])
    assert not (tmp_path / "decision.json").exists()
