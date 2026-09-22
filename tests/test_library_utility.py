"""Independent accounting and no-future-data acceptance checks for the new study."""
import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.library_utility.evaluate import allocation, ledger, paired_blocks
from benchmarks.library_utility.run import context, library_evidence, parse_action, run_arm


def test_next_observation_execution_cannot_earn_the_earlier_jump():
    prices = np.array([[1.], [10.], [20.], [20.]])
    report = ledger(prices, [{"decision_index": 0, "weights": [1.]}], start=0, end=3, cost_bps=0)
    assert report["daily_returns"] == pytest.approx([0, 1, 0])
    assert report["metrics"]["total_return"] == pytest.approx(1)


def test_exact_entry_exit_cost_and_cash():
    prices = np.ones((4, 1))
    report = ledger(prices, [{"decision_index": 0, "weights": [1.]}], start=0, end=3, cost_bps=100)
    assert report["nav"][-1] == pytest.approx(.99 / 1.01)
    cash = ledger(prices, [], start=0, end=3)
    assert cash["metrics"]["total_return"] == 0
    assert cash["metrics"]["sharpe_zero_cash_rate"] is None


def test_invalid_action_holds_units_instead_of_secretly_closing():
    prices = np.array([[1.], [1.], [2.], [4.]])
    actions = [{"decision_index": 0, "weights": [1.]}, {"decision_index": 1, "weights": None}]
    report = ledger(prices, actions, start=0, end=3, cost_bps=0)
    assert report["metrics"]["total_return"] == pytest.approx(3)


@pytest.mark.parametrize("value", [[True], [-.1], [1.1], [float("nan")], [float("inf")], []])
def test_invalid_allocations_rejected(value):
    with pytest.raises(ValueError):
        allocation(value, 1)


def test_json_protocol_accepts_one_fence_but_not_explanatory_trailing_text():
    assert parse_action('```json\n{"weights":[0.5],"reason":"risk"}\n```', 1)["weights"] == [.5]
    with pytest.raises(ValueError):
        parse_action('{"weights":[0.5],"reason":"risk"} extra', 1)


def test_actual_tool_receipts_and_base_context_are_future_invariant():
    rng = np.random.default_rng(5)
    prices = pd.DataFrame(np.exp(rng.normal(.0002, .01, (180, 4)).cumsum(axis=0)),
                          index=pd.date_range("2024-01-01", periods=180, tz="UTC"),
                          columns=[f"asset_{i}" for i in range(4)])
    base, past = context(prices, 150)
    altered = prices.copy()
    altered.iloc[151:] *= 10
    other, other_past = context(altered, 150)
    assert base == other
    evidence = library_evidence(past)
    assert evidence == library_evidence(other_past)
    assert len(evidence["calls"]) == 3


def test_paired_bootstrap_does_not_treat_models_as_extra_dates():
    result = paired_blocks(np.ones(84) * .001, np.zeros(84), draws=20)
    assert result["descriptive_95pct_interval"] == pytest.approx([.001, .001])


def test_raw_provider_response_frozen_and_invalid_counted(tmp_path):
    class FakeChat:
        def __call__(self, messages):
            return {"choices": [{"message": {"content": 'not JSON'}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    prices = pd.DataFrame(np.tile(np.arange(1., 161)[:, None], (1, 4)),
                          index=pd.date_range("2024-01-01", periods=160, tz="UTC"))
    actions, usage = run_arm(FakeChat(), prices, [140], "no_library", 11, tmp_path, [], cost_bps=5)
    assert usage["invalid_hold"] == 1 and actions[0]["weights"] is None
    saved = json.loads((tmp_path / "decision-000.json").read_text())
    assert saved["response"]["choices"][0]["message"]["content"] == "not JSON"
    assert (tmp_path / "frozen_actions.json").exists()
