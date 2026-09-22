"""Financial and causal checks before submitting the market experiment."""
import math

import numpy as np
import pandas as pd
import pytest

from benchmarks.fly_paper.run_control import Learner
from benchmarks.fly_paper.run_market import simulate
from benchmarks.fly_paper.test_model import simple_params
from benchmarks.library_utility.evaluate import ledger
from benchmarks.verified_memory.episode_credit import audit_option


def prices(n=200):
    rng = np.random.default_rng(13)
    return pd.DataFrame(np.exp(np.cumsum(rng.normal(0, .006, (n, 4)), axis=0)),
                        index=pd.bdate_range("2005-01-03", periods=n, tz="UTC"))


class Fixed:
    def __init__(self):
        self.targets = []

    def choose(self, state, *args):
        return int(not state[-2])  # enter, then hold until forced/terminal exit

    def learn(self, state, action, receipt, now):
        result = audit_option(receipt, now)
        assert result["status"] == "ready"
        self.targets.append(result["target"])


def test_market_ledger_matches_independent_scorer_and_charges_terminal_exit():
    px = prices()
    result = simulate(px, Fixed(), start=100, end=199)
    score = ledger(px, result["actions"], start=100, end=199, cost_bps=5)
    np.testing.assert_allclose(result["nav"], score["nav"], atol=1e-12, rtol=0)
    assert result["audit"]["forced_exits"] > 0
    assert result["audit"]["early_updates"] == 0
    assert result["audit"]["duplicate_updates"] == 0
    assert result["fees"][-1] > 0


def test_entry_target_includes_later_holds_and_both_fees():
    px = prices(200)
    px.iloc[:] = np.exp(np.arange(200)[:, None]*.001)
    result = simulate(px, Fixed(), start=100, end=199)
    first = next(r for r in result["receipts"] if r["origin"] >= 100 and r["action"] == 1)
    assert first["closed_index"]-first["origin"] == 61
    expected = .06+math.log(1-.0005)-math.log(1+.0005)
    assert first["target"] == pytest.approx(expected)


@pytest.mark.parametrize("arm", ["fly_actor_critic", "ordinary_mc_value"])
def test_future_changes_preserve_market_actions_and_learning(arm):
    px = prices()
    changed = px.copy()
    changed.iloc[150:] *= np.linspace(.5, 3, 50)[:, None]
    a = simulate(px, Learner(arm, simple_params(), 11), start=100, end=199, stop_before=150)
    b = simulate(changed, Learner(arm, simple_params(), 11), start=100, end=199, stop_before=150)
    assert a == b
