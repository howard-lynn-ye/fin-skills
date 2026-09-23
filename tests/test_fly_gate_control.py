"""Isolate gate/freeze control logic; real upstream integration lives in test_core."""
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.fly_reuse.core import Compact
from benchmarks.fly_reuse.run import selected_arms


def policy(kind):
    # A tiny deterministic KC state avoids requiring either upstream checkout.
    instance = Compact.__new__(Compact)
    instance.kind = kind
    instance.brain = SimpleNamespace(w_buy=np.ones((2, 1)), w_sell=np.zeros((2, 1)))
    instance.rng = np.random.default_rng(11)
    instance.risk_threshold = .1
    instance.gates = instance.updates = 0
    instance.batch_size = 16
    instance.batch = []
    return instance


@pytest.mark.parametrize("vol,expected", [(0.05, False), (.1, False), (.2, True)])
def test_learned_and_frozen_share_gate_and_prelearning_decisions(vol, expected):
    state = {"x":np.ones(2)/2, "kc":np.ones(2), "vol":vol}
    learned, frozen = policy("fly_gated"), policy("frozen_gated")
    a, b = learned.decide(state, True), frozen.decide(state, True)
    assert a["gated"] == b["gated"] == expected
    assert a["action"] == b["action"]
    assert a["score"] == b["score"]
    assert a["probability"] == b["probability"]
    assert learned.gates == frozen.gates == int(expected)
    if expected:
        assert a["action"] == b["action"] == 0


def test_frozen_gate_never_queues_learning_but_learned_control_does():
    frozen, learned = policy("frozen_gated"), policy("fly_gated")
    for reward in (-.02, 0., .02):
        frozen.receive({}, reward, {}, 10, True)
        learned.receive({}, reward, {}, 10, True)
    assert frozen.batch == [] and frozen.updates == 0
    assert len(learned.batch) == 3


def test_paired_preset_and_full_mode_are_explicit():
    assert selected_arms("compact", True) == ("fly_gated", "frozen_gated")
    assert "frozen_gated" in selected_arms("compact", False)
    with pytest.raises(ValueError, match="compact"):
        selected_arms("full", True)
