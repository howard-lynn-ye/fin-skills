"""Mechanism checks: causality, accounting, valid losses, and meaningful negative controls."""
import numpy as np
import pytest

from benchmarks.library_utility.evaluate import ledger
from benchmarks.verified_memory.model import (OnlineValue, audit_packet, expert_weights,
    features, payoff, prepare, simulate, synthetic)
from fin_skills.api import get


@pytest.fixture(scope="module")
def prices():
    return synthetic(91, 210)


def test_feature_causality(prices):
    assert get("assert_causal").run(fn=features, df=prices, k=103).passed


def test_sparse_dense_parameter_match_and_activity():
    a, b = OnlineValue("dense", 1), OnlineValue("fly_sparse", 1)
    np.testing.assert_array_equal(a.projection, b.projection)
    assert a.readout.shape == b.readout.shape
    z = b.encode(np.linspace(-1, 1, 20))
    assert np.count_nonzero(z) == 16
    assert np.linalg.norm(z) == pytest.approx(1)


def test_clean_conditions_identical_and_keep_losses(prices):
    data = prepare(prices, "clean")
    expected = None
    for mode in ("unchecked", "output_audit", "verified_update"):
        actions, updates, audit = simulate(data, "fly_sparse", mode, 3, start=90, end=209)
        assert audit["early_updates"] == audit["invalid_updates"] == 0
        assert audit["negative_targets_learned"] > 0
        if expected is not None:
            assert actions == expected
        expected = actions


@pytest.mark.parametrize("scenario", ["early_feedback", "wrong_payoff"])
def test_real_guard_blocks_injected_faults(prices, scenario):
    data = prepare(prices, scenario)
    a, _, au = simulate(data, "fly_sparse", "unchecked", 3, start=90, end=209)
    b, _, ao = simulate(data, "fly_sparse", "output_audit", 3, start=90, end=209)
    _, _, ag = simulate(data, "fly_sparse", "verified_update", 3, start=90, end=209)
    assert a == b
    assert au["invalid_updates"] > 0
    assert ao["final_audit_passed"] is False
    assert ag["invalid_updates"] == ag["early_updates"] == 0
    assert ag["blocked_packets"] > 0
    assert ag["negative_targets_learned"] > 0
    assert ag["final_audit_passed"] is True


@pytest.mark.parametrize("k", [70, 105, 140])
def test_future_perturbation_including_memory(prices, k):
    altered = prices.copy()
    altered.iloc[k:] *= np.linspace(0.3, 3, len(prices) - k)[:, None]
    for scenario in ("clean", "early_feedback", "wrong_payoff"):
        first = simulate(prepare(prices, scenario), "fly_sparse", "verified_update", 7,
                         start=64, end=k)[0]
        second = simulate(prepare(altered, scenario), "fly_sparse", "verified_update", 7,
                          start=64, end=k)[0]
        assert first == second
    # The control must actually detect a harmful channel, not merely pass every arm.
    first = simulate(prepare(prices, "early_feedback"), "fly_sparse", "unchecked", 7,
                     start=64, end=k)[0]
    second = simulate(prepare(altered, "early_feedback"), "fly_sparse", "unchecked", 7,
                      start=64, end=k)[0]
    assert first != second


def test_payoff_matches_independent_cash_share_ledger(prices):
    t, end = 74, 80
    weights = expert_weights(prices, t)
    targets = payoff(prices.iloc[t + 1].to_numpy(), prices.iloc[end].to_numpy(), weights)
    for i, w in enumerate(weights):
        result = ledger(prices, [{"decision_index": t, "weights": w.tolist()}],
                        start=t, end=end, cost_bps=5)
        assert targets[i] == pytest.approx(result["metrics"]["total_return"], abs=1e-12)
    clean = audit_packet(prices, t, end, end, weights, targets)
    assert clean["passed"]
    broken = targets.copy()
    broken[1] += 0.05
    assert not audit_packet(prices, t, end, end, weights, broken)["accounting_passed"]


def test_cash_payoff_zero_and_loss_allowed():
    weights = np.array([[1., 0, 0, 0], [0., 0, 0, 0]])
    reward = payoff(np.ones(4), np.full(4, 0.9), weights)
    assert reward[0] < -0.1
    assert reward[1] == 0


def test_mature_revision_not_double_learned(prices):
    data = prepare(prices, "early_feedback")
    _, updates, _ = simulate(data, "linear", "verified_update", 3, start=64, end=209)
    origins = [u["origin"] for u in updates if u["applied"]]
    assert len(origins) == len(set(origins))
    assert all(u["at"] > u["required"] for u in updates if u["applied"])
