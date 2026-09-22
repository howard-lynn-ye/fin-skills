import numpy as np
import pytest

from benchmarks.verified_memory.fly_v2 import Eligibility, MushroomBody, PARAMETERS, run_market
from benchmarks.verified_memory.model import prepare, synthetic


def test_matched_projection_and_sparse_activity():
    sparse, dense = MushroomBody("fly_trace", 1), MushroomBody("dense_trace", 1)
    np.testing.assert_array_equal(sparse.inputs, dense.inputs)
    assert sparse.approach.shape == dense.approach.shape
    z = sparse.encode(np.linspace(-1, 1, 20))
    assert np.count_nonzero(z) == PARAMETERS["active"]
    assert np.linalg.norm(z) == pytest.approx(1)


@pytest.mark.parametrize("reward,expected_pool", [(1.0, "avoid"), (-1.0, "approach")])
def test_local_compartment_plasticity_and_opponent_sign(reward, expected_pool):
    brain = MushroomBody("fly_trace", 1)
    z = brain.encode(np.linspace(-1, 1, 20))
    memory = Eligibility(0, z, 2, 0.)
    brain.reinforce(memory, reward, 4)
    pool = getattr(brain, expected_pool)
    assert np.all(pool[z == 0] == 0.5)
    assert np.all(pool[:, [0, 1, 3, 4]] == 0.5)
    assert np.all(pool[z > 0, 2] < 0.5)
    assert np.sign(brain.values(z)[2]) == np.sign(reward)


def test_original_eligibility_survives_current_observation():
    a, b = MushroomBody("fly_trace", 3), MushroomBody("fly_trace", 3)
    z = a.encode(np.linspace(-1, 1, 20))
    memory = Eligibility(1, z.copy(), 1, 0.)
    a.reinforce(memory, 1, 7, current_z=np.ones_like(z))
    b.reinforce(memory, 1, 7, current_z=np.zeros_like(z))
    np.testing.assert_array_equal(a.avoid, b.avoid)
    assert np.array_equal(memory.z, z)


def test_prediction_error_reduces_with_learning():
    brain = MushroomBody("fly_trace", 4)
    z = brain.encode(np.linspace(-1, 1, 20))
    errors = []
    for t in range(30):
        memory = Eligibility(t * 5, z.copy(), 1, float(brain.values(z)[1]))
        errors.append(brain.reinforce(memory, 0.5, t * 5 + 4)["delta"])
    assert 0 <= errors[-1] < errors[0] / 3


def test_frozen_circuit_and_finite_weight_bounds():
    for variant in ("fly_frozen", "fly_trace", "fly_absolute"):
        brain = MushroomBody(variant, 2)
        z = brain.encode(np.linspace(-1, 1, 20))
        for i in range(100):
            memory = Eligibility(i, z.copy(), 2, float(brain.values(z)[2]))
            brain.reinforce(memory, (-1.) ** i, i + 1)
        assert np.isfinite(brain.approach).all()
        assert np.min(brain.approach) >= 0 and np.max(brain.approach) <= 1
        if variant == "fly_frozen":
            assert np.all(brain.approach == 0.5) and np.all(brain.avoid == 0.5)


def test_market_checked_feedback_causality_and_clean_equivalence():
    prices = synthetic(73, 180)
    clean = prepare(prices, "clean")
    a = run_market(clean, "fly_trace", "unchecked", 1, start=64, end=179)[0]
    b = run_market(clean, "fly_trace", "verified_update", 1, start=64, end=179)[0]
    assert a == b
    altered = prices.copy()
    altered.iloc[105:] *= 3
    for variant in ("fly_trace", "dense_trace", "fly_no_trace", "fly_frozen", "linear_trace"):
        first = run_market(prepare(prices, "early_feedback"), variant,
                           "verified_update", 1, start=64, end=105)
        second = run_market(prepare(altered, "early_feedback"), variant,
                            "verified_update", 1, start=64, end=105)
        assert first[0] == second[0]
        assert first[2]["early_updates"] == first[2]["invalid_updates"] == 0


def test_negative_selected_rewards_retained_and_other_actions_not_learned():
    prepared = prepare(synthetic(3, 300), "clean")
    _, updates, stats = run_market(prepared, "fly_trace", "verified_update", 2, start=64, end=299)
    assert stats["negative_labels"] > 0
    assert all(isinstance(u["reward"], float) for u in updates if u["applied"])
    assert stats["updates"] == stats["received_labels"]
