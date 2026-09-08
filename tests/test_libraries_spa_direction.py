"""fin_skills.libraries.spa_direction - SPA/StepM take LOSSES; returns invert the test."""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.libraries.spa_direction import (BENCH_SHARPE, PERIODS, REPS, T, RefSPA,
                                                _stationary_bootstrap_indices, ann_sharpe,
                                                main, make_strategies, ref_stepm)


@pytest.fixture(scope="module")
def strategies():
    return make_strategies()


def test_strategies_realise_their_target_sharpes_exactly(strategies):
    returns, benchmark, target = strategies
    assert list(returns.columns) == list(target)
    for name, sharpe in target.items():
        assert ann_sharpe(returns[name]) == pytest.approx(sharpe, abs=1e-9)
    assert ann_sharpe(benchmark) == pytest.approx(BENCH_SHARPE, abs=1e-9)
    assert len(returns) == T and PERIODS == 252
    again, _, _ = make_strategies()
    assert np.array_equal(returns.to_numpy(), again.to_numpy())


def test_stationary_bootstrap_indices_are_circular_and_seeded():
    idx = _stationary_bootstrap_indices(np.random.default_rng(1), 100, 10)
    assert idx.shape == (100,) and idx.min() >= 0 and idx.max() < 100
    assert np.array_equal(idx, _stationary_bootstrap_indices(np.random.default_rng(1), 100, 10))


def test_correct_orientation_crowns_the_excellent_strategy(strategies):
    returns, benchmark, _ = strategies
    ok = RefSPA(-benchmark, -returns, reps=200, seed=7).compute()
    assert ok.pvalues["consistent"] < 0.05
    assert int(np.argmax(ok.loss_diff.mean(0))) == 5                  # S6_excellent
    assert 5 in ok.better_models(0.05).tolist() and 0 not in ok.better_models(0.05).tolist()
    assert ref_stepm(-benchmark, -returns, 0.05, 200, 7) == [5]


def test_returns_passed_as_losses_crown_the_terrible_strategy(strategies):
    returns, benchmark, _ = strategies
    bad = RefSPA(benchmark, returns, reps=200, seed=7).compute()
    assert bad.pvalues["consistent"] < 0.05                           # rejects just as confidently
    assert int(np.argmax(bad.loss_diff.mean(0))) == 0                 # S1_terrible
    assert 0 in bad.better_models(0.05).tolist() and 5 not in bad.better_models(0.05).tolist()
    assert 5 not in ref_stepm(benchmark, returns, 0.05, 200, 7)


def test_refspa_is_seeded(strategies):
    returns, benchmark, _ = strategies
    a = RefSPA(-benchmark, -returns, reps=100, seed=3).compute()
    b = RefSPA(-benchmark, -returns, reps=100, seed=3).compute()
    assert a.pvalues == b.pvalues
    assert set(a.critical_values(0.05)) == {"lower", "consistent", "upper"}
    assert a.block_size == int(np.sqrt(T)) and REPS == 1000


def test_main_demonstrates_the_swing_and_states_the_rule(capsys):
    main()
    out = capsys.readouterr().out
    assert "trap -> S1_terrible" in out and "correct -> S6_excellent" in out
    assert "Rule: losses = -returns before SPA / StepM / RealityCheck / MCS." in out
