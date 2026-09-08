"""fin_skills.core.spa_test - SPA / StepM / MCS wrappers that take RETURNS and negate once.

The self-contained stationary bootstrap is tested directly; the arch-backed wrappers run
only when `arch` is installed and otherwise assert the documented ArchMissing message.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import has_module, requires
from fin_skills.core.spa_test import (ArchMissing, MCSResult, SPAResult, _as_matrix,
                                      _as_vector, _panel, _stationary_bootstrap_indices,
                                      choose_block_size, mcs_test, naive_best_of_n_pvalue,
                                      spa_test, spa_test_selfcontained, stepm_test)


@pytest.fixture(scope="module")
def null_panel():
    rng = np.random.default_rng(3)
    bench, models, names = _panel(rng, edge=0.0)
    return bench, pd.DataFrame(models, columns=names)


@pytest.fixture(scope="module")
def edge_panel():
    rng = np.random.default_rng(3)
    bench, models, names = _panel(rng, edge=0.0012, edge_on=4)
    return bench, pd.DataFrame(models, columns=names)


def test_input_coercion_and_validation():
    arr, names = _as_matrix(np.zeros((5, 2)), "model")
    assert arr.shape == (5, 2) and names == ["model_0", "model_1"]
    arr, names = _as_matrix(pd.Series(np.zeros(5), name="s"), "model")
    assert arr.shape == (5, 1) and names == ["s"]
    with pytest.raises(ValueError, match="NaN"):
        _as_matrix(np.array([1.0, np.nan]), "model")
    with pytest.raises(ValueError, match="single series"):
        _as_vector(np.zeros((5, 2)), "benchmark_returns")
    with pytest.raises(ValueError, match="at least 2 candidates"):
        mcs_test(np.zeros(10))


def test_stationary_bootstrap_indices_are_valid_and_seeded():
    rng = np.random.default_rng(0)
    idx = _stationary_bootstrap_indices(50, 5, 20, rng)
    assert idx.shape == (20, 50) and idx.min() >= 0 and idx.max() < 50
    again = _stationary_bootstrap_indices(50, 5, 20, np.random.default_rng(0))
    assert np.array_equal(idx, again)


def test_block_size_override_and_provenance(null_panel):
    bench, mdf = null_panel
    assert choose_block_size(-mdf.to_numpy(), override=7) == (7, "caller-specified")
    bs, src = choose_block_size(-mdf.to_numpy())
    assert 1 <= bs <= len(mdf) // 2
    assert src.startswith("arch.optimal_block_length" if has_module("arch") else "fallback")


def test_three_pvalues_are_ordered_and_the_null_is_not_rejected(null_panel):
    bench, mdf = null_panel
    res = spa_test_selfcontained(bench, mdf, reps=300, seed=12)
    assert isinstance(res, SPAResult)
    assert set(res.pvalues) == {"lower", "consistent", "upper"}
    assert res.pvalues["lower"] <= res.pvalues["consistent"] <= res.pvalues["upper"]
    assert res.pvalue == res.pvalues["consistent"] and res.pvalue > 0.05
    assert res.n_obs == 750 and res.model_names == list(mdf.columns)
    assert "REPORT THIS" in res.report() and "CANNOT REJECT" in res.report()


def test_naive_best_of_n_is_more_eager_than_spa_on_the_same_null(null_panel):
    bench, mdf = null_panel
    name, p_naive = naive_best_of_n_pvalue(bench, mdf)
    assert name in mdf.columns and 0 <= p_naive <= 1
    assert p_naive < spa_test_selfcontained(bench, mdf, reps=300, seed=12).pvalue


def test_a_real_edge_survives_the_search_correction(edge_panel):
    bench, mdf = edge_panel
    res = spa_test_selfcontained(bench, mdf, reps=300, seed=12)
    assert res.pvalue < 0.05 and res.best_model == "strat_04"
    assert res.better_models == ["strat_04"]
    assert res.mean_excess.idxmax() == "strat_04"
    assert "REJECT H0" in res.report()


def test_feeding_returns_as_losses_inverts_the_test(edge_panel):
    bench, mdf = edge_panel
    wrong = spa_test_selfcontained(bench, mdf, reps=300, seed=12, already_losses=True)
    assert "strat_04" not in wrong.better_models          # the genuinely best one vanishes
    d_wrong = np.asarray(bench)[:, None] - mdf.to_numpy()
    assert mdf.columns[int(np.argmax(d_wrong.mean(axis=0)))] == mdf.mean().idxmin()  # worst ranked first


def test_selfcontained_is_seeded(edge_panel):
    bench, mdf = edge_panel
    a = spa_test_selfcontained(bench, mdf, reps=200, seed=5)
    b = spa_test_selfcontained(bench, mdf, reps=200, seed=5)
    assert a.pvalues == b.pvalues and a.better_models == b.better_models


@pytest.mark.skipif(has_module("arch"), reason="arch is installed; the missing-arch path is not reachable")
def test_wrappers_name_the_missing_package(null_panel):
    bench, mdf = null_panel
    with pytest.raises(ArchMissing, match="pip install arch"):
        spa_test(bench, mdf, reps=50)
    assert issubclass(ArchMissing, ImportError)


@requires("arch")
def test_arch_backed_wrappers_find_the_planted_edge(edge_panel):
    bench, mdf = edge_panel
    res = spa_test(bench, mdf, reps=200, seed=1)
    assert res.source == "arch.bootstrap.SPA"
    assert res.pvalue < 0.05 and res.best_model == "strat_04"
    assert set(res.pvalues) == {"lower", "consistent", "upper"}
    assert stepm_test(bench, mdf, reps=200, seed=1) == ["strat_04"]
    mcs = mcs_test(mdf, reps=200, seed=1)
    assert isinstance(mcs, MCSResult) and "strat_04" in mcs.included
    assert set(mcs.included) | set(mcs.excluded) == set(mdf.columns)
    assert "INCLUDED" in mcs.report()
    with pytest.raises(ValueError, match="aligned"):
        spa_test(bench[:-1], mdf, reps=50)


@pytest.mark.slow
def test_demo_measures_the_false_positive_rates_and_the_trap(run_main):
    out = run_main("fin_skills.core.spa_test")
    assert "naive t-test on the best of 10" in out
    assert "It picked the WORST strategy of the ten and called it the best." in out
