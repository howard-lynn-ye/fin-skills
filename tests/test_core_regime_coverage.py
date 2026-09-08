"""fin_skills.core.regime_coverage - which regimes a test period contained, with episodes counted.

Beyond `episodes` and `regime_table` the module is its __main__ demo (the regime axes, the
per-episode table and the result_manifest gate), so the demo is run under capsys and its
closing lines are asserted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.core.regime_coverage import episodes, regime_table


def test_episodes_are_contiguous_runs_with_exclusive_ends():
    assert episodes(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]
    assert episodes(np.array([True, True])) == [(0, 2)]
    assert episodes(np.array([False, False])) == []
    assert episodes(np.array([], dtype=bool)) == []


@pytest.fixture
def toy():
    n = 300
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=n)
    labels = pd.Series(np.where((np.arange(n) // 50) % 2 == 0, "calm", "turbulent"))
    asset = rng.normal(0.0005, 0.01, n)
    pos = np.ones(n)
    return dates, asset, pos * asset, pos, labels


def test_regime_table_counts_observations_and_episodes(toy):
    dates, asset, strat, pos, labels = toy
    tab, strings = regime_table(dates, asset, strat, pos, labels, "ex-ante rule")
    assert set(tab.index) == {"calm", "turbulent"}
    assert tab["n_obs"].sum() == 300 and tab["share"].sum() == pytest.approx(1.0)
    assert tab.loc["calm", "n_episodes"] == 3 and tab.loc["turbulent", "n_episodes"] == 3
    assert tab.loc["calm", "longest_ep"] == 50
    assert tab.loc["calm", "first"] == dates[0].date() and tab.loc["calm", "in_mkt"] == 1.0
    assert len(strings) == 2 and all("[ex-ante rule]" in s and "episodes" in s for s in strings)
    assert tab.loc["calm", "asset_ann"] == pytest.approx(asset[labels.to_numpy() == "calm"].mean() * 252)


def test_regime_table_reports_flat_positions_as_out_of_market(toy):
    dates, asset, _, _, labels = toy
    pos = np.zeros(300)
    tab, _ = regime_table(dates, asset, pos * asset, pos, labels, "ex-post label")
    assert (tab["in_mkt"] == 0.0).all() and tab["hit_rate"].isna().all()


def test_demo_prints_the_regimes_covered_strings_and_runtime(run_main):
    # the module's substance is its __main__ demo: three regime axes on the simulated DGP,
    # the per-episode table and, when importable, the result_manifest gate
    out = run_main("fin_skills.core.regime_coverage")
    assert "=== regimes_covered = [" in out
    assert "[ex-ante rule]" in out and "[ex-post label" in out
    assert "One crisis is one observation" in out
    assert "total runtime" in out.strip().splitlines()[-1]
