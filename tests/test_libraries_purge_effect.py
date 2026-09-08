"""fin_skills.libraries.purge_effect - plain KFold on overlapping labels manufactures skill.

`main()` averages 24 datasets and then re-runs the comparison with sklearn's RandomForest
(about 30 s), so it is marked slow; the gap itself is reproduced here on three datasets.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires
from fin_skills.libraries import purge_effect as pe


def test_dataset_is_seeded_and_labels_overlap_by_construction():
    x, y = pe.make_dataset(0)
    x2, y2 = pe.make_dataset(0)
    assert np.array_equal(x, x2) and np.array_equal(y, y2)
    assert not np.array_equal(y, pe.make_dataset(1)[1])
    assert x.shape == (pe.N, pe.N_FEATURES) and set(np.unique(y)) == {0, 1}
    assert np.corrcoef(y[:-1], y[1:])[0, 1] > 0.8              # consecutive labels share H-1 bars
    assert abs(np.corrcoef(y[:-pe.H], y[pe.H:])[0, 1]) < 0.2     # windows no longer overlap


def test_auc_is_the_mann_whitney_statistic():
    y = np.array([0, 0, 1, 1])
    assert pe.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert pe.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert pe.auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert np.isnan(pe.auc(np.array([1, 1]), np.array([0.1, 0.2])))


def test_splitters_partition_the_index():
    n = 100
    for tr, te in pe.kfold(n, 5):
        assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == n
    folds = [te for _, te in pe.kfold(n, 5, shuffle=True, seed=1)]
    assert sorted(np.concatenate(folds).tolist()) == list(range(n))
    for tr, te in pe.time_series_split(n, 5):
        assert tr.max() + 1 == te.min()                          # abutting, still leaky


def test_purged_kfold_removes_every_overlapping_and_embargoed_row():
    for tr, te in pe.purged_kfold(pe.N, pe.N_SPLITS, pe.H, pe.EMBARGO):
        start, end = te[0], te[-1]
        overlap = (tr + pe.H >= start) & (tr <= end + pe.H)
        embargoed = (tr > end + pe.H) & (tr <= end + pe.H + pe.EMBARGO)
        assert not overlap.any() and not embargoed.any()
        assert len(tr) < pe.N - len(te)


def test_naive_splitters_inflate_auc_on_pure_noise_and_purging_removes_it():
    acc = {"shuffle": [], "kfold": [], "tss": [], "purged": []}
    for seed in range(3):
        x, y = pe.make_dataset(seed)
        acc["shuffle"] += pe.fold_aucs(x, y, pe.kfold(pe.N, pe.N_SPLITS, True, seed))
        acc["kfold"] += pe.fold_aucs(x, y, pe.kfold(pe.N, pe.N_SPLITS))
        acc["tss"] += pe.fold_aucs(x, y, pe.time_series_split(pe.N, pe.N_SPLITS))
        acc["purged"] += pe.fold_aucs(x, y, pe.purged_kfold(pe.N, pe.N_SPLITS, pe.H, pe.EMBARGO))
    mean = {k: float(np.mean(v)) for k, v in acc.items()}
    assert mean["shuffle"] > 0.9                                 # near-perfect skill on noise
    assert mean["kfold"] > mean["purged"] + 0.05
    assert mean["tss"] > mean["purged"] + 0.05
    assert abs(mean["purged"] - 0.5) < 0.08                      # the honest answer
    m, se, n = pe.summarise(acc["purged"])
    assert n == len(acc["purged"]) and se > 0


@requires("sklearn")
def test_reference_pieces_match_scikit_learn():
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold, TimeSeriesSplit
    x, y = pe.make_dataset(0)
    for (a, b), (c, d) in zip(pe.kfold(pe.N, pe.N_SPLITS), KFold(n_splits=pe.N_SPLITS).split(x)):
        assert np.array_equal(a, c) and np.array_equal(b, d)
    for (a, b), (c, d) in zip(pe.time_series_split(pe.N, pe.N_SPLITS),
                              TimeSeriesSplit(n_splits=pe.N_SPLITS).split(x)):
        assert np.array_equal(a, c) and np.array_equal(b, d)
    assert pe.auc(y[:500], x[:500, 0]) == pytest.approx(roc_auc_score(y[:500], x[:500, 0]), abs=1e-12)


@pytest.mark.slow
def test_main_reports_the_measured_gap_and_the_rule(capsys):
    pe.main()
    out = capsys.readouterr().out
    assert "MEASURED GAP" in out and "Rule: state the label horizon and purge it" in out
