"""fin_skills.ml.feature_importance - MDI, MDA, SFI and the split that decides all three.

The properties the SKILL.md claims:

  * the CART splits on exact variance reduction and its MDI is the weighted impurity decrease;
  * the reconstruction of sklearn's feature_importances_ from the public tree_ arrays IS
    sklearn's number, for a tree and for a forest;
  * MDI favours the high-cardinality noise column over the binary one, splits credit between the
    collinear pair, and gives a random walk a real share;
  * MDA has no cardinality bias but under-states each member of the collinear pair;
  * clustered MDA recovers the pair's joint importance and leaves a singleton cluster alone;
  * a shuffled k-fold gives the autocorrelated noise column a large importance and an embargoed
    forward split does not.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires
from fin_skills.ml import feature_importance as fim


@pytest.fixture(scope="module")
def data():
    return fim.design()


@pytest.fixture(scope="module")
def folds():
    return fim.embargoed_forward_folds(fim.N_SAMPLES)


@pytest.fixture(scope="module")
def forest(data):
    X, y, _ = data
    return fim.fit_forest(X, y)


@pytest.fixture(scope="module")
def mda_embargo(data, folds):
    X, y, _ = data
    return fim.mda(X, y, folds)


def test_the_design_is_seeded_and_is_what_it_says(data):
    X, y, names = data
    X2, y2, _ = fim.design()
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    assert not np.array_equal(y, fim.design(seed=fim.SEED + 1)[1])
    assert X.shape == (fim.N_SAMPLES, 7) and len(names) == 7
    assert len(np.unique(X[:, 3])) == fim.N_SAMPLES        # high cardinality
    assert len(np.unique(X[:, 4])) == 2 and len(np.unique(X[:, 5])) == 5
    assert np.corrcoef(X[:, 0], X[:, 1])[0, 1] > 0.98      # the collinear copy
    # the noise columns really are unrelated to y, and x0/x2 really are related
    resid = y - fim.BETA[0] * X[:, 0] - fim.BETA[2] * X[:, 2]
    assert abs(resid.std() - fim.NOISE_SD) < 0.15
    assert np.corrcoef(resid[:-1], resid[1:])[0, 1] > 0.9  # the residual OVERLAPS across rows
    for j in (3, 4, 5):
        assert abs(np.corrcoef(X[:, j], y)[0, 1]) < 0.06


def test_the_split_search_maximises_variance_reduction():
    x = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
    y = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0])
    sse, thr = fim._best_split(x, y, min_leaf=1)
    assert sse == pytest.approx(0.0) and 3.0 < thr < 10.0
    assert fim._best_split(x, y, min_leaf=4) == (sse, thr)  # the 4/4 split is still allowed
    assert fim._best_split(np.zeros(8), y, min_leaf=1) is None       # no distinct values
    assert fim._best_split(x, y, min_leaf=5) is None                 # no room for two leaves


def test_the_tree_is_deterministic_and_its_mdi_is_the_weighted_impurity_decrease():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 2))
    y = X[:, 0] + 0.2 * rng.standard_normal(400)
    t1 = fim.grow_tree(X, y, np.random.default_rng(7), max_depth=3, max_features=2)
    t2 = fim.grow_tree(X, y, np.random.default_rng(7), max_depth=3, max_features=2)
    assert np.array_equal(t1["mdi"], t2["mdi"])
    assert t1["mdi"][0] > 10 * max(t1["mdi"][1], 1e-12)     # feature 0 carries the signal
    # sum of the per-feature MDI equals the total variance explained by the tree
    pred = fim.predict_tree(t1["root"], X)
    explained = float(np.mean((y - y.mean()) ** 2) - np.mean((y - pred) ** 2))
    assert t1["mdi"].sum() == pytest.approx(explained, rel=1e-9)


def test_the_forest_predicts_and_its_mdi_sums_to_one(data, forest):
    X, y, _ = data
    assert len(forest["trees"]) > 0
    assert forest["mdi"].sum() == pytest.approx(1.0)
    assert (forest["mdi"] >= 0).all()
    assert fim.r2(y, fim.predict_forest(forest, X)) > 0.2
    assert np.array_equal(fim.fit_forest(X, y)["mdi"], forest["mdi"])   # seeded
    empty = fim.fit_forest(X[:5], y[:5], n_trees=2)
    assert fim.predict_forest(empty, X[:5]).shape == (5,)


@requires("sklearn")
def test_the_sklearn_reconstruction_is_sklearns_number(data):
    X, y, _ = data
    chk = fim.sklearn_mdi_check(X, y)
    assert chk is not None
    assert chk["tree_diff"] < 1e-12
    assert chk["forest_diff"] < 1e-12
    assert chk["doc_warns_high_cardinality"] is True
    # sklearn agrees with this script's forest about which features are noise
    assert chk["rf_mdi"][4] < chk["rf_mdi"][3] < chk["rf_mdi"][0]


def test_mdi_is_cardinality_biased_splits_credit_and_rewards_a_time_proxy(forest):
    mdi = forest["mdi"]
    assert mdi[3] > 2.0 * mdi[4]                    # 4.1x in the demo: noise vs noise
    assert mdi[3] > mdi[5]                          # 1500 levels beats 5 levels
    assert mdi[0] + mdi[1] > 0.45                   # 60.8 % shared by the pair
    assert min(mdi[0], mdi[1]) > 0.3 * max(mdi[0], mdi[1])   # neither is ignored
    assert mdi[6] > mdi[3] + mdi[4] + mdi[5]        # the random walk beats all the iid noise


def test_mda_has_no_cardinality_bias_but_understates_the_collinear_pair(mda_embargo):
    m = mda_embargo
    for j in (3, 4, 5, 6):
        assert abs(m[j]) < 0.02                     # every irrelevant column is ~0
    assert m[2] > 0.05 and m[0] > 0.05
    assert abs(m[3] - m[4]) < 0.02                  # high and low cardinality noise now agree
    assert m[0] + m[1] < 0.30                       # 0.175 in the demo, half the joint value


def test_clustered_mda_recovers_the_pairs_joint_importance(data, folds, mda_embargo):
    X, y, _ = data
    clusters = fim.correlation_clusters(X, 0.8)
    assert clusters[0] == [0, 1]
    assert [len(c) for c in clusters[1:]] == [1, 1, 1, 1, 1]
    cm = fim.mda(X, y, folds, groups=clusters)
    assert cm[0] > 1.5 * max(mda_embargo[0], mda_embargo[1])       # 2.9x in the demo
    assert cm[0] > mda_embargo[0] + mda_embargo[1]
    assert cm[1] == pytest.approx(mda_embargo[2], abs=0.05)        # a singleton is unchanged
    assert int(np.argmax(cm)) == 0
    # a threshold above every pairwise correlation leaves every feature on its own
    assert fim.correlation_clusters(X, 0.999) == [[j] for j in range(7)]


def test_a_shuffled_split_invents_importance_for_the_random_walk(data, mda_embargo):
    X, y, _ = data
    sh = fim.shuffled_folds(fim.N_SAMPLES)
    m_sh = fim.mda(X, y, sh)
    assert m_sh[6] > 0.02                           # +0.0521 in the demo
    assert m_sh[6] > 10 * abs(mda_embargo[6])
    assert abs(mda_embargo[6]) < 0.01
    # the shuffled out-of-sample R2 is inflated too
    r_sh = np.mean([fim.r2(y[te], fim.predict_forest(fim.fit_forest(X[tr], y[tr],
                    seed=fim.SEED + i), X[te])) for i, (tr, te) in enumerate(sh)])
    r_em = np.mean([fim.r2(y[te], fim.predict_forest(fim.fit_forest(X[tr], y[tr],
                    seed=fim.SEED + i), X[te]))
                    for i, (tr, te) in enumerate(fim.embargoed_forward_folds(fim.N_SAMPLES))])
    assert r_sh > r_em


def test_the_folds_are_what_they_say():
    sh = fim.shuffled_folds(90, k=3)
    assert len(sh) == 3
    assert sorted(np.concatenate([te for _, te in sh]).tolist()) == list(range(90))
    for tr, te in sh:
        assert set(tr).isdisjoint(te) and len(tr) + len(te) == 90
    em = fim.embargoed_forward_folds(300, k=3, embargo=10)
    assert len(em) == 3
    for tr, te in em:
        assert set(tr).isdisjoint(te)
        assert min(abs(int(a) - int(b)) for a in tr for b in (te[0], te[-1])) >= 10
    assert list(em[0][1]) == list(range(100))
    assert len(em[0][0]) == 300 - 100 - 10          # the embargo really removes rows


def test_sfi_scores_each_feature_alone(data, folds):
    X, y, _ = data
    s = fim.sfi(X, y, folds)
    assert s.shape == (7,)
    assert s[0] > 0.05 and s[1] > 0.05              # BOTH members of the pair are informative
    assert s[2] > 0.0
    for j in (3, 4, 5, 6):
        assert s[j] < 0.01                          # a useless feature cannot beat the mean


@pytest.mark.slow
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.ml.feature_importance")
    for head in ("=== 1. The design", "=== 2. MDI: in-sample, and biased",
                 "=== 3. MDA on an embargoed forward split", "=== 4. Clustered MDA",
                 "=== 5. The split matters more than the metric"):
        assert head in out
    assert out.isascii()
    rule = [ln for ln in out.splitlines() if ln.startswith("Rule: ")]
    assert len(rule) == 1 and "MDI" in rule[0] and "cluster" in rule[0]
    assert out.strip().splitlines()[-1].startswith("total runtime")
