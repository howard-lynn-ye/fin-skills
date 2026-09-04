#!/usr/bin/env python3
"""Plain KFold on overlapping labels manufactures skill out of pure noise.

A financial label stamped at bar t is not resolved until bar t+H. Consecutive labels
therefore share H-1 of their H bars and are almost the same number - here the lag-1
label autocorrelation is 0.90. Features are smooth too. So a training row sitting one
bar outside the test fold carries nearly the test row's features AND nearly its answer,
and any model with memory looks it up.

The dataset below has NO predictive relationship whatsoever: the features are persistent
AR(1) series drawn from a generator that never touches the returns defining the labels.
The true out-of-sample AUC is 0.500 by construction. Plain `KFold` still reports better
than that, and so does `TimeSeriesSplit`, because both butt a training row directly
against the fold boundary. `KFold(shuffle=True)` - the default in `train_test_split` and
in most tutorial code - reports near-perfect skill on the same noise.

Nothing raises. There is no diagnostic other than splitting correctly and watching the
score fall, which is why an inflated CV score gets blamed on regime change instead.

Run:  python purge_effect.py
scikit-learn is optional. Without it the numpy k-NN, the hand-written KFold /
TimeSeriesSplit and the reference purged splitter below reproduce the same gap.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

N = 2000          # observations
H = 100           # label horizon: the label stamped at i resolves at i+H
N_FEATURES = 5
PHI = 0.995       # feature persistence - a slow indicator, not a jump process
N_SPLITS = 5
EMBARGO = 100     # observations dropped after the test block, on top of the purge
K = 5             # k-NN neighbours
REPLICATIONS = 24  # independent datasets; one draw is far too noisy to conclude from


# --------------------------------------------------------------------------
# Synthetic data: known overlapping horizon, no signal at all
# --------------------------------------------------------------------------
def make_dataset(seed: int):
    """y[i] = 1{ sum of e over (i, i+H] > 0 }; X = AR(1) noise independent of e.

    Labels overlap by construction. Features are unrelated to `e`, so E[y|X] = 0.5
    everywhere and the honest out-of-sample AUC is exactly 0.500.
    """
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 1.0, N + H)                    # the return series being labelled
    c = np.concatenate([[0.0], np.cumsum(e)])
    y = ((c[1 + H: N + 1 + H] - c[1: N + 1]) > 0).astype(int)

    x = np.zeros((N, N_FEATURES))
    eps = rng.normal(0.0, 1.0, (N, N_FEATURES))        # independent of e
    for i in range(1, N):
        x[i] = PHI * x[i - 1] + np.sqrt(1 - PHI ** 2) * eps[i]
    return x, y


# --------------------------------------------------------------------------
# Reference pieces: numpy + scipy only
# --------------------------------------------------------------------------
def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney form of the ROC AUC, so sklearn is not needed to score."""
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def knn_scores(x_tr, y_tr, x_te, k: int = K) -> np.ndarray:
    """Score = share of the k nearest training labels that are 1. Memory, nothing else."""
    d2 = (x_te ** 2).sum(1)[:, None] + (x_tr ** 2).sum(1)[None, :] - 2 * x_te @ x_tr.T
    nn = np.argpartition(d2, kth=min(k, d2.shape[1] - 1), axis=1)[:, :k]
    return y_tr[nn].mean(axis=1)


def kfold(n: int, n_splits: int, shuffle: bool = False, seed: int = 0):
    """sklearn's KFold: contiguous folds, or a random permutation of rows if shuffled."""
    idx = np.arange(n)
    if shuffle:
        idx = idx.copy()
        np.random.default_rng(seed).shuffle(idx)
    for fold in np.array_split(idx, n_splits):
        test = np.sort(fold)
        yield np.setdiff1d(np.arange(n), test), test


def time_series_split(n: int, n_splits: int):
    """sklearn's TimeSeriesSplit: expanding train, the next contiguous block as test.

    Forward-only, and still leaky: the last training row abuts the first test row.
    """
    test_size = n // (n_splits + 1)
    for start in range(n - n_splits * test_size, n, test_size):
        yield np.arange(start), np.arange(start, start + test_size)


def purged_kfold(n: int, n_splits: int, horizon: int, embargo: int):
    """Contiguous folds, then PURGE the overlapping labels and EMBARGO what follows.

    Purge: observation i carries a label spanning [i, i+horizon]. Drop it from train if
    that window touches the test block's own label span [test_start, test_end+horizon].
    Embargo: drop a further `embargo` observations after that, to kill leakage through
    serially correlated FEATURES - a separate channel that purging does not touch.
    """
    idx = np.arange(n)
    t1 = idx + horizon                                 # label resolution index
    for fold in np.array_split(idx, n_splits):
        start, end = fold[0], fold[-1]
        overlaps = (t1 >= start) & (idx <= end + horizon)
        embargoed = (idx > end + horizon) & (idx <= end + horizon + embargo)
        yield idx[~(overlaps | embargoed)], fold


def fold_aucs(x, y, splitter, score_fn=knn_scores) -> list[float]:
    """AUC per fold. Folds whose test block is single-class are undefined and skipped."""
    out = []
    for train, test in splitter:
        if len(train) == 0 or len(np.unique(y[train])) < 2:
            continue
        a = auc(y[test], score_fn(x[train], y[train], x[test]))
        if not np.isnan(a):
            out.append(a)
    return out


def summarise(vals: list[float]) -> tuple[float, float, int]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))), len(a)


# --------------------------------------------------------------------------
def main() -> None:
    x0, y0 = make_dataset(0)
    lag1 = float(np.corrcoef(y0[:-1], y0[1:])[0, 1])
    lagh = float(np.corrcoef(y0[:-H], y0[H:])[0, 1])

    print(f"Synthetic data: N={N} observations, label horizon H={H}, {N_FEATURES} AR(1)")
    print(f"features (phi={PHI}) drawn independently of the returns that make the labels.")
    print("There is no signal. The honest out-of-sample AUC is 0.500 exactly.\n")
    print(f"  label autocorrelation  lag 1   = {lag1:.4f}  <- consecutive labels share "
          f"{H - 1}/{H} bars")
    print(f"  label autocorrelation  lag {H} = {lagh:+.4f}  <- windows no longer overlap")
    print(f"  class balance          P(y=1)  = {y0.mean():.4f}      (all three: dataset "
          f"seed 0)")
    print(f"\nAveraged over {REPLICATIONS} independent datasets (seeds 0-"
          f"{REPLICATIONS - 1}); one draw is far too noisy to read.\n")

    schemes = {
        "KFold(shuffle=True)": lambda n, s: kfold(n, N_SPLITS, True, s),
        "KFold(shuffle=False)": lambda n, s: kfold(n, N_SPLITS),
        "TimeSeriesSplit": lambda n, s: time_series_split(n, N_SPLITS),
        f"Purged H={H}, no embargo": lambda n, s: purged_kfold(n, N_SPLITS, H, 0),
        f"Purged H={H} + embargo {EMBARGO}": lambda n, s: purged_kfold(n, N_SPLITS, H, EMBARGO),
    }
    acc = {k: [] for k in schemes}
    for seed in range(REPLICATIONS):
        x, y = make_dataset(seed)
        for name, mk in schemes.items():
            acc[name] += fold_aucs(x, y, mk(len(y), seed))

    stats = {k: summarise(v) for k, v in acc.items()}
    bar = "=" * 78
    print(bar)
    print(f"  {'splitter':<30}{'mean AUC':>10}{'std err':>10}{'skill (AUC-0.5)':>18}{'folds':>8}")
    print(bar)
    for name in schemes:
        m, s, n = stats[name]
        print(f"  {name:<30}{m:>10.4f}{s:>10.4f}{m - 0.5:>+18.4f}{n:>8}")
    print(bar)

    clean = stats[f"Purged H={H} + embargo {EMBARGO}"][0]
    for name in ("KFold(shuffle=False)", "TimeSeriesSplit", "KFold(shuffle=True)"):
        print(f"  {name:<30} inflates measured skill by "
              f"{stats[name][0] - clean:+.4f} AUC")
    print(f"\n  MEASURED GAP: plain KFold {stats['KFold(shuffle=False)'][0]:.4f} vs "
          f"purged+embargoed {clean:.4f}  =  "
          f"{stats['KFold(shuffle=False)'][0] - clean:+.4f} AUC of skill that is not there.")
    purge_only = stats[f"Purged H={H}, no embargo"][0]
    print(f"  Purging alone already lands at {purge_only:.4f}; the embargo moves it to "
          f"{clean:.4f},\n  a change of {clean - purge_only:+.4f} against a standard error of "
          f"{stats[f'Purged H={H} + embargo {EMBARGO}'][1]:.4f} -- i.e. NOT measurable here,\n"
          f"  because these features are independent of the labelled return series. The\n"
          f"  embargo earns its keep when features are ROLLING FUNCTIONS of that series.")

    # ---- verify the reference implementations against the real library ----
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import KFold, TimeSeriesSplit
    except ImportError:
        print("\n  scikit-learn not installed - reference implementations only; the gap "
              "above\n  is computed entirely from numpy/scipy and needs no library")
        return

    ref_kf = list(kfold(N, N_SPLITS))
    sk_kf = list(KFold(n_splits=N_SPLITS).split(x0))
    kf_same = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                  for a, b in zip(ref_kf, sk_kf))
    ref_ts = list(time_series_split(N, N_SPLITS))
    sk_ts = list(TimeSeriesSplit(n_splits=N_SPLITS).split(x0))
    ts_same = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                  for a, b in zip(ref_ts, sk_ts))
    ref_auc = auc(y0[:500], x0[:500, 0])
    sk_auc = float(roc_auc_score(y0[:500], x0[:500, 0]))

    print("\n" + bar)
    print("  VERIFIED against installed scikit-learn")
    print(bar)
    print(f"  reference KFold folds identical to sklearn KFold:              {kf_same}")
    print(f"  reference TimeSeriesSplit folds identical to sklearn:          {ts_same}")
    print(f"  reference AUC {ref_auc:.6f} vs roc_auc_score {sk_auc:.6f}, "
          f"|diff| = {abs(ref_auc - sk_auc):.2e}")

    def rf_scores(x_tr, y_tr, x_te):
        clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        clf.fit(x_tr, y_tr)
        return clf.predict_proba(x_te)[:, 1]

    reps = 6
    rf_acc = {"KFold(shuffle=False)": [], "TimeSeriesSplit": [],
              f"Purged + embargo {EMBARGO}": []}
    for seed in range(reps):
        x, y = make_dataset(seed)
        rf_acc["KFold(shuffle=False)"] += fold_aucs(
            x, y, KFold(n_splits=N_SPLITS).split(x), rf_scores)
        rf_acc["TimeSeriesSplit"] += fold_aucs(
            x, y, TimeSeriesSplit(n_splits=N_SPLITS).split(x), rf_scores)
        rf_acc[f"Purged + embargo {EMBARGO}"] += fold_aucs(
            x, y, purged_kfold(len(y), N_SPLITS, H, EMBARGO), rf_scores)

    print(f"\n  Not a k-NN artefact: RandomForestClassifier(n_estimators=100) on sklearn's")
    print(f"  own splitters, {reps} datasets --")
    rf_stats = {k: summarise(v) for k, v in rf_acc.items()}
    for name, (m, s, n) in rf_stats.items():
        print(f"    {name:<28} mean AUC {m:.4f} (se {s:.4f})   skill {m - 0.5:+.4f}")
    rf_gap = (rf_stats["KFold(shuffle=False)"][0]
              - rf_stats[f"Purged + embargo {EMBARGO}"][0])
    print(f"    gap KFold - purged           {rf_gap:+.4f} AUC")

    print("\n  Rule: state the label horizon and purge it, then embargo the feature")
    print("        lookback on top. purgedcv makes `evaluation_times` mandatory for")
    print("        exactly this reason - understating it purges nothing and the folds")
    print("        still look clean.")


if __name__ == "__main__":
    main()
