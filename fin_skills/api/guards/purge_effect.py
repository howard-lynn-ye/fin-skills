"""Guard: overlapping labels across CV folds (lib-purgedcv / purge_effect.py).

purge_effect.py is a demonstration, not a library: its functions build the synthetic
data, score folds, and implement the reference splitters. The smallest sensible check
is therefore (a) the purge criterion from `purged_kfold`, applied to the caller's own
folds - a training row whose label span [i, i+horizon] touches the test block's label
span is a leak - and (b) the script's own measurement: AUC under the caller's splits
against AUC under a purged split on the same data, via `fold_aucs`.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import as_1d, as_2d, require_int
from fin_skills.libraries.purge_effect import (N_SPLITS, fold_aucs, kfold, knn_scores,
                                               purged_kfold)


@register
class PurgeGuard(Guard):
    """Plain KFold on overlapping labels manufactures skill; purge the horizon, then embargo.

    Inputs
        x, y     : features (n, p) and binary labels (n,), where y[i] resolves at i+horizon.
        horizon  : label horizon in bars (the evaluation_times purgedcv makes mandatory).
        splits   : optional iterable of (train_idx, test_idx) from YOUR splitter, e.g.
                   list(KFold(5).split(x)). Without it the guard scores plain KFold.
        n_splits : folds for the reference splitters. Default 5.
        embargo  : bars dropped after each test block on top of the purge. Default 0.
        score_fn : (x_tr, y_tr, x_te) -> scores. Default the script's k-NN.
        auc_tol  : AUC inflation above the purged reference that earns a warning.
                   Default 0.01.

    Fails when any of your training rows carries a label overlapping the test block's
    label span (purge violation). Rows inside the embargo window are warnings. The AUC
    comparison is corroboration: your splits scoring above the purged reference by more
    than `auc_tol` is reported as a warning, since a single draw is noisy.
    """

    name = "purge_effect"
    skill = "lib-purgedcv"
    summary = "Checks your CV folds for label overlap (purge) and measures the AUC they manufacture."
    wraps = ("fin_skills.libraries.purge_effect.purged_kfold",
             "fin_skills.libraries.purge_effect.fold_aucs",
             "fin_skills.libraries.purge_effect.kfold",
             "fin_skills.libraries.purge_effect.knn_scores")
    required = ("x", "y", "horizon")
    optional = ("splits", "n_splits", "embargo", "score_fn", "auc_tol")

    def check(self, x: Any, y: Any, horizon: int,
              splits: Iterable[tuple[Sequence[int], Sequence[int]]] | None = None,
              n_splits: int = N_SPLITS, embargo: int = 0,
              score_fn: Callable[..., np.ndarray] | None = None, auc_tol: float = 0.01) -> Outcome:
        out = Outcome()
        xa = as_2d(x, "x")
        ya = as_1d(y, "y")
        n = len(ya)
        if xa.shape[0] != n:
            raise TypeError(f"x has {xa.shape[0]} rows, y has {n}")
        if not set(np.unique(ya)).issubset({0.0, 1.0}):
            raise TypeError("y must be binary (0/1) labels")
        ya = ya.astype(int)
        horizon = require_int(horizon, "horizon", minimum=1)
        n_splits = require_int(n_splits, "n_splits", minimum=2)
        embargo = require_int(embargo, "embargo", minimum=0)
        score = knn_scores if score_fn is None else score_fn
        if not callable(score):
            raise TypeError("score_fn must be callable")

        user_label = "your splits"
        if splits is not None:
            folds = []
            for i, pair in enumerate(splits):
                if len(pair) != 2:
                    raise TypeError(f"splits[{i}] must be a (train_idx, test_idx) pair")
                tr, te = (np.asarray(pair[0], dtype=int), np.asarray(pair[1], dtype=int))
                if tr.size == 0 or te.size == 0 or tr.max() >= n or te.max() >= n:
                    raise TypeError(f"splits[{i}] has empty or out-of-range indices")
                folds.append((tr, te))
            if len(folds) < 2:
                raise TypeError("splits needs at least two folds")
            n_over_total = n_emb_total = 0
            for i, (tr, te) in enumerate(folds):
                start, end = int(te.min()), int(te.max())
                overlap = ((tr + horizon) >= start) & (tr <= end + horizon)
                n_over = int(overlap.sum())
                n_over_total += n_over
                if n_over:
                    out.error(f"{n_over} training row(s) carry labels overlapping the test "
                              f"block [{start}, {end}] + horizon {horizon} (purge violation)",
                              where=f"fold {i}")
                if embargo:
                    emb = (tr > end + horizon) & (tr <= end + horizon + embargo)
                    n_emb = int(emb.sum())
                    n_emb_total += n_emb
                    if n_emb:
                        out.warning(f"{n_emb} training row(s) sit inside the {embargo}-bar "
                                    f"embargo after the test block", where=f"fold {i}")
            out.note(n_overlapping_rows=n_over_total, n_embargo_rows=n_emb_total,
                     n_folds=len(folds))
            if n_over_total == 0:
                out.info(f"no training row overlaps a test label span (horizon {horizon})",
                         where="purge")
        else:
            folds = list(kfold(n, n_splits))
            user_label = f"plain KFold({n_splits})"
            out.note(n_folds=len(folds))

        user_aucs = fold_aucs(xa, ya, folds, score)
        ref_aucs = fold_aucs(xa, ya, purged_kfold(n, n_splits, horizon, embargo), score)
        if not user_aucs or not ref_aucs:
            raise TypeError("could not score any fold (single-class test blocks?)")
        inflation = float(np.mean(user_aucs) - np.mean(ref_aucs))
        out.note(auc_user=float(np.mean(user_aucs)), auc_purged=float(np.mean(ref_aucs)),
                 auc_inflation=inflation, user_label=user_label, fold_aucs_user=user_aucs,
                 fold_aucs_purged=ref_aucs)
        msg = (f"{user_label} mean AUC {np.mean(user_aucs):.4f} vs purged"
               f"{'+embargo' if embargo else ''} {np.mean(ref_aucs):.4f} "
               f"({inflation:+.4f})")
        if inflation > auc_tol:
            out.warning(msg + f": skill above auc_tol={auc_tol:g} that a purged split does not see",
                        where="auc")
        else:
            out.info(msg, where="auc")
        return out
