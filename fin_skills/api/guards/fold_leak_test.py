"""Guard: shared mutable state between folds (market-data-engineering / fold_leak_test.py)."""
from __future__ import annotations

from typing import Any, Callable, Sequence

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_callable
from fin_skills.market_data.fold_leak_test import assert_folds_independent, find_shared_state


@register
class FoldLeakGuard(Guard):
    """Each fold must be a pure function of (fold, config); two checks, neither sufficient alone.

    Inputs
        run_fold : callable (fold, config) -> result.
        folds    : the sequence of folds (at least 2).
        config   : passed through unchanged. Default None.
        tol      : numeric tolerance when comparing fold results. Default 0.0.
        workers  : thread-pool size for the interleaving test. Default 4.
        seed     : seed for the shuffled order. Default 0.

    Fails when the serial, shuffled and thread-pooled runs disagree (order- or
    interleaving-dependent state), or when the closure/defaults/globals scan finds a
    shared random stream or an already-fitted estimator - the deterministic leak no
    re-run can see. Other captured mutables (frames, arrays, containers) are warnings:
    safe only if never written.
    """

    name = "fold_leak_test"
    skill = "market-data-engineering"
    summary = "Detects state shared across walk-forward folds: re-run equivalence plus a closure scan."
    wraps = ("fin_skills.market_data.fold_leak_test.assert_folds_independent",
             "fin_skills.market_data.fold_leak_test.find_shared_state")
    required = ("run_fold", "folds")
    optional = ("config", "tol", "workers", "seed")

    def check(self, run_fold: Callable[[Any, Any], Any], folds: Sequence[Any],
              config: Any = None, tol: float = 0.0, workers: int = 4,
              seed: int = 0) -> Outcome:
        out = Outcome()
        run_fold = require_callable(run_fold, "run_fold")
        folds = list(folds)
        if len(folds) < 2:
            raise TypeError("folds needs at least 2 entries to compare")

        scan = find_shared_state(run_fold, warn=False)
        out.note(shared_state=scan)
        for s in scan:
            msg = f"{s['name']} ({s['type']}): {s['why']}"
            if s["kind"] == "rng":
                out.error(msg, where="scan")
            elif s["kind"] == "estimator":
                (out.error if "ALREADY FITTED" in s["why"] else out.warning)(msg, where="scan")
            else:
                out.warning(msg, where="scan")

        try:
            results = assert_folds_independent(run_fold, folds, config, tol=tol,
                                               workers=workers, seed=seed)
        except AssertionError as exc:
            out.error(str(exc), where="reruns")
        else:
            out.note(results=results)
            out.info("serial, shuffled-order and thread-pool runs agree", where="reruns")
        return out
