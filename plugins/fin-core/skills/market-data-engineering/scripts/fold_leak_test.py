"""Detect shared mutable state between walk-forward folds, before it becomes a paper.

WHY this exists: "we get different numbers when we parallelise the backtest" is almost
never a parallelism bug. It is a pre-existing leak that parallelism EXPOSED. If each fold
is a pure function of (fold, config), then running the folds serially, in a shuffled
order, or across a pool must give byte-identical results. If it does not, something
mutable is shared -- a scaler fitted outside the loop, a global feature cache, an
accumulating position book, or one RNG drawn from by every fold.

Two complementary checks, because neither is sufficient alone:

  assert_folds_independent  catches ORDER- and INTERLEAVING-dependent state: shared RNGs,
      accumulators, caches that are written as well as read. It cannot see state that is
      merely read, because reading it gives the same answer every time.

  find_shared_state         catches exactly that blind spot by inspecting the callable's
      closure -- and the module globals it actually references -- for captured mutable
      objects. A StandardScaler fitted on the full sample before the loop leaks the test
      set into every fold and is perfectly deterministic, so only the scan can find it.

Usage:
    from fold_leak_test import assert_folds_independent, find_shared_state
    find_shared_state(run_fold)                      # warns about captured mutables
    assert_folds_independent(run_fold, folds, cfg)   # raises if results are not stable
"""
from __future__ import annotations

import functools
import inspect
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from types import CodeType
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

FoldFn = Callable[[Any, Any], Any]

# Attribute pairs that identify a stateful, fit-then-apply estimator by duck typing.
# Matching on the API rather than on `import sklearn` means this also catches your own
# hand-rolled scalers, the ones nobody thinks to check.
_ESTIMATOR_APPLY = ("transform", "predict", "predict_proba", "score")
_MUTABLE_CONTAINERS = (dict, list, set, bytearray)
_RNG_TYPES = (np.random.Generator, np.random.RandomState, random.Random)


def _looks_fitted(obj: Any) -> bool:
    """sklearn convention: learned attributes end in a single trailing underscore."""
    return any(
        a.endswith("_") and not a.startswith("_") and not a.endswith("__")
        for a in vars(obj)
    ) if hasattr(obj, "__dict__") else False


def _classify(obj: Any) -> tuple[str, str] | None:
    """Return (kind, why) if `obj` is mutable state worth warning about, else None."""
    # Modules, classes and functions are referenced by every fold function alive and are
    # not per-run state. Excluding them is what keeps the global scan signal-only --
    # note a CLASS would otherwise duck-type as an estimator (it has .fit and .transform).
    if isinstance(obj, type) or inspect.ismodule(obj) or inspect.isroutine(obj):
        return None
    if isinstance(obj, _RNG_TYPES):
        return ("rng", "a shared random stream: draw order decides each fold's numbers, "
                       "so results depend on scheduling. Seed per fold instead.")
    if hasattr(obj, "fit") and any(hasattr(obj, a) for a in _ESTIMATOR_APPLY):
        fitted = " ALREADY FITTED --" if _looks_fitted(obj) else ""
        return ("estimator",
                f"a fit/apply estimator captured from outside the loop.{fitted} if it saw "
                f"data outside this fold's training window, every fold is contaminated. "
                f"Construct and fit it INSIDE run_fold.")
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return ("frame", "a mutable pandas object shared across folds. Safe only if never "
                         "written; an in-place assignment in one fold is seen by the rest.")
    if isinstance(obj, np.ndarray) and obj.flags.writeable:
        return ("array", "a writeable numpy array shared across folds. Set "
                         "arr.flags.writeable = False if it is meant to be constant.")
    if isinstance(obj, _MUTABLE_CONTAINERS):
        return ("container", f"a mutable {type(obj).__name__} captured by the closure -- "
                             f"the usual shape of a feature cache or an accumulator.")
    return None


def _referenced_globals(code: CodeType) -> set[str]:
    """Names this code object (and its nested lambdas/comprehensions) can read globally."""
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, CodeType):
            names |= _referenced_globals(const)
    return names


def find_shared_state(run_fold: FoldFn, warn: bool = True) -> list[dict[str, Any]]:
    """Inspect a fold function for captured mutable state: closure, defaults, globals.

    Catches the deterministic leaks that no re-run comparison can see -- above all the
    scaler, imputer or encoder fitted on the whole sample before the walk-forward loop.
    Globals are scanned too, and restricted to names the function's bytecode actually
    references: module level is where this state usually lives in real research code,
    and `co_names` keeps the scan from reporting the whole module namespace.
    """
    fn = run_fold
    findings: list[dict[str, Any]] = []
    if isinstance(fn, functools.partial):
        for i, arg in enumerate(fn.args):
            _collect(findings, f"partial:arg[{i}]", arg)
        for k, v in fn.keywords.items():
            _collect(findings, f"partial:{k}", v)
        fn = fn.func

    code = getattr(fn, "__code__", None)
    cells = getattr(fn, "__closure__", None) or ()
    for name, cell in zip(getattr(code, "co_freevars", ()) or (), cells):
        try:
            value = cell.cell_contents
        except ValueError:  # cell not yet populated (recursive definition)
            continue
        _collect(findings, f"closure:{name}", value)

    for name, value in (getattr(fn, "__kwdefaults__", None) or {}).items():
        _collect(findings, f"default:{name}", value)
    for i, value in enumerate(getattr(fn, "__defaults__", None) or ()):
        _collect(findings, f"default[{i}]", value)

    g = getattr(fn, "__globals__", {}) or {}
    if code is not None:
        for name in sorted(_referenced_globals(code) & g.keys()):
            _collect(findings, f"global:{name}", g[name])

    if warn:
        for f in findings:
            warnings.warn(
                f"shared state in {getattr(fn, '__name__', fn)!r}: "
                f"{f['name']} ({f['type']}) -- {f['why']}",
                stacklevel=2,
            )
    return findings


def _collect(out: list[dict[str, Any]], name: str, value: Any) -> None:
    hit = _classify(value)
    if hit is not None:
        kind, why = hit
        out.append({"name": name, "kind": kind, "type": type(value).__name__, "why": why})


def _equal(a: Any, b: Any, tol: float) -> bool:
    """Structural comparison that copes with the things folds actually return."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (pd.Series, pd.DataFrame)) or isinstance(b, (pd.Series, pd.DataFrame)):
        try:
            (pd.testing.assert_series_equal if isinstance(a, pd.Series)
             else pd.testing.assert_frame_equal)(a, b, atol=tol, rtol=tol)
            return True
        except (AssertionError, TypeError):
            return False
    if isinstance(a, (int, float, np.number, np.ndarray)) and not isinstance(a, bool):
        return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                                atol=tol, rtol=tol, equal_nan=True))
    return bool(a == b)


def _first_difference(a: Sequence[Any], b: Sequence[Any], tol: float) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if not _equal(x, y, tol):
            return i
    return None


def assert_folds_independent(
    run_fold: FoldFn,
    folds: Sequence[Any],
    config: Any,
    tol: float = 0.0,
    workers: int = 4,
    seed: int = 0,
) -> list[Any]:
    """Raise AssertionError unless run_fold(fold, config) is a pure function of its args.

    Runs the same folds three ways and demands identical results:
      1. serially, in the given order        -- the baseline everyone reports
      2. serially, in a SHUFFLED order       -- deterministic detector of order-dependent
                                                state (shared RNG, accumulator, cache)
      3. across a thread pool                -- detector of interleaving-dependent state

    Threads, not processes, on purpose. A process pool gives every worker its own COPY of
    the closure, which HIDES exactly the shared-object leak we are hunting (and on Windows
    a closure will not pickle at all). Threads share the interpreter's memory, so if two
    folds touch the same object the result moves.

    Returns the serial results, so this can wrap a real run rather than duplicate it.
    """
    if len(folds) < 2:
        raise ValueError("need at least 2 folds to compare")

    serial = [run_fold(f, config) for f in folds]

    order = list(range(len(folds)))
    random.Random(seed).shuffle(order)
    if order == list(range(len(folds))):  # tiny fold counts can shuffle to identity
        order.reverse()
    shuffled_raw = [run_fold(folds[i], config) for i in order]
    shuffled = [None] * len(folds)
    for slot, res in zip(order, shuffled_raw):
        shuffled[slot] = res

    with ThreadPoolExecutor(max_workers=min(workers, len(folds))) as ex:
        pooled = list(ex.map(lambda f: run_fold(f, config), folds))

    for label, other in (("SHUFFLED-ORDER serial", shuffled), ("THREAD-POOL", pooled)):
        i = _first_difference(serial, other, tol)
        if i is None:
            continue
        suspects = find_shared_state(run_fold, warn=False)
        causes = ("\n  suspected cause(s) reachable from the fold function:\n"
                  + "\n".join(f"    - {s['name']} ({s['type']}): {s['why']}"
                              for s in suspects)) if suspects else (
                  "\n  nothing mutable in the closure, defaults or referenced globals -- "
                  "look at a process-wide cache, an imported module's state, or a "
                  "file/DB the folds both write.")
        raise AssertionError(
            f"FOLD LEAK: {label} results differ from the in-order serial run.\n"
            f"  first divergent fold index {i}: serial={serial[i]!r} vs {other[i]!r}\n"
            f"  each fold must be a pure function of (fold, config). It is not."
            f"{causes}"
        )
    return serial


def report(run_fold: FoldFn, folds: Sequence[Any], config: Any) -> str:
    """One-line pass/fail plus the closure scan. Safe to call in CI."""
    lines = [f"fold-leak check: {getattr(run_fold, '__name__', run_fold)}"]
    for s in find_shared_state(run_fold, warn=False):
        lines.append(f"  [scan   ] {s['name']} ({s['type']}): {s['why']}")
    try:
        assert_folds_independent(run_fold, folds, config)
        lines.append("  [reruns ] PASS - serial, shuffled and pooled runs agree")
    except AssertionError as e:
        lines.append("  [reruns ] FAIL - " + str(e).replace("\n", "\n  "))
    return "\n".join(lines)


if __name__ == "__main__":
    # A hand-rolled, sklearn-shaped scaler: fit/transform plus a trailing-underscore
    # learned attribute. Duck-typed on purpose -- keeps this file numpy/pandas-only while
    # tripping exactly the same detector a real StandardScaler would.
    class Scaler:
        def fit(self, x: np.ndarray) -> "Scaler":
            self.mean_ = float(np.mean(x))
            self.scale_ = float(np.std(x)) or 1.0
            return self

        def transform(self, x: np.ndarray) -> np.ndarray:
            return (x - self.mean_) / self.scale_

    DATA = np.random.default_rng(42).normal(0, 1, 1200)
    DATA.flags.writeable = False   # shared INPUT data is fine -- once it cannot be written
    FOLDS = [(i * 200, (i + 1) * 200) for i in range(6)]
    CONFIG = {"n_boot": 8}

    # ------------------------------------------------------------------- THE LEAK
    # Two textbook mistakes, one function. Both are invisible in a single serial run.
    shared_scaler = Scaler().fit(DATA)          # fitted on ALL folds, test set included
    shared_rng = np.random.default_rng(0)       # one stream consumed by every fold

    def run_fold_leaky(fold: tuple[int, int], config: dict) -> float:
        lo, hi = fold
        z = shared_scaler.transform(DATA[lo:hi])
        boot = shared_rng.integers(0, len(z), size=(config["n_boot"], len(z)))
        return float(np.mean(z[boot]))

    # ------------------------------------------------------------------- THE FIX
    def run_fold_clean(fold: tuple[int, int], config: dict) -> float:
        lo, hi = fold
        z = Scaler().fit(DATA[lo:hi]).transform(DATA[lo:hi])   # fitted in-fold only
        rng = np.random.default_rng(lo)                        # seed derived from the fold
        boot = rng.integers(0, len(z), size=(config["n_boot"], len(z)))
        return float(np.mean(z[boot]))

    print("=" * 78)
    print("CLEAN fold function")
    print("=" * 78)
    print(report(run_fold_clean, FOLDS, CONFIG))

    print("\n" + "=" * 78)
    print("LEAKY fold function")
    print("=" * 78)
    print(report(run_fold_leaky, FOLDS, CONFIG))

    print("\n" + "=" * 78)
    print("Note the division of labour: the RE-RUN test caught the shared RNG, but the")
    print("full-sample scaler is perfectly deterministic -- it returns the same numbers")
    print("every time -- so only the STATE SCAN could see it. Run both. The deterministic")
    print("leak is the one that survives review and gets published.")
    print("=" * 78)
