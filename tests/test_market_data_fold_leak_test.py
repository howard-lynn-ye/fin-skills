"""fin_skills.market_data.fold_leak_test - shared mutable state between walk-forward folds."""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from fin_skills.market_data.fold_leak_test import (_equal, assert_folds_independent,
                                            find_shared_state, report)

DATA = np.random.default_rng(42).normal(0, 1, 1200)
DATA.flags.writeable = False
FOLDS = [(i * 200, (i + 1) * 200) for i in range(6)]
CONFIG = {"n_boot": 8}


class Scaler:
    """sklearn-shaped fit/transform with a trailing-underscore learned attribute."""

    def fit(self, x):
        self.mean_ = float(np.mean(x))
        self.scale_ = float(np.std(x)) or 1.0
        return self

    def transform(self, x):
        return (x - self.mean_) / self.scale_


def clean_fold(fold, config):
    lo, hi = fold
    z = Scaler().fit(DATA[lo:hi]).transform(DATA[lo:hi])
    boot = np.random.default_rng(lo).integers(0, len(z), size=(config["n_boot"], len(z)))
    return float(np.mean(z[boot]))


def test_clean_fold_function_passes_and_returns_the_serial_results():
    out = assert_folds_independent(clean_fold, FOLDS, CONFIG)
    assert out == [clean_fold(f, CONFIG) for f in FOLDS]
    assert find_shared_state(clean_fold, warn=False) == []


def test_shared_rng_is_caught_by_the_rerun_and_named_as_the_cause():
    shared_rng = np.random.default_rng(0)

    def leaky(fold, config):
        lo, hi = fold
        z = DATA[lo:hi]
        boot = shared_rng.integers(0, len(z), size=(config["n_boot"], len(z)))
        return float(np.mean(z[boot]))

    with pytest.raises(AssertionError, match="FOLD LEAK") as exc:
        assert_folds_independent(leaky, FOLDS, CONFIG)
    assert "closure:shared_rng (Generator)" in str(exc.value)


def test_accumulator_makes_results_order_dependent():
    seen = []

    def accumulating(fold, config):
        seen.append(fold)
        return len(seen)

    with pytest.raises(AssertionError, match="SHUFFLED-ORDER"):
        assert_folds_independent(accumulating, FOLDS, CONFIG)


def test_full_sample_scaler_is_deterministic_so_only_the_scan_sees_it():
    fitted = Scaler().fit(DATA)                     # saw every fold, including test sets

    def deterministic_leak(fold, config):
        lo, hi = fold
        return float(np.mean(fitted.transform(DATA[lo:hi])))

    assert_folds_independent(deterministic_leak, FOLDS, CONFIG)   # reruns cannot see it
    hits = find_shared_state(deterministic_leak, warn=False)
    assert [h["kind"] for h in hits] == ["estimator"]
    assert "ALREADY FITTED" in hits[0]["why"]


def test_scan_classifies_captured_objects():
    frame, arr, frozen, cache = pd.DataFrame({"a": [1]}), np.zeros(3), np.zeros(3), {}
    frozen.flags.writeable = False

    def fn(fold, config, default_list=[]):        # noqa: B006 - the mutable default is the point
        return (frame, arr, frozen, cache, default_list, np, Scaler, clean_fold)

    kinds = {h["name"]: h["kind"] for h in find_shared_state(fn, warn=False)}
    assert kinds == {"closure:frame": "frame", "closure:arr": "array",
                     "closure:cache": "container", "default[0]": "container"}
    # modules, classes and functions are never per-run state; a read-only array is fine


def test_scan_walks_into_partials_and_referenced_globals():
    rng = np.random.default_rng(1)
    fn = functools.partial(clean_fold, rng)
    names = [h["name"] for h in find_shared_state(fn, warn=False)]
    assert names == ["partial:arg[0]"]
    # DATA is a referenced global but is read-only, so it is not reported
    assert find_shared_state(clean_fold, warn=False) == []


def test_scan_warns_by_default():
    shared = {}

    def fn(fold, config):
        return shared.setdefault(fold, 1)

    with pytest.warns(UserWarning, match="shared state in 'fn'"):
        find_shared_state(fn)


def test_equal_handles_frames_nans_and_tolerances():
    a = pd.DataFrame({"x": [1.0, np.nan]})
    assert _equal(a, a.copy(), 0.0)
    assert _equal(np.array([1.0, np.nan]), np.array([1.0, np.nan]), 0.0)
    assert _equal(1.0, 1.0 + 1e-9, 1e-6) and not _equal(1.0, 1.1, 1e-6)
    assert _equal({"k": [1, 2]}, {"k": [1, 2]}, 0.0) and not _equal({"k": 1}, {"j": 1}, 0.0)


def test_report_and_minimum_fold_count():
    assert "[reruns ] PASS" in report(clean_fold, FOLDS, CONFIG)
    seen = []

    def accumulating(fold, config):
        seen.append(fold)
        return len(seen)

    assert "[reruns ] FAIL" in report(accumulating, FOLDS, CONFIG)
    with pytest.raises(ValueError, match="at least 2 folds"):
        assert_folds_independent(clean_fold, FOLDS[:1], CONFIG)


def test_demo_shows_both_detectors(run_main):
    out = run_main("fin_skills.market_data.fold_leak_test")
    assert "[reruns ] PASS" in out and "[reruns ] FAIL" in out and "[scan   ]" in out
