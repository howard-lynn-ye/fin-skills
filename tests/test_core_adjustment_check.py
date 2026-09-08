"""fin_skills.core.adjustment_check - raw / back- / forward-adjusted detection and reconciliation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.core.adjustment_check import (_synthetic, detect_convention,
                                              detect_convention_from_vintages, reconcile,
                                              reconcile_report)


@pytest.fixture(scope="module")
def variants():
    return _synthetic()


def test_synthetic_is_deterministic_for_a_seed():
    a = _synthetic(seed=3)
    b = _synthetic(seed=3)
    for x, y in zip(a[:3], b[:3]):
        pd.testing.assert_series_equal(x, y)
    assert not _synthetic(seed=4)[0].equals(a[0])


def test_raw_series_shows_the_split_jump(variants):
    raw, _, _, splits = variants
    d = detect_convention(raw, splits)
    assert d["convention"] == "raw"
    assert d["confidence"] == "high"
    assert d["cum_factor"] == 21.0                # 7:1 then 3:1


def test_adjusted_variants_are_told_apart_by_tick_alignment(variants):
    _, back, fwd, splits = variants
    assert detect_convention(back, splits)["convention"] == "back-adjusted"
    fwd_d = detect_convention(fwd, splits)
    assert fwd_d["convention"] == "forward-adjusted"
    assert fwd_d["confidence"] == "medium"        # inferred, not measured


def test_a_raw_reference_upgrades_the_anchor_test_to_high_confidence(variants):
    raw, back, fwd, splits = variants
    for series, want in ((back, "back-adjusted"), (fwd, "forward-adjusted")):
        d = detect_convention(series, splits, raw_reference=raw)
        assert (d["convention"], d["confidence"]) == (want, "high")


def test_no_action_inside_the_sample_is_unknown(variants):
    raw, _, _, splits = variants
    d = detect_convention(raw.iloc[:50], splits)
    assert d["convention"] == "unknown"


def test_two_vintage_test_is_definitive(variants):
    _, back, fwd, _ = variants
    new_date = back.index[-1]
    v = detect_convention_from_vintages(back, back / 2.0, new_date)
    assert v["convention"] == "back-adjusted"
    assert v["history_rewritten"] is True
    assert v["implied_factor"] == pytest.approx(0.5)
    v = detect_convention_from_vintages(fwd, fwd.copy(), new_date)
    assert v["convention"] == "forward-adjusted"
    assert v["history_rewritten"] is False
    assert detect_convention_from_vintages(fwd, fwd, fwd.index[0])["convention"] == "unknown"


def test_reconcile_attributes_steps_to_corporate_actions(variants):
    raw, back, _, splits = variants
    r = reconcile(raw, back, splits)
    assert r["verdict"].startswith("ADJUSTMENT DIFFERENCE")
    assert len(r["steps"]) == 2
    assert r["n_unexplained_steps"] == 0
    assert all(r["steps"]["attribution"].str.startswith("split"))


def test_back_and_forward_adjusted_differ_by_one_global_constant(variants):
    _, back, fwd, splits = variants
    r = reconcile(back, fwd, splits)
    assert r["verdict"].startswith("ADJUSTMENT CONVENTION ONLY")
    assert r["max_abs_return_diff"] < 1e-12        # identical returns
    assert len(r["steps"]) == 0
    ratio = (back / fwd).dropna()
    assert ratio.max() - ratio.min() < 1e-12


def test_a_bad_tick_is_a_data_error_not_a_convention(variants):
    _, back, _, splits = variants
    corrupt = back.copy()
    corrupt.iloc[400] *= 1.35
    r = reconcile(back, corrupt, splits)
    assert r["verdict"].startswith("DATA ERROR")
    assert r["n_unexplained_steps"] == 2          # the print, and the step straight back
    assert "UNEXPLAINED" in reconcile_report(r)


def test_identical_and_non_overlapping_inputs(variants):
    _, back, _, splits = variants
    assert reconcile(back, back, splits)["verdict"].startswith("IDENTICAL")
    other = pd.Series(1.0, index=pd.bdate_range("1990-01-01", periods=10))
    assert reconcile(back, other)["verdict"] == "NO OVERLAP"


def test_report_carries_the_verdict(variants):
    raw, back, _, splits = variants
    text = reconcile_report(reconcile(raw, back, splits))
    assert "VERDICT" in text and "ADJUSTMENT DIFFERENCE" in text


def test_demo_runs_and_states_the_conclusion(run_main):
    out = run_main("fin_skills.core.adjustment_check")
    assert "Same tool, three answers" in out
