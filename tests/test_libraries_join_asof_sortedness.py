"""fin_skills.libraries.join_asof_sortedness - an unsorted left frame matches quotes from the future."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import has_module, requires
from fin_skills.libraries.join_asof_sortedness import (asof_backward_correct,
                                                       asof_backward_cursor, demo, panel,
                                                       tiny_case)


def test_tiny_case_the_cursor_cannot_rewind():
    sig, qt = tiny_case()
    correct = asof_backward_correct(sig, qt, "time", "symbol", "px")
    cursor = asof_backward_cursor(sig, qt, "time", "symbol", "px")
    assert correct.tolist()[:3] == [12.0, 12.0, 21.0] and np.isnan(correct.iloc[3])
    assert cursor.tolist() == [12.0, 12.0, 21.0, 21.0]     # B@t=2 got the quote from t=5
    assert int(sum(a != b and not (np.isnan(a) and np.isnan(b))
                   for a, b in zip(cursor, correct))) == 1


def test_on_sorted_input_the_cursor_is_the_correct_join():
    sig, qt = tiny_case()
    srt = sig.sort_values(["symbol", "time"]).reset_index(drop=True)
    a = asof_backward_cursor(srt, qt, "time", "symbol", "px")
    b = asof_backward_correct(srt, qt, "time", "symbol", "px")
    pd.testing.assert_series_equal(a, b)


def test_correct_join_is_order_independent_at_scale():
    sig, qt = panel(seed=0)
    shuffled = sig.sample(frac=1.0, random_state=0)
    a = asof_backward_correct(sig, qt, "time", "symbol", "px")
    b = asof_backward_correct(shuffled, qt, "time", "symbol", "px").sort_index()
    pd.testing.assert_series_equal(a, b)
    c = asof_backward_cursor(shuffled.reset_index(drop=True), qt, "time", "symbol", "px")
    assert (c.to_numpy() != b.reset_index(drop=True).to_numpy()).sum() > 0.5 * len(sig)


def test_panel_and_demo_are_seeded():
    a, b = panel(seed=3), panel(seed=3)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
    assert not panel(seed=4)[0].equals(a[0])
    r1, r2 = demo(seed=1), demo(seed=1)
    assert r1["tiny_diff"] == r2["tiny_diff"] == 1
    assert r1["panel_diff"] == r2["panel_diff"] > 0 and r1["n_rows"] == 450


@pytest.mark.skipif(has_module("polars"), reason="polars is installed")
def test_demo_has_no_live_keys_without_polars():
    assert not any(k.startswith("LIVE_") for k in demo())


@requires("polars")
def test_installed_polars_returns_future_quotes_on_unsorted_by_joins():
    r = demo()
    assert r["LIVE_cursor_matches"] is True            # polars reproduces the cursor model
    assert r["LIVE_correct_matches"] is True           # sorted input is right
    assert r["LIVE_lookahead_bad"] > 0 and r["LIVE_lookahead_ok"] == 0
    assert r["LIVE_height_ok"] is True                 # row count is no guard
    assert r["LIVE_heights"] == (450, 450)
    matrix = dict(r["LIVE_matrix"])
    for label, behaviour in matrix.items():
        if label.startswith("NO by="):
            assert behaviour.startswith("RAISES")
        else:
            assert behaviour.startswith("returns rows")
    assert any("Sortedness" in w for w in r["LIVE_warnings"])


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.libraries.join_asof_sortedness")
    assert "Rule: polars does NOT raise on unsorted input" in out
