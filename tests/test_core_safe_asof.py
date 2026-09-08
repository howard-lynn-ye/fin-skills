"""fin_skills.core.safe_asof - the as-of join with the defaults inverted and invariants asserted."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.core.safe_asof import (RIGHT_TS_COL, assert_row_count_preserved,
                                       assert_strictly_prior, check_key_dtypes,
                                       safe_merge_asof)

ts = pd.Timestamp


@pytest.fixture
def signals() -> pd.DataFrame:
    return pd.DataFrame({"time": [ts("2024-01-02 09:30:01")], "symbol": ["AAA"], "signal": [1]})


@pytest.fixture
def quotes() -> pd.DataFrame:
    return pd.DataFrame({
        "time": [ts("2024-01-02 09:29:59"), ts("2024-01-02 09:30:01"), ts("2024-01-02 09:30:05")],
        "symbol": ["AAA", "AAA", "AAA"],
        "px": [100.0, 101.0, 102.0],
    })


def test_same_timestamp_quote_is_not_matched(signals, quotes):
    # pandas' default (allow_exact_matches=True) hands back the 09:30:01 quote; this does not
    naive = pd.merge_asof(signals, quotes, on="time", by="symbol")
    assert naive.px.iloc[0] == 101.0
    out = safe_merge_asof(signals, quotes, on="time", by="symbol", tolerance="5min")
    assert out.px.iloc[0] == 100.0
    assert out[RIGHT_TS_COL].iloc[0] == ts("2024-01-02 09:29:59")
    assert (out[RIGHT_TS_COL] < out["time"]).all()


def test_tolerance_is_mandatory_and_positive(signals, quotes):
    with pytest.raises(ValueError, match="tolerance is REQUIRED"):
        safe_merge_asof(signals, quotes, on="time", by="symbol")
    with pytest.raises(ValueError, match="positive"):
        safe_merge_asof(signals, quotes, on="time", by="symbol", tolerance="0min")


def test_refuses_forward_and_nearest_joins_unless_told_they_look_ahead(signals, quotes):
    for direction in ("forward", "nearest"):
        with pytest.raises(ValueError, match="looks into the future BY DESIGN"):
            safe_merge_asof(signals, quotes, on="time", by="symbol", tolerance="5min",
                            direction=direction)
    out = safe_merge_asof(signals, quotes, on="time", by="symbol", tolerance="5min",
                          direction="forward", require_prior=False)
    assert out.px.iloc[0] == 102.0                 # the next quote, by explicit request


def test_refuses_exact_matches_unless_prior_requirement_is_waived(signals, quotes):
    with pytest.raises(ValueError, match="contradicts require_prior"):
        safe_merge_asof(signals, quotes, on="time", by="symbol", tolerance="5min",
                        allow_exact_matches=True)
    out = safe_merge_asof(signals, quotes, on="time", by="symbol", tolerance="5min",
                          allow_exact_matches=True, require_prior=False)
    assert out.px.iloc[0] == 101.0


def test_stale_quote_beyond_tolerance_leaves_the_row_unmatched():
    late = pd.DataFrame({"time": [ts("2024-03-01 10:00")], "symbol": ["ZZZ"], "signal": [1]})
    stale = pd.DataFrame({"time": [ts("2024-01-05 15:59")], "symbol": ["ZZZ"], "px": [42.0]})
    assert pd.merge_asof(late, stale, on="time", by="symbol").px.iloc[0] == 42.0
    out = safe_merge_asof(late, stale, on="time", by="symbol", tolerance="3D")
    assert len(out) == 1
    assert np.isnan(out.px.iloc[0])
    assert pd.isna(out[RIGHT_TS_COL].iloc[0])


def test_row_count_is_preserved_when_a_symbol_has_no_quotes(quotes):
    multi = pd.DataFrame({
        "time": pd.to_datetime(["2024-01-02 09:31", "2024-01-02 09:31", "2024-01-02 09:32"]),
        "symbol": ["AAA", "BBB", "AAA"], "signal": [1, -1, 1],
    })
    inner = multi.merge(quotes, on="symbol", how="inner")
    with pytest.raises(AssertionError, match="row count changed"):
        assert_row_count_preserved(multi, inner, context="pd.merge")
    out = safe_merge_asof(multi, quotes, on="time", by="symbol", tolerance="5min")
    assert len(out) == len(multi)
    assert np.isnan(out.loc[out.symbol == "BBB", "px"].iloc[0])


def test_output_carries_left_index_labels_and_is_sorted_by_on(quotes):
    left = pd.DataFrame({"time": [ts("2024-01-02 09:31"), ts("2024-01-02 09:30:01")],
                         "symbol": ["AAA", "AAA"], "signal": [2, 1]}, index=["b", "a"])
    out = safe_merge_asof(left, quotes, on="time", by="symbol", tolerance="5min")
    assert list(out.index) == ["a", "b"]
    assert out["time"].is_monotonic_increasing
    assert out.px.tolist() == [100.0, 102.0]


def test_category_vs_object_key_is_refused_before_the_join(signals, quotes):
    cat = signals.assign(symbol=signals.symbol.astype("category"))
    with pytest.raises(TypeError, match="category-vs-object"):
        safe_merge_asof(cat, quotes, on="time", by="symbol", tolerance="5min")
    with pytest.raises(KeyError):
        check_key_dtypes(signals, quotes, "nope")


def test_null_keys_and_column_collisions_are_refused(signals, quotes):
    bad = quotes.copy()
    bad.loc[0, "time"] = pd.NaT
    with pytest.raises(ValueError, match="contains nulls"):
        safe_merge_asof(signals, bad, on="time", by="symbol", tolerance="5min")
    with pytest.raises(ValueError, match="already exists"):
        safe_merge_asof(signals.assign(**{RIGHT_TS_COL: 0}), quotes, on="time", by="symbol",
                        tolerance="5min")


def test_numeric_on_key_uses_a_numeric_tolerance():
    left = pd.DataFrame({"t": [5.0, 10.0], "v": [1, 2]})
    right = pd.DataFrame({"t": [4.0, 5.0, 9.5], "px": [1.0, 2.0, 3.0]})
    out = safe_merge_asof(left, right, on="t", tolerance=2.0)
    assert out.px.tolist() == [1.0, 3.0]           # 5.0 is an exact match and is skipped


def test_assert_strictly_prior_catches_a_hand_built_lookahead():
    out = pd.DataFrame({"time": [ts("2024-01-02 09:30")],
                        RIGHT_TS_COL: [ts("2024-01-02 09:30")]})
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_strictly_prior(out, "time")
    unmatched = pd.DataFrame({"time": [ts("2024-01-02 09:30")], RIGHT_TS_COL: [pd.NaT]})
    assert assert_strictly_prior(unmatched, "time") is None


def test_demo_prints_its_closing_rule(run_main):
    out = run_main("fin_skills.core.safe_asof")
    assert "Four defaults, four leaks" in out
