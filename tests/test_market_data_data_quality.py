"""Delivered daily-bar contracts: actual defects, no imputation and clock sensitivity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.market_data.data_quality import (annualised_vol, clean_panel,
    delivered_panel, extreme_returns, fill_cost, mechanism_table, missing_sessions,
    stale_mask, stale_runs, validate_panel, vol_bias)


def test_clean_delivery_passes_without_mutation():
    sessions, bars = clean_panel()
    original = bars.copy(deep=True)
    report = validate_panel(bars, sessions)
    assert report.passed and not report.errors and not report.warnings
    pd.testing.assert_frame_equal(bars, original)


def test_missing_sessions_report_the_exact_holes_without_filling():
    sessions, bars = clean_panel()
    holes = sessions[[2, 3, 25]]
    delivered = bars.drop(index=holes)
    before = delivered.copy(deep=True)
    report = validate_panel(delivered, sessions)
    pd.testing.assert_index_equal(missing_sessions(delivered.index, sessions), holes)
    assert not report.passed and report.get("missing_sessions").count == len(holes)
    pd.testing.assert_frame_equal(delivered, before)


def test_no_calendar_is_a_warning_not_a_claim_of_coverage():
    _, bars = clean_panel()
    report = validate_panel(bars)
    assert report.passed
    assert report.get("calendar").severity == "warning"
    assert report.get("missing_sessions") is None


def test_delivered_demo_exercises_each_planted_defect_without_crashing():
    sessions, bars = delivered_panel()
    report = validate_panel(bars, sessions)
    assert not report.passed
    for name in ("duplicate_rows", "non_positive_prices", "ohlc_ordering", "zero_volume",
                 "stale_closes", "extreme_returns", "missing_sessions", "non_session_rows"):
        assert report.get(name).count > 0, name


@pytest.mark.parametrize("column,value,check", [
    ("close", np.nan, "nan_values"), ("open", np.inf, "infinite_values"),
    ("low", -1., "non_positive_prices"), ("volume", -1., "negative_volume"),
    ("high", 0.01, "ohlc_ordering")])
def test_defect_values_are_reported(column, value, check):
    sessions, bars = clean_panel(n_sessions=40)
    bars.loc[sessions[3], column] = value
    report = validate_panel(bars, sessions)
    assert not report.passed and report.get(check).count >= 1


def test_zero_volume_is_distinct_from_negative_volume():
    sessions, bars = clean_panel(n_sessions=40)
    bars.loc[sessions[3], "volume"] = 0
    report = validate_panel(bars, sessions)
    assert report.passed and report.get("zero_volume").severity == "warning"
    assert report.get("negative_volume").count == 0
    assert validate_panel(bars.drop(columns="volume"), sessions, require_volume=False).passed


def test_malformed_schema_index_and_sorting_have_readable_failures():
    sessions, bars = clean_panel(n_sessions=40)
    assert validate_panel(bars.drop(columns="low")).get("schema").severity == "error"
    assert validate_panel(bars.reset_index(drop=True)).get("index_type").severity == "error"
    assert validate_panel(bars.iloc[::-1]).get("sorted").severity == "error"
    assert validate_panel(bars.assign(close="bad")).get("dtypes").severity == "error"
    duplicate = pd.concat([bars, bars[["close"]]], axis=1)
    assert validate_panel(duplicate).get("schema").severity == "error"
    bad_index = bars.copy()
    bad_index.index = pd.DatetimeIndex([pd.NaT, *sessions[1:]])
    assert validate_panel(bad_index).get("missing_timestamp").severity == "error"


def test_constant_and_empty_series_do_not_divide_by_zero():
    close = pd.Series([20.] * 5)
    assert annualised_vol(close) == 0
    assert np.isnan(vol_bias(close)["ratio_dropped_over_retained"])
    assert stale_mask(pd.Series([], dtype=float)).size == 0
    assert stale_runs(pd.Series([], dtype=float)).empty
    sessions, bars = clean_panel(n_sessions=40)
    bars[["open", "high", "low", "close"]] = 20.
    assert not validate_panel(bars, sessions).passed


def test_stale_runs_count_repeats_excluding_first_print_and_nan():
    close = pd.Series([1., 1., 1., 2., np.nan, np.nan, 2., 2.])
    assert stale_runs(close)["length"].tolist() == [2, 1]
    assert stale_mask(close).sum() == 3


def test_stale_detection_accepts_copy_on_write_series_without_mutating_input():
    with pd.option_context('mode.copy_on_write', True):
        close = pd.Series([1., 1., 2., 2.])
        original = close.copy(deep=True)
        assert stale_mask(close).tolist() == [False, True, False, True]
        assert stale_runs(close)['length'].tolist() == [1, 1]
        pd.testing.assert_series_equal(close, original)


def test_stale_filtering_changes_clock_and_does_not_recover_latent_volatility():
    table = mechanism_table().set_index("mechanism")
    assert table.loc["deferred", "vol_retained"] == table.loc["fabricated", "vol_retained"]
    assert table.loc["deferred", "vol_dropped"] == table.loc["fabricated", "vol_dropped"]
    assert table.loc["deferred", "vol_dropped"] > table.loc["deferred", "vol_truth"]
    assert table.loc["fabricated", "vol_dropped"] == table.loc["fabricated", "vol_truth"]
    assert table.loc["deferred", "dropped_over_retained"] > 1


def test_absolute_return_cap_is_symmetric_on_simple_returns():
    close = pd.Series([100., 70., 100., 140.])
    flags = extreme_returns(close, cap=0.35, sigma=1000.)
    assert flags["date"].tolist() == [2, 3]
    assert flags["pct_move"].tolist() == pytest.approx([100 * 30 / 70, 40.])


def test_invalid_price_does_not_bridge_a_return():
    flags = extreme_returns(pd.Series([100., np.nan, 300., 300.]), cap=0.5)
    assert flags.empty


def test_nullable_numeric_columns_are_supported():
    sessions, bars = clean_panel(n_sessions=40)
    bars["volume"] = bars["volume"].astype("Int64")
    assert validate_panel(bars, sessions).passed
    bars.loc[sessions[3], "volume"] = pd.NA
    assert validate_panel(bars, sessions).get("nan_values").count == 1


def test_calendar_compares_local_date_labels_without_timezone_shifts():
    sessions, bars = clean_panel(n_sessions=40)
    bars.index = bars.index.tz_localize("Asia/Tokyo") + pd.Timedelta(hours=15)
    assert validate_panel(bars, sessions).passed


@pytest.mark.parametrize("kwargs", [{"max_stale_run": -1}, {"max_stale_frac": 2},
                                    {"max_abs_return": 0}, {"outlier_sigma": np.nan},
                                    {"periods_per_year": -1}])
def test_invalid_thresholds_reject(kwargs):
    _, bars = clean_panel(n_sessions=40)
    with pytest.raises(ValueError):
        validate_panel(bars, **kwargs)


def test_seeded_demos_are_deterministic_and_filling_changes_signals():
    pd.testing.assert_frame_equal(clean_panel()[1], clean_panel()[1])
    assert fill_cost() == fill_cost()
    assert fill_cost()["sma_rows_moved"] > 0
    assert fill_cost()["reported"] == fill_cost()["n_missing"]


def test_guard_reports_failure_and_preserves_input():
    from fin_skills.api.guards.data_quality import DataQualityGuard
    sessions, bars = clean_panel()
    before = bars.copy(deep=True)
    guard = DataQualityGuard()
    assert guard.run(bars=bars, dates=sessions).passed
    result = guard.run(bars=bars.drop(index=sessions[2]), dates=sessions)
    assert not result.passed and result.evidence["counts"]["missing_sessions"] == 1
    pd.testing.assert_frame_equal(bars, before)
    with pytest.raises(TypeError):
        guard.run(bars=bars, max_stale_frac=2)


def test_demo_runs_offline_and_prints_rule(run_main):
    out = run_main("fin_skills.market_data.data_quality")
    assert "PASS" in out and "FAIL" in out
    assert out.strip().splitlines()[-1].startswith("The rule:")
    assert out.isascii()
