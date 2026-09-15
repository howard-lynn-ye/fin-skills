"""Calendar, venue and quote-currency accounting for the crypto market-structure module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _helpers import is_ascii
from fin_skills.api import conventions
from fin_skills.crypto import market_structure as ms


def test_calendar_factor_is_delegated_to_the_library_convention(monkeypatch):
    calls = []
    real = conventions.annualization_factor

    def traced(calendar):
        calls.append(calendar)
        return real(calendar)

    monkeypatch.setattr(conventions, "annualization_factor", traced)
    assert ms.annualization_factor("crypto") == 365
    assert ms.annualization_factor("hourly") == 8760
    assert calls == ["crypto", "hourly"]


def test_same_returns_change_sharpe_by_the_exact_square_root_factor():
    r = ms.calendar_series()
    result = ms.annualisation_table(r)
    assert result["sharpe_252"] / result["sharpe_365"] == pytest.approx(np.sqrt(252 / 365))
    assert result["sharpe_365"] == pytest.approx(0.8432, abs=0.00005)
    assert result["sharpe_252"] == pytest.approx(0.7006, abs=0.00005)
    assert ms.sharpe(r, 365) == pytest.approx(conventions.annualize_sharpe(r, "crypto"))
    assert np.array_equal(r, ms.calendar_series())


@pytest.mark.parametrize("returns,periods", [([1], 365), ([1, np.nan], 365), ([1, 2], 0),
                                           ([[1, 2]], 365)])
def test_sharpe_rejects_bad_rows_or_factors(returns, periods):
    with pytest.raises(ValueError):
        ms.sharpe(returns, periods)


def test_constant_returns_have_undefined_sharpe():
    assert np.isnan(ms.sharpe([0, 0], 365))


def test_rolling_year_definitions_differ_on_calendar_rows():
    result = ms.window_disagreement(ms.calendar_series())
    assert result["observations"] == 1095
    assert result["sign_disagreement"] == pytest.approx(0.177, abs=0.001)
    assert result["correlation"] == pytest.approx(0.7791, abs=0.0001)


def test_cross_venue_dispersion_uses_prices_at_the_same_instant():
    feeds = pd.DataFrame({"a": [100.0, 100.0], "b": [101.0, 102.0]})
    d = ms.cross_venue_dispersion(feeds)
    assert d["median_range_bps"] == pytest.approx(150)
    assert d["max_range_bps"] == pytest.approx(200)
    assert d["highest_venue_changes"] == 0  # initial row is not a leader change
    with pytest.raises(ValueError):
        ms.cross_venue_dispersion(feeds.assign(a=0))


def test_synthetic_venue_dispersion_is_deterministic_and_nonzero():
    feeds = ms.venue_feeds()
    pd.testing.assert_frame_equal(feeds, ms.venue_feeds())
    d = ms.cross_venue_dispersion(feeds)
    assert d["median_range_bps"] == pytest.approx(21.4, abs=0.05)
    assert d["p99_range_bps"] > d["median_range_bps"]
    assert ms.same_rule_five_feeds(feeds)["spread"] > 0


def test_first_breaches_include_times_outside_the_watch_window():
    result = ms.unwatched_breaches(days=15, n_paths=150)
    assert 0 < result["breached"] <= 150
    assert 0 < result["breaches_unwatched"] < 1
    assert result["breaches_unwatched"] + result["breaches_in_hours"] == pytest.approx(1)
    assert result == ms.unwatched_breaches(days=15, n_paths=150)


def test_sampling_hour_changes_intervals_and_endpoint_statistics():
    result = ms.close_hour_effect(n_days=60, hours={"midnight": 0, "eight": 8})
    assert set(result["per_hour"]) == {"midnight", "eight"}
    assert all(row["rows"] == 59 for row in result["per_hour"].values())
    assert result["sharpe_spread"] > 0


def test_only_negative_outage_moves_are_not_the_unconditional_expectation():
    result = ms.outage_cost(n_paths=1000)
    assert result["expected_annual_drag"] < result["unconditional_annual_change"]
    assert result["p05_move"] < result["median_move"]


def test_forced_close_price_reduces_the_marked_gain():
    result = ms.adl_haircut(30000, 21000, 27000)
    assert result["pnl_at_mark"] == pytest.approx(0.30)
    assert result["pnl_after_adl"] == pytest.approx(0.10)
    assert result["share_given_up"] == pytest.approx(2 / 3)


def test_stablecoin_conversion_changes_intermediate_returns_even_if_peg_recovers():
    result = ms.depeg_shock(annual_vol=0)
    assert result["peak_phantom"] == pytest.approx(1 / result["worst_peg"] - 1)
    assert result["phantom_return_in_window"] == pytest.approx(0)
    assert result["tracking_error_bps"] > 0
    with pytest.raises(ValueError):
        ms.depeg_shock(n_days=100, depeg_start=95, depeg_days=6)


def test_main_is_ascii_and_prints_the_rule(run_main):
    out = run_main("fin_skills.crypto.market_structure")
    assert is_ascii(out)
    assert ms.THE_RULE in out
    assert "conventions: fin_skills.api.conventions (imported)" in out
    assert "NOT expected drag" in out
    assert max(map(len, out.splitlines())) <= 98
