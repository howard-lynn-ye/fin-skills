"""fin_skills.core.survivorship_audit - zero delistings is a missing dataset, not a clean one."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.core.survivorship_audit import (_synthetic_panel, audit_universe, dead_names,
                                                first_observation, last_observation,
                                                survivorship_inflation)


@pytest.fixture(scope="module")
def panel():
    full, listings = _synthetic_panel()
    alive = [c for c in full.columns if pd.notna(full[c].iloc[-1])]
    return full, listings, full[alive]


def test_synthetic_panel_is_seeded_and_dead_names_stop_reporting(panel):
    full, listings, _ = panel
    again, _ = _synthetic_panel()
    pd.testing.assert_frame_equal(full, again)
    dead = dead_names(full)
    assert len(dead) > 0
    delisted = listings.loc[listings["delisting_date"].notna(), "ticker"]
    assert set(dead) == set(delisted)
    last = last_observation(full)
    assert (last[dead] < full.index[-1] - pd.Timedelta(days=30)).all()
    assert (first_observation(full) == full.index[0]).all()


def test_bias_free_panel_is_accepted(panel):
    full, listings, _ = panel
    a = audit_universe(full, listings=listings)
    assert a.verdict.startswith("PLAUSIBLY BIAS-FREE")
    assert a.n_ended_early == len(dead_names(full)) and a.n_missing_delisted == 0
    assert a.inflation["measurable"] and a.inflation["inflation_bps"] > 0
    assert "VERDICT" in a.report()


def test_survivor_only_panel_is_caught_with_and_without_a_listing_table(panel):
    _, listings, survivors = panel
    with_table = audit_universe(survivors, listings=listings)
    assert with_table.verdict.startswith("SURVIVOR-BIASED (confirmed)")
    assert with_table.n_missing_delisted == listings["delisting_date"].notna().sum()
    tell_only = audit_universe(survivors)
    assert tell_only.verdict.startswith("SURVIVOR-ONLY SNAPSHOT")
    assert any("UPPER BOUND" in n for n in tell_only.notes)
    assert not tell_only.inflation["measurable"]


def test_inflation_is_the_survivors_only_premium_and_is_unmeasurable_on_survivors(panel):
    full, _, survivors = panel
    infl = survivorship_inflation(full)
    assert infl["survivors_only_cagr"] > infl["full_universe_cagr"]
    assert infl["inflation_bps"] == pytest.approx(infl["inflation_ann"] * 1e4)
    assert infl["n_dead"] == len(dead_names(full))
    assert survivorship_inflation(survivors)["inflation_bps"] is None


def test_partial_coverage_is_likely_biased():
    full, _ = _synthetic_panel(n_names=120, years=6, seed=2)
    dead = list(dead_names(full))
    keep_dead = dead[: max(1, len(dead) // 8)]
    partial = full[[c for c in full.columns if c not in dead or c in keep_dead]]
    a = audit_universe(partial, expected_annual_delist_rate=0.10)
    assert a.verdict.startswith("LIKELY SURVIVOR-BIASED")


def test_bare_ticker_lists_and_pre_listing_data(panel):
    full, listings, _ = panel
    with pytest.raises(ValueError, match="explicit sample_start"):
        audit_universe(list(full.columns))
    a = audit_universe(list(full.columns), sample_start=full.index[0], sample_end=full.index[-1])
    assert a.verdict.startswith("UNTESTABLE")
    late = listings.copy()
    late.loc[late["ticker"] == "N000", "listing_date"] = full.index[100]   # data before listing
    b = audit_universe(full, listings=late)
    assert b.n_prelisting_data == 1 and b.prelisting_names == ["N000"]
    assert any("before their listing date" in n for n in b.notes)


def test_late_starts_are_counted_but_are_not_bias(panel):
    full, _, _ = panel
    ipo = full.copy()
    ipo.iloc[:400, 0] = np.nan
    a = audit_universe(ipo)
    assert a.n_started_late == 1 and a.verdict.startswith("PLAUSIBLY BIAS-FREE")


def test_demo_prints_the_three_verdicts(run_main):
    out = run_main("fin_skills.core.survivorship_audit")
    assert "PLAUSIBLY BIAS-FREE" in out and "SURVIVOR-BIASED (confirmed)" in out
    assert "SURVIVOR-ONLY SNAPSHOT" in out
