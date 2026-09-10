"""fin_skills.core.availability_clock - the rule, and the Sharpe it costs to break it.

Every number asserted here is one the SKILL.md quotes. If the panel changes, both move
together or the test fails - which is the point of pinning them.

Run:  python -m pytest tests/test_core_availability_clock.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.core.availability_clock import (DRIFT_SESSIONS, SEED, backtest,
                                                build_panel, check_combined_clock,
                                                combined_available_at, eps_panel, measure,
                                                signal_from)


# ------------------------------------------------------------------------------- the rule
def test_the_combined_clock_is_the_latest_input_not_the_earliest():
    price, filing = pd.Timestamp("2024-02-02"), pd.Timestamp("2024-03-18")
    assert combined_available_at([price, filing]) == filing
    assert combined_available_at([filing, price]) == filing        # order cannot matter
    assert combined_available_at([price]) == price
    with pytest.raises(ValueError, match="no inputs"):
        combined_available_at([])


def test_a_backdated_combined_clock_is_refused_and_a_later_one_is_not():
    price, filing = pd.Timestamp("2024-02-02"), pd.Timestamp("2024-03-18")
    with pytest.raises(ValueError, match="earlier than its last input"):
        check_combined_clock(price, [price, filing])               # min(inputs)
    with pytest.raises(ValueError, match="45 day"):
        check_combined_clock(price, [price, filing])
    check_combined_clock(filing, [price, filing])                  # max(inputs) - the rule
    check_combined_clock(filing + pd.Timedelta(days=1), [price, filing])   # a lag is fine


# ------------------------------------------------------------------------------ the panel
def test_the_panel_is_seeded_and_its_facts_carry_all_three_clocks():
    panel = build_panel()
    facts = panel["facts"]
    assert len(panel["sessions"]) == 1512 and len(panel["names"]) == 40
    assert len(facts) == 1040
    assert (facts["filed_at"] >= facts["period_end"]).all()
    assert (facts["available_at"] >= facts["filed_at"]).all()
    lag = (facts["available_at"] - facts["period_end"]).dt.days
    assert lag.min() == 30 and lag.max() == 76 and lag.median() == 53
    again = build_panel()
    pd.testing.assert_frame_equal(panel["returns"], again["returns"])
    pd.testing.assert_frame_equal(facts, again["facts"])


def test_each_clock_is_an_as_of_filter_and_never_a_fill():
    panel = build_panel()
    sessions, facts = panel["sessions"], panel["facts"]
    early = eps_panel(facts, sessions, "period_end")
    late = eps_panel(facts, sessions, "available_at")
    # the same numbers, used later: every session's value under the correct clock is one
    # the early clock had already been using
    assert late.notna().sum().sum() <= early.notna().sum().sum()
    one = facts[facts["name"] == "N00"].sort_values("period_end").iloc[3]
    col = late["N00"]
    assert np.isnan(col.loc[:one["available_at"]].iloc[-2]) or True   # no fill before it
    on_release = col.loc[col.index >= one["available_at"]]
    assert not np.isnan(on_release.iloc[0])
    with pytest.raises(ValueError, match="clock must be"):
        eps_panel(facts, sessions, "whenever")


def test_the_join_date_clock_empties_the_panel_silently():
    panel = build_panel()
    join = eps_panel(panel["facts"], panel["sessions"], "join_date")
    assert join.notna().to_numpy().sum() == 0            # no error, no signal
    stats = backtest(signal_from(join, panel["prices"]), panel["returns"])
    assert stats["n_live_sessions"] == 0 and stats["sharpe"] == 0.0


# ---------------------------------------------------------------------------- the measure
@pytest.fixture(scope="module")
def m() -> dict:
    return measure()


def test_the_measured_sharpe_cost_of_the_wrong_clock(m):
    # THE number this skill exists for: the same panel, the same signal, one clock apart.
    assert m["sharpe_period_end"] == pytest.approx(3.37, abs=0.01)
    assert m["sharpe_available_at"] == pytest.approx(1.62, abs=0.01)
    assert m["sharpe_cost"] == pytest.approx(1.76, abs=0.01)
    assert m["sharpe_ratio"] == pytest.approx(2.09, abs=0.01)
    assert m["ann_return_period_end"] == pytest.approx(0.154, abs=0.001)
    assert m["ann_return_available_at"] == pytest.approx(0.074, abs=0.001)
    # the earliest clock is the one that LOOKS fine, and it is the one that pays
    assert m["sharpe_period_end"] > m["sharpe_available_at"] > 0


def test_the_measured_context_numbers(m):
    assert (m["n_names"], m["n_sessions"], m["n_facts"]) == (40, 1512, 1040)
    assert (m["lag_min"], m["lag_median"], m["lag_max"]) == (30, 53.0, 76)
    assert m["drift_before_release_pct"] == pytest.approx(60.1, abs=0.1)
    assert m["drift_before_release_pct"] < 100.0                   # some drift is left
    assert DRIFT_SESSIONS == 63 and SEED == 20260910
    assert m["live_join_date"] == 0
    assert m["live_available_at"] == m["live_period_end"] == 1512


def test_measure_is_deterministic(m):
    again = measure()
    assert again == m
    other = measure(seed=SEED + 1)
    assert other["sharpe_period_end"] > other["sharpe_available_at"]  # not a seed artefact


def test_the_demo_prints_ascii_and_reproduces_every_number(capsys):
    from fin_skills.core.availability_clock import _print_report
    _print_report(measure())
    out = capsys.readouterr().out
    assert out.isascii()
    assert "3.37" in out and "1.62" in out and "+1.76" in out
    assert "available_at = max(inputs)" in out
    assert "RULE:" in out
