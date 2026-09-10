"""fin_skills.alt_data.thirteen_f - quarter-end staleness, coverage, confidential treatment.

The documented properties: the transcribed rule constants say what the skill says; the age
of a position at disclosure is the sum of two legs and the 45-day one is the smaller; a
position closed inside its own quarter never appears; the quarter-end keying beats the
filing-date keying; a long-only reconstruction has a materially different market beta from
the book it is meant to represent; and omitting the highest-conviction positions costs more
than omitting the same number at random.
"""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.alt_data import thirteen_f as tf


@pytest.fixture(scope="module")
def panel():
    return tf.make_panel(seed=tf.SEED)


def test_rule_constants_are_what_the_skill_quotes():
    assert tf.DEADLINE_DAYS_AFTER_QUARTER_END == 45
    assert tf.THRESHOLD_USD == 100_000_000
    assert tf.THRESHOLD_SET_YEAR == 1978
    assert tf.PROPOSED_THRESHOLD_USD == 3_500_000_000
    assert tf.PROPOSAL_WITHDRAWN == "2021-05-11"
    assert tf.XML_INFORMATION_TABLE_SINCE == "2013-05-20"
    assert tf.CT_MAX_PERIOD_DAYS == 365
    assert tf.CT_AMENDMENT_BUSINESS_DAYS == 6


def test_the_coverage_table_says_shorts_are_not_reported():
    cov = tf.coverage_table().set_index("instrument")
    assert cov.loc["SHORT positions", "status"] == "NOT reported"
    assert "should not include short positions" in cov.loc["SHORT positions", "source"]
    assert cov.loc["long netted against short", "status"] == "NOT permitted"
    assert cov.loc["securities on non-US exchanges", "status"] == "NOT reported"
    assert cov.loc["US exchange-traded stock", "status"] == "reported"
    # the one row that is inferred rather than quoted has to say so
    assert "INFERRED" in cov.loc["bonds, commodities, currencies, cash", "source"]
    quoted = cov.drop(index="bonds, commodities, currencies, cash")
    assert not quoted["source"].str.contains("INFERRED").any()


def test_the_filing_lag_piles_against_the_deadline(panel):
    pos, _, _ = panel
    s = tf.lag_summary(pos["lag_cal"].to_numpy())
    assert 38.0 < s["median"] < 45.0
    assert s["p95"] <= tf.DEADLINE_DAYS_AFTER_QUARTER_END + 0.5
    assert s["share_at_or_past_44"] > 0.15      # a deadline is a target, not a mean
    assert s["max"] > tf.DEADLINE_DAYS_AFTER_QUARTER_END   # amendments and late filers
    assert s["p05"] > 20.0


def test_the_age_is_two_legs_and_forty_five_days_is_the_smaller_one(panel):
    pos, _, _ = panel
    a = tf.age_summary(pos)
    assert a["n"] > 8_000
    # the age decomposes, and neither leg dominates - but the quarter leg is not smaller
    assert a["mean"] == pytest.approx(a["mean_quarter_leg"] + a["mean_filing_leg"], rel=0.02)
    assert a["mean_quarter_leg"] > 25.0
    assert a["mean_filing_leg"] <= tf.DEADLINE_DAYS_AFTER_QUARTER_END
    # the whole claim: the median position is far older than the 45-day deadline
    assert a["median"] > 2 * 45 * 0.8
    assert a["share_over_90"] > 0.25
    assert a["max"] > 150
    # and it is a DISTRIBUTION - the same filings carry very different ages
    assert a["p95"] - a["p05"] > 60


def test_a_position_closed_inside_its_quarter_is_never_disclosed(panel):
    pos, _, _ = panel
    never = pos[~pos["disclosed"]]
    assert (never["exit_day"] <= never["q_end"]).all()
    seen = pos[pos["disclosed"]]
    assert (seen["exit_day"] > seen["q_end"]).all()
    inv = tf.invisibility(pos)
    assert 0.10 < inv["never_disclosed"] < 0.40
    # the invisible book is the FAST book - that is the selection
    assert inv["mean_hold_never_td"] < inv["mean_hold_seen_td"] / 3
    assert 0.05 < inv["closed_by_filing_date"] < 0.40
    assert inv["closed_by_quarter_end_plus_90"] > inv["closed_by_filing_date"]


def test_the_panel_is_deterministic_in_its_seed():
    a, ra, ma = tf.make_panel(seed=tf.SEED)
    b, rb, mb = tf.make_panel(seed=tf.SEED)
    assert a.equals(b) and np.array_equal(ra, rb) and np.array_equal(ma, mb)
    c, _, _ = tf.make_panel(seed=tf.SEED + 1)
    assert not np.array_equal(a["lag_cal"].to_numpy(), c["lag_cal"].to_numpy())


def test_entry_is_the_next_day_and_neutral_is_optional():
    import pandas as pd
    d = pd.DataFrame({"name": [0], "q_end": [10]})
    rets = np.zeros((30, 4))
    rets[10, 0] = 1.0
    rets[11, 0] = 0.5
    r = tf.portfolio(d, rets, "q_end", hold=5)
    assert r[10] == 0.0 and r[11] > 0.0
    raw = tf.portfolio(d, rets, "q_end", hold=5, neutral=False)
    assert raw[11] == pytest.approx(0.5)
    assert r[11] == pytest.approx(0.5 - rets[11].mean())


def test_the_quarter_end_key_is_a_look_ahead():
    tab = tf.run_ab(seed=tf.SEED)
    assert set(tab.index) == {"quarter-end date", "filing date"}
    qe, fd = tab.loc["quarter-end date", "sharpe"], tab.loc["filing date", "sharpe"]
    assert qe > fd > 0
    assert tab.loc["filing date", "sharpe_gap"] == 0.0
    assert tab.loc["quarter-end date", "sharpe_gap"] == pytest.approx(qe - fd)
    assert 0.2 < tab.loc["quarter-end date", "retained"] < 0.9
    assert tab.loc["quarter-end date", "ann_vol"] == pytest.approx(
        tab.loc["filing date", "ann_vol"], rel=0.15)


def test_the_ab_holds_on_average_and_the_skill_reports_its_own_noise():
    sw = tf.ab_over_seeds(n_seeds=10)
    assert set(sw.index) == {"quarter_end", "filing", "gap"}
    assert sw.loc["quarter_end", "mean"] > sw.loc["filing", "mean"] > 0
    assert sw.loc["gap", "mean"] > 0.2
    # the honest part: this A/B is underpowered on one panel, and the skill says so
    assert sw.loc["gap", "sd"] > 0.15
    assert 0.3 < sw.attrs["pooled_retained"] < 0.85


def test_a_long_only_reconstruction_has_the_wrong_market_exposure():
    re_ = tf.reconstruction_error(n_managers=12)
    assert re_.loc["corr", "mean"] > 0.85            # it LOOKS like a noisy copy
    assert re_.loc["beta_reconstructed", "mean"] == pytest.approx(1.0, abs=0.15)
    # and the true book's beta is materially lower, because the shorts are invisible
    assert re_.loc["beta_true", "mean"] < re_.loc["beta_reconstructed", "mean"] - 0.15
    expected = (tf.SLEEVES["long_13f"] + tf.SLEEVES["short_equity"]
                + 0.30 * tf.SLEEVES["non_13f"])
    assert re_.loc["beta_true", "mean"] == pytest.approx(expected, abs=0.15)
    # the error is an exposure, not noise: whole percent of tracking error a year
    assert re_.loc["tracking_error", "mean"] > 0.03
    assert re_.loc["vol_true", "mean"] > re_.loc["vol_reconstructed", "mean"]


def test_omitting_the_best_positions_costs_more_than_omitting_at_random():
    ct = tf.confidential_treatment(n_seeds=6, shares=(0.0, 0.20))
    assert list(ct.index) == ["0%", "20%"]
    assert ct.loc["0%", "cost_of_selection"] == 0.0
    assert ct.loc["0%", "top-conviction omitted"] == pytest.approx(
        ct.loc["0%", "randomly omitted"])
    assert ct.loc["20%", "cost_of_selection"] > 0.0
    # the adversarial column falls; the random one is roughly flat
    assert ct.loc["20%", "top-conviction omitted"] < ct.loc["0%", "top-conviction omitted"]
    assert ct.loc["20%", "randomly omitted"] == pytest.approx(
        ct.loc["0%", "randomly omitted"], abs=0.15)
    assert (ct["se"] > 0).all()


def test_sharpe_convention_is_daily_annualised():
    r = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    assert tf.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert np.isnan(tf.sharpe(np.zeros(5)))


def test_decay_weights_sum_to_one_and_halve_on_schedule():
    w = tf.decay_weights(128, 32.0)
    assert w.sum() == pytest.approx(1.0)
    assert w[32] / w[0] == pytest.approx(0.5, rel=1e-9)


@pytest.mark.slow
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.alt_data.thirteen_f")
    for head in ("1. The rule, transcribed", "2. The age of a position",
                 "3. What the table never contains", "4. The A/B",
                 "5. A 13F is not the manager's portfolio",
                 "6. Confidential treatment"):
        assert head in out
    assert "$100,000,000" in out and "2021-05-11" in out and "SHORT positions" in out
    assert out.strip().splitlines()[-1].startswith("Rule:")
    assert all(ord(c) < 128 for c in out)
