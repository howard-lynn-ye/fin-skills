"""fin_skills.macro.recession_indicators - the label that arrives years late.

Properties under test: the transcribed NBER announcement dates reproduce the committee's
own elapsed-months column; 64% of US recession months since 1980 carried no label at the
time; the vintage USREC prints 0 rather than abstaining and is late at both ends; the
trap fires when USREC is read retrospectively and clears when it is read as of the month;
the Sahm rule is defined as FRED defines it and its hard threshold degrades with input
revisions; everything is seeded.
"""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.macro import recession_indicators as ri


def test_announcement_lags_reproduce_nbers_own_elapsed_months_column():
    at = ri.announcement_table()
    assert len(at) == 12
    assert at["match"].all(), at[~at["match"]]
    peaks = at[at["kind"] == "peak"]["computed_lag"]
    troughs = at[at["kind"] == "trough"]["computed_lag"]
    assert peaks.mean() == pytest.approx(44 / 6)          # 5+6+9+8+12+4
    assert troughs.mean() == pytest.approx(91 / 6)        # 12+8+21+20+15+15
    # the two the skill quotes by name
    row = at[at["turning_point"] == "2007-12"].iloc[0]
    assert row["announced"] == "2008-12-01" and row["computed_lag"] == 12
    row = at[at["turning_point"] == "2020-04"].iloc[0]
    assert row["announced"] == "2021-07-19" and row["computed_lag"] == 15


def test_recession_spans_follow_freds_own_usrec_convention():
    recs = ri.recessions_from_turning_points()
    assert len(recs) == 6
    # "the recession begins the first day of the period following a peak and ends on the
    # last day of the period of the trough"
    assert str(recs[4]["start"]) == "2008-01" and str(recs[4]["end"]) == "2009-06"
    assert recs[4]["n_months"] == 18
    assert str(recs[5]["start"]) == "2020-03" and recs[5]["n_months"] == 2
    assert sum(r["n_months"] for r in recs) == 58


def test_sixty_four_percent_of_recession_months_had_no_label_at_the_time():
    tab = ri.unlabelled_recession_months()
    assert int(tab["months"].sum()) == 58
    assert int(tab["unlabelled"].sum()) == 37
    assert tab["unlabelled"].sum() / tab["months"].sum() == pytest.approx(0.6379, abs=1e-3)
    # 1990-91 and 2020 were over before they were declared
    assert (tab["share"] == 1.0).sum() == 2


def test_the_vintage_label_prints_zero_rather_than_abstaining_and_is_late_both_ends():
    state = np.zeros(60, dtype=int)
    state[20:32] = 1                                     # a 12-month recession
    peak_lag, trough_lag = np.array([5]), np.array([15])
    early = ri.usrec_known_at(state, peak_lag, trough_lag, 22)
    assert not np.isnan(early[20:23]).any()              # it does not abstain
    assert early[20] == 0.0 and early[22] == 0.0         # peak announced at month 25
    at_26 = ri.usrec_known_at(state, peak_lag, trough_lag, 26)
    assert (at_26[20:27] == 1.0).all()                   # now the whole run reads 1
    assert np.isnan(at_26[27:]).all()                    # the future is unpublished
    late = ri.usrec_known_at(state, peak_lag, trough_lag, 44)
    assert (late[32:45] == 1.0).all()                    # OVERSTAYS: trough not declared
    final = ri.usrec_known_at(state, peak_lag, trough_lag, 59)
    assert np.array_equal(final[:60], state.astype(float))


def test_announcement_lags_are_nbers_own_values_tiled():
    p, t = ri.announcement_lags(12)
    assert list(p) == [5, 6, 9, 8, 12, 4] * 2
    assert list(t) == [12, 8, 21, 20, 15, 15] * 2


@pytest.fixture(scope="module")
def sim():
    return ri.simulate()


def test_simulation_is_seeded_and_uses_the_real_cycle_lengths(sim):
    again = ri.simulate()
    assert np.array_equal(sim["state"], again["state"])
    assert not np.array_equal(sim["state"], ri.simulate(900, ri.SEED + 1)["state"])
    lens = [e - s + 1 for s, e in ri.episodes(sim["state"])]
    assert 0.06 < sim["state"].mean() < 0.14              # a realistic recession share
    assert min(lens) >= 2 and np.mean(lens) < 16
    assert sim["u"].min() >= 3.2 and sim["u"].max() <= 12.0


def test_live_usrec_reproduces_the_real_worlds_recall(sim):
    p, t = ri.announcement_lags()
    live = ri.live_usrec(sim["state"], p, t)
    rec = sim["state"] == 1
    recall = live[rec].mean()
    # the real figure from section 1 is 1 - 0.638 = 0.362
    assert 0.20 < recall < 0.55
    assert live[~rec].mean() > 0.0                        # and it overstays into expansions


def test_the_trap_fires_retrospectively_and_clears_on_the_as_of_reading(sim):
    res = ri.train_two_ways(sim)
    a = res["auc"]
    assert a["USREC as a signal, retrospective"] == pytest.approx(1.0)
    assert a["USREC as a signal, as of that month"] < 0.75
    assert (a["USREC as a signal, retrospective"]
            - a["USREC as a signal, as of that month"]) > 0.20
    # the classifier, by contrast, hardly moves - that is the documented finding
    gap = (a["classifier, walk-forward retrospective label"]
           - a["classifier, walk-forward vintage label"])
    assert abs(gap) < 0.05
    assert res["detect"]["USREC as a signal, as of that month"]["median_delay"] >= 3
    err = res["label_error"]
    assert err["all_rows"] < 0.01 < err["recent_rows"]    # the damage is at the edge


def test_sahm_is_the_rule_fred_documents():
    # a flat series can never trigger; a 0.5pp step over three months must
    flat = np.full(40, 5.0)
    assert np.nanmax(ri.sahm(flat)) == pytest.approx(0.0)
    rising = np.r_[np.full(20, 4.0), np.full(20, 4.6)]
    s = ri.sahm(rising)
    assert np.nanmax(s) == pytest.approx(0.6, abs=1e-9)
    assert (np.nan_to_num(s) >= ri.SAHM_THRESHOLD).any()
    assert ri.SAHM_THRESHOLD == 0.50
    # the statistic is a 3-month mean minus the min of the previous twelve 3-month means
    u = np.arange(40, dtype=float) * 0.1 + 5.0
    assert s.shape == (40,) and np.isnan(ri.sahm(u)[:14]).all()


def test_unemployment_vintages_are_causal_and_only_the_recent_years_move(sim):
    U = ri.unemployment_vintages(sim["u"])
    n = sim["u"].size
    for m in (50, 300, n - 1):
        assert np.isnan(U[m, m + 1:]).all()               # no future publication
    assert np.allclose(U[n - 1, :12], U[n - 200, :12])    # the far past is frozen
    assert not np.allclose(U[400], U[500][:900])          # the recent years do move


def test_sahm_barely_moves_at_u3s_own_revision_size_but_the_threshold_breaks(sim):
    tab = ri.sahm_sensitivity(sim, sds=(0.0, 0.075, 0.6))
    assert tab.loc[0.0, "mean_abs_gap"] == 0.0
    assert tab.loc[0.0, "threshold_disagreements"] == 0
    assert tab.loc[0.075, "mean_abs_gap"] < 0.05          # hundredths of a point
    assert tab.loc[0.6, "mean_abs_gap"] > tab.loc[0.075, "mean_abs_gap"]
    assert tab.loc[0.6, "threshold_disagreements"] > 10 * tab.loc[0.075,
                                                                  "threshold_disagreements"]
    assert tab.loc[0.6, "same_trigger_month"] < tab.loc[0.0, "same_trigger_month"]
    assert tab.loc[0.6, "false_alarms"] > tab.loc[0.075, "false_alarms"]


def test_inversion_lead_is_long_and_dispersed_with_false_alarms(sim):
    ir = ri.inversion_record(sim)
    assert ir["n_inversions"] > ir["n_recessions"]
    assert ir["false_alarms"] > 0
    assert 4 <= ir["median_lead"] <= 22
    assert ir["iqr"][1] - ir["iqr"][0] >= 3               # not a point estimate
    assert ir["recessions_preceded"] <= ir["n_recessions"]


def test_auc_and_logit_helpers():
    y = np.array([0.0, 0, 1, 1])
    assert ri.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert ri.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert ri.auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 1))
    p = 1 / (1 + np.exp(-(0.4 + 1.8 * X[:, 0])))
    beta = ri.fit_logit(X, (rng.random(600) < p).astype(float))
    assert beta[0] == pytest.approx(0.4, abs=0.25)
    assert beta[1] == pytest.approx(1.8, abs=0.35)


@pytest.mark.slow
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.macro.recession_indicators")
    for head in ("1. The NBER's own announcement record",
                 "2-3. A classifier trained on the answer",
                 "4. The Sahm rule on vintage unemployment",
                 "5. Yield-curve inversion"):
        assert head in out
    assert "63.8%" in out and "2008-12-01" in out and "2021-07-19" in out
    assert out.strip().splitlines()[-1].startswith("Rule:")
    assert all(ord(c) < 128 for c in out)
