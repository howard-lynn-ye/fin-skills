"""fin_skills.alt_data.congress - the 45-day disclosure lag and the amount brackets.

The documented properties: the transcribed statutory constants say what the skill says;
the bracket map reproduces the House PTR form including the spouse/dependent-child
collapse; the lag panel reproduces the distribution its generator was given; a position is
never entered on the key day itself; the transaction-date keying beats the disclosure-date
keying by roughly the claimed margin in every seed; the retained fraction falls with the
lag and rises with the half-life; and the bucketing costs a size-weighted signal about half
of what the size information was worth.
"""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.alt_data import congress as cg


@pytest.fixture(scope="module")
def panel():
    return cg.make_panel(seed=cg.SEED)


def test_statutory_constants_are_what_the_skill_quotes():
    assert cg.PTR_DEADLINE_DAYS == 45 and cg.PTR_NOTIFICATION_DAYS == 30
    assert cg.PTR_THRESHOLD_USD == 1_000
    assert cg.LATE_FILING_FEE_USD == 200 and cg.LATE_FEE_GRACE_DAYS == 30
    assert cg.CIVIL_PENALTY_BASE_USD == 50_000
    assert cg.CIVIL_PENALTY_2025_USD == 75_540
    # ten numeric columns A-J, then column K, which has no numeric equivalent
    assert len(cg.NUMERIC_BRACKETS) == 10
    assert cg.NUMERIC_BRACKETS[0] == (1_001, 15_000)
    assert cg.NUMERIC_BRACKETS[-1] == (50_000_001, None)
    assert cg.SPOUSE_OVER_1M == (1_000_001, None)
    assert cg.K_INDEX == 10 and len(cg.AMOUNT_BRACKETS) == 11
    assert len(cg.BRACKET_LABELS) == 11


def test_the_lowest_bracket_spans_15x_and_the_top_two_are_open():
    w = cg.bracket_widths()
    assert w.loc[0, "width_x"] == pytest.approx(15_000 / 1_001)   # 15.0x to one decimal
    assert not np.isfinite(w.loc[9, "width_x"])       # over $50,000,000
    assert not np.isfinite(w.loc[10, "width_x"])      # column K


def test_bracket_of_maps_the_edges_and_collapses_column_k():
    amt = np.array([500.0, 1_001.0, 15_000.0, 15_001.0, 999_999.0,
                    1_200_000.0, 60_000_000.0])
    got = cg.bracket_of(amt)
    assert got[0] == -1                # under $1,001 - never reported at all
    assert got[1] == 0 and got[2] == 0  # both ends of the lowest bracket
    assert got[3] == 1
    assert got[4] == 5
    assert got[5] == 6 and got[6] == 9
    # the same $1.2m and $60m trades, in a spouse asset, are ONE disclosure
    spouse = np.ones(len(amt), dtype=bool)
    k = cg.bracket_of(amt, spouse)
    assert k[5] == cg.K_INDEX and k[6] == cg.K_INDEX
    assert k[4] == 5                   # under $1,000,000 the collapse does not apply


def test_imputations_are_the_five_choices_the_skill_prices():
    b = np.array([0, 5, 9, cg.K_INDEX])
    truth = np.array([7_000.0, 700_000.0, 90_000_000.0, 4_000_000.0])
    assert np.allclose(cg.impute_size(b, "true", truth), truth)
    assert np.allclose(cg.impute_size(b, "equal", truth), 1.0)
    assert np.allclose(cg.impute_size(b, "low", truth),
                       [1_001, 500_001, 50_000_001, 1_000_001])
    mid = cg.impute_size(b, "midpoint", truth)
    assert mid[0] == pytest.approx((1_001 + 15_000) / 2)
    assert mid[2] == 50_000_001 and mid[3] == 1_000_001   # open: lower edge at x1
    hi = cg.impute_size(b, "midpoint", truth, open_top_multiple=5.0)
    assert hi[2] == pytest.approx(5 * 50_000_001) and hi[0] == mid[0]
    geo = cg.impute_size(b, "geometric", truth)
    assert geo[0] == pytest.approx(np.sqrt(1_001 * 15_000))
    assert geo[0] < mid[0]
    with pytest.raises(ValueError):
        cg.impute_size(b, "true")
    with pytest.raises(ValueError):
        cg.impute_size(b, "nope", truth)


def test_the_panel_reproduces_the_lag_distribution_it_was_given(panel):
    filings, _ = panel
    s = cg.lag_summary(filings["lag_cal"].to_numpy())
    assert s["n"] > 1_400
    assert 25.0 < s["median"] < 38.0            # clusters short of the 45-day deadline
    assert s["p05"] > 1.0
    assert s["share_late"] == pytest.approx(cg.LATE_SHARE, abs=0.04)
    assert s["p95"] > cg.PTR_DEADLINE_DAYS      # the tail is real, not decoration
    assert s["max"] > 200.0
    assert 0.0 < s["share_over_365"] < 0.05
    # scaling the draw scales the distribution
    big, _ = cg.make_panel(seed=cg.SEED, lag_scale=2.0)
    assert big["lag_cal"].median() > 1.8 * filings["lag_cal"].median()


def test_no_amount_below_the_reporting_threshold_survives(panel):
    filings, _ = panel
    assert (filings["size"] >= cg.NUMERIC_BRACKETS[0][0]).all()
    assert (filings["bracket"] >= 0).all()


def test_the_panel_is_deterministic_in_its_seed():
    a, ra = cg.make_panel(seed=cg.SEED)
    b, rb = cg.make_panel(seed=cg.SEED)
    assert a.equals(b) and np.array_equal(ra, rb)
    c, _ = cg.make_panel(seed=cg.SEED + 1)
    assert not np.array_equal(a["lag_cal"].to_numpy(), c["lag_cal"].to_numpy())


def test_entry_is_the_next_day_not_the_key_day():
    """A signal read off a close cannot be filled at that close."""
    import pandas as pd
    f = pd.DataFrame({"name": [0], "disc_day": [10], "txn_day": [3]})
    rets = np.zeros((30, 4))
    rets[10, 0] = 1.0            # a huge move ON the disclosure day
    rets[11, 0] = 0.5            # and the day after
    r = cg.portfolio(f, rets, "disc_day", hold=5)
    assert r[10] == 0.0          # not captured
    assert r[11] > 0.0           # captured


def test_the_ab_gap_is_positive_and_the_vol_is_unchanged(panel):
    tab = cg.run_ab(seed=cg.SEED)
    assert set(tab.index) == {"transaction-date", "disclosure-date"}
    txn, disc = tab.loc["transaction-date"], tab.loc["disclosure-date"]
    assert txn["sharpe"] > disc["sharpe"] > 0
    assert tab.loc["disclosure-date", "sharpe_gap"] == 0.0
    assert txn["sharpe_gap"] == pytest.approx(txn["sharpe"] - disc["sharpe"])
    assert txn["retained"] == pytest.approx(1.0)
    assert 0.2 < disc["retained"] < 0.8
    # the look-ahead is a return, not a smoother ride
    assert txn["ann_vol"] == pytest.approx(disc["ann_vol"], rel=0.15)
    assert txn["ann_return"] > disc["ann_return"]


def test_the_gap_holds_in_every_seed_and_is_of_the_claimed_size():
    gaps = []
    for k in range(20):
        t = cg.run_ab(seed=cg.SEED + k)
        gaps.append(t.loc["transaction-date", "sharpe_gap"])
    assert min(gaps) > 0.0
    assert 0.35 < float(np.mean(gaps)) < 0.90     # the skill quotes 0.589 over 40 seeds


def test_ab_over_seeds_summarises_the_same_three_series():
    sw = cg.ab_over_seeds(n_seeds=12)
    assert set(sw.index) == {"txn", "disc", "gap"}
    assert sw.loc["txn", "mean"] > sw.loc["disc", "mean"]
    assert sw.loc["gap", "mean"] == pytest.approx(
        sw.loc["txn", "mean"] - sw.loc["disc", "mean"], abs=1e-9)
    assert sw.loc["gap", "beats_zero"] > 0.8


def test_retained_falls_with_the_lag_and_rises_with_the_half_life():
    ret, lags = cg.lag_grid(lag_scales=(0.25, 1.0, 2.0), half_lives=(5.0, 63.0),
                            n_seeds=4)
    assert list(ret.index) == ["0.25x", "1x", "2x"]
    for col in ret.columns:
        v = ret[col].to_numpy()
        assert v[0] > v[1] > v[2]              # more lag, less signal
    for row in ret.index:
        assert ret.loc[row, "hl=63d"] > ret.loc[row, "hl=5d"]
    assert (ret.to_numpy() < 1.0).all()
    # the observed mean lag scales with the multiplier that produced it
    m = lags["mean_lag_cal_days"].to_numpy()
    assert m[1] == pytest.approx(4 * m[0], rel=0.15)


def test_bucketing_destroys_ordering_that_no_imputation_can_restore():
    ol = cg.ordering_loss(n_seeds=4)
    assert ol["tied_pair_share"] > 0.15
    assert 0.80 < ol["r2"] < 0.99            # the middle survives, the ends do not
    assert ol["resid_sd"] > 0.3
    assert ol["sd_log_midpoint"] < ol["sd_log_true"]
    occ = cg.bracket_occupancy(n_seeds=4)
    assert len(occ) == 11
    assert occ["share"].sum() == pytest.approx(1.0)
    # column K is the widest disclosed bucket in the scheme
    assert occ.loc["K: spouse/child over $1,000,000", "p90_over_p10"] > 5.0


def test_the_dollar_aggregate_is_a_free_parameter():
    ag = cg.aggregate_bias(n_seeds=4)
    assert list(ag.index) == ["midpoint", "geometric", "low"]
    mid = ag.loc["midpoint"]
    assert mid["top x1"] < 1.0 < mid["top x5"]       # it crosses truth on an assumption
    assert mid["top x1"] < mid["top x2"] < mid["top x5"]
    # the lower-bound imputation cannot move at all: the open bracket IS its lower edge
    lo = ag.loc["low"]
    assert lo["top x1"] == pytest.approx(lo["top x5"])
    assert ag.loc["midpoint", "filings_in_open_brackets"] < 0.05
    assert ag.loc["midpoint", "dollars_in_open_brackets"] > 0.25


def test_the_bucketing_costs_about_half_of_what_size_was_worth():
    wt = cg.weighting_table(n_seeds=6)
    assert set(wt.index) == set(cg.IMPUTATIONS)
    true, mid, eq = (wt.loc[m, "sharpe"] for m in ("true", "midpoint", "equal"))
    assert true > mid > eq                    # size helps; the bucket keeps only some of it
    assert wt.loc["true", "cost_vs_true"] == 0.0
    assert wt.loc["midpoint", "cost_vs_true"] < 0.0
    lost = (true - mid) / (true - eq)
    assert 0.2 < lost < 0.8
    assert wt.loc["equal", "effective_n"] > wt.loc["midpoint", "effective_n"]
    # the open-bracket assumption moves the midpoint run and not the equal-weight one
    hi = cg.weighting_table(n_seeds=6, open_top_multiple=5.0)
    assert hi.loc["midpoint", "sharpe"] != wt.loc["midpoint", "sharpe"]
    assert hi.loc["equal", "sharpe"] == pytest.approx(wt.loc["equal", "sharpe"])


def test_sharpe_convention_is_daily_annualised():
    r = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    assert cg.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert np.isnan(cg.sharpe(np.zeros(5)))


def test_decay_weights_sum_to_one_and_halve_on_schedule():
    w = cg.decay_weights(64, 16.0)
    assert w.sum() == pytest.approx(1.0)
    assert w[16] / w[0] == pytest.approx(0.5, rel=1e-9)


@pytest.mark.slow
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.alt_data.congress")
    for head in ("1. The regime, transcribed", "2. The seeded panel",
                 "3. The A/B", "4. How the gap scales with the lag",
                 "5. Amount ranges"):
        assert head in out
    assert "13105(l)" in out and "$200" in out and "K: spouse/child" in out
    assert out.strip().splitlines()[-1].startswith("Rule:")
    assert all(ord(c) < 128 for c in out)
