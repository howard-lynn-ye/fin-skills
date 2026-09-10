"""fin_skills.alt_data.form4 - the two-day deadline, the 22:00 ET window, the code filter.

The documented properties: the transcribed code list has all twenty codes in the SEC's own
groups; the seeded panel reproduces the lag and acceptance-hour distributions it was given;
the first tradeable session is the filing day only when the filing landed before the close;
the transaction-date and filing-day-close keyings both beat the honest one; the gap comes
from the announcement rather than the deadline, so it vanishes as announce_share goes to
zero; and filtering to code P beats every more inclusive signal.
"""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.alt_data import form4 as f4


@pytest.fixture(scope="module")
def panel():
    return f4.make_panel(seed=f4.SEED)


def test_the_rule_constants_are_what_the_skill_quotes():
    assert f4.FORM4_DEADLINE_BUSINESS_DAYS == 2
    assert f4.DEFERRED_NOTIFICATION_CAP_BUSINESS_DAYS == 3
    assert f4.DEFERRED_OUTER_LIMIT_BUSINESS_DAYS == 5
    assert f4.EDGAR_CUTOFF_HOUR_ET == 17.5
    assert f4.OWNERSHIP_CUTOFF_HOUR_ET == 22.0
    assert f4.MARKET_CLOSE_HOUR_ET == 16.0
    assert f4.EDGAR_MAX_REQUESTS_PER_SECOND == 10


def test_there_are_twenty_codes_in_the_secs_own_five_groups():
    assert len(f4.TRANSACTION_CODES) == 20
    assert "V" in f4.TRANSACTION_CODES          # the one usually dropped
    groups = {g for g, _, _ in f4.TRANSACTION_CODES.values()}
    assert groups == {"general", "16b-3", "derivative", "exempt", "other"}
    assert f4.TRANSACTION_CODES["P"][1].startswith("Open market or private purchase")
    assert f4.TRANSACTION_CODES["G"][1] == "Bona fide gift"
    assert f4.TRANSACTION_CODES["V"][1] == "Transaction voluntarily reported earlier than required"
    assert set(f4.EXEMPT_HEADED_CODES) == {"G", "L", "W", "Z"}
    assert set(f4.RULE_16B3_CODES) == {"A", "D", "F", "I", "M"}
    for c in f4.EXEMPT_HEADED_CODES:
        assert f4.TRANSACTION_CODES[c][0] == "exempt"
    for c in f4.RULE_16B3_CODES:
        assert f4.TRANSACTION_CODES[c][0] == "16b-3"


def test_seven_codes_carry_no_direction_of_their_own():
    """The code is not the direction - that is a separate flag on the transaction line."""
    amb = {c for c, (_, _, d) in f4.TRANSACTION_CODES.items() if d is None}
    assert amb == {"V", "I", "E", "H", "G", "W", "Z", "J", "K"}
    assert "P" not in amb and "S" not in amb


def test_the_simulated_mix_is_a_distribution_where_purchases_are_the_minority():
    assert sum(f4.CODE_MIX.values()) == pytest.approx(1.0)
    assert set(f4.CODE_MIX) <= set(f4.TRANSACTION_CODES)
    assert f4.CODE_MIX["P"] < 0.2
    assert "P" in f4.ACQUISITION_CODES and "S" in f4.DISPOSITION_CODES
    assert "G" not in f4.ACQUISITION_CODES and "G" not in f4.DISPOSITION_CODES
    tab = f4.code_table()
    assert len(tab) == 20
    assert tab["simulated_share"].sum() == pytest.approx(1.0)


def test_the_panel_reproduces_the_lag_and_hour_it_was_given(panel):
    f, _ = panel
    s = f4.lag_summary(f["lag_bd"].to_numpy(), f["hour"].to_numpy())
    assert s["n"] == 6_000
    assert s["median"] == 2.0
    assert 0.70 < s["share_within_2"] < 0.82
    assert s["share_within_5"] > 0.94
    assert 0.0 < s["share_over_5"] < 0.06
    assert s["share_post_close"] == pytest.approx(f4.POST_CLOSE_SHARE, abs=0.03)
    assert s["max_hour"] <= f4.OWNERSHIP_CUTOFF_HOUR_ET
    assert f["hour"].min() >= 9.5
    # the deferred branch is the reason anything sits past the two-day deadline; the
    # separate late branch can still land on top of it, which is why this is a quantile
    dl = f.loc[f["deferred"], "lag_bd"]
    assert dl.min() >= 3
    assert dl.quantile(0.95) <= f4.DEFERRED_OUTER_LIMIT_BUSINESS_DAYS
    assert f["deferred"].mean() == pytest.approx(f4.DEFERRED_SHARE, abs=0.03)
    assert f.loc[~f["deferred"] & (f["lag_bd"] <= 4), "lag_bd"].max() <= 4


def test_the_first_tradeable_session_is_the_filing_day_only_before_the_close(panel):
    f, _ = panel
    pre = f["hour"] < f4.MARKET_CLOSE_HOUR_ET
    assert (f.loc[pre, "react_day"] == f.loc[pre, "file_day"]).all()
    assert (f.loc[~pre, "react_day"] == f.loc[~pre, "file_day"] + 1).all()
    assert (f["file_day"] >= f["txn_day"]).all()


def test_the_panel_is_deterministic_in_its_seed():
    a, ra = f4.make_panel(seed=f4.SEED)
    b, rb = f4.make_panel(seed=f4.SEED)
    assert a.equals(b) and np.array_equal(ra, rb)
    c, _ = f4.make_panel(seed=f4.SEED + 1)
    assert not np.array_equal(a["lag_bd"].to_numpy(), c["lag_bd"].to_numpy())


def test_only_code_p_carries_information(panel):
    f, _ = panel
    assert (f.loc[f["code"] == "P", "informed"]).all()
    assert not (f.loc[f["code"] != "P", "informed"]).any()
    assert f.loc[f["code"] != "P", "alpha"].abs().max() == 0.0


def test_scheduled_codes_cluster_and_purchases_do_not(panel):
    f, _ = panel
    per_day = lambda sub: sub.groupby(["name", sub["txn_day"] % 252]).size().max()
    assert per_day(f[f["code"] == "A"]) > per_day(f[f["code"] == "P"])


def test_entry_is_the_next_day_and_shorts_are_hedged():
    import pandas as pd
    d = pd.DataFrame({"name": [0], "react_day": [10]})
    rets = np.zeros((30, 4))
    rets[10, 0] = 1.0
    rets[11, 0] = 0.5
    r = f4.portfolio(d, rets, "react_day", hold=5)
    assert r[10] == 0.0 and r[11] > 0.0
    short = f4.portfolio(d, rets, "react_day", hold=5, side=np.array([-1.0]))
    assert short[11] < 0.0
    flat = f4.portfolio(d, rets, "react_day", hold=5, side=np.array([0.0]))
    assert np.allclose(flat, 0.0)


def test_both_look_ahead_keys_beat_the_honest_one():
    tab = f4.run_ab(seed=f4.SEED)
    assert list(tab.index) == ["transaction-date", "filing-date, same close",
                               "first tradeable session"]
    txn = tab.loc["transaction-date", "sharpe"]
    close = tab.loc["filing-date, same close", "sharpe"]
    honest = tab.loc["first tradeable session", "sharpe"]
    assert txn > close > honest > 0
    assert tab.loc["first tradeable session", "sharpe_gap"] == 0.0
    assert tab.loc["transaction-date", "sharpe_gap"] == pytest.approx(txn - honest)
    # the look-ahead buys return, not a smoother ride
    assert tab.loc["transaction-date", "ann_vol"] == pytest.approx(
        tab.loc["first tradeable session", "ann_vol"], rel=0.10)


def test_the_gap_is_the_announcement_and_not_the_two_day_deadline():
    sp = f4.announce_split(n_seeds=5, shares=(0.0, 0.4))
    assert sp.loc["0%", "retained"] > 0.8       # a two-day lag on its own is nearly free
    assert sp.loc["40%", "retained"] < sp.loc["0%", "retained"]
    assert sp.loc["40%", "gap"] > 2 * sp.loc["0%", "gap"]
    # the transaction-date run barely notices where the alpha sits; it holds through both
    assert sp.loc["40%", "txn"] == pytest.approx(sp.loc["0%", "txn"], rel=0.10)


def test_filling_at_the_filing_days_close_is_worth_a_measurable_sharpe():
    pc = f4.post_close_cost(n_seeds=5)
    naive = pc.loc["filing day's close", "sharpe"]
    honest = pc.loc["first tradeable session", "sharpe"]
    assert naive > honest
    assert naive - honest > 0.15
    assert pc.loc["filing day's close", "post_close_share"] == pytest.approx(
        f4.POST_CLOSE_SHARE, abs=0.05)


def test_the_code_filter_beats_every_more_inclusive_signal():
    cf = f4.code_filter_table(n_seeds=5)
    assert set(cf.index) == set(f4.SIGNALS)
    p_only = cf.loc["open-market purchases (P)"]
    assert p_only["sharpe"] == cf["sharpe"].max()
    assert p_only["informed_share_of_used"] == pytest.approx(1.0)
    assert p_only["share_of_filings_used"] == pytest.approx(f4.CODE_MIX["P"], abs=0.02)
    net = cf.loc["net insider buying"]
    assert net["share_of_filings_used"] > 0.9      # reads nearly everything
    assert net["informed_share_of_used"] < 0.2     # and almost none of it is a decision
    assert p_only["sharpe"] - cf["sharpe"].min() > 0.3
    # more filings used is not more signal
    assert cf.loc["every Form 4 acquisition", "sharpe"] < p_only["sharpe"]
    assert cf.loc["purchases and sales (P, S)", "sharpe"] < p_only["sharpe"]


def test_signal_sides_are_the_four_the_skill_prices(panel):
    f, _ = panel
    p = f4.signal_sides(f, "open-market purchases (P)")
    assert set(np.unique(p)) == {0.0, 1.0}
    net = f4.signal_sides(f, "net insider buying")
    assert set(np.unique(net)) == {-1.0, 0.0, 1.0}
    assert (net[f["code"].to_numpy() == "F"] == -1.0).all()   # tax withholding reads short
    assert (net[f["code"].to_numpy() == "G"] == 0.0).all()    # a gift has no direction
    with pytest.raises(ValueError):
        f4.signal_sides(f, "nope")


def test_sharpe_convention_is_daily_annualised():
    r = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    assert f4.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert np.isnan(f4.sharpe(np.zeros(5)))


@pytest.mark.slow
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.alt_data.form4")
    for head in ("1. The rule, transcribed", "2. The seeded panel",
                 "3. The A/B", "4. Where the gap comes from",
                 "5. The 22:00 ET window", "6. The code filter"):
        assert head in out
    assert "16a-3(g)(4)" in out and "T+5" in out and "20 transaction codes" in out
    assert out.strip().splitlines()[-1].startswith("Rule:")
    assert all(ord(c) < 128 for c in out)
