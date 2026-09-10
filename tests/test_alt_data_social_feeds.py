"""fin_skills.alt_data.social_feeds - what you cannot get, and what the rest does.

The documented properties: the access table marks the pages that could not be reached
rather than filling them in; the cost table is arithmetic on X's published price and the
cap; a post's outcome is the forward return from its creation day; deletion is correlated
with being wrong, so the surviving sample's hit rate is inflated above the true 50% and the
inflation grows with the asymmetry; selecting accounts in-sample manufactures a large
Sharpe in a panel where nobody has any skill, and selecting out-of-sample does not; and
keying on created_at rather than the indexed day is worth a measurable Sharpe.
"""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.alt_data import social_feeds as sf


@pytest.fixture(scope="module")
def panel():
    return sf.make_panel(seed=sf.SEED)


def test_the_access_table_marks_what_could_not_be_reached():
    at = sf.access_table()
    assert set(at["platform"]) == {"X", "Reddit", "StockTwits"}
    assert set(at["status"]) <= {"OK", "BLOCKED", "ABSENT"}
    blocked = at[at["status"] == "BLOCKED"]
    assert len(blocked) == 3
    topics = set(blocked["topic"])
    assert "developer agreement" in topics                  # X, HTTP 402
    assert "developer portal / plan list" in topics         # X, login wall
    assert "Data API Terms themselves" in topics            # Reddit, unreachable
    # a blocked row has to SAY it is blocked in its own note, not just in the column
    for note in blocked["note"]:
        assert any(w in note.lower() for w in ("could not", "402", "login wall"))
    # the one row that is an absence rather than a block is marked differently
    absent = at[at["status"] == "ABSENT"]
    assert list(absent["topic"]) == ["academic research product"]


def test_the_access_table_carries_the_quotes_the_skill_relies_on():
    notes = " ".join(sf.access_table()["note"])
    assert "pay-per-usage pricing" in notes            # X has no tiers
    assert "3 million Post reads" in notes
    assert "1,500,000 Post IDs" in notes and "30 day period" in notes
    assert "Negative financial status" in notes
    assert "Aggregate analysis" in notes               # the carve-out
    assert "100 queries per minute" in notes
    assert "will be blocked" in notes                  # Reddit unauthenticated
    assert "Reddit For Researchers" in notes
    assert "404" in notes                              # StockTwits docs


def test_the_hiq_record_does_not_overstate_the_cfaa_ruling():
    assert "serious question" in sf.HIQ["cfaa_holding"]
    assert "NOT a merits holding" in sf.HIQ["cfaa_holding"]
    assert "PRELIMINARY INJUNCTION" in sf.HIQ["cfaa_posture"]
    assert "Van Buren" in sf.HIQ["cfaa_posture"]
    assert sf.HIQ["money_usd"] == 500_000
    assert "algorithms developed at hiQ" in sf.HIQ["injunction"]
    assert "BREACHED" in sf.HIQ["contract_result"]
    assert "secondhand" in sf.HIQ["source_note"]


def test_the_cost_table_is_arithmetic_on_the_published_price():
    assert sf.X_PRICES_USD["post_read"] == 0.005
    assert sf.X_MONTHLY_POST_READ_CAP == 3_000_000
    assert sf.X_ID_REDISTRIBUTION_PER_30_DAYS == 1_500_000
    assert sf.X_HYDRATED_PER_RECIPIENT_PER_DAY == 50_000
    assert sf.REDDIT_QPM_PER_OAUTH_CLIENT == 100
    assert sf.REDDIT_QPM_WINDOW_MINUTES == 10
    c = sf.read_cost(1_000, 365)
    assert c["posts"] == 365_000
    assert c["total_usd"] == pytest.approx(1_825.0)
    assert c["over_cap"] is False
    big = sf.read_cost(200_000, 365)
    assert big["over_cap"] is True
    tab = sf.cost_table()
    assert tab.loc["at the cap", "usd/month"] == pytest.approx(15_000.0)
    assert tab.loc["at the cap", "pct of monthly cap"] == pytest.approx(1.0)
    assert tab.loc["1,000/day", "usd/year"] == pytest.approx(1_825.0)


def test_attention_is_concentrated_so_a_post_count_is_not_a_bet_count():
    p = sf.attention_weights(200)
    assert p.sum() == pytest.approx(1.0)
    assert (np.diff(p) < 0).all()                 # ranked
    eff = sf.effective_names(200)
    assert 5 < eff < 200 / 2
    assert sf.effective_names(200, 2.0) < eff     # a steeper tail concentrates further


def test_forward_returns_exclude_the_day_itself():
    r = np.arange(12, dtype=float).reshape(6, 2)
    f = sf.forward_returns(r, 2)
    assert f[0, 0] == pytest.approx(r[1, 0] + r[2, 0])
    assert f[3, 1] == pytest.approx(r[4, 1] + r[5, 1])
    assert np.isnan(f[5]).all()                   # nothing after the last row


def test_the_panel_is_deterministic_and_has_the_three_timestamps(panel):
    posts, rets = panel
    a, ra = sf.make_panel(seed=sf.SEED)
    assert posts.equals(a) and np.array_equal(rets, ra)
    b, _ = sf.make_panel(seed=sf.SEED + 1)
    assert not np.array_equal(posts["call"].to_numpy(), b["call"].to_numpy())
    assert (posts["index_day"] >= posts["create_day"]).all()
    assert (posts.loc[posts["deleted"], "delete_day"] > posts.loc[posts["deleted"],
                                                                  "index_day"]).all()
    assert (posts.loc[~posts["deleted"], "delete_day"] == -1).all()
    il = sf.index_lag_summary(posts)
    assert il["share_same_day"] == pytest.approx(sf.INDEX_LAG_PMF[0], abs=0.02)
    assert il["share_next_day_or_later"] == pytest.approx(1 - sf.INDEX_LAG_PMF[0], abs=0.02)
    assert il["max"] == 5


def test_nobody_has_skill_by_default_and_deletion_targets_the_wrong_calls(panel):
    posts, _ = panel
    assert not posts["skilled"].any()
    assert posts["correct"].mean() == pytest.approx(0.5, abs=0.01)
    # a wrong call is deleted more often - that is the whole mechanism
    d_wrong = posts.loc[~posts["correct"], "deleted"].mean()
    d_right = posts.loc[posts["correct"], "deleted"].mean()
    assert d_wrong > d_right + 0.10
    assert d_right == pytest.approx(0.10, abs=0.02)
    assert d_wrong == pytest.approx(0.25, abs=0.02)


def test_entry_is_the_next_day_and_the_side_is_the_call():
    import pandas as pd
    p = pd.DataFrame({"name": [0, 1], "index_day": [10, 10], "call": [1.0, -1.0]})
    rets = np.zeros((30, 4))
    rets[10, 0] = 1.0
    rets[11, 0] = 0.4
    rets[11, 1] = -0.4
    r = sf.portfolio(p, rets, "index_day", hold=3)
    assert r[10] == 0.0                     # nothing earned on the key day
    assert r[11] > 0.0                      # long the up name, short the down one
    assert r[14] == 0.0                     # the hold has expired


def test_a_historical_pull_inflates_the_hit_rate_of_a_coin_flip():
    sv = sf.survivorship(n_seeds=3)
    live = sv.loc["every post (live)"]
    hist = sv.loc["surviving posts (historical pull)"]
    assert live["hit_rate"] == pytest.approx(0.50, abs=0.01)
    assert hist["hit_rate"] > live["hit_rate"] + 0.03
    assert hist["sharpe"] > live["sharpe"] + 1.0
    assert live["share_of_posts"] == 1.0
    assert 0.75 < hist["share_of_posts"] < 0.90


def test_the_inflation_scales_with_the_deletion_asymmetry_and_vanishes_at_zero():
    g = sf.survivorship_grid(n_seeds=2, extras=(0.0, 0.15, 0.30))
    assert (g["true_hit_rate"] == 0.5).all()
    hits = g["apparent_hit_rate"].to_numpy()
    sharpes = g["apparent_sharpe"].to_numpy()
    assert hits[0] == pytest.approx(0.50, abs=0.01)      # the control
    assert abs(sharpes[0]) < 0.6
    assert hits[1] > hits[0] and hits[2] > hits[1]
    assert sharpes[1] > sharpes[0] and sharpes[2] > sharpes[1]
    assert hits[2] > 0.55


def test_in_sample_account_selection_manufactures_the_result():
    ps = sf.panel_selection(n_seeds=3)
    assert set(ps.index) == {"all accounts", "selected out-of-sample",
                             "selected in-sample"}
    base = ps.loc["all accounts", "sharpe"]
    assert abs(base) < 1.0                                   # zero skill, so ~0
    assert ps.loc["selected in-sample", "sharpe"] > base + 1.5
    # out-of-sample selection finds nothing, because there is nothing to find
    assert ps.loc["selected out-of-sample", "sharpe"] < base + 1.0
    assert ps.loc["all accounts", "vs all accounts"] == 0.0
    assert ps.loc["selected in-sample", "vs all accounts"] > 1.5


def test_account_scores_ranks_accounts_on_realised_calls(panel):
    posts, _ = panel
    s = sf.account_scores(posts)
    assert len(s) > 300 and s.index.name == "account"
    half = sf.account_scores(posts, slice(0, 600))
    assert not np.allclose(s.reindex(half.index).to_numpy(), half.to_numpy())


def test_keying_on_created_at_is_a_look_ahead():
    ta = sf.timestamp_ab(n_seeds=3)
    created = ta.loc["created_at (look-ahead)", "sharpe"]
    indexed = ta.loc["indexed_at (honest)", "sharpe"]
    assert created > indexed > 0
    assert ta.loc["indexed_at (honest)", "gap"] == 0.0
    assert ta.loc["created_at (look-ahead)", "gap"] == pytest.approx(created - indexed)
    assert 0.5 < ta.loc["created_at (look-ahead)", "retained"] < 1.0


def test_sharpe_convention_is_daily_annualised():
    r = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    assert sf.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert np.isnan(sf.sharpe(np.zeros(5)))


def test_no_scraper_and_no_credential_path_anywhere_in_the_module():
    src = __import__("inspect").getsource(sf)
    for banned in ("requests", "urllib", "httpx", "aiohttp", "praw", "tweepy",
                   "bearer_token", "api_key", "client_secret", "os.environ"):
        assert banned not in src, f"{banned!r} must not appear in a no-scraper module"


@pytest.mark.slow
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.alt_data.social_feeds")
    for head in ("1. What you can actually get", "2. What reading costs",
                 "3. Three timestamps", "4. Deletion survivorship",
                 "5. Panel selection", "6. The timestamp A/B"):
        assert head in out
    assert "[BLOCKED]" in out and "hiQ" in out and "$500,000" in out
    assert "pay-per-usage" in out
    assert out.strip().splitlines()[-1].startswith("Rule:")
    assert all(ord(c) < 128 for c in out)
