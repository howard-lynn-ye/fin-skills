"""fin_skills.core.microstructure_measures - estimators scored on streams with planted answers.

The full demo (four parts on a 10-session book and a 40-session trade stream) takes
10-20 s and is marked slow; the invariants it relies on are checked on one session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.core import microstructure_measures as mm


def test_tick_rule_and_prevailing_quote_conventions():
    assert mm.tick_rule(np.array([10.0, 10.01, 10.01, 10.0])).tolist() == [0.0, 1.0, 1.0, -1.0]
    # last quote STRICTLY before (at - lag); -1 when none
    assert mm.prevailing(np.array([1.0, 2.0, 3.0]), np.array([2.5, 0.5, 2.0])).tolist() == [1, -1, 0]
    assert mm.prevailing(np.array([1.0, 2.0, 3.0]), np.array([2.5]), lag=1.0).tolist() == [0]


def test_ols_and_session_bar_ids():
    assert mm.ols(np.array([1.0, 3.0, 5.0, 7.0]), np.array([0.0, 1.0, 2.0, 3.0])) == (2.0, 0.0, 1.0)
    ids = mm.session_bar_ids(np.ones(5), np.array([0, 0, 0, 1, 1]), 2, 2)
    assert ids.tolist() == [0, 0, 1, 2, 2]        # the cumulative weight restarts each session
    assert mm.norm_cdf(np.array([0.0]))[0] == pytest.approx(0.5)
    m = mm.moments(np.random.default_rng(0).normal(size=2000))
    assert m["n"] == 2000 and abs(m["exkurt"]) < 0.5 and abs(m["skew"]) < 0.3


@pytest.fixture(scope="module")
def book():
    return mm.simulate_book(n_days=1, seed=1)


def test_effective_minus_realized_equals_impact_by_construction(book):
    trades, quotes = book["trades"], book["quotes"]
    qt = quotes["t"].to_numpy()
    qmid = 0.5 * (quotes["bid"].to_numpy() + quotes["ask"].to_numpy())
    t, px, q = trades["t"].to_numpy(), trades["px"].to_numpy(), trades["side"].to_numpy()
    idx0 = mm.prevailing(qt, t)
    ok = (idx0 >= 0) & (t < mm.SESSION - 300.0)
    mid0 = qmid[idx0]
    mid1 = qmid[np.searchsorted(qt, t + 300.0, side="right") - 1]
    effective = 2 * q * (px - mid0)
    realized = 2 * q * (px - mid1)
    impact = 2 * q * (mid1 - mid0)
    assert np.allclose((effective - realized)[ok], impact[ok], atol=1e-12)
    assert ok.sum() > 1000
    # a trade at the touch pays the quoted half-spread twice: effective > 0 on average
    assert effective[ok].mean() > 0


def test_book_stream_has_the_documented_shape(book):
    trades, quotes = book["trades"], book["quotes"]
    assert set(trades.columns) >= {"t", "day", "px", "sz", "side", "loc", "delay", "t_vendor"}
    assert set(trades["side"].unique()) == {-1.0, 1.0}
    assert (trades["t_vendor"] > trades["t"]).all()        # the vendor stamps late
    assert quotes["t"].is_monotonic_increasing
    assert (quotes["ask"] > quotes["bid"]).all()
    assert book["lambdas"].tolist() == [mm.LAMBDAS[0]]


def test_simulations_are_seeded():
    a = mm.simulate_book(n_days=1, seed=3)
    b = mm.simulate_book(n_days=1, seed=3)
    pd.testing.assert_frame_equal(a["trades"], b["trades"])
    pd.testing.assert_frame_equal(a["quotes"], b["quotes"])
    x = mm.simulate_trades(2, 5, 1.0, 0.0015, 1.0, 0.0015)
    y = mm.simulate_trades(2, 5, 1.0, 0.0015, 1.0, 0.0015)
    pd.testing.assert_frame_equal(x["trades"], y["trades"])
    assert not x["trades"].equals(mm.simulate_trades(2, 6, 1.0, 0.0015, 1.0, 0.0015)["trades"])


def test_kyle_lambda_regression_recovers_the_planted_impact():
    # no news, no jumps: every mid move is planted impact, so the slope is lambda
    sim = mm.simulate_trades(3, mm.SEED, 1.0, 0.0, 0.0, 0.0)
    assert sim["trade_var_share"] == pytest.approx(1.0)
    tr = sim["trades"]
    for d in range(3):
        m = tr["day"].to_numpy() == d
        tod = tr["t"].to_numpy()[m] - d * 86400.0
        iv = (tod // 300).astype(int)
        last = pd.Series(tr["px"].to_numpy()[m]).groupby(iv).last().to_numpy()
        flow = pd.Series((tr["side"] * tr["sz"]).to_numpy()[m]).groupby(iv).sum().to_numpy()
        slope, _, r2 = mm.ols(np.diff(last), flow[1:])
        assert slope == pytest.approx(sim["lambdas"][d], rel=0.15)
        assert r2 > 0.8
    with_news = mm.simulate_trades(1, mm.SEED, 1.0, 0.005, 4.0, 0.003)
    assert with_news["trade_var_share"] < 1.0


def test_classification_rules_agree_with_the_truth_far_above_chance(book):
    preds = mm.classify(book["trades"], book["quotes"], "t", 0.0)
    truth = book["trades"]["side"].to_numpy()
    assert set(preds) == {"tick rule", "quote rule (mid=unclassified)", "Lee-Ready", "EMO"}
    assert np.mean(preds["Lee-Ready"] == truth) > 0.8
    assert np.mean(preds["tick rule"] == truth) > 0.6


@pytest.mark.slow
def test_demo_runs_all_four_parts_and_states_the_rule(run_main):
    out = run_main("fin_skills.core.microstructure_measures")
    for part in ("PART 1", "PART 2", "PART 3", "PART 4"):
        assert part in out
    assert "Rule: score every microstructure estimator on data where you planted the answer" in out
