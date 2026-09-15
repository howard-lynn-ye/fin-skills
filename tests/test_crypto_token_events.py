"""fin_skills.crypto.token_events - crypto's corporate actions and what they do to a series.

Each test asserts the PROPERTY the SKILL.md documents rather than the printed digits: the
rebase identity (1+r_holder) = (1+r_price)(1+r_supply) holds to machine precision and the
price-only return is wrong by the documented amount; a redenomination prints exactly one day of
-(1 - 1/ratio) and dividing the pre-event prints by the ratio restores the holder's series
EXACTLY; survivorship runs in both directions, is monotone in the hazard, and a survivor-only
cohort overstates; the wrapper's basis is a few bps with a fat tail and still costs tracking
error; a fork conserves value; and the demo is deterministic, ASCII, and prints the rule.
"""
from __future__ import annotations

import numpy as np
import pytest

from _helpers import is_ascii
from fin_skills.crypto import token_events as te


# ------------------------------------------------------------------------------- 1. rebase
def test_rebase_identity_is_exact_at_every_horizon():
    m = te.rebase_returns(te.rebase_history())
    assert m["identity_residual"] < 1e-12
    assert m["daily_identity_max"] < 1e-12
    # the identity restated: the holder return is the price return COMPOUNDED with the supply
    assert (1 + m["value_return"]) == pytest.approx(
        (1 + m["price_return"]) * (1 + m["supply_return"]), rel=1e-12)


def test_price_only_return_is_wrong_by_the_documented_amount():
    m = te.rebase_returns(te.rebase_history())
    # SKILL.md sec 2: price +2.40%, supply +136.82%, holder +142.49%, gap 140.09%
    assert m["price_return"] == pytest.approx(0.0240, abs=5e-4)
    assert m["supply_return"] == pytest.approx(1.3682, abs=5e-4)
    assert m["value_return"] == pytest.approx(1.4249, abs=5e-4)
    assert m["value_return"] - m["price_return"] == pytest.approx(1.4009, abs=1e-3)
    # and the Sharpe is understated by a factor of about 4.6
    assert m["price_sharpe"] == pytest.approx(0.2608, abs=5e-4)
    assert m["value_sharpe"] == pytest.approx(1.1973, abs=5e-4)
    assert m["value_sharpe"] / m["price_sharpe"] > 4.0
    # the two series are near-perfectly correlated day by day - that is what hides it
    assert m["corr"] > 0.95


def test_rebase_holder_value_is_the_network_value_and_the_run_is_deterministic():
    df = te.rebase_history()
    value = df["price"] * df["supply_index"]
    ratio = value / df["mcap"]
    assert np.allclose(ratio, ratio.iloc[0], rtol=1e-12)      # non-dilutive: a constant share
    again = te.rebase_history()
    assert np.array_equal(df.to_numpy(), again.to_numpy())
    assert not np.array_equal(df.to_numpy(), te.rebase_history(seed=1).to_numpy())
    for bad in (dict(n_days=1), dict(lag=0.0), dict(target=-1.0)):
        with pytest.raises(ValueError):
            te.rebase_history(**bad)


# ---------------------------------------------------------------------- 2. redenomination
@pytest.mark.parametrize("ratio", [10.0, 60.0, 0.5])
def test_redenomination_prints_exactly_one_factor_day_and_the_factor_repairs_it(ratio):
    event = 250
    df = te.redenominated_series(new_per_old=ratio, event_day=event)
    raw = (1.0 + df["raw"].pct_change().dropna()).to_numpy()
    true = (1.0 + df["true_old_unit"].pct_change().dropna()).to_numpy()
    contaminated = raw / true                                 # 1.0 everywhere except the event
    assert contaminated[event - 1] == pytest.approx(1.0 / ratio, rel=1e-12)
    assert np.allclose(np.delete(contaminated, event - 1), 1.0, rtol=1e-12)
    d = te.redenomination_damage(df)
    assert d["adjusted_matches_true"] < 1e-12                 # the factor restores it EXACTLY
    assert d["raw_total"] != pytest.approx(d["true_total"], abs=1e-6)


def test_the_stratis_case_is_the_one_that_fails_silently():
    assert te.STRAX_SWAP["ticker_before"] == te.STRAX_SWAP["ticker_after"]   # same symbol
    assert te.GAL_SWAP["ticker_before"] != te.GAL_SWAP["ticker_after"]       # join breaks loudly
    assert te.STRAX_SWAP["new_per_old"] == 10.0 and te.GAL_SWAP["new_per_old"] == 60.0
    d = te.redenomination_damage(te.redenominated_series(
        new_per_old=te.STRAX_SWAP["new_per_old"]))
    # SKILL.md sec 3: unadjusted -89.11% / worst day -90.73% against the holder's +8.86%
    assert d["raw_total"] == pytest.approx(-0.8911, abs=5e-4)
    assert d["true_total"] == pytest.approx(0.0886, abs=5e-4)
    assert d["raw_worst_day"] == pytest.approx(-0.9073, abs=5e-4)
    assert d["raw_vol"] > d["true_vol"] and d["raw_maxdd"] < d["true_maxdd"]
    with pytest.raises(ValueError, match="inside the sample"):
        te.redenominated_series(n_days=100, event_day=100)


# ------------------------------------------------------------------------- 3. survivorship
def test_survivorship_runs_in_both_directions_and_is_monotone_in_the_hazard():
    rates = [te.survivorship(te.venue_history(annual_delist=h)) for h in (0.10, 0.20, 0.35)]
    fwd = [s["survival_rate"] for s in rates]
    assert fwd == sorted(fwd, reverse=True)                   # a higher hazard kills more
    for s in rates:
        assert 0.0 < s["survival_rate"] < 1.0
        assert 0.0 < s["backward_rate"] < 1.0                 # today's list is partly post-hoc
        assert s["realised_annual_delist"] > 0.0
        # In these fixed synthetic paths the survivor-only selection looks better.
        assert s["survivor_mean_return"] > s["cohort_mean_return"]
        assert s["overstatement"] > 0.0


def test_the_documented_middle_row():
    s = te.survivorship(te.venue_history(annual_delist=0.20))
    assert s["realised_annual_delist"] == pytest.approx(0.232, abs=2e-3)
    assert s["survived"] == 33 and s["survival_rate"] == pytest.approx(0.66)
    assert s["existed_then"] == 29 and s["backward_rate"] == pytest.approx(0.58)
    assert s["cohort_mean_return"] == pytest.approx(0.596, abs=2e-3)
    assert s["survivor_mean_return"] == pytest.approx(1.110, abs=2e-3)
    assert s["overstatement"] == pytest.approx(0.514, abs=3e-3)


def test_top_n_as_of_only_ever_returns_pairs_trading_on_that_day():
    u = te.venue_history(annual_delist=0.35)
    for day in (0, 400, u["n_days"] - 1):
        picks = te.top_n_as_of(u, day, n=40)
        assert len(picks) == len(set(picks.tolist())) == 40
        assert (u["listed_on"][picks] <= day).all()
        assert (u["dead_on"][picks] > day).all()
    with pytest.raises(ValueError):
        te.venue_history(annual_delist=1.5)


def test_short_universe_is_not_padded_with_dead_or_unlisted_pairs():
    u = {"n_days": 4, "logv": np.zeros((3, 4)), "listed_on": np.array([0, 0, 3]),
         "dead_on": np.array([4, 1, 4])}
    assert te.top_n_as_of(u, 2, n=50).tolist() == [0]
    u["dead_on"][:] = 0
    assert te.top_n_as_of(u, 2).size == 0
    with pytest.raises(ValueError):
        te.top_n_as_of(u, 4)


def test_ranking_cannot_use_volume_before_listing_or_after_the_decision():
    u = {"n_days": 4, "logv": np.log(np.array([[10, 10, 10, 10], [1e6, 1e6, 1, 1e9]])),
         "listed_on": np.array([0, 2]), "dead_on": np.array([4, 4])}
    assert te.top_n_as_of(u, 2, n=1).tolist() == [0]
    u["logv"][1, :2] = 100
    u["logv"][1, 3] = 100
    assert te.top_n_as_of(u, 2, n=1).tolist() == [0]


# ------------------------------------------------------------------- 4. wrapped / bridged
def test_wrapped_basis_is_tight_with_a_fat_tail_and_still_costs_tracking_error():
    m = te.wrapped_metrics(te.wrapped_basis())
    assert m["median_abs_bps"] < 10.0                         # "a few bps" in the calm state
    assert m["p99_abs_bps"] > 20 * m["median_abs_bps"]        # and a tail two orders wider
    assert m["worst_bps"] < -100.0
    assert m["correlation"] > 0.99                            # what makes it look substitutable
    assert m["tracking_error_bps"] > 100.0                    # and what it actually costs
    assert m["days_beyond"] > 0
    # SKILL.md sec 5
    assert m["median_abs_bps"] == pytest.approx(2.7, abs=0.1)
    assert m["worst_bps"] == pytest.approx(-515.0, abs=2.0)
    assert m["tracking_error_bps"] == pytest.approx(576.0, abs=2.0)


# -------------------------------------------------------------------------------- 5. forks
@pytest.mark.parametrize("share", [0.0, 0.02, 0.10, 0.25, 1.0])
def test_the_value_preserving_fork_illustration_keeps_both_assets(share):
    f = te.fork_split(100.0, share)
    assert f["total"] == pytest.approx(100.0)                 # nothing is created at a split
    assert f["ticker_only_return"] == pytest.approx(-share)
    assert f["holder_return"] == 0.0
    assert f["understatement"] == pytest.approx(-share)


def test_fork_split_rejects_a_share_outside_zero_one():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        te.fork_split(100.0, 1.5)


# --------------------------------------------------------------------------------- 6. demo
def test_main_prints_the_rule_in_ascii_and_is_deterministic(run_main, capsys):
    out = run_main("fin_skills.crypto.token_events")
    assert is_ascii(out)
    assert te.THE_RULE in out
    assert out.count("=" * 96) == 4                          # a banner and a rule box
    for heading in ("REBASE", "REDENOMINATION", "SURVIVORSHIP", "WRAPPED", "FORKS"):
        assert heading in out
    assert max(len(line) for line in out.splitlines()) <= 98
    te.main()
    assert capsys.readouterr().out == out
