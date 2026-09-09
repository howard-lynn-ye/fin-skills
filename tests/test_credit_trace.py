"""fin_skills.credit.trace - the TRACE 15-minute window and the dissemination size caps.

The documented properties: the reporting table says 15 minutes for corporate bonds and 60 for
Treasuries and never says one minute; a seeded tape is reproducible; the caps censor size from
above so reported volume is a LOWER BOUND and the median survives while the mean does not; an
Amihud ratio built on censored volume is biased UP but not by the hidden-volume fraction; two
bonds with identical liquidity are ranked apart by the caps alone; and a right-censored
lognormal fit turns a 47% error into a 1.6% one.
"""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.credit.trace import (CAP_HIGH_YIELD, CAP_INVESTMENT_GRADE, LOG_SIZE_SD, amihud,
                                     amihud_bias, apply_cap, censored_lognormal_mle,
                                     censoring_report, cross_sectional_distortion,
                                     dissemination_caps, recovery_report, reporting_timeframes,
                                     simulate_tape, staleness)


# ------------------------------------------------------------------ the regulatory facts
def test_the_reporting_table_says_fifteen_minutes_and_never_one_minute():
    rows = reporting_timeframes()
    corp = [r for r in rows if "Corporate" in r["security"]]
    assert len(corp) == 1
    assert "15 minutes" in corp[0]["limit"]
    assert "6730(a)(1)" in corp[0]["rule"]
    # the whole point of the skill: the string "one minute" must not appear as a live limit
    assert not any("one minute" in r["limit"].lower() or "1 minute" in r["limit"].lower()
                   for r in rows)
    # Treasuries are 60 minutes, so a single latency assumption across the feed is wrong
    tsy = [r for r in rows if "Treasury" in r["security"]]
    assert len(tsy) == 1 and "60 minutes" in tsy[0]["limit"]


def test_the_two_dissemination_caps_are_five_and_one_million():
    caps = {c["grade"]: c for c in dissemination_caps()}
    assert caps["Investment grade"]["cap"] == CAP_INVESTMENT_GRADE == 5_000_000.0
    assert caps["Investment grade"]["printed_as"] == "5MM+"
    assert caps["Non-investment grade"]["cap"] == CAP_HIGH_YIELD == 1_000_000.0
    assert caps["Non-investment grade"]["printed_as"] == "1MM+"


def test_staleness_is_sqrt_of_time_and_fifteen_minutes_is_3_87x_one_minute():
    rows = staleness()
    assert rows[0]["minutes"] == 1.0 and rows[1]["minutes"] == 15.0
    assert rows[1]["vs_one_minute"] == pytest.approx(np.sqrt(15.0), abs=1e-12)
    assert rows[1]["vs_one_minute"] == pytest.approx(3.87, abs=0.005)
    assert rows[0]["sd_bp"] == pytest.approx(1.59, abs=0.005)
    assert rows[1]["sd_bp"] == pytest.approx(6.18, abs=0.005)
    with pytest.raises(ValueError):
        staleness(annual_vol=0.0)


# ------------------------------------------------------------------ the tape
def test_the_seeded_tape_is_deterministic_and_reproducible():
    a, b = simulate_tape(), simulate_tape()
    assert np.array_equal(a.size, b.size) and np.array_equal(a.price, b.price)
    assert len(a) == 3000 and a.day.max() == 249
    assert simulate_tape(seed=1).size[0] != a.size[0]
    with pytest.raises(ValueError):
        simulate_tape(n_days=0)


def test_applying_a_cap_can_only_shrink_a_size():
    t = simulate_tape()
    for cap in (CAP_INVESTMENT_GRADE, CAP_HIGH_YIELD):
        rep = apply_cap(t.size, cap)
        assert np.all(rep <= t.size) and np.all(rep <= cap)
        assert rep.sum() < t.size.sum(), "reported volume is a strict lower bound here"
    with pytest.raises(ValueError):
        apply_cap(t.size, 0.0)


# ------------------------------------------------------------------ what the cap does
def test_two_percent_of_prints_hide_sixteen_percent_of_the_volume():
    ig = censoring_report(simulate_tape(), CAP_INVESTMENT_GRADE)
    assert ig["capped_trades_pct"] == pytest.approx(1.97, abs=0.005)
    assert ig["hidden_volume_pct"] == pytest.approx(16.28, abs=0.005)
    assert ig["vwap_error_bp"] == pytest.approx(-3.92, abs=0.005)
    hy = censoring_report(simulate_tape(), CAP_HIGH_YIELD)
    assert hy["capped_trades_pct"] == pytest.approx(17.90, abs=0.005)
    assert hy["hidden_volume_pct"] == pytest.approx(47.49, abs=0.005)
    # the trade count barely moves while the volume collapses - that is the whole trap
    assert ig["capped_trades_pct"] < 2.0 < 16.0 < ig["hidden_volume_pct"]


def test_the_median_survives_the_censoring_and_the_mean_does_not():
    for cap in (CAP_INVESTMENT_GRADE, CAP_HIGH_YIELD):
        r = censoring_report(simulate_tape(), cap)
        assert r["reported_median_size"] == pytest.approx(r["true_median_size"], abs=1e-9)
        assert r["reported_mean_size"] < r["true_mean_size"]
        # mean size and total volume carry the identical bias, by construction
        assert r["mean_size_error_pct"] == pytest.approx(-r["hidden_volume_pct"], abs=1e-9)


def test_the_vwap_moves_away_from_the_mid_because_big_trades_pay_less():
    t = simulate_tape()
    r = censoring_report(t, CAP_INVESTMENT_GRADE)
    assert r["vwap_reported"] != r["vwap_true"]
    assert abs(r["vwap_error_bp"]) > 1.0, "not a rounding error on an execution benchmark"


# ------------------------------------------------------------------ Amihud
def test_the_amihud_bias_is_always_up_and_never_the_volume_fraction():
    rows = amihud_bias(simulate_tape())
    for r in rows:
        assert r["amihud_reported"] > r["amihud_true"], "censoring can only raise the ratio"
        assert 0.0 < r["overstatement_pct"] < r["naive_guess_pct"], \
            "1/(1-hidden) overshoots: Amihud is dominated by days with no capped print"
    assert rows[0]["overstatement_pct"] == pytest.approx(2.89, abs=0.005)
    assert rows[1]["overstatement_pct"] == pytest.approx(41.08, abs=0.005)
    assert rows[0]["naive_guess_pct"] / rows[0]["overstatement_pct"] == pytest.approx(6.7, abs=0.1)


def test_the_caps_alone_rank_two_identical_bonds_apart():
    x = cross_sectional_distortion(simulate_tape())
    assert x["amihud_true_ratio"] == 1.0
    assert x["amihud_reported_ratio"] == pytest.approx(1.3712, abs=5e-5)
    assert x["hy_looks_more_illiquid_pct"] == pytest.approx(37.12, abs=0.005)


def test_amihud_defaults_to_the_true_size():
    t = simulate_tape()
    assert amihud(t) == amihud(t, t.size)
    assert amihud(t, apply_cap(t.size, CAP_HIGH_YIELD)) > amihud(t)


# ------------------------------------------------------------------ the fix
def test_the_censored_fit_recovers_what_the_naive_average_cannot():
    for cap, naive, mle in ((CAP_INVESTMENT_GRADE, -16.28, -0.49), (CAP_HIGH_YIELD, -47.49, 1.64)):
        r = recovery_report(simulate_tape(), cap)
        assert r["naive_error_pct"] == pytest.approx(naive, abs=0.005)
        assert r["mle_error_pct"] == pytest.approx(mle, abs=0.005)
        assert abs(r["mle_error_pct"]) < abs(r["naive_error_pct"]) / 10.0
        assert r["naive_total"] < r["true_total"]
        assert r["sigma_hat"] == pytest.approx(LOG_SIZE_SD, abs=0.05)


def test_the_fit_uses_the_capped_prints_as_information_not_as_their_face_value():
    t = simulate_tape()
    rep = apply_cap(t.size, CAP_HIGH_YIELD)
    fit = censored_lognormal_mle(rep, CAP_HIGH_YIELD)
    assert fit["converged"] == 1.0
    assert fit["n_censored"] == pytest.approx((t.size > CAP_HIGH_YIELD).sum(), abs=0)
    # taking "1MM+" at face value is exactly the naive answer, and it is far off
    assert fit["mean_size"] > rep.mean() * 1.5
    with pytest.raises(ValueError):
        censored_lognormal_mle(np.array([1.0, -2.0]), CAP_HIGH_YIELD)


def test_an_uncensored_sample_leaves_the_fit_at_the_plain_lognormal_mle():
    t = simulate_tape()
    huge = float(t.size.max()) * 10.0
    fit = censored_lognormal_mle(apply_cap(t.size, huge), huge)
    assert fit["n_censored"] == 0.0
    lx = np.log(t.size)
    assert fit["mu"] == pytest.approx(lx.mean(), abs=1e-4)
    assert fit["sigma"] == pytest.approx(lx.std(ddof=0), abs=1e-4)


# ------------------------------------------------------------------ the demo
def test_the_demo_prints_the_rule_and_stays_ascii(run_main):
    out = run_main("fin_skills.credit.trace")
    assert all(ord(c) < 128 for c in out), "a non-ASCII char dies on a stock Windows console"
    assert "THE RULE:" in out
    assert "15 MINUTES, not one" in out
    assert "25-17" in out and "2026-06-08" in out
    for token in ("16.28%", "47.49%", "1.97%", "37.12%", "-3.92 bp"):
        assert token in out
