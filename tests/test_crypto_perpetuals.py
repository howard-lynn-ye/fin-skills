"""fin_skills.crypto.perpetuals - funding is a cash flow, and the mark is what closes you.

Each test asserts the PROPERTY the SKILL.md documents: the venue's clamp returns the interest
rate whenever the premium is inside the band; the /(8/N) term makes a 4h rate exactly half an
8h rate on the same premium path, so `x 3 x 365` understates by exactly the interval ratio;
funding accumulates to the measured bill and the drag is the difference between two equity
curves; the mark-versus-last liquidation gap fires in BOTH directions and an index-only trigger
fires the rarer one more often; the liquidation level moves toward a long as funding is paid;
open interest is uncorrelated with volume; and the demo is deterministic, ASCII and prints the
rule.
"""
from __future__ import annotations

import functools

import numpy as np
import pytest

from _helpers import is_ascii
from fin_skills.crypto import perpetuals as pp


@functools.lru_cache(maxsize=1)
def _sweep():
    return tuple(pp.liquidation_gap_sweep())


# ------------------------------------------------------------------------------ 1. funding
def test_the_clamp_returns_the_interest_rate_whenever_the_premium_is_inside_the_band():
    r = pp.funding_rates(premium_sd=1e-9)                     # a premium pinned at zero
    assert np.allclose(r, pp.BINANCE_FUNDING["interest_per_interval"], atol=1e-12)
    # and the median of the real path is that same interest component
    assert np.median(pp.funding_rates()) == pytest.approx(
        pp.BINANCE_FUNDING["interest_per_interval"], abs=1e-9)


def test_the_venues_own_divide_by_eight_over_n_halves_a_four_hour_rate():
    eight = pp.funding_rates(n_settlements=400, interval_hours=8)
    four = pp.funding_rates(n_settlements=400, interval_hours=4)
    one = pp.funding_rates(n_settlements=400, interval_hours=1)
    assert np.allclose(four, eight / 2.0, rtol=1e-12)         # same premium path, venue formula
    assert np.allclose(one, eight / 8.0, rtol=1e-12)
    with pytest.raises(ValueError):
        pp.funding_rates(n_settlements=0)


def test_annualising_with_the_wrong_interval_is_wrong_by_exactly_the_interval_ratio():
    r8 = pp.funding_rates(n_settlements=400, interval_hours=8).mean()
    r4 = pp.funding_rates(n_settlements=400, interval_hours=4).mean()
    assert pp.annualise_funding(r8, 8) == pytest.approx(pp.annualise_funding(r4, 4), rel=1e-12)
    naive = r4 * 3.0 * 365.0                                  # the "three settlements a day" habit
    assert naive == pytest.approx(pp.annualise_funding(r4, 4) / 2.0, rel=1e-12)
    with pytest.raises(ValueError):
        pp.annualise_funding(0.0001, 0)


def test_funding_accumulates_to_the_measured_bill_and_scales_with_leverage():
    rates = pp.funding_rates()
    c = pp.funding_cost(rates)
    # SKILL.md sec 1: $9,587 on $100,000 = 9.59% of notional = 95.9% of margin at 10x
    assert c["total_paid"] == pytest.approx(9587.0, abs=2.0)
    assert c["pct_of_notional"] == pytest.approx(0.0959, abs=1e-4)
    assert c["pct_of_margin"] == pytest.approx(c["pct_of_notional"] * 10.0, rel=1e-12)
    assert c["mean_rate"] == pytest.approx(8.76e-5, abs=1e-7)
    assert c["positive_share"] > 0.9
    short = pp.funding_cost(rates, side="short")
    assert short["total_paid"] == pytest.approx(-c["total_paid"], rel=1e-12)
    with pytest.raises(ValueError, match="side"):
        pp.funding_cost(rates, side="flat")


def test_a_price_only_perp_backtest_is_wrong_by_the_accumulated_funding():
    bt = pp.perp_backtest(pp.funding_rates())
    # Fixed quantity; charge funding on each settlement's moving mark, in a cash ledger.
    assert bt["gross_return"] == pytest.approx(0.4767, abs=5e-4)
    assert bt["net_return"] == pytest.approx(0.3682, abs=5e-4)
    assert bt["funding_drag"] == pytest.approx(0.1084, abs=5e-4)
    assert bt["gross_return"] - bt["net_return"] == pytest.approx(bt["funding_cash"])
    assert bt["net_return"] < bt["gross_return"]
    # a short over the same path receives it instead
    flipped = pp.perp_backtest(-pp.funding_rates())
    assert flipped["net_return"] > flipped["gross_return"]


# ------------------------------------------------------------------- 2. mark vs index vs last
def test_the_mark_is_always_between_the_other_two_prices_and_the_run_is_deterministic():
    p = pp.venue_paths(n_paths=40, n_steps=200)
    stack = np.stack([p["price1"], p["price2"], p["last"]])
    assert (p["mark"] >= stack.min(axis=0) - 1e-12).all()
    assert (p["mark"] <= stack.max(axis=0) + 1e-12).all()
    again = pp.venue_paths(n_paths=40, n_steps=200)
    assert np.array_equal(p["mark"], again["mark"])
    assert not np.array_equal(p["mark"], pp.venue_paths(n_paths=40, n_steps=200, seed=1)["mark"])
    for bad in (dict(n_paths=0, n_steps=50), dict(n_paths=4, n_steps=2),
                dict(n_paths=4, n_steps=50, book_lag=0.0),
                dict(n_paths=4, n_steps=50, bar_seconds=0.0)):
        with pytest.raises(ValueError):
            pp.venue_paths(**bad)


def test_the_mark_versus_last_liquidation_gap_fires_in_both_directions():
    rows = {(g["trigger"], g["leverage"]): g for g in _sweep()}
    for (_, lev), g in rows.items():
        assert g["paths"] == 6000
        assert g["last_hits"] > 0
        assert g["disagree_share"] > 0.0                       # the two never agree completely
        assert g["max_gap_bps"] > 10.0
    med, idx = rows[("mark", 50.0)], rows[("index", 50.0)]
    # LAST-only is the big one: a wick the MA has not absorbed does not move a median mark
    assert med["last_only"] > 100 and med["last_only_share"] > 0.05
    # TRIGGER-only - liquidated where the chart never printed - is rarer, and an index-only
    # mark fires it many times more often than the median formula does
    assert 0 < med["trigger_only"] < idx["trigger_only"]
    assert idx["trigger_only"] >= 5 * med["trigger_only"]
    assert idx["max_gap_bps"] > med["max_gap_bps"]


def test_the_documented_fifty_x_row():
    rows = {(g["trigger"], g["leverage"]): g for g in _sweep()}
    med, idx = rows[("mark", 50.0)], rows[("index", 50.0)]
    assert med["liq_level"] - 1.0 == pytest.approx(-0.0161, abs=1e-4)
    assert (med["trigger_hits"], med["last_hits"]) == (2717, 3033)
    assert (med["trigger_only"], med["last_only"]) == (3, 319)
    assert (idx["trigger_only"], idx["last_only"]) == (37, 371)
    assert med["max_gap_bps"] == pytest.approx(265.0, abs=1.0)
    assert idx["max_gap_bps"] == pytest.approx(402.0, abs=1.0)


def test_liquidation_gap_rejects_a_trigger_that_is_not_a_price():
    p = pp.venue_paths(n_paths=8, n_steps=60)
    with pytest.raises(ValueError, match="trigger"):
        pp.liquidation_gap(p, trigger="close")


# --------------------------------------------------------------------------- 3. liquidation
@pytest.mark.parametrize("lev,expect", [(3.0, -0.3307), (5.0, -0.1968), (10.0, -0.0964)])
def test_liquidation_levels_and_what_is_left_at_them(lev, expect):
    lo = pp.liq_ratio(lev)
    assert lo - 1.0 == pytest.approx(expect, abs=1e-4)
    assert lo > 1.0 - 1.0 / lev                                # above bankruptcy, by the mmr
    # the short's level is the mirror image, a shade nearer because the mmr is a haircut
    assert pp.liq_ratio(lev, side="short") - 1.0 == pytest.approx(-expect, abs=0.006)
    assert pp.liq_ratio(lev, side="short") - 1.0 < -expect
    left = (1.0 / lev - (1.0 - lo)) * lev
    assert 0.0 < left < 0.05                                   # a few percent of margin, at most


def test_liq_ratio_rejects_nonsense():
    for bad in (dict(leverage=1.0), dict(leverage=10.0, mmr=1.5)):
        with pytest.raises(ValueError):
            pp.liq_ratio(**bad)
    with pytest.raises(ValueError, match="side"):
        pp.liq_ratio(10.0, side="both")


def test_funding_walks_the_liquidation_level_toward_a_long_with_the_price_flat():
    fw = pp.funding_walks_the_line()
    assert fw["settlements"] == 90 and fw["paid_pct_notional"] == pytest.approx(0.009)
    assert fw["paid_pct_margin"] == pytest.approx(0.09)
    assert fw["liq_end"] > fw["liq_start"]                     # closer to a long's entry
    # the move is exactly the funding paid, grossed by the maintenance haircut
    assert fw["moved_pct"] == pytest.approx(
        fw["paid_pct_notional"] / (1.0 - pp.OKX_TIER1_MMR), rel=1e-12)
    assert fw["liq_start"] - 1.0 == pytest.approx(-0.0964, abs=1e-4)
    assert fw["liq_end"] - 1.0 == pytest.approx(-0.0873, abs=1e-4)


# -------------------------------------------------------------------------------- 4. basis
def test_the_dated_curve_is_upward_sloping_and_the_two_annualisations_bracket_it():
    c = pp.deribit_curve()
    assert list(c.index)[0] == "BTC-25SEP26" and len(c) == 6
    assert c["days"].is_monotonic_increasing and c["basis"].is_monotonic_increasing
    assert (c["compound_yr"] >= c["simple_yr"]).all()           # contango: compounding adds
    assert c.loc["BTC-25SEP26", "simple_yr"] == pytest.approx(0.0310, abs=1e-4)
    assert c.loc["BTC-25JUN27", "simple_yr"] == pytest.approx(0.0483, abs=1e-4)
    assert c.loc["BTC-25DEC26", "days"] == pytest.approx(107.8, abs=0.05)
    with pytest.raises(ValueError):
        pp.annualised_basis(1.0, 1.0, 0.0)


# ------------------------------------------------------------------------ 5. open interest
def test_synthetic_volume_does_not_uniquely_determine_position_opening():
    oi = pp.oi_tape()
    assert abs(oi["corr_daily_volume_oi"]) < 0.05
    assert oi["share_transfers"] == pytest.approx(0.45, abs=0.01)
    assert 0.0 < oi["oi_per_volume"] < 0.2
    assert oi["oi_end"] == pytest.approx(oi["oi_start"] + oi["oi_change"])
    # with no transfers at all OI still cannot be read off volume, only its sign changes
    closed = pp.oi_tape(p_open=0.5, p_close=0.5)
    assert closed["share_transfers"] == pytest.approx(0.0, abs=0.01)
    with pytest.raises(ValueError):
        pp.oi_tape(p_open=0.8, p_close=0.5)
    with pytest.raises(ValueError):
        pp.oi_tape(n_trades=10, per_day=500)


def test_funding_is_on_end_mark_and_kept_in_cash_for_a_fixed_quantity():
    rates = np.array([0.01, -0.02, 0.03])
    flat = pp.perp_backtest(rates, annual_vol=0, drift=0)
    assert flat["gross_return"] == 0
    assert flat["net_return"] == pytest.approx(-rates.sum())
    # No reinvestment: zero price return and funding must sum, not compound.
    assert flat["net_return"] != pytest.approx(np.prod(1 - rates) - 1)
    growing = pp.perp_backtest(rates, annual_vol=0, drift=0.2, interval_hours=24)
    settlement_marks = np.exp(0.2 * np.arange(1, 4) / 365)
    assert growing["funding_cash"] == pytest.approx(np.dot(settlement_marks, rates))


def test_funding_inputs_and_impossible_closes_fail_loudly():
    for rates in ([], [np.nan], [[0.01, 0.02]]):
        with pytest.raises(ValueError):
            pp.funding_cost(rates)
        with pytest.raises(ValueError):
            pp.perp_backtest(rates)
    with pytest.raises(ValueError):
        pp.funding_cost(np.array([0.01]), leverage=0)
    with pytest.raises(ValueError, match="initial_oi"):
        pp.oi_tape(p_open=0, p_close=1, initial_oi=0)


# --------------------------------------------------------------------------------- 6. demo
def test_main_prints_the_rule_in_ascii_and_is_deterministic(run_main, capsys):
    out = run_main("fin_skills.crypto.perpetuals")
    assert is_ascii(out)
    assert pp.THE_RULE in out
    assert out.count("=" * 96) == 4
    for heading in ("FUNDING", "INTERVAL", "MARK vs INDEX vs LAST", "LIQUIDATION", "BASIS",
                    "OPEN INTEREST"):
        assert heading in out
    assert max(len(line) for line in out.splitlines()) <= 98
    pp.main()
    assert capsys.readouterr().out == out
