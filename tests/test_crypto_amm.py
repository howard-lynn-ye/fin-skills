"""fin_skills.crypto.amm - constant product, concentrated liquidity, and what an LP is short.

Each test asserts the PROPERTY the SKILL.md documents, not the printed digits: the closed-form
swap output equals the same trade walked along the curve in 200,000 slices (path independence);
the executed price moves exactly half as far in log terms as the pool price; the impermanent-loss
closed form 2*sqrt(r)/(1+r) reproduces reserve arithmetic to 1e-15 and is symmetric in r and 1/r;
the break-even volume-to-TVL grows with the square of vol; a range's loss amplification equals
its capital multiple exactly for a move that stays inside the range; and a sandwich extracts a
constant share of whatever slippage tolerance is set.
"""
from __future__ import annotations

import numpy as np
import pytest

from _helpers import is_ascii
from fin_skills.crypto import amm


# --------------------------------------------------------------------------- 1. the invariant
def test_the_closed_form_swap_is_the_same_fill_as_walking_the_curve():
    x = y = 1_000_000.0
    exact = amm.cp_swap(x, y, 50_000.0, fee=0.0)["dy"]
    sliced = amm.swap_by_slices(x, y, 50_000.0, slices=200_000, fee=0.0)
    assert abs(exact - sliced) / exact < 1e-12               # path independence, to precision
    # and the invariant is preserved exactly when the fee is zero
    d = amm.cp_swap(x, y, 50_000.0, fee=0.0)
    assert d["k_after"] == pytest.approx(d["k_before"], rel=1e-12)
    # with a fee it can only grow - that growth IS the LP's income
    withfee = amm.cp_swap(x, y, 50_000.0)
    assert withfee["k_after"] > withfee["k_before"]
    assert withfee["dy"] < d["dy"]


@pytest.mark.parametrize("u", [0.001, 0.01, 0.05, 0.10, 0.50, 2.0])
def test_the_executed_price_moves_exactly_half_as_far_in_logs_as_the_pool(u):
    c = amm.cp_impact(u)
    assert np.log(c["exec_ratio"]) / np.log(c["pool_ratio"]) == pytest.approx(0.5, rel=1e-12)
    assert c["slippage"] < 0.0 and c["pool_move"] < c["slippage"]
    # the closed form agrees with an actual swap on a pool of any size
    d = amm.cp_swap(1_000_000.0, 1_000_000.0, u * 1_000_000.0, fee=0.0)
    assert d["exec_price"] / d["spot_before"] == pytest.approx(c["exec_ratio"], rel=1e-12)
    assert d["spot_after"] / d["spot_before"] == pytest.approx(c["pool_ratio"], rel=1e-12)


def test_swap_and_impact_reject_nonsense():
    for bad in (dict(x=0.0, y=1.0, dx=1.0), dict(x=1.0, y=1.0, dx=-1.0),
                dict(x=1.0, y=1.0, dx=1.0, fee=1.5)):
        with pytest.raises(ValueError):
            amm.cp_swap(**bad)
    with pytest.raises(ValueError):
        amm.cp_impact(-0.1)
    with pytest.raises(ValueError):
        amm.swap_by_slices(1.0, 1.0, 0.1, slices=0)


@pytest.mark.parametrize("fee", [0.0005, 0.003, 0.01])
@pytest.mark.parametrize("u", [0.01, 0.5])
def test_fee_aware_impact_matches_the_actual_reserve_change(fee, u):
    trade = amm.cp_swap(400, 900, u * 400, fee=fee)
    impact = amm.cp_impact(u, fee=fee)
    assert impact["pool_ratio"] == pytest.approx(trade["spot_after"] / trade["spot_before"])
    assert impact["exec_ratio"] == pytest.approx(trade["exec_price"] / trade["spot_before"])
    assert impact["pool_ratio"] > (1 + u) ** -2  # fee stays in the input reserve


def test_retained_fees_break_the_zero_fee_slicing_identity():
    one = amm.cp_swap(100, 100, 30, fee=0.01)["dy"]
    sliced = amm.swap_by_slices(100, 100, 30, slices=50, fee=0.01)
    assert sliced < one
    with pytest.raises(ValueError):
        amm.il_vs_fees(0.3, fee=0)
    with pytest.raises(ValueError):
        amm.sandwich(100, 100, 0, 0.01)


# ---------------------------------------------------------------------------------- 2. fees
def test_a_fee_on_the_input_is_a_bigger_fraction_of_the_output():
    assert amm.fee_on_withdrawn(0.003) == pytest.approx(3.0 / 997.0, rel=1e-14)
    assert amm.fee_on_withdrawn(0.003) > 0.003                # 0.300903% vs 0.3000%
    assert amm.fee_on_withdrawn(0.0) == 0.0
    assert amm.V2["lp_share"] + amm.V2["protocol_share"] == pytest.approx(amm.V2["fee"])
    assert amm.V2["protocol_share"] == pytest.approx(amm.V2["fee"] * amm.V2["protocol_cut"])
    assert amm.V2["fee"] in amm.V3["fee_tiers"]               # 0.30% is the middle v3 tier


def test_the_v2_fee_accumulator_is_the_growth_in_sqrt_k():
    t = amm.simulate_tape()
    assert t["k1"] > t["k0"]                                  # fees can only raise the invariant
    g = t["sqrt_k_growth"]
    assert t["fee_growth"] == pytest.approx(g / (1.0 + g), rel=1e-12)   # 1 - sqrt(k0/k1)
    assert 0.0 < t["fee_growth"] < 1.0
    assert amm.simulate_tape()["k1"] == t["k1"]               # deterministic
    with pytest.raises(ValueError):
        amm.fee_growth_from_k(0.0, 1.0)


# -------------------------------------------------------------------- 3-4. impermanent loss
@pytest.mark.parametrize("r", [0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0])
def test_the_impermanent_loss_closed_form_reproduces_the_reserves(r):
    closed = amm.il_closed_form(r)
    sim = amm.il_from_reserves(1_000_000.0, 1_000_000.0, r)
    assert abs(closed - sim) < 1e-15                          # an identity, not a fit
    assert closed <= 1e-15                                    # never positive
    assert amm.il_closed_form(1.0 / r) == pytest.approx(closed, rel=1e-12)   # symmetric in 1/r


def test_the_documented_impermanent_loss_values():
    assert amm.il_closed_form(2.0) == pytest.approx(-0.057191, abs=1e-6)
    assert amm.il_closed_form(4.0) == pytest.approx(-0.20, abs=1e-9)
    assert amm.il_closed_form(10.0) == pytest.approx(-0.425040, abs=1e-6)
    assert amm.il_closed_form(1.0) == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        amm.il_closed_form(0.0)
    with pytest.raises(ValueError):
        amm.il_from_reserves(1.0, 1.0, -1.0)


def test_the_break_even_volume_to_tvl_grows_with_the_square_of_volatility():
    rows = [amm.il_vs_fees(v) for v in (0.30, 0.60, 1.00, 1.50)]
    ils = [-r["mean_il"] for r in rows]
    bes = [r["breakeven_daily_vol_to_tvl"] for r in rows]
    assert ils == sorted(ils) and bes == sorted(bes)
    assert ils[1] / ils[0] == pytest.approx(4.0, rel=0.10)    # doubling vol quadruples IL
    assert bes[1] / bes[0] == pytest.approx(4.0, rel=0.10)
    # the tail is the point: the mean is worse than the median at every vol
    for r in rows:
        assert r["mean_il"] < r["median_il"] < 0.0
        assert r["worst_decile_il"] < r["mean_il"]
    # SKILL.md sec 4, the 60% row
    assert rows[1]["mean_il"] == pytest.approx(-0.00370, abs=5e-5)
    assert rows[1]["breakeven_daily_vol_to_tvl"] == pytest.approx(0.0411, abs=5e-4)
    with pytest.raises(ValueError):
        amm.il_vs_fees(0.0)


# ------------------------------------------------------------------ 5. concentrated liquidity
@pytest.mark.parametrize("m", [1.05, 1.10, 1.25, 2.00, 4.00])
def test_a_range_amplifies_the_loss_by_exactly_its_capital_multiple(m):
    eff = amm.capital_efficiency(m)
    assert eff == pytest.approx(1.0 / (1.0 - 1.0 / np.sqrt(m)), rel=1e-12)
    assert eff > 1.0
    # a move small enough to stay inside the range: amplification == capital multiple
    small = amm.cl_loss_vs_hold(m, 1.02) / amm.il_closed_form(1.02)
    assert small == pytest.approx(eff, rel=2e-3)
    # and a big move out of range costs more than the full-range position
    assert amm.cl_loss_vs_hold(m, 1.2) < amm.il_closed_form(1.2) < 0.0


def test_out_of_range_a_position_holds_exactly_one_asset():
    L, pa, pb = 1.0, 0.8, 1.25
    x_hi, y_hi = amm.cl_amounts(L, 10.0, pa, pb)              # price above the range
    assert x_hi == pytest.approx(0.0, abs=1e-15) and y_hi > 0.0
    x_lo, y_lo = amm.cl_amounts(L, 0.01, pa, pb)              # price below the range
    assert y_lo == pytest.approx(0.0, abs=1e-15) and x_lo > 0.0
    inside = amm.cl_amounts(L, 1.0, pa, pb)
    assert inside[0] > 0.0 and inside[1] > 0.0
    for bad in (dict(p=1.0, pa=1.0, pb=1.0), dict(p=-1.0, pa=0.5, pb=2.0)):
        with pytest.raises(ValueError):
            amm.cl_amounts(L, **bad)
    with pytest.raises(ValueError):
        amm.capital_efficiency(1.0)


def test_time_in_range_falls_as_the_range_narrows():
    tirs = [amm.time_in_range(m, 0.60) for m in (1.05, 1.10, 1.25, 2.00)]
    assert tirs == sorted(tirs)
    assert tirs[0] < 0.5 < tirs[-1]
    assert amm.time_in_range(2.0, 0.60) >= amm.time_in_range(2.0, 1.50)
    with pytest.raises(ValueError):
        amm.time_in_range(0.9, 0.60)


# --------------------------------------------------------------------------- 6. the sandwich
def test_a_sandwich_takes_a_constant_share_of_whatever_tolerance_you_set():
    takes = []
    for tol in (0.001, 0.005, 0.01, 0.05):
        s = amm.sandwich(1_000_000.0, 1_000_000.0, 20_000.0, tol)
        assert s["victim_shortfall"] == pytest.approx(-tol, rel=1e-4)   # pushed to the limit
        assert s["attacker_profit_x"] > 0.0
        takes.append(s["capture_of_tolerance"])
    assert all(t == pytest.approx(0.86, abs=0.02) for t in takes)
    assert max(takes) - min(takes) < 0.01                     # the share barely moves
    with pytest.raises(ValueError):
        amm.sandwich(1e6, 1e6, 1e4, 0.0)


# --------------------------------------------------------------------------------- 7. demo
def test_main_prints_the_rule_in_ascii_and_is_deterministic(run_main, capsys):
    out = run_main("fin_skills.crypto.amm")
    assert is_ascii(out)
    assert amm.THE_RULE in out
    assert out.count("=" * 96) == 4
    for heading in ("x*y = k", "FEES", "IMPERMANENT LOSS", "BREAK-EVEN", "RANGE",
                    "NOT MEASURABLE OFFLINE"):
        assert heading in out
    assert max(len(line) for line in out.splitlines()) <= 98
    amm.main()
    assert capsys.readouterr().out == out
