"""fin_skills.models.credit_models - Merton, constant hazard, CDS, and the measure trap.

The documented properties: the Merton solve reproduces a published textbook example and never
returns a wrong root, mu = r makes the physical distance-to-default equal N(-d2) exactly,
lambda*(1-R) is the CONTINUOUS-premium par spread with no approximation in it, the implied
hazard is a function of the recovery you assumed, and pricing with a physical PD is wrong in a
known direction.
"""
from __future__ import annotations

import math

import pytest

from conftest import has_module, requires
from fin_skills.models.credit_models import (approximation_error_table, cds_legs,
                                             cds_mark_to_market, cds_par_spread,
                                             distance_to_default,
                                             hazard_from_default_probability, implied_hazard,
                                             measure_trap, merton_equity, merton_solve,
                                             merton_solver_scan, quantlib_cds_cross_check,
                                             recovery_sensitivity, survival_probability)

# Hull, Options Futures and Other Derivatives, the credit-risk chapter example
HULL = dict(E=3.0, sigma_E=0.80, D=10.0, r=0.05, T=1.0)
H, R, RF, TC = 0.02, 0.40, 0.03, 5.0


# ------------------------------------------------------------------ Merton
def test_merton_reproduces_the_published_textbook_example():
    m = merton_solve(**HULL)
    assert m.V == pytest.approx(12.40, abs=0.005)
    assert m.sigma_V == pytest.approx(0.2123, abs=5e-5)
    assert m.d2 == pytest.approx(1.1408, abs=5e-5)
    assert m.pd_risk_neutral == pytest.approx(0.127, abs=5e-4)
    assert m.debt_value == pytest.approx(9.40, abs=0.005)
    assert m.pv_promised == pytest.approx(9.51, abs=0.005)
    assert m.expected_loss == pytest.approx(0.012, abs=5e-4)
    assert m.residual < 1e-10


def test_the_solve_inverts_the_forward_map_exactly():
    m = merton_solve(**HULL)
    e, se = merton_equity(m.V, m.sigma_V, HULL["D"], HULL["r"], HULL["T"])
    assert e == pytest.approx(HULL["E"], abs=1e-9)
    assert se == pytest.approx(HULL["sigma_E"], abs=1e-9)
    # debt is the residual claim, and the credit spread is what discounts it to that value
    assert m.debt_value == pytest.approx(m.V - HULL["E"], abs=1e-12)
    assert m.debt_value == pytest.approx(
        m.pv_promised * math.exp(-m.credit_spread * HULL["T"]), abs=1e-12)
    assert 0.0 < m.debt_value < m.pv_promised          # risky debt is worth less than riskless


def test_the_solver_refuses_bad_input_and_never_returns_a_wrong_root():
    for bad in (dict(E=0.0), dict(sigma_E=0.0), dict(D=-1.0), dict(T=0.0)):
        with pytest.raises(ValueError):
            merton_solve(**{**HULL, **bad})
    with pytest.raises(ValueError, match="must be positive"):
        merton_equity(-1.0, 0.2, 10.0, 0.05, 1.0)
    scan = merton_solver_scan()
    assert scan["total"] == 270
    assert scan["wrong_root"] == [], "a converged-but-wrong root is the only dangerous outcome"
    assert scan["solved"] >= 250
    assert len(scan["raised"]) == scan["total"] - scan["solved"]
    # the only corner it cannot do is a 300% equity vol over five years
    assert {(sE, T) for _, sE, _, T, _ in scan["raised"]} == {(3.0, 5.0)}


def test_a_zero_equity_value_gives_an_infinite_equity_vol_rather_than_a_nan():
    # deeply distressed: the call is worthless, so sigma_E is undefined, not nan
    e, se = merton_equity(1e-8, 0.2, 100.0, 0.05, 1.0)
    assert e == 0.0 or e < 1e-12
    assert not math.isnan(se)


# ------------------------------------------------------------------ the two measures
def test_the_physical_distance_to_default_equals_n_minus_d2_when_mu_is_r():
    m = merton_solve(**HULL)
    dd, pd = distance_to_default(m.V, m.sigma_V, HULL["D"], HULL["r"], HULL["T"])
    assert dd == pytest.approx(m.d2, abs=1e-12)
    assert pd == pytest.approx(m.pd_risk_neutral, abs=1e-12)


def test_a_higher_asset_drift_lowers_the_physical_pd_and_never_the_risk_neutral_one():
    m = merton_solve(**HULL)
    pds = {}
    for mu in (0.05, 0.08, 0.10, 0.15):
        dd, pd = distance_to_default(m.V, m.sigma_V, HULL["D"], mu, HULL["T"])
        pds[mu] = pd
    assert pds[0.05] > pds[0.08] > pds[0.10] > pds[0.15]
    assert pds[0.05] == pytest.approx(m.pd_risk_neutral, abs=1e-12)
    assert m.pd_risk_neutral / pds[0.15] == pytest.approx(2.37, abs=0.01)


# ------------------------------------------------------------------ hazard and CDS
def test_survival_and_hazard_are_inverses():
    assert survival_probability(5.0, 0.02) == pytest.approx(math.exp(-0.10))
    pd5 = 1.0 - survival_probability(5.0, 0.02)
    assert hazard_from_default_probability(pd5, 5.0) == pytest.approx(0.02, abs=1e-14)
    assert hazard_from_default_probability(0.0, 1.0) == 0.0
    with pytest.raises(ValueError, match="PD must be"):
        hazard_from_default_probability(1.0, 1.0)


def test_the_legs_add_up_and_the_par_spread_makes_the_swap_worth_nothing():
    legs = cds_legs(H, R, RF, TC)
    assert legs["rpv01"] == pytest.approx(legs["annuity"] + legs["accrual"], abs=1e-15)
    assert legs["par_spread"] == pytest.approx(legs["protection"] / legs["rpv01"], abs=1e-15)
    assert cds_mark_to_market(legs["par_spread"], H, R, RF, TC) == pytest.approx(0.0, abs=1e-15)
    # a contract struck below the fair spread is worth something to the protection buyer
    assert cds_mark_to_market(legs["par_spread"] / 2, H, R, RF, TC) > 0.0
    for bad in (dict(hazard=-0.01), dict(recovery=1.0), dict(recovery=-0.1), dict(T=0.0),
                dict(freq=0)):
        with pytest.raises(ValueError, match="need hazard"):
            cds_legs(**{**dict(hazard=H, recovery=R, r=RF, T=TC), **bad})
    with pytest.raises(ValueError, match="whole number of premium periods"):
        cds_legs(H, R, RF, 5.1)


def test_lambda_times_one_minus_recovery_is_the_continuous_premium_spread_exactly():
    # the documented explanation: the gap is premium FREQUENCY, and it vanishes as freq -> inf
    approx = H * (1.0 - R)
    ladder = {f: cds_par_spread(H, R, RF, TC, freq=f) for f in (1, 2, 4, 12, 52, 365)}
    assert all(s > approx for s in ladder.values())            # arrears defers the annuity
    assert ladder[1] > ladder[2] > ladder[4] > ladder[12] > ladder[52] > ladder[365]
    assert (ladder[365] - approx) * 1e4 < 0.01                 # daily is within 0.01 bp
    assert (ladder[1] - approx) * 1e4 == pytest.approx(1.81, abs=0.01)
    assert (ladder[4] - approx) * 1e4 == pytest.approx(0.45, abs=0.01)
    # halving the period roughly halves the gap: an O(1/freq) error
    assert (ladder[4] - approx) / (ladder[12] - approx) == pytest.approx(3.0, rel=0.05)


def test_accrual_on_default_lowers_the_fair_spread_at_every_frequency():
    for f in (1, 2, 4, 12):
        with_acc = cds_par_spread(H, R, RF, TC, freq=f)
        without = cds_par_spread(H, R, RF, TC, freq=f, accrual_on_default=False)
        assert with_acc < without
    assert (cds_par_spread(H, R, RF, TC, freq=4, accrual_on_default=False)
            - cds_par_spread(H, R, RF, TC, freq=4)) * 1e4 == pytest.approx(0.30, abs=0.01)


def test_the_approximation_error_is_a_constant_fraction_of_the_spread():
    rows = approximation_error_table((0.005, 0.01, 0.02, 0.05, 0.10, 0.20), R, RF, TC)
    pcts = [row["error_pct"] for row in rows]
    assert all(p == pytest.approx(-0.37, abs=0.01) for p in pcts)   # understates, always
    assert all(row["approx_bp"] < row["exact_bp"] for row in rows)
    assert rows[-1]["error_bp"] == pytest.approx(-4.47, abs=0.01)


def test_the_implied_hazard_inverts_the_par_spread_and_depends_on_the_recovery():
    for R_ in (0.0, 0.2, 0.4, 0.6, 0.8):
        h = implied_hazard(0.0100, R_, RF, TC)
        assert cds_par_spread(h, R_, RF, TC) == pytest.approx(0.0100, abs=1e-12)
    rows = {r["recovery"]: r for r in recovery_sensitivity(0.0100, (0.0, 0.2, 0.4, 0.6, 0.8),
                                                           RF, TC)}
    assert rows[0.6]["hazard"] / rows[0.2]["hazard"] == pytest.approx(2.0, abs=0.01)
    assert rows[0.8]["pd_T"] == pytest.approx(0.220, abs=0.001)
    for r_ in rows.values():                                # s/(1-R) is close but always above
        assert r_["approx_hazard"] > r_["hazard"]
    with pytest.raises(ValueError, match="spread must be positive"):
        implied_hazard(0.0, R, RF, TC)


# ------------------------------------------------------------------ the trap
def test_pricing_with_a_physical_pd_is_wrong_in_a_known_direction():
    out = measure_trap(mu=0.10, recovery=R, cds_T=1.0, **HULL)
    assert out["pd_phys"] < out["pd_rn"]
    assert out["hazard_phys"] < out["hazard_rn"]
    assert out["spread_phys_bp"] < out["spread_rn_bp"]
    assert out["spread_error_bp"] < -250.0                  # roughly -288 bp on this firm
    # revaluing a FAIR contract with the physical hazard invents a loss that is not there
    assert out["mtm_error_pct_notional"] < -2.0
    # and with mu = r the trap closes completely: same measure, same price
    same = measure_trap(mu=HULL["r"], recovery=R, cds_T=1.0, **HULL)
    assert same["spread_error_bp"] == pytest.approx(0.0, abs=1e-9)
    assert same["mtm_error_pct_notional"] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ QuantLib
@pytest.mark.skipif(has_module("QuantLib"), reason="QuantLib is installed")
def test_cross_check_is_none_without_quantlib():
    assert quantlib_cds_cross_check(H, R, RF, TC) is None


@requires("QuantLib")
def test_quantlib_agrees_to_a_hundredth_of_a_basis_point():
    live = quantlib_cds_cross_check(H, R, RF, TC)
    assert live is not None
    assert live["survival_T"] == pytest.approx(survival_probability(TC, H), abs=1e-12)
    mine = cds_par_spread(H, R, RF, TC)
    mine_no_accrual = cds_par_spread(H, R, RF, TC, accrual_on_default=False)
    assert abs(live["midpoint"] - mine) * 1e4 < 0.01
    assert abs(live["isda"] - mine) * 1e4 < 0.05
    assert abs(live["midpoint_no_accrual"] - mine_no_accrual) * 1e4 < 0.01
    # the two engines are not the same model: they differ on identical inputs
    assert live["midpoint"] != live["isda"]
    assert abs(live["midpoint"] - live["isda"]) * 1e4 == pytest.approx(0.0154, abs=0.005)
    # and both agree that accrual on default lowers the fair spread
    assert live["midpoint"] < live["midpoint_no_accrual"]


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.models.credit_models")
    assert "N(-d2) is risk-neutral" in out
    assert "never quote a hazard without its recovery" in out
    assert out.isascii()
