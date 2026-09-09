"""fin_skills.strategies.execution_algos - schedules, Almgren-Chriss (2000), and shortfall."""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.strategies.execution_algos import (AC_TABLE1, ac_cost, ac_kappa, ac_trade_list,
                                                   ac_trajectory, cost_bps, direct_cost,
                                                   efficient_frontier, eq16_residual, eta_tilde,
                                                   fill_prices, implementation_shortfall,
                                                   linear_cost, pov_schedule, pov_shortfall,
                                                   session_vwap, simulate_session, twap_schedule,
                                                   vwap_schedule, vwap_self_contamination)

P = AC_TABLE1
ARGS = (P["X"], P["T"], P["N"], P["sigma"], P["eta"], P["gamma"], P["eps"])
TAU = P["T"] / P["N"]


@pytest.fixture(scope="module")
def session():
    return simulate_session(n_bins=120, seed=4, drift_bps=40.0)


# ------------------------------------------------------ Almgren-Chriss against the paper itself
def test_kappa_reproduces_the_papers_printed_value():
    """AC (2000) p. 24: 'we have from (19) that kappa ~ 0.6/day, so kappa*T ~ 3'."""
    k = ac_kappa(P["lam_u"], P["sigma"], P["eta"], P["gamma"], TAU)
    assert k["eta_tilde"] == pytest.approx(P["eta"] - 0.5 * P["gamma"] * TAU)
    assert k["kappa_tilde_sq"] == pytest.approx(P["lam_u"] * P["sigma"] ** 2 / k["eta_tilde"])
    assert k["kappa_eq19"] == pytest.approx(0.6, abs=0.005)      # eq. (19)
    assert k["kappa"] == pytest.approx(0.6, abs=0.01)            # exact cosh relation
    assert k["kappa"] * P["T"] == pytest.approx(3.0, abs=0.05)
    assert k["half_life"] == pytest.approx(1.0 / k["kappa"])


def test_buy_and_hold_variance_reproduces_the_papers_2_12():
    """AC (2000) p. 24: holding untraded gives sigma*sqrt(T) = 2.12 $/share, sqrt(V) = $2.12M."""
    per_share = P["sigma"] * np.sqrt(P["T"])
    assert per_share == pytest.approx(2.12, abs=0.005)
    assert per_share * P["X"] / 1e6 == pytest.approx(2.12, abs=0.005)


def test_the_cosh_relation_defines_kappa_exactly():
    """(2/tau^2)(cosh(kappa*tau) - 1) = kappa~^2, the unnumbered display after eq. (16)."""
    for lam in (2e-6, 1e-6, 2e-7):
        k = ac_kappa(lam, P["sigma"], P["eta"], P["gamma"], TAU)
        lhs = 2.0 / TAU ** 2 * (np.cosh(k["kappa"] * TAU) - 1.0)
        assert lhs == pytest.approx(k["kappa_tilde_sq"], rel=1e-12)


def test_the_trajectory_solves_the_difference_equation_in_all_three_branches():
    """eq. (17) substituted into eq. (16). lam < 0 is the trigonometric branch (trajectory C)."""
    for lam in (2e-6, 1e-6, 0.0, -2e-7):
        k = ac_kappa(lam, P["sigma"], P["eta"], P["gamma"], TAU)
        x, _ = ac_trajectory(P["X"], P["T"], P["N"], k["kappa"], k["trigonometric"])
        assert x[0] == pytest.approx(P["X"]) and abs(x[-1]) < 1e-6
        assert eq16_residual(x, P["T"], k["kappa_tilde_sq"]) < 1e-6 * P["X"]
    assert ac_kappa(-2e-7, P["sigma"], P["eta"], P["gamma"], TAU)["trigonometric"] is True
    assert ac_kappa(1e-6, P["sigma"], P["eta"], P["gamma"], TAU)["trigonometric"] is False


def test_eq18_written_out_equals_minus_diff_of_eq17():
    k = ac_kappa(P["lam_u"], P["sigma"], P["eta"], P["gamma"], TAU)
    _, n = ac_trajectory(P["X"], P["T"], P["N"], k["kappa"])
    assert np.allclose(n, ac_trade_list(P["X"], P["T"], P["N"], k["kappa"]), atol=1e-6)
    assert np.allclose(twap_schedule(P["X"], P["N"]),
                       ac_trade_list(P["X"], P["T"], P["N"], 0.0))
    assert (n > 0).all()                     # AC: a sell program never buys


def test_eq20_closed_form_equals_eq8_and_eq5_summed_on_the_trajectory():
    """The independent oracle for the long closed form - the check to run if you reimplement."""
    k = ac_kappa(P["lam_u"], P["sigma"], P["eta"], P["gamma"], TAU)
    x, _ = ac_trajectory(P["X"], P["T"], P["N"], k["kappa"])
    E_cf, V_cf = ac_cost(P["X"], P["T"], P["N"], k["kappa"], P["sigma"], P["eta"],
                         P["gamma"], P["eps"])
    E_d, V_d = direct_cost(x, P["T"], P["sigma"], P["eta"], P["gamma"], P["eps"])
    assert E_cf == pytest.approx(E_d, rel=1e-12)
    assert V_cf == pytest.approx(V_d, rel=1e-12)


def test_eq10_and_eq11_equal_the_same_sums_on_the_linear_trajectory():
    x, _ = ac_trajectory(P["X"], P["T"], P["N"], 0.0)
    E_cf, V_cf = linear_cost(*ARGS)
    E_d, V_d = direct_cost(x, P["T"], P["sigma"], P["eta"], P["gamma"], P["eps"])
    assert E_cf == pytest.approx(E_d, rel=1e-12) and V_cf == pytest.approx(V_d, rel=1e-12)
    # eq. (10) written out term by term
    assert E_cf == pytest.approx(0.5 * P["gamma"] * P["X"] ** 2 + P["eps"] * P["X"]
                                 + eta_tilde(P["eta"], P["gamma"], TAU) * P["X"] ** 2 / P["T"])
    assert V_cf == pytest.approx(P["sigma"] ** 2 * P["X"] ** 2 * P["T"] / 3.0
                                 * (1 - 1 / P["N"]) * (1 - 1 / (2 * P["N"])))


def test_ac_cost_reduces_to_the_linear_cost_as_kappa_goes_to_zero():
    """AC (2000): eq. (20) 'reduce to (10-13) in the limits kappa -> 0, infinity'."""
    E_lin, V_lin = linear_cost(*ARGS)
    k = ac_kappa(1e-14, P["sigma"], P["eta"], P["gamma"], TAU)
    E, V = ac_cost(P["X"], P["T"], P["N"], k["kappa"], P["sigma"], P["eta"], P["gamma"],
                   P["eps"])
    assert E == pytest.approx(E_lin, rel=1e-6) and V == pytest.approx(V_lin, rel=1e-6)
    assert ac_cost(P["X"], P["T"], P["N"], 0.0, P["sigma"], P["eta"], P["gamma"],
                   P["eps"]) == (E_lin, V_lin)


def test_trap2_ac_costs_more_in_expectation_than_twap_and_wins_only_on_the_objective():
    """Section 4's headline: TWAP IS the minimum-expected-cost trajectory in this model."""
    lam = P["lam_u"]
    k = ac_kappa(lam, P["sigma"], P["eta"], P["gamma"], TAU)
    E_ac, V_ac = ac_cost(P["X"], P["T"], P["N"], k["kappa"], P["sigma"], P["eta"],
                         P["gamma"], P["eps"])
    E_lin, V_lin = linear_cost(*ARGS)
    assert E_ac > E_lin                                   # AC is DEARER in expectation
    assert V_ac < V_lin                                   # and less variable
    assert E_ac + lam * V_ac < E_lin + lam * V_lin        # it wins only on U = E + lam*V


def test_the_frontier_is_monotone_and_the_half_life_ignores_the_deadline():
    """AC section 2.3: theta = 1/kappa is 'independent of the exogenously specified T'."""
    fr = efficient_frontier([2e-6, 1e-6, 2e-7, 0.0], *ARGS).set_index("lambda")
    assert fr["E"].is_monotonic_decreasing                # less risk aversion, cheaper
    assert fr["sqrt_V"].is_monotonic_increasing           # and more variable
    assert fr["kappa"].is_monotonic_decreasing
    # kappa depends on tau, not on T, so a 5-day and a 10-day deadline share a half-life
    kap = ac_kappa(P["lam_u"], P["sigma"], P["eta"], P["gamma"], 1.0)["kappa"]
    x5, _ = ac_trajectory(P["X"], 5.0, 5, kap)
    x10, _ = ac_trajectory(P["X"], 10.0, 10, kap)
    # doubling the deadline does not slow the EARLY trade down: the first days are the same
    # decay, and both track exp(-kappa*t) - the deadline only changes where the tail lands
    for day in (1, 2):
        assert x10[day] / P["X"] == pytest.approx(x5[day] / P["X"], abs=0.01)
        assert x10[day] / P["X"] == pytest.approx(np.exp(-kap * day), abs=0.01)
    # the two only part company as the shorter deadline bends the tail down to zero
    assert x5[3] < x10[3]
    assert x5[5] == pytest.approx(0.0, abs=1e-6)          # T=5 is flat by day 5
    assert x10[5] / P["X"] > 0.04                          # T=10 still holds 4% of it


def test_ac_rejects_a_non_convex_or_impossible_parameterisation():
    with pytest.raises(ValueError, match="eta~"):
        ac_kappa(1e-6, P["sigma"], 1e-7, 1e-5, 1.0)          # eta~ <= 0
    with pytest.raises(ValueError, match="too negative"):
        ac_kappa(-1e-4, P["sigma"], P["eta"], P["gamma"], TAU)
    with pytest.raises(ValueError):
        ac_trajectory(P["X"], P["T"], 0, 0.5)


# ------------------------------------------------------------------------ intraday schedules
def test_schedules_sum_to_the_order_and_have_their_documented_shapes(session):
    vol, n = session["volume"], session["n_bins"]
    tw, vw = twap_schedule(1000.0, n), vwap_schedule(1000.0, vol)
    assert tw.sum() == pytest.approx(1000.0) and vw.sum() == pytest.approx(1000.0)
    assert np.allclose(tw, tw[0])                                    # TWAP is flat
    assert np.corrcoef(vw, vol)[0, 1] == pytest.approx(1.0)          # VWAP tracks volume
    with pytest.raises(ValueError, match="non-negative"):
        vwap_schedule(100.0, -vol)
    with pytest.raises(ValueError, match="n_bins"):
        twap_schedule(100.0, 0)


def test_pov_caps_each_bin_and_can_fail_to_finish(session):
    vol = session["volume"]
    big = 10 * vol.sum()
    fills = pov_schedule(big, vol, 0.05)
    assert np.all(fills <= 0.05 * vol + 1e-9)                 # never exceeds the cap
    assert fills.sum() == pytest.approx(0.05 * vol.sum())     # so it cannot finish
    assert pov_shortfall(big, vol, 0.05) == pytest.approx(big - 0.05 * vol.sum())
    small = 0.001 * vol.sum()
    assert pov_schedule(small, vol, 0.5).sum() == pytest.approx(small)
    assert pov_shortfall(small, vol, 0.5) == 0.0
    with pytest.raises(ValueError, match="rate"):
        pov_schedule(100.0, vol, 0.0)


def test_impact_is_signed_and_permanent_impact_moves_the_whole_session(session):
    mid, vol = session["mid"], session["volume"]
    own = vwap_schedule(50_000.0, vol)
    buy_mid, buy_fill = fill_prices(mid, own, vol, side=1)
    sell_mid, sell_fill = fill_prices(mid, own, vol, side=-1)
    assert (buy_fill >= buy_mid).all() and (sell_fill <= sell_mid).all()
    assert buy_mid[-1] > mid[-1] and sell_mid[-1] < mid[-1]     # permanent, and it persists
    assert buy_mid[0] > mid[0]                                  # from the first bin onward
    zero_mid, zero_fill = fill_prices(mid, np.zeros_like(own), vol)
    assert np.allclose(zero_mid, mid) and np.allclose(zero_fill, mid)


def test_trap1_the_benchmark_absorbs_a_growing_share_of_the_true_cost(session):
    """Section 3: the measured cost understates the true cost by more as size grows."""
    tab = vwap_self_contamination(session, (0.01, 0.05, 0.10, 0.25))
    assert (tab["cost_vs_clean"] > tab["cost_vs_realised"]).all()      # one-signed
    assert tab["cost_vs_clean"].is_monotonic_increasing                # true cost rises
    share = tab["hidden_bps"] / tab["cost_vs_clean"]
    assert share.is_monotonic_increasing and share.iloc[0] < 0.10 < share.iloc[-1]
    with pytest.raises(ValueError, match="participation"):
        vwap_self_contamination(session, (1.0,))


def test_cost_bps_and_session_vwap_conventions():
    assert cost_bps(101.0, 100.0, side=1) == pytest.approx(100.0)
    assert cost_bps(101.0, 100.0, side=-1) == pytest.approx(-100.0)
    assert session_vwap(np.array([10.0, 20.0]), np.array([3.0, 1.0])) == pytest.approx(12.5)
    with pytest.raises(ValueError, match="at least 10 bins"):
        simulate_session(n_bins=3)


# ------------------------------------------------------------------ implementation shortfall
def test_shortfall_components_sum_to_the_paper_minus_real_identity():
    r = implementation_shortfall(decision_px=100.0, arrival_px=100.5, avg_fill_px=101.0,
                                 final_px=103.0, target_qty=1000.0, filled_qty=600.0,
                                 fees=25.0)
    assert r["delay"] == pytest.approx(600 * 0.5)
    assert r["execution"] == pytest.approx(600 * 0.5)
    assert r["opportunity"] == pytest.approx(400 * 3.0)
    assert r["total"] == pytest.approx(300 + 300 + 1200 + 25)
    assert r["identity_error"] == pytest.approx(0.0, abs=1e-9)
    assert r["fill_rate"] == pytest.approx(0.6)
    assert r["total_bps"] == pytest.approx(r["total"] / (1000.0 * 100.0) * 1e4)
    assert (r["delay_bps"] + r["execution_bps"] + r["opportunity_bps"] + r["fees_bps"]
            == pytest.approx(r["total_bps"]))


def test_a_fully_filled_order_has_no_opportunity_term_and_a_sell_flips_the_sign():
    full = implementation_shortfall(100.0, 100.0, 101.0, 90.0, 1000.0, 1000.0)
    assert full["opportunity"] == 0.0 and full["delay"] == 0.0
    assert full["execution"] == pytest.approx(1000.0)
    sell = implementation_shortfall(100.0, 100.0, 99.0, 90.0, 1000.0, 1000.0, side=-1)
    assert sell["execution"] == pytest.approx(1000.0)      # selling 1 below arrival also costs
    assert sell["identity_error"] == pytest.approx(0.0, abs=1e-9)


def test_shortfall_validates_its_inputs():
    with pytest.raises(ValueError, match="filled_qty"):
        implementation_shortfall(100.0, 100.0, 100.0, 100.0, 100.0, 200.0)
    with pytest.raises(ValueError, match="filled_qty"):
        implementation_shortfall(100.0, 100.0, 100.0, 100.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="side"):
        implementation_shortfall(100.0, 100.0, 100.0, 100.0, 10.0, 5.0, side=0)


def test_a_lower_participation_cap_moves_cost_into_the_opportunity_column(session):
    """Section 5: capping participation does not remove cost, it relocates it."""
    mid, vol = session["mid"], session["volume"]
    qty = 0.05 * vol.sum()
    out = {}
    for rate in (0.20, 0.02):
        own = pov_schedule(qty, vol, rate)
        _, px = fill_prices(mid, own, vol)
        filled = float(own.sum())
        out[rate] = implementation_shortfall(
            decision_px=float(mid[0]), arrival_px=float(mid[0]),
            avg_fill_px=float(np.sum(px * own) / filled), final_px=float(mid[-1]),
            target_qty=qty, filled_qty=filled)
    assert out[0.02]["execution_bps"] < out[0.20]["execution_bps"]     # looks cheaper
    assert out[0.02]["opportunity_bps"] > out[0.20]["opportunity_bps"]  # is not
    assert out[0.02]["fill_rate"] < 1.0 and out[0.20]["fill_rate"] == pytest.approx(1.0)


# ------------------------------------------------------------------------- panel and demo
def test_session_is_deterministic_and_the_drift_is_imposed_not_sampled():
    a, b = simulate_session(n_bins=100, seed=9, drift_bps=50.0), \
        simulate_session(n_bins=100, seed=9, drift_bps=50.0)
    assert np.array_equal(a["mid"], b["mid"]) and np.array_equal(a["volume"], b["volume"])
    flat = simulate_session(n_bins=100, seed=9, drift_bps=0.0)
    drifted = simulate_session(n_bins=100, seed=9, drift_bps=100.0)
    assert (drifted["mid"][-1] - drifted["mid"][0]) > (flat["mid"][-1] - flat["mid"][0])
    assert a["volume"].sum() == pytest.approx(5e6)


def test_demo_reproduces_the_papers_numbers_and_prints_the_rule(run_main):
    out = run_main("fin_skills.strategies.execution_algos")
    assert "kappa ~ 0.6/day" in out and "3.04" in out       # the paper's kappa and kappa*T
    assert "2.1243 $/share, $2.12M" in out                  # the paper's buy-and-hold sqrt(V)
    assert "TRAP 1" in out and "TRAP 2" in out
    assert "MORE in expectation" in out
    assert "Rule: pick the benchmark before the trade" in out
    assert out.isascii()
