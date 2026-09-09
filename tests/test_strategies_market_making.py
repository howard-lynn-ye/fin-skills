"""fin_skills.strategies.market_making - Avellaneda & Stoikov (2008), reproduced and stressed."""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.strategies.market_making import (AS_PARAMS, AS_TABLES, arrival_intensity,
                                                 best_spread_multiplier, fill_probability,
                                                 mean_optimal_spread, optimal_spread,
                                                 reservation_price, simulate)

P = AS_PARAMS


# ------------------------------------------------------- the closed forms, against the paper
@pytest.mark.parametrize("gamma,printed", [(0.1, 1.49), (0.01, 1.35), (1.0, 3.02)])
def test_mean_spread_reproduces_the_papers_printed_average_spread(gamma, printed):
    """AS Tables 1-3 print 1.49, 1.35 and 3.02 - the time-average of eq. (30) over [0, T]."""
    assert mean_optimal_spread(gamma, P["sigma"], P["T"], P["k"]) == pytest.approx(printed,
                                                                                   abs=0.005)


def test_mean_spread_is_the_integral_of_eq30():
    """The closed form must equal a numerical average of eq. (30) over the horizon."""
    g, s, T, k = 0.1, 2.0, 1.0, 1.5
    grid = np.linspace(0.0, T, 200_001)
    numeric = np.mean([optimal_spread(g, s, float(t), T, k) for t in grid[::200]])
    assert mean_optimal_spread(g, s, T, k) == pytest.approx(numeric, rel=1e-3)
    # and the two terms are what the paper writes
    assert optimal_spread(g, s, 0.0, T, k) == pytest.approx(
        g * s ** 2 * T + (2.0 / g) * np.log(1.0 + g / k))
    assert optimal_spread(g, s, T, T, k) == pytest.approx((2.0 / g) * np.log(1.0 + g / k))


def test_the_spread_does_not_depend_on_inventory():
    """AS: 'the bid-ask spread in (25) is independent of the inventory.'"""
    a = optimal_spread(0.1, 2.0, 0.3, 1.0, 1.5)
    assert optimal_spread(0.1, 2.0, 0.3, 1.0, 1.5) == a          # no q argument to pass
    with pytest.raises(ValueError, match="positive"):
        optimal_spread(0.0, 2.0, 0.0, 1.0, 1.5)
    with pytest.raises(ValueError, match="positive"):
        optimal_spread(0.1, 2.0, 0.0, 1.0, 0.0)


def test_reservation_price_skews_against_the_inventory_and_vanishes_at_T():
    """eq. (8): r = s - q*gamma*sigma^2*(T - t). Long inventory quotes lower."""
    assert reservation_price(100.0, 0.0, 0.1, 2.0, 0.0, 1.0) == pytest.approx(100.0)
    assert reservation_price(100.0, 5.0, 0.1, 2.0, 0.0, 1.0) == pytest.approx(98.0)
    assert reservation_price(100.0, -5.0, 0.1, 2.0, 0.0, 1.0) == pytest.approx(102.0)
    assert reservation_price(100.0, 5.0, 0.1, 2.0, 1.0, 1.0) == pytest.approx(100.0)
    assert np.allclose(reservation_price(np.full(3, 100.0), np.array([-1.0, 0.0, 1.0]),
                                         0.1, 2.0, 0.0, 1.0), [100.4, 100.0, 99.6])
    with pytest.raises(ValueError, match="t must not exceed"):
        reservation_price(100.0, 0.0, 0.1, 2.0, 2.0, 1.0)


def test_arrival_intensity_is_the_exponential_of_eq12():
    assert arrival_intensity(0.0, 140.0, 1.5) == pytest.approx(140.0)
    assert arrival_intensity(1.0, 140.0, 1.5) == pytest.approx(140.0 * np.exp(-1.5))
    assert arrival_intensity(-0.5, 140.0, 1.5) > 140.0       # a quote through the mid
    assert arrival_intensity(2.0, 140.0, 1.5) < arrival_intensity(1.0, 140.0, 1.5)
    with pytest.raises(ValueError, match="positive"):
        arrival_intensity(0.0, 0.0, 1.5)


# ------------------------------------------------------ TRAP: lambda*dt is not a probability
def test_the_two_fill_probability_readings_diverge_at_the_papers_parameters():
    lam0 = arrival_intensity(0.0, P["A"], P["k"])
    assert fill_probability(lam0, P["dt"], "linear") == pytest.approx(0.70)
    assert fill_probability(lam0, P["dt"], "poisson") == pytest.approx(0.503, abs=0.001)
    # the linear form has to be clipped once a quote goes through the mid
    lam_neg = arrival_intensity(-0.5, P["A"], P["k"])
    assert lam_neg * P["dt"] > 1.0
    assert fill_probability(lam_neg, P["dt"], "linear") == 1.0
    assert 0.0 < fill_probability(lam_neg, P["dt"], "poisson") < 1.0
    # and they do agree where lambda*dt is genuinely small
    tiny = arrival_intensity(5.0, P["A"], P["k"])
    assert fill_probability(tiny, P["dt"], "linear") == pytest.approx(
        fill_probability(tiny, P["dt"], "poisson"), rel=1e-3)
    with pytest.raises(ValueError, match="poisson"):
        fill_probability(1.0, 0.1, "binomial")


def test_the_poisson_reading_reports_less_profit_on_the_same_strategy():
    a = simulate(prob_model="linear", n_paths=1500, seed=1)
    b = simulate(prob_model="poisson", n_paths=1500, seed=1)
    assert b["fills"] < a["fills"]
    assert b["pnl_mean"] < a["pnl_mean"]
    assert b["pnl_mean"] / a["pnl_mean"] - 1 < -0.05          # a materially different number


# ------------------------------------------------------------- the paper's own experiment
@pytest.mark.parametrize("gamma", [0.1, 0.01, 1.0])
def test_the_simulation_reproduces_the_papers_tables(gamma):
    """All six rows of Tables 1-3, within a few percent, on an independent implementation."""
    want = AS_TABLES[gamma]
    for strat, key in (("inventory", "inv"), ("symmetric", "sym")):
        r = simulate(strat, gamma=gamma, n_paths=3000)
        pnl, pnl_sd, _, q_sd = want[key]
        assert r["avg_spread"] == pytest.approx(want["spread"], abs=0.02)
        assert r["pnl_mean"] == pytest.approx(pnl, rel=0.06)
        assert r["pnl_std"] == pytest.approx(pnl_sd, rel=0.15)
        assert r["q_std"] == pytest.approx(q_sd, rel=0.15)


def test_the_symmetric_strategy_earns_more_and_risks_much_more():
    """The paper's own reading of Table 1, and the reason to use the inventory strategy."""
    inv = simulate("inventory", n_paths=3000)
    sym = simulate("symmetric", n_paths=3000)
    assert sym["pnl_mean"] > inv["pnl_mean"]            # it sits on the mid, takes more volume
    assert sym["pnl_std"] > 1.7 * inv["pnl_std"]        # at about twice the dispersion
    assert sym["q_std"] > 2.0 * inv["q_std"]            # and far more inventory
    assert abs(sym["q_absmax"]) > abs(inv["q_absmax"])


def test_the_two_strategies_converge_as_gamma_goes_to_zero():
    """AS: 'in the limit as gamma -> 0 the two strategies are identical'."""
    def gap(g):
        return abs(simulate("symmetric", gamma=g, n_paths=1500)["q_std"]
                   - simulate("inventory", gamma=g, n_paths=1500)["q_std"])
    assert gap(0.001) < gap(0.1)


def test_the_inventory_skew_is_what_controls_inventory():
    """Turn the skew off (symmetric) and the terminal inventory disperses."""
    inv = simulate("inventory", n_paths=3000)
    sym = simulate("symmetric", n_paths=3000)
    assert abs(inv["q_mean"]) < 0.5 and abs(sym["q_mean"]) < 0.7   # both are unbiased
    assert inv["q_std"] < sym["q_std"]                              # only one is controlled


def test_simulate_validates_its_inputs_and_respects_an_inventory_cap():
    with pytest.raises(ValueError, match="strategy"):
        simulate("aggressive")
    with pytest.raises(ValueError, match="informed_frac"):
        simulate(informed_frac=1.5)
    with pytest.raises(ValueError, match="spread_mult"):
        simulate(spread_mult=0.0)
    with pytest.raises(ValueError, match="informed_jump"):
        simulate(informed_jump=-1.0)
    capped = simulate("symmetric", q_max=3.0, n_paths=500)
    assert capped["q_absmax"] <= 3.0                    # a cap AS does not have, but people add


def test_the_simulation_is_deterministic_under_its_seed():
    a = simulate(n_paths=400, seed=7)
    b = simulate(n_paths=400, seed=7)
    assert np.array_equal(a["pnl"], b["pnl"]) and np.array_equal(a["final_q"], b["final_q"])
    assert not np.array_equal(a["pnl"], simulate(n_paths=400, seed=8)["pnl"])


# ------------------------------------------------------------------------ adverse selection
def test_informed_flow_degrades_pnl_monotonically_and_thins_the_flow():
    """Section 4: the risk AS has no term for, on quotes optimal by eq. (30) at every step."""
    runs = {phi: simulate(informed_frac=phi, n_paths=3000) for phi in (0.0, 0.25, 0.5)}
    assert runs[0.0]["toxic"] == 0.0 and runs[0.5]["toxic"] > runs[0.25]["toxic"] > 0
    pnls = [runs[p]["pnl_mean"] for p in (0.0, 0.25, 0.5)]
    assert pnls == sorted(pnls, reverse=True)
    assert runs[0.5]["pnl_mean"] < 0.8 * runs[0.0]["pnl_mean"]
    assert runs[0.5]["fills"] < runs[0.0]["fills"]             # informed flow is also thinner
    assert runs[0.5]["pnl_per_fill"] < runs[0.0]["pnl_per_fill"]   # and worse per fill


def test_the_inventory_skew_is_no_defence_against_toxicity():
    """Both strategies lose about the same share - the skew is an inventory control."""
    inv0, inv5 = simulate("inventory", n_paths=3000), simulate("inventory",
                                                               informed_frac=0.5, n_paths=3000)
    sym0, sym5 = simulate("symmetric", n_paths=3000), simulate("symmetric",
                                                               informed_frac=0.5, n_paths=3000)
    assert (inv5["pnl_mean"] / inv0["pnl_mean"]) == pytest.approx(
        sym5["pnl_mean"] / sym0["pnl_mean"], rel=0.15)


def test_widening_pays_only_when_the_adverse_move_dwarfs_the_half_spread():
    """Section 5: the governing ratio, not the risk aversion."""
    mults = (1.0, 1.25, 1.5, 2.0)
    small, _ = best_spread_multiplier(0.25, mults, informed_jump=0.0, n_paths=2000)
    large, _ = best_spread_multiplier(0.25, mults, informed_jump=4.0, n_paths=2000)
    assert small == 1.0                     # at AS's own scale, do not widen
    assert large > small                    # once an informed fill really moves the price, do
    half = mean_optimal_spread(P["gamma"], P["sigma"], P["T"], P["k"]) / 2.0
    step = P["sigma"] * np.sqrt(P["dt"])
    assert half / step > 5.0                # AS quotes 5x a mid step: hard to pick off


@pytest.mark.slow
def test_demo_reproduces_the_papers_tables_and_prints_the_rule(run_main):
    out = run_main("fin_skills.strategies.market_making")
    assert "1.4908" in out and "3.0217" in out              # eq. (30) vs the paper's tables
    assert "paper 1.49 / 65.0 / 6.6 / 2.9" in out           # Table 1 alongside the rerun
    assert "TRAP" in out and "picked off" in out
    assert "Rule: quote eq. (30)'s spread" in out
    assert out.isascii()
