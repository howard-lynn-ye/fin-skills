"""fin_skills.ml.bet_sizing - probability to position, and the three decisions in between.

The properties the SKILL.md claims:

  * m = 2*Phi(z) - 1 is 0 at the pivot, monotone, saturating, and pivots at 1/n_classes rather
    than at 0.5 - so a three-class probability has the opposite sign under the binary formula;
  * the curve sits below 2p-1 near the coin flip and above it at the extremes;
  * averaging active bets caps gross exposure at 1 while summing them does not, and the two are
    not proportional;
  * plain rounding barely reduces turnover (and at a fine step increases it) while requiring a
    full step of drift does, at the cost of tracking;
  * discretisation loses at zero cost and wins at a high one;
  * an expanding-maximum concurrency budget leverages more than a full-sample one, especially
    early in the sample.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fin_skills.ml import bet_sizing as bs


@pytest.fixture(scope="module")
def sim():
    d = bs.simulate_bets()
    n = d["ret"].shape[0]
    sig = d["side"][d["bet"]] * bs.bet_size(d["prob"])
    return d, n, sig


def test_the_size_curve_has_the_shape_it_claims():
    assert bs.bet_size(0.5) == pytest.approx(0.0, abs=1e-12)
    assert bs.bet_size(1.0) == pytest.approx(1.0, abs=1e-12)
    assert bs.bet_size(0.0) == pytest.approx(-1.0, abs=1e-12)
    grid = np.linspace(0.01, 0.99, 99)
    m = bs.bet_size(grid)
    assert (np.diff(m) > 0).all()                        # monotone in p
    assert (np.abs(m) <= 1.0 + 1e-12).all()
    # below the linear size near the coin flip, above it at the extremes
    lin = bs.linear_size(grid)
    assert m[grid < 0.85].max() < lin[grid < 0.85].max()
    assert bs.bet_size(0.55) < bs.linear_size(0.55)
    assert bs.bet_size(0.95) > bs.linear_size(0.95)
    assert bs.bet_size(0.55) / bs.linear_size(0.55) == pytest.approx(0.801, abs=0.01)
    # symmetry about the pivot
    assert bs.bet_size(0.3) == pytest.approx(-bs.bet_size(0.7), abs=1e-12)


def test_the_pivot_is_one_over_n_classes():
    assert bs.bet_size(1 / 3, 3) == pytest.approx(0.0, abs=1e-12)
    assert bs.bet_size(0.40, 3) > 0 > bs.bet_size(0.40, 2)   # opposite signs on the same p
    assert bs.bet_size(0.40, 3) == pytest.approx(0.1082, abs=1e-3)
    assert bs.bet_size(0.40, 2) == pytest.approx(-0.1617, abs=1e-3)
    assert bs.norm_cdf(0.0) == pytest.approx(0.5)
    assert bs.norm_cdf(-8.0) < 1e-12 and bs.norm_cdf(8.0) > 1 - 1e-12


def test_discretise_rounds_to_the_grid_and_clips():
    s = np.array([-2.0, -0.3, 0.0, 0.26, 0.74, 1.5])
    assert list(bs.discretise(s, 0.5)) == [-1.0, -0.5, 0.0, 0.5, 0.5, 1.0]
    assert np.array_equal(bs.discretise(s, 0.0), np.clip(s, -1.0, 1.0))
    # half-to-even: exactly 0.25 at step 0.5 goes to 0.0, and 0.75 goes to 1.0
    assert bs.discretise(np.array([0.25]), 0.5)[0] == 0.0
    assert bs.discretise(np.array([0.75]), 0.5)[0] == 1.0
    assert set(np.unique(bs.discretise(np.linspace(-1, 1, 101), 0.5))) == {-1.0, -0.5, 0.0,
                                                                          0.5, 1.0}


def test_the_sticky_rule_only_moves_after_a_full_step():
    s = np.array([0.0, 0.3, 0.4, 0.6, 0.55, 0.2, -0.6])
    out = bs.discretise_sticky(s, 0.5)
    assert out[0] == 0.0 and out[1] == 0.0               # 0.3 is not a full step from 0
    assert out[3] == 0.5                                 # 0.6 is, so it moves to the 0.5 level
    assert out[4] == 0.5                                 # 0.55 is not a full step from 0.5
    assert out[6] == -0.5
    assert bs.turnover(bs.discretise_sticky(s, 0.5)) <= bs.turnover(bs.discretise(s, 0.5))
    assert np.array_equal(bs.discretise_sticky(s, 0.0), np.clip(s, -1.0, 1.0))
    # a signal that never moves a full step never trades
    assert bs.turnover(bs.discretise_sticky(np.full(50, 0.4), 0.5)) == 0.0


def test_the_simulation_is_seeded_and_calibrated(sim):
    d, n, _ = sim
    d2 = bs.simulate_bets()
    assert np.array_equal(d["prob"], d2["prob"]) and np.array_equal(d["side"], d2["side"])
    assert not np.array_equal(bs.simulate_bets(seed=bs.SEED + 1)["prob"], d["prob"])
    assert (d["q"] > 0.5).all() and (d["q"] < 0.9).all()
    # P(correct) = q by construction, so the two agree up to sampling error
    k = d["q"].shape[0]
    assert abs(d["correct"].mean() - d["q"].mean()) < 3.0 / math.sqrt(k)
    assert abs(d["prob"].mean() - d["q"].mean()) < 0.05
    assert (d["side"] * d["fwd"] > 0)[d["correct"]].all()
    assert (d["bar"] >= 0).all() and (d["bar"] < n).all()


def test_averaging_caps_exposure_and_summing_does_not(sim):
    d, n, sig = sim
    pos_sum = bs.active_signals(n, d["bar"], sig, "sum")
    pos_avg = bs.active_signals(n, d["bar"], sig, "avg")
    assert np.max(np.abs(pos_avg)) <= 1.0 + 1e-12
    assert np.max(np.abs(pos_sum)) > 2.0                 # 3.196 in the demo
    assert np.mean(np.abs(pos_sum) > 1.0) > 0.05
    assert np.mean(np.abs(pos_avg) > 1.0) == 0.0
    assert bs.turnover(pos_sum) > 1.5 * bs.turnover(pos_avg)
    # not a rescaling: the divisor moves with concurrency
    r = np.corrcoef(pos_sum, pos_avg)[0, 1]
    assert 0.7 < r < 0.99
    conc = bs.concurrency(n, d["bar"])
    assert conc.max() >= 2 and np.mean(conc == 0) > 0.0
    assert np.allclose(pos_avg[conc > 0], pos_sum[conc > 0] / conc[conc > 0])
    with pytest.raises(ValueError):
        bs.active_signals(n, d["bar"], sig, "median")


def test_rounding_barely_reduces_turnover_but_the_sticky_rule_does(sim):
    d, n, sig = sim
    base = bs.active_signals(n, d["bar"], sig, "avg")
    t0 = bs.turnover(base)
    fine = bs.turnover(bs.discretise(base, 0.05))
    assert fine > 0.98 * t0                              # 100.3 % in the demo
    coarse = bs.turnover(bs.discretise(base, 1.0))
    assert coarse < 0.85 * t0
    sticky = bs.turnover(bs.discretise_sticky(base, 0.5))
    assert sticky < 0.5 * t0                             # 32 % in the demo
    assert np.corrcoef(bs.discretise_sticky(base, 0.5), base)[0, 1] < \
           np.corrcoef(bs.discretise(base, 0.5), base)[0, 1]


def test_discretisation_loses_at_zero_cost_and_wins_at_a_high_one(sim):
    d, n, sig = sim
    base = bs.active_signals(n, d["bar"], sig, "avg")
    sticky = bs.discretise_sticky(base, 0.5)
    free = (bs.sharpe(bs.pnl_series(base, d["ret"], 0.0)),
            bs.sharpe(bs.pnl_series(sticky, d["ret"], 0.0)))
    dear = (bs.sharpe(bs.pnl_series(base, d["ret"], 0.0020)),
            bs.sharpe(bs.pnl_series(sticky, d["ret"], 0.0020)))
    assert free[0] > free[1]                             # 0.90 vs 0.68
    assert dear[1] > dear[0]                             # +0.02 vs -1.40
    assert free[0] > dear[0]                             # costs always hurt
    assert math.isnan(bs.sharpe(np.zeros(10)))


def test_pnl_is_causal_and_charges_on_turnover(sim):
    d, n, _ = sim
    pos = np.zeros(n)
    pos[10:20] = 1.0
    pnl = bs.pnl_series(pos, d["ret"], cost=0.0)
    assert pnl[10] == pytest.approx(d["ret"][11])
    assert pnl[19] == pytest.approx(d["ret"][20])
    assert pnl[9] == 0.0 and pnl[20] == 0.0
    shocked = d["ret"].copy()
    shocked[30] += 1.0
    assert np.allclose(bs.pnl_series(pos, shocked, 0.0), pnl)     # nothing before t+1 leaks
    costed = bs.pnl_series(pos, d["ret"], cost=0.01)
    assert bs.turnover(pos) == pytest.approx(2.0)
    assert (pnl.sum() - costed.sum()) == pytest.approx(0.01 * 2.0)


def test_the_full_sample_concurrency_budget_is_a_look_ahead(sim):
    d, n, sig = sim
    pos_sum = bs.active_signals(n, d["bar"], sig, "sum")
    conc = bs.concurrency(n, d["bar"])
    causal = bs.expanding_max(conc)
    assert (np.diff(causal) >= 0).all() and causal[-1] == conc.max()
    first = int(np.argmax(causal >= conc.max()))
    assert first > 0.1 * n                               # bar 1236 of 5000 in the demo
    full_budget = pos_sum / conc.max()
    causal_budget = pos_sum / np.maximum(causal, 1.0)
    assert np.max(np.abs(causal_budget)) > 1.5 * np.max(np.abs(full_budget))
    early = slice(0, n // 4)
    assert np.mean(np.abs(causal_budget[early])) > 1.2 * np.mean(np.abs(full_budget[early]))
    assert np.mean(np.abs(causal_budget) > np.max(np.abs(full_budget))) > 0.0


def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.ml.bet_sizing")
    for head in ("=== 1. The size curve", "=== 2. Averaging active bets",
                 "=== 3. Discretisation", "=== 4. Budgeting concurrent bets"):
        assert head in out
    assert out.isascii()
    rule = [ln for ln in out.splitlines() if ln.startswith("Rule: ")]
    assert len(rule) == 1 and "Phi(z)" in rule[0] and "average" in rule[0]
    assert out.strip().splitlines()[-1].startswith("total runtime")
