"""fin_skills.microstructure.lob_models - the order book as a queueing system.

Each test asserts the PROPERTY the SKILL.md documents, not the printed digits: the exact
absorption probability reproduces Cont-Stoikov-Talreja (2010) Table 3 to the three decimals it
is printed in and agrees with an independent Monte Carlo; Proposition 5 as eqs. (14)/(16) state
it does NOT reproduce Table 4 (documented, and asserted so a future fix is visible); a seeded
book run is deterministic and its built-in symmetry check holds; and fill probability falls with
queue position while the volume-only rule and the value of a fill fall with it too.
"""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.microstructure.lob_models import (CST_PARAMS, CST_TABLE3, CST_TABLE4,
                                                  book_experiment, cst_table_checks,
                                                  fill_prob_before_move, position_sweep,
                                                  prob_mid_up, prob_mid_up_grid,
                                                  prob_mid_up_sim, stationary_queue_dist)


# ---------------------------------------------------------------- the stationary queue law
def test_stationary_queue_law_is_proper_and_matches_detailed_balance():
    pi = stationary_queue_dist(**CST_PARAMS)
    assert pi.sum() == pytest.approx(1.0, abs=1e-12)
    lam, mu, th = CST_PARAMS["lam"], CST_PARAMS["mu"], CST_PARAMS["theta"]
    for x in range(1, 12):                       # lam*pi[x-1] == (mu + x*theta)*pi[x]
        assert lam * pi[x - 1] == pytest.approx((mu + x * th) * pi[x], rel=1e-12)
    mean = float((np.arange(len(pi)) * pi).sum())
    assert 1.5 < mean < 1.8                      # the SKILL.md quotes 1.62 units
    # the linear cancellation rate is what makes this proper for ANY lambda
    assert stationary_queue_dist(lam=50.0, mu=0.1, theta=0.7).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive"):
        stationary_queue_dist(lam=-1.0, mu=1.0, theta=1.0)
    with pytest.raises(ValueError, match="n_max"):
        stationary_queue_dist(1.0, 1.0, 1.0, n_max=0)


# --------------------------------------------------- Proposition 3: direction of the move
def test_prob_mid_up_reproduces_the_papers_table_3():
    got = prob_mid_up_grid(5)
    want = np.array(CST_TABLE3)
    assert np.abs(got - want).max() < 1e-3       # Table 3 is printed to three decimals
    for i in range(5):
        assert got[i, i] == pytest.approx(0.5, abs=1e-9)   # equal queues, no edge


def test_prob_mid_up_is_antisymmetric_and_monotone_in_both_queues():
    for a in range(1, 6):
        for b in range(1, 6):
            assert prob_mid_up(a, b) + prob_mid_up(b, a) == pytest.approx(1.0, abs=1e-9)
    for b in (1, 3, 5):                          # a deeper ask makes an up move less likely
        vals = [prob_mid_up(a, b) for a in range(1, 7)]
        assert all(x > y for x, y in zip(vals, vals[1:]))
    for a in (1, 3, 5):                          # a deeper bid makes it more likely
        vals = [prob_mid_up(a, b) for b in range(1, 7)]
        assert all(x < y for x, y in zip(vals, vals[1:]))


def test_prob_mid_up_edge_cases_and_validation():
    assert prob_mid_up(0, 3) == 1.0              # the ask is already gone
    assert prob_mid_up(3, 0) == 0.0
    assert prob_mid_up(0, 0) == 0.5
    with pytest.raises(ValueError, match="non-negative"):
        prob_mid_up(-1, 2)
    with pytest.raises(ValueError, match="n_max"):
        prob_mid_up(80, 2, n_max=60)
    with pytest.raises(ValueError, match="positive"):
        prob_mid_up(1, 1, mu=0.0)


def test_the_truncation_is_harmless_and_the_simulation_agrees():
    for n in (30, 45):
        assert prob_mid_up(5, 1, n_max=n) == pytest.approx(prob_mid_up(5, 1, n_max=60), abs=1e-9)
    for (a, b) in ((1, 5), (5, 1), (2, 3)):
        p, se = prob_mid_up_sim(a, b, 60_000, seed=11)
        assert abs(p - prob_mid_up(a, b)) < 4.0 * se
    assert prob_mid_up_sim(2, 3, 5_000, seed=7) == prob_mid_up_sim(2, 3, 5_000, seed=7)
    assert prob_mid_up_sim(2, 3, 5_000, seed=7) != prob_mid_up_sim(2, 3, 5_000, seed=8)


# ---------------------------------------------- Proposition 5: the documented non-reproduction
def test_proposition_5_as_written_does_not_reproduce_table_4():
    chk = cst_table_checks()
    assert chk["table3_maxabs"] < 1e-3           # Table 3 does reproduce
    # and Table 4 does not - asserted so that a future correction is not silent
    assert chk["table4_maxabs"] > 0.1
    assert np.abs(np.array(chk["table4"]) - np.array(CST_TABLE4)).max() == \
        pytest.approx(chk["table4_maxabs"])


def test_fill_prob_before_move_is_monotone_and_validated():
    for a in (1, 3, 5):                          # deeper in the bid queue -> less likely
        vals = [fill_prob_before_move(b, a) for b in range(1, 7)]
        assert all(x > y for x, y in zip(vals, vals[1:]))
    for b in (1, 3, 5):                          # a deeper ask buys you time
        vals = [fill_prob_before_move(b, a) for a in range(1, 7)]
        assert all(x < y for x, y in zip(vals, vals[1:]))
    assert 0.0 < fill_prob_before_move(1, 1) < 1.0
    with pytest.raises(ValueError, match="at least 1"):
        fill_prob_before_move(0, 1)
    with pytest.raises(ValueError, match="positive"):
        fill_prob_before_move(1, 1, theta=0.0)


# -------------------------------------------------------- queue position, fills, toxicity
def test_a_seeded_book_run_is_deterministic():
    a = book_experiment(3, 10, n_trials=2_000, seed=5)
    assert a == book_experiment(3, 10, n_trials=2_000, seed=5)
    assert a["fill_prob"] != book_experiment(3, 10, n_trials=2_000, seed=6)["fill_prob"]


def test_a_symmetric_book_has_an_unconditional_up_probability_of_one_half():
    r = book_experiment(5, 20, n_trials=20_000, seed=3)
    assert r["p_mid_up_unconditional"] == pytest.approx(0.5, abs=0.02)
    assert r["unresolved"] == 0                  # every episode ends inside max_steps
    # and on an asymmetric book the simulated race matches the exact Proposition 3 value -
    # two independent implementations of the same quantity
    for depth, ask in ((20, 3), (4, 9), (7, 7)):
        sim = book_experiment(0, depth, ask_depth=ask, n_trials=20_000, seed=3)
        assert sim["p_mid_up_unconditional"] == pytest.approx(prob_mid_up(ask, depth), abs=0.015)


def test_fill_probability_falls_with_queue_position_and_volume_alone_cannot_see_it():
    rows = position_sweep(20, (0, 2, 5, 10, 19), n_trials=20_000, seed=3)
    fp = [r["fill_prob"] for r in rows]
    assert all(x > y for x, y in zip(fp, fp[1:]))          # strictly falling
    assert fp[0] / fp[-1] > 1.5                            # 0.977 vs 0.533 in the SKILL.md
    # the volume-only rule is exact at the front (nothing is ahead of you) and collapses
    assert rows[0]["volume_only"] == pytest.approx(rows[0]["fill_prob"], abs=1e-12)
    assert rows[3]["volume_only"] < rows[3]["fill_prob"] / 20.0
    # ... because most of the queue ahead leaves by cancelling, which never prints
    shares = [r["cancel_share_of_advance"] for r in rows[1:]]
    assert all(0.4 < s < 0.95 for s in shares)
    assert all(x < y for x, y in zip(shares, shares[1:]))
    # from the back of a symmetric queue, being filled IS the bid queue emptying
    assert rows[-1]["fill_prob"] == pytest.approx(1.0 - rows[-1]["p_mid_up_unconditional"],
                                                  abs=0.05)


def test_a_fill_is_worth_less_than_the_half_spread_with_no_informed_trader_anywhere():
    rows = position_sweep(20, (0, 2, 5, 10, 19), n_trials=20_000, seed=3)
    for r in rows:
        assert r["p_mid_up_given_fill"] < r["p_mid_up_unconditional"]
        assert r["mark_to_mid_ticks"] < r["half_spread_ticks"]
        assert r["adverse_selection_ticks"] == pytest.approx(0.5 - r["mark_to_mid_ticks"])
    quality = [r["mark_to_mid_ticks"] for r in rows]
    assert all(x > y for x, y in zip(quality, quality[1:]))   # deeper -> worse fills
    assert rows[0]["adverse_selection_ticks"] < 0.03          # the front barely gives anything
    assert rows[-1]["adverse_selection_ticks"] > 0.10         # the back gives back ~a third


def test_book_experiment_validates_its_arguments():
    for bad in (dict(position=-1, depth=5), dict(position=5, depth=5), dict(position=0, depth=0)):
        with pytest.raises(ValueError, match="depth"):
            book_experiment(n_trials=10, **bad)
    with pytest.raises(ValueError, match="ask_depth"):
        book_experiment(0, 5, ask_depth=0, n_trials=10)
    with pytest.raises(ValueError, match="positive"):
        book_experiment(0, 5, n_trials=10, lam=0.0)


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.microstructure.lob_models")
    assert "Rule: a passive order's fill probability is a function of your QUEUE POSITION" in out
    assert "8.27e-04" in out                     # the Table 3 reproduction
    assert out.isascii()
