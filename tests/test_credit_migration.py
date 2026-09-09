"""fin_skills.credit.migration - cohort vs duration estimators, withdrawals, and embedding.

The documented properties: the published-shape matrix is exactly stochastic and has structural
zeros; the negative off-diagonals of logm(P) are EXACTLY those structural zeros, so the implied
six-month matrix carries negative probabilities while still squaring back to P to machine
precision; the clip-and-rebalance repair buys a valid generator at the cost of reproducing P;
`5 x PD_1` errs in both directions and the sign flips inside single-B; on one seeded panel the
cohort estimator reports AAA -> D as exactly zero where the duration estimator reports a
positive number; and treating NR as an absorbing state understates every default probability
while censoring does not.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from fin_skills.credit.migration import (DEFAULT, GRADES, SEED, check_stochastic,
                                         cohort_estimator, cumulative_pd, duration_estimator,
                                         embedding_report, estimator_comparison, fill_scan,
                                         fill_structural_zeros, generator_log, horizon_trap,
                                         is_valid_generator, matrix_power,
                                         published_matrix, regularised_generator,
                                         simulate_panel, structural_zeros, true_generator,
                                         withdrawal_bias)


# ------------------------------------------------------------------ the matrix itself
def test_the_matrix_is_exactly_stochastic_with_an_absorbing_default():
    p = published_matrix()
    chk = check_stochastic(p)
    assert chk["max_row_sum_error"] < 1e-12
    assert chk["min_entry"] == 0.0                      # the structural zeros
    assert chk["is_stochastic"] == 1.0
    assert p[DEFAULT, DEFAULT] == 1.0 and p[DEFAULT, :DEFAULT].sum() == 0.0
    assert p[0, DEFAULT] == 0.0, "AAA -> D is the structural zero the skill is about"


def test_the_structural_zeros_are_the_cells_the_skill_names():
    assert set(structural_zeros(published_matrix())) == {
        ("AAA", "D"), ("B", "AAA"), ("CCC", "AAA"), ("CCC", "AA")}


def test_matrix_powers_stay_stochastic_and_default_only_accumulates():
    p = published_matrix()
    prev = np.zeros(len(GRADES) - 1)
    for h in (1, 2, 5, 10, 30):
        ph = matrix_power(p, h)
        assert np.max(np.abs(ph.sum(axis=1) - 1.0)) < 1e-12
        assert ph.min() >= -1e-15
        pd = ph[:DEFAULT, DEFAULT]
        assert np.all(pd >= prev - 1e-15), "cumulative default cannot fall with the horizon"
        prev = pd
    with pytest.raises(ValueError):
        matrix_power(p, -1)
    assert cumulative_pd(p).shape == (len(GRADES) - 1, 5)


# ------------------------------------------------------------------ the embedding trap
def test_the_structural_zero_forces_a_negative_generator_entry():
    e = embedding_report()
    assert e.aaa_default_1y == 0.0
    assert e.aaa_default_2y == pytest.approx(0.000056, abs=5e-7)
    assert e.n_negative_generator_cells == 4
    assert e.min_generator_offdiag == pytest.approx(-6.507e-05, rel=1e-3)
    assert e.generator_offdiag_cell == ("CCC", "AA")
    assert e.aaa_default_generator == pytest.approx(-2.6000e-05, rel=1e-3)
    assert not is_valid_generator(generator_log(published_matrix()))


def test_the_six_month_matrix_holds_a_negative_probability_and_still_squares_back():
    e = embedding_report()
    assert e.n_negative_root_cells == 4
    assert e.min_root_entry < 0.0
    assert e.min_root_entry == pytest.approx(-1.494e-05, rel=1e-3)
    assert e.root_cell == ("CCC", "AA")
    assert e.aaa_default_root == pytest.approx(-6.6174e-06, rel=1e-3)
    # the check people actually run passes, which is exactly why the trap survives
    assert e.root_squared_error < 1e-12


def test_the_negative_cells_are_precisely_the_structural_zeros():
    e = embedding_report()
    assert e.zeros_explain_negatives
    assert set(e.negative_cells) == set(e.structural_zeros)
    assert len(e.negative_cells) == 4


def test_a_token_epsilon_does_not_buy_embeddability_but_a_basis_point_does():
    rows = {r["fill"]: r for r in fill_scan()}
    assert rows[1e-6]["valid_generator"] == 0.0, "an epsilon is barely better than zero"
    assert rows[1e-6]["min_root_entry"] < 0.0
    assert rows[5e-5]["valid_generator"] == 0.0
    assert rows[1e-4]["valid_generator"] == 1.0
    assert rows[1e-4]["min_generator_offdiag"] >= 0.0
    # the generator becomes valid strictly LATER than the root becomes non-negative
    assert rows[3e-5]["min_root_entry"] >= 0.0 and rows[3e-5]["valid_generator"] == 0.0
    # more fill is monotonically better
    mins = [rows[f]["min_generator_offdiag"] for f in sorted(rows)]
    assert all(b >= a for a, b in zip(mins, mins[1:]))
    filled = fill_structural_zeros(published_matrix(), 1e-4)
    assert np.max(np.abs(filled.sum(axis=1) - 1.0)) < 1e-12
    assert structural_zeros(filled) == []
    assert is_valid_generator(generator_log(filled))
    assert expm(generator_log(filled) / 2.0).min() >= 0.0


def test_the_repair_buys_a_valid_generator_and_pays_for_it():
    reg = regularised_generator()
    assert reg["valid"] is True
    assert reg["min_root_entry"] >= 0.0
    assert reg["reproduction_error_bp"] == pytest.approx(0.51, abs=0.005)
    assert reg["reproduction_error_bp"] > 0.0, "you cannot have both properties"
    assert np.max(np.abs(reg["generator"].sum(axis=1))) < 1e-10


# ------------------------------------------------------------------ the horizon trap
def test_five_times_the_one_year_pd_errs_in_both_directions():
    rows = {r["grade"]: r for r in horizon_trap()}
    assert rows["AAA"]["naive_pct"] == 0.0 and rows["AAA"]["true_pct"] > 0.0
    assert rows["AAA"]["true_pct"] == pytest.approx(0.0636, abs=5e-5)
    assert rows["BBB"]["error_bp"] == pytest.approx(-91.3, abs=0.05)
    assert rows["CCC"]["error_bp"] == pytest.approx(4304.5, abs=0.05)
    assert rows["CCC"]["naive_pct"] == pytest.approx(99.0, abs=1e-9)
    assert rows["CCC"]["true_pct"] == pytest.approx(55.9554, abs=5e-4)
    signs = [np.sign(rows[g]["error_bp"]) for g in GRADES[:-1]]
    assert -1 in signs and 1 in signs, "there is no conservative direction to round in"


def test_the_horizon_trap_respects_the_horizon_argument():
    p = published_matrix()
    one = horizon_trap(p, horizon=1)
    for r in one:
        assert r["error_bp"] == pytest.approx(0.0, abs=1e-9), "at one year the two agree"


# ------------------------------------------------------------------ cohort vs duration
def test_the_seeded_panel_is_deterministic():
    a, b = simulate_panel(n_firms=300, years=5), simulate_panel(n_firms=300, years=5)
    assert np.array_equal(a.yearly, b.yearly) and np.array_equal(a.counts, b.counts)
    c = simulate_panel(n_firms=300, years=5, seed=SEED + 1)
    assert not np.array_equal(a.yearly, c.yearly)


def test_the_simulation_generator_is_valid_with_a_zero_aaa_default_intensity():
    q = true_generator()
    assert is_valid_generator(q)
    assert q[0, DEFAULT] == 0.0
    assert np.all(q[DEFAULT] == 0.0), "default is absorbing"
    # but the one-year matrix it produces has a POSITIVE AAA -> D: two-step paths exist
    assert expm(q)[0, DEFAULT] > 0.0


def test_the_cohort_estimator_reports_zero_where_the_duration_estimator_does_not():
    c = estimator_comparison()
    assert c["cohort_aaa_default"] == 0.0
    assert c["duration_aaa_default"] > 0.0
    assert c["true_aaa_default"] == pytest.approx(0.00002472, abs=5e-9)
    assert c["duration_aaa_default"] == pytest.approx(0.00002175, abs=5e-9)
    assert c["duration_valid_generator"] is True
    assert c["duration_root_min"] >= 0.0
    assert c["cohort_root_min"] < 0.0
    assert len(c["rows"]) == len(GRADES) - 1


def test_both_estimators_return_valid_objects():
    panel = simulate_panel(n_firms=1500, years=10)
    p = cohort_estimator(panel.yearly)
    assert np.max(np.abs(p.sum(axis=1) - 1.0)) < 1e-12 and p.min() >= 0.0
    assert p[DEFAULT, DEFAULT] == 1.0
    q = duration_estimator(panel.counts, panel.exposure)
    assert is_valid_generator(q)
    assert np.all(np.diag(q)[:DEFAULT] <= 0.0)


# ------------------------------------------------------------------ withdrawals
def test_treating_nr_as_a_state_understates_default_and_censoring_does_not():
    wb = withdrawal_bias()
    rows = {r["grade"]: r for r in wb["rows"]}
    assert wb["n_withdrawn"] == 2433
    assert rows["CCC"]["nr_error_bp"] == pytest.approx(-1748, abs=1.0)
    assert rows["B"]["nr_error_bp"] == pytest.approx(-593, abs=1.0)
    assert rows["BBB"]["nr_error_bp"] == pytest.approx(-54, abs=1.0)
    # the NR bias is one-directional and grows as the grade falls
    for g in ("A", "BBB", "BB", "B", "CCC"):
        assert rows[g]["nr_error_bp"] < 0.0
    order = [abs(rows[g]["nr_error_bp"]) for g in ("A", "BBB", "BB", "B", "CCC")]
    assert all(b > a for a, b in zip(order, order[1:]))
    # censoring is noise, not bias: both signs, and much smaller at the worst grade
    signs = {np.sign(rows[g]["censored_error_bp"]) for g in GRADES[:-1]}
    assert signs == {-1.0, 1.0}
    assert abs(rows["CCC"]["censored_error_bp"]) < abs(rows["CCC"]["nr_error_bp"]) / 10.0
    assert wb["nr_mass_5y"] == pytest.approx(0.2276, abs=5e-5)


# ------------------------------------------------------------------ the demo
def test_the_demo_prints_the_rule_and_stays_ascii(run_main):
    out = run_main("fin_skills.credit.migration")
    assert all(ord(c) < 128 for c in out), "a non-ASCII char dies on a stock Windows console"
    assert "THE RULE:" in out
    assert "never take a ROOT of it without checking" in out
    assert "NEGATIVE PROBABILITY" in out
    for token in ("0.0056%", "-6.507e-05", "-1.494e-05", "+4304.5", "0.002175%", "-1748"):
        assert token in out
