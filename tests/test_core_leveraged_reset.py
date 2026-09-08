"""fin_skills.core.leveraged_reset - 'k times the index' is a one-day statement.

The Monte Carlo demo (section E, 20,000 paths x 252 days x 3 vols) lives under `main()`
and is marked slow; the arithmetic it rests on is tested directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.core import leveraged_reset as lr


def test_one_day_leverage_is_exactly_k_times_the_index():
    assert lr.leveraged_wealth(np.array([0.01]), 3.0).tolist() == pytest.approx([1.03])
    assert lr.leveraged_wealth(np.array([0.01]), -1.0).tolist() == pytest.approx([0.99])
    assert lr.leveraged_wealth(np.array([0.01]), -3.0).tolist() == pytest.approx([0.97])


def test_financing_and_expense_are_daily_drags_on_borrowed_and_total_nav():
    r = np.zeros(252)
    assert lr.leveraged_wealth(r, 3.0)[-1] == pytest.approx(1.0)
    fin = lr.leveraged_wealth(r, 3.0, financing=0.04)[-1]
    assert fin == pytest.approx((1 - 2 * 0.04 / 252) ** 252)          # 2x NAV borrowed
    assert lr.leveraged_wealth(r, 1.0, financing=0.04)[-1] == pytest.approx(1.0)  # nothing borrowed
    assert lr.leveraged_wealth(r, 3.0, expense=0.0082)[-1] == pytest.approx((1 - 0.0082 / 252) ** 252)


def test_shaped_path_pins_the_endpoints_it_promises():
    z = np.random.default_rng(0).standard_normal(252)
    for vol in (0.16, 0.25, 0.40):
        r = lr.shaped_path(z, vol, 0.0)
        assert np.prod(1 + r) - 1 == pytest.approx(0.0, abs=1e-12)
        assert np.log1p(r).std() * np.sqrt(252) == pytest.approx(vol, abs=1e-12)
    r = lr.shaped_path(z, 0.16, 0.30)
    assert np.prod(1 + r) - 1 == pytest.approx(0.30, abs=1e-12)


def test_flat_but_volatile_year_loses_money_at_3x():
    z = np.random.default_rng(0).standard_normal(252)
    for vol in (0.16, 0.25, 0.40):
        r = lr.shaped_path(z, vol, 0.0)
        assert np.prod(1 + 3 * r) - 1 < 0
        assert np.prod(1 - 1 * r) - 1 < 0        # the inverse product bleeds too
    losses = [np.prod(1 + 3 * lr.shaped_path(z, v, 0.0)) - 1 for v in (0.16, 0.25, 0.40)]
    assert losses[0] > losses[1] > losses[2]      # more vol, more drag


def test_analytic_decay_matches_the_exact_path_to_second_order():
    z = np.random.default_rng(0).standard_normal(252)
    r = lr.shaped_path(z, 0.16, 0.0)
    exact = np.prod(1 + 3 * r) - 1
    approx = lr.analytic_lev_return(0.0, float((r ** 2).sum()), 3)
    assert approx < 0
    assert exact == pytest.approx(approx, abs=2e-3)
    assert lr.analytic_lev_return(0.10, 0.0, 3) == pytest.approx(1.1 ** 3 - 1)   # no variance, no drag


def test_fast_sections_print_their_headings(capsys):
    rng = np.random.default_rng(0)
    lr.section_a()
    r16 = lr.section_bc(rng)
    lr.section_d(rng)
    lr.section_e(rng, n_paths=200)
    lr.section_f(r16)
    lr.section_g(rng)
    lr.section_h(rng)
    lr.section_i(rng)
    lr.section_j(rng)
    out = capsys.readouterr().out
    for letter in "ABCDEFGHIJ":
        assert f"\n{letter}. " in "\n" + out
    assert len(r16) == 252 and np.prod(1 + r16) - 1 == pytest.approx(0.0, abs=1e-12)


def test_sections_are_deterministic_for_the_same_generator_state(capsys):
    lr.section_d(np.random.default_rng(5))
    first = capsys.readouterr().out
    lr.section_d(np.random.default_rng(5))
    assert capsys.readouterr().out == first


@pytest.mark.slow
def test_main_runs_the_monte_carlo_and_ends_with_the_rule(capsys):
    lr.main()
    out = capsys.readouterr().out
    assert "E. Holding-period sensitivity: 20,000 Monte Carlo paths" in out
    assert out.rstrip().endswith("period the answer depends on the path, and the sign of the gap is not fixed.")
    lr.main()
    assert capsys.readouterr().out == out           # SEED = 0 throughout
