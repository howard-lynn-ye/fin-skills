"""fin_skills.core.overfitting - PBO via CSCV, and the Minimum Backtest Length.

The load-bearing assertion is that PBO on pure noise sits on its own no-skill line. It is
checked across many seeded panels, never on one, because a single panel's PBO has an
across-panel standard deviation of 0.14 to 0.24 - all C(S,S/2) splits reuse the same rows.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from fin_skills.core.overfitting import (EULER_GAMMA, credible, cscv, crossover_grid, logit,
                                         min_backtest_length, min_backtest_length_bound,
                                         min_backtest_length_table, noise_panel, null_pbo,
                                         pbo, relative_rank, expected_max_sharpe)

T = 1008          # 4.0 years of daily observations, divisible by 16
S = 16


def _panels(n_configs: int, reps: int, base: int = 1000, edges=()):
    for s in range(reps):
        yield noise_panel(np.random.default_rng(base + s), T, n_configs, edges_annual=edges)


# ------------------------------------------------------------------ the definitions
def test_relative_rank_and_logit_agree_on_where_zero_is():
    ranks = np.arange(1, 11)
    w = relative_rank(ranks, 10)
    assert np.allclose(w, ranks / 11.0)                 # the paper's rank/(N+1)
    lam = logit(w)
    # logit <= 0 exactly when w <= 0.5, i.e. rank <= (N+1)/2 = 5.5 -> ranks 1..5
    assert (lam <= 0).sum() == 5
    assert np.isfinite(lam).all()
    assert not np.isfinite(logit(relative_rank(np.array([10]), 10, "n"))).all()
    with pytest.raises(ValueError):
        relative_rank(ranks, 10, "nope")


def test_null_pbo_is_half_only_for_even_n():
    assert null_pbo(10) == 0.5 and null_pbo(50) == 0.5 and null_pbo(100) == 0.5
    assert null_pbo(5) == pytest.approx(0.6)            # 3 of 5 ranks satisfy rank <= 3
    assert null_pbo(25) == pytest.approx(0.52)
    for n in range(2, 60):
        assert null_pbo(n) == math.ceil(n / 2) / n


def test_cscv_shapes_and_the_combination_count():
    x = next(_panels(8, 1))
    r = cscv(x, 8)
    assert r.n_combinations == math.comb(8, 4) == 70
    assert r.lam.shape == r.w.shape == r.os_rank.shape == (70,)
    assert set(np.unique(r.os_rank)) <= set(range(1, 9))
    assert 0.0 <= r.pbo <= 1.0
    assert r.n_configs == 8 and r.n_obs == T and r.n_blocks == 8
    assert r.report().isascii() and "PBO" in r.report()
    # C(16,8) is 12,870 - the PBO preprint prints 12,780, and an implementation that
    # copies that number is wrong.
    assert cscv(next(_panels(4, 1)), 16).n_combinations == 12870 != 12780


def test_cscv_rejects_bad_input():
    x = next(_panels(4, 1))
    with pytest.raises(ValueError, match="even"):
        cscv(x, 7)
    with pytest.raises(ValueError, match="at least 2 configurations"):
        cscv(x[:, :1], 8)
    with pytest.raises(ValueError, match="NaN"):
        bad = x.copy()
        bad[3, 2] = np.nan
        cscv(bad, 8)
    with pytest.raises(ValueError, match="exceeds"):
        cscv(x[:8], 16)
    with pytest.raises(ValueError, match=r"\(T, N\)"):
        cscv(x[:, 0], 8)


# ------------------------------------------------------------------ THE property
@pytest.mark.parametrize("n_configs", [10, 50])
def test_pbo_on_pure_noise_sits_on_the_no_skill_line(n_configs):
    """The load-bearing measurement. Mean over 24 independent panels, against the
    no-skill line for this N, with a tolerance of three standard errors."""
    vals = np.array([cscv(x, S).pbo for x in _panels(n_configs, 24)])
    mean = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(vals.size)
    line = null_pbo(n_configs)
    assert abs(mean - line) < 3 * se + 0.02, (mean, se, line)
    # and one panel is NOT a precise estimate: the spread is far above a binomial's
    assert vals.std(ddof=1) > 10 * math.sqrt(0.25 / math.comb(S, S // 2))


def test_planting_a_real_strategy_pulls_pbo_down():
    """Paired: the same panels, the edge added to column 0 only."""
    base = list(_panels(50, 12, base=1000))
    drift = 0.01 / np.sqrt(252)
    out = {}
    for sr in (0.0, 2.0):
        vals = []
        for x in base:
            y = x.copy()
            y[:, 0] += sr * drift
            vals.append(cscv(y, S).pbo)
        out[sr] = float(np.mean(vals))
    assert out[2.0] < out[0.0] - 0.15, out
    assert out[0.0] > 0.35


def test_the_winners_out_of_sample_standing_decays_as_the_grid_grows():
    """Nested and paired: one 200-column panel per seed, each N keeping the first N."""
    wide = list(_panels(200, 8, base=3000, edges=(1.0,)))
    small = [cscv(x[:, :5], S) for x in wide]
    large = [cscv(x, S) for x in wide]
    p_real_small = np.mean([float((r.is_best == 0).mean()) for r in small])
    p_real_large = np.mean([float((r.is_best == 0).mean()) for r in large])
    sr_small = np.mean([float(np.median(r.os_perf)) for r in small])
    sr_large = np.mean([float(np.median(r.os_perf)) for r in large])
    assert p_real_small > 2 * p_real_large, (p_real_small, p_real_large)
    assert sr_small > sr_large > 0, (sr_small, sr_large)
    assert np.mean([r.pbo for r in large]) > np.mean([r.pbo for r in small])


def test_a_correlated_grid_is_noisier_than_independent_columns():
    grid = [cscv(crossover_grid(np.random.default_rng(4000 + s), T), S).pbo for s in range(10)]
    indep = [cscv(x, S).pbo for x in _panels(48, 10, base=1000)]
    assert np.std(grid, ddof=1) > np.std(indep, ddof=1)
    g0 = crossover_grid(np.random.default_rng(4000), T)
    cc = np.corrcoef(g0, rowvar=False)
    assert float(np.abs(cc[~np.eye(g0.shape[1], dtype=bool)]).mean()) > 0.4


def test_pbo_helper_matches_cscv():
    x = next(_panels(12, 1))
    assert pbo(x, 8) == cscv(x, 8).pbo


# ------------------------------------------------------------------ MinBTL
def test_expected_max_sharpe_matches_the_formula_and_the_published_anchor():
    for n in (5, 10, 100, 1000):
        want = ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n)
                + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e)))
        assert expected_max_sharpe(n) == pytest.approx(want)
    # Notices of the AMS 61(5), 2014: "N = 10 ... a Sharpe ratio IS of 1.57"
    assert expected_max_sharpe(10) == pytest.approx(1.57, abs=0.005)
    assert expected_max_sharpe(10, 0.5) == pytest.approx(0.5 * expected_max_sharpe(10))
    with pytest.raises(ValueError):
        expected_max_sharpe(1)


def test_expected_max_sharpe_is_close_to_the_true_expected_maximum():
    rng = np.random.default_rng(9)
    for n in (10, 100):
        emp = rng.standard_normal((40_000, n)).max(axis=1).mean()
        err = expected_max_sharpe(n) - emp
        assert 0.0 < err < 0.08, (n, err)      # overstates, slightly, always


def test_min_backtest_length_reproduces_the_papers_two_prose_anchors():
    # "if only five years of data are available, no more than forty-five ... configurations"
    assert min_backtest_length(1.0, 45) == pytest.approx(5.0, abs=0.01)
    # "After trying only seven independent strategy configurations ... a two-year backtest"
    assert min_backtest_length(1.0, 7) == pytest.approx(2.0, abs=0.08)
    # Theorem 2's own upper bound
    for n in (10, 100, 1000):
        assert min_backtest_length(1.0, n) < min_backtest_length_bound(1.0, n)
    # halving the claimed Sharpe quadruples the requirement
    assert min_backtest_length(0.5, 50) == pytest.approx(4 * min_backtest_length(1.0, 50))
    with pytest.raises(ValueError):
        min_backtest_length(0.0, 50)


def test_min_backtest_length_table_is_monotone_in_the_trial_count():
    rows = min_backtest_length_table(1.0)
    years = [y for _, _, y in rows]
    assert years == sorted(years)
    assert dict((n, round(y, 2)) for n, _, y in rows)[50] == 5.18


def test_credible_contrasts_the_sample_with_the_requirement():
    c = credible(1.0, 50, T, 252)
    assert c["sample_years"] == pytest.approx(4.0)
    assert not c["credible"] and c["shortfall_years"] > 1.0
    assert c["expected_max_sharpe_at_this_length"] > 1.0    # noise alone beats the claim
    assert credible(2.0, 50, T, 252)["credible"]


# ------------------------------------------------------------------ determinism / demo
def test_a_seeded_run_is_deterministic():
    a = cscv(noise_panel(np.random.default_rng(5), T, 20), S)
    b = cscv(noise_panel(np.random.default_rng(5), T, 20), S)
    assert a.pbo == b.pbo
    assert np.array_equal(a.os_rank, b.os_rank) and np.array_equal(a.is_best, b.is_best)


@pytest.mark.slow
def test_main_prints_the_measurements_and_the_rule(run_main):
    out = run_main("fin_skills.core.overfitting")
    assert out.isascii()
    assert "RULE:" in out
    assert "no-skill line" in out and "12870" in out
    assert "MinBTL" in out and "1.5746" in out
