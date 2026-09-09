"""fin_skills.strategies.alpha_combine - IC, neutralization, and the three ways an IC table lies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fin_skills.strategies.alpha_combine import (PERIODS_PER_YEAR, clipped_fraction, combine_rank,
                                                 combine_zscore, cost_aware_weights, cs_rank,
                                                 cs_zscore, forward_return, full_sample_beta,
                                                 ic_series, ic_weights, icir, ls_returns,
                                                 net_of_cost, neutralize, newey_west_se,
                                                 newey_west_tstat, overlap_tstats, plain_tstat,
                                                 realized_beta, rolling_beta, sharpe,
                                                 simulate_panel, smooth, to_weights, turnover,
                                                 winsorize)


@pytest.fixture(scope="module")
def panel():
    return simulate_panel(n_names=60, n_days=500, n_sectors=4, seed=5)


# --------------------------------------------------------------- IC against an independent impl
def test_rank_ic_matches_scipy_spearmanr_row_by_row(panel):
    """The vectorised row correlation is the documented equal of scipy, not an approximation."""
    a = panel["alphas"]["idio_fast"].iloc[50:70]
    f = forward_return(panel["returns"], 1).iloc[50:70]
    ours = ic_series(a, f, "spearman").to_numpy()
    theirs = np.array([stats.spearmanr(a.iloc[i], f.iloc[i])[0] for i in range(len(ours))])
    assert np.max(np.abs(ours - theirs)) < 1e-12


def test_pearson_ic_matches_scipy_and_differs_from_the_rank_version(panel):
    a = panel["alphas"]["idio_fast"].iloc[50:70]
    f = forward_return(panel["returns"], 1).iloc[50:70]
    ours = ic_series(a, f, "pearson").to_numpy()
    theirs = np.array([stats.pearsonr(a.iloc[i], f.iloc[i])[0] for i in range(len(ours))])
    assert np.max(np.abs(ours - theirs)) < 1e-12
    assert not np.allclose(ours, ic_series(a, f, "spearman").to_numpy())
    with pytest.raises(ValueError, match="pearson"):
        ic_series(a, f, "kendall")


def test_ic_of_a_perfect_forecast_is_one_and_of_its_negation_minus_one(panel):
    fwd = forward_return(panel["returns"], 1).iloc[:-1]
    assert ic_series(fwd, fwd).mean() == pytest.approx(1.0)
    assert ic_series(-fwd, fwd).mean() == pytest.approx(-1.0)


def test_forward_return_is_the_compounded_forward_window():
    r = pd.DataFrame({"a": [0.1, 0.2, -0.1, 0.05]})
    f3 = forward_return(r, 3)
    assert f3["a"].iloc[0] == pytest.approx(1.2 * 0.9 * 1.05 - 1.0)
    assert f3["a"].iloc[-3:].isna().all()
    assert forward_return(r, 1)["a"].iloc[0] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="horizon"):
        forward_return(r, 0)


# ---------------------------------------------------------- TRAP 1: the HAC correction, both ways
def test_newey_west_reduces_to_the_plain_t_at_zero_lags():
    x = pd.Series(np.random.default_rng(0).normal(0.1, 1.0, 400))
    # nw uses the population sd, plain_tstat the sample sd, so they differ by sqrt(n/(n-1))
    assert newey_west_tstat(x, 0) == pytest.approx(plain_tstat(x) * np.sqrt(x.size / (x.size - 1)))
    with pytest.raises(ValueError, match="lags"):
        newey_west_se(x, -1)


def test_newey_west_widens_the_error_on_a_positively_autocorrelated_series():
    """The documented mechanism: AC(1) > 0 -> the i.i.d. se is too small -> the t is too big."""
    rng = np.random.default_rng(3)
    e = rng.normal(0.0, 1.0, 3000)
    ar = np.empty_like(e)
    v = 0.0
    for t in range(e.size):
        v = 0.9 * v + e[t]
        ar[t] = v + 1.0
    ar = pd.Series(ar)
    assert ar.autocorr(1) > 0.8
    assert newey_west_se(ar, 40) > 2.0 * newey_west_se(ar, 0)
    assert abs(newey_west_tstat(ar, 40)) < 0.6 * abs(plain_tstat(ar))
    iid = pd.Series(rng.normal(1.0, 1.0, 3000))
    assert newey_west_se(iid, 40) == pytest.approx(newey_west_se(iid, 0), rel=0.25)


def test_overlap_inflates_only_when_the_ALPHA_is_persistent(panel):
    """Section 3's headline, and the half of it everyone gets wrong."""
    ret = panel["returns"]
    slow = overlap_tstats(panel["alphas"]["idio_slow"], ret, 21)
    fast = overlap_tstats(panel["alphas"]["idio_fast"], ret, 21)
    assert slow["ic_autocorr1"] > 0.7 and abs(fast["ic_autocorr1"]) < 0.3
    # persistent alpha: the naive overlapping t is far above the HAC-corrected one
    assert slow["t_overlapping"] > 2.0 * slow["t_newey_west"]
    # near-i.i.d. alpha: HAC barely moves it, so the overlapping sample was honest
    assert fast["t_newey_west"] == pytest.approx(fast["t_overlapping"], rel=0.35)
    # and both keep far more observations than sampling every h-th day
    assert slow["n_overlapping"] > 15 * slow["n_independent"]


def test_lags_equal_h_minus_one_gives_no_correction_at_horizon_one(panel):
    """The rule of thumb is about the overlap, not the alpha - at h=1 it does nothing."""
    ic = ic_series(panel["alphas"]["idio_slow"], forward_return(panel["returns"], 1))
    assert ic.autocorr(1) > 0.2
    assert newey_west_tstat(ic, 0) == pytest.approx(
        plain_tstat(ic) * np.sqrt(ic.size / (ic.size - 1)))
    assert abs(newey_west_tstat(ic, 21)) < abs(plain_tstat(ic))


# ------------------------------------------------------------------------------ neutralization
def test_residual_is_orthogonal_to_every_column_it_was_neutralized_against(panel):
    """The defining property: zero cross-sectional correlation with the design, per period."""
    a, sec = panel["alphas"]["sector_bet"], panel["sector"]
    beta = rolling_beta(panel["returns"], panel["market"], window=60, lag=1)
    resid = neutralize(a, sector=sec, beta=beta)
    t = 300
    row, b = resid.iloc[t], beta.iloc[t]
    assert np.isfinite(row).all()
    assert abs(np.corrcoef(row, b)[0, 1]) < 1e-8               # orthogonal to beta
    means = row.groupby(sec).mean()                            # and to every sector dummy
    assert float(means.abs().max()) < 1e-8


def test_neutralizing_kills_the_planted_exposure_and_spares_the_others(panel):
    """The acceptance test of section 4: remove what you named, and nothing else."""
    fwd = forward_return(panel["returns"], 1)
    sec = panel["sector"]
    raw_sec = ic_series(panel["alphas"]["sector_bet"], fwd).mean()
    neu_sec = ic_series(neutralize(panel["alphas"]["sector_bet"], sector=sec), fwd).mean()
    raw_idio = ic_series(panel["alphas"]["idio_fast"], fwd).mean()
    neu_idio = ic_series(neutralize(panel["alphas"]["idio_fast"], sector=sec), fwd).mean()
    assert abs(neu_sec) < 0.5 * abs(raw_sec)                   # most of the sector bet is gone
    assert neu_idio == pytest.approx(raw_idio, rel=0.25)       # the idio alpha is not


def test_neutralize_validates_its_inputs(panel):
    a = panel["alphas"]["idio_fast"]
    with pytest.raises(ValueError, match="sector map, a beta frame"):
        neutralize(a)
    with pytest.raises(ValueError, match="does not cover"):
        neutralize(a, sector=panel["sector"].iloc[:5])
    with pytest.raises(ValueError, match="window"):
        rolling_beta(panel["returns"], panel["market"], window=2)


def test_a_static_beta_series_is_accepted_and_broadcast(panel):
    b = full_sample_beta(panel["returns"], panel["market"])
    assert isinstance(b, pd.Series) and b.notna().all()
    resid = neutralize(panel["alphas"]["low_beta"], beta=b)
    assert resid.shape == panel["alphas"]["low_beta"].shape


# ------------------------------------------------------- TRAP 2: measured on the BOOK, not the IC
def test_a_peeking_beta_hedges_better_than_anything_causal(panel):
    """Section 5's INVARIANT. The script measures this on three panel sizes; only the
    peeking row is stable, which is why the skill does not claim the stale-vs-causal
    ordering (it flips at 60 names) or a signed Sharpe effect."""
    ret, mkt = panel["returns"], panel["market"]
    a = panel["alphas"]["low_beta"]

    def book_beta(b=None):
        sig = a if b is None else neutralize(a, beta=b)
        return abs(realized_beta(ls_returns(to_weights(cs_zscore(sig)), ret), mkt))

    none_ = book_beta()
    causal_ = book_beta(rolling_beta(ret, mkt, window=126, lag=1))
    stale_ = book_beta(full_sample_beta(ret, mkt))
    ahead_ = book_beta(rolling_beta(ret, mkt, window=126, lag=0).shift(-63))
    assert none_ > 0.15                            # the raw book really is beta-exposed
    assert causal_ < none_ and stale_ < none_      # any hedge helps
    assert ahead_ < 0.5 * min(causal_, stale_)     # peeking beats what was achievable
    assert ahead_ < 0.2 * none_                    # and removes ~85% of it


def test_realized_beta_recovers_a_known_exposure():
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0, 0.01, 2000))
    p = 0.7 * m + pd.Series(rng.normal(0, 0.001, 2000))
    assert realized_beta(p, m) == pytest.approx(0.7, rel=0.02)


# ------------------------------------------------------------- TRAP 3: the clip that clips nothing
def test_winsorizing_a_rank_moves_exactly_zero_cells(panel):
    """Section 6's headline. A rank spans [-0.5, 0.5]; 3 cross-sectional sd is +/-0.87."""
    fat = panel["alphas"]["idio_fast"].copy()
    fat.iloc[::13, 0] *= 60.0
    assert clipped_fraction(fat, 3.0) > 0.0
    assert clipped_fraction(cs_rank(fat), 3.0) == 0.0
    assert clipped_fraction(cs_rank(fat), 1.5) > 0.0           # a tight enough clip does bite
    pd.testing.assert_frame_equal(winsorize(cs_rank(fat), 3.0), cs_rank(fat))


def test_winsorize_and_zscore_are_affine_images_but_not_equal_frames(panel):
    """Section 6(b): the two orders give DIFFERENT frames and the SAME book, because each is
    an affine image of the other and to_weights is affine-invariant. A rank is not affine."""
    fat = panel["alphas"]["idio_fast"].copy()
    fat.iloc[::13, 0] *= 60.0
    wz, zw = cs_zscore(winsorize(fat, 3.0)), winsorize(cs_zscore(fat), 3.0)
    assert np.nanmax(np.abs(wz.to_numpy() - zw.to_numpy())) > 1.0        # frames differ
    assert np.nanmax(np.abs(to_weights(wz).to_numpy()
                            - to_weights(zw).to_numpy())) < 1e-12        # books do not
    # to_weights is invariant to any positive affine row transform, which is why
    assert np.nanmax(np.abs(to_weights(wz).to_numpy()
                            - to_weights(3.5 * wz + 2.0).to_numpy())) < 1e-12
    # and a rank is NOT an affine image of a z-score, so its book genuinely differs
    assert np.nanmax(np.abs(to_weights(cs_rank(fat)).to_numpy()
                            - to_weights(wz).to_numpy())) > 1e-3


def test_an_unclipped_outlier_takes_over_the_book(panel):
    fat = panel["alphas"]["idio_fast"].copy()
    fat.iloc[::13, 0] *= 60.0
    unclipped = float(to_weights(cs_zscore(fat)).abs().max().max())
    clipped = float(to_weights(cs_zscore(winsorize(fat, 3.0))).abs().max().max())
    ranked = float(to_weights(cs_rank(fat)).abs().max().max())
    # the separation grows with the number of names (0.44 vs 0.18 on the 200-name demo panel),
    # because a 3-sd clip is computed from an sd the outlier itself inflated
    assert unclipped > clipped > ranked
    with pytest.raises(ValueError, match="n_sd"):
        winsorize(fat, 0.0)


# ------------------------------------------------------- weights, turnover, cost and combination
def test_weights_are_dollar_neutral_with_unit_gross(panel):
    w = to_weights(cs_zscore(panel["alphas"]["idio_fast"])).dropna()
    assert float(w.sum(axis=1).abs().max()) < 1e-12
    assert w.abs().sum(axis=1).dropna().round(12).eq(1.0).all()


def test_turnover_is_one_when_the_book_flips_and_zero_when_it_is_held():
    held = pd.DataFrame({"a": [0.5, 0.5, 0.5], "b": [-0.5, -0.5, -0.5]})
    flip = pd.DataFrame({"a": [0.5, -0.5, 0.5], "b": [-0.5, 0.5, -0.5]})
    assert turnover(held) == pytest.approx(0.0)
    assert turnover(flip) == pytest.approx(1.0)


def test_cost_is_charged_on_one_way_turnover(panel):
    w = to_weights(cs_zscore(panel["alphas"]["idio_fast"]))
    g = ls_returns(w, panel["returns"])
    n = net_of_cost(g, w, 10.0)
    assert (n <= g.reindex(n.index) + 1e-15).all()
    assert (g.reindex(n.index) - n).mean() == pytest.approx(turnover(w) * 10.0 / 1e4, rel=0.05)
    assert net_of_cost(g, w, 0.0).equals(g.reindex(net_of_cost(g, w, 0.0).index))


def test_a_high_turnover_alpha_loses_more_of_its_sharpe_to_the_same_cost(panel):
    ret = panel["returns"]
    fast_w = to_weights(cs_zscore(panel["alphas"]["idio_fast"]))
    slow_w = to_weights(cs_zscore(panel["alphas"]["idio_slow"]))
    assert turnover(fast_w) > 2.0 * turnover(slow_w)
    fast_drop = sharpe(ls_returns(fast_w, ret)) - sharpe(
        net_of_cost(ls_returns(fast_w, ret), fast_w, 10.0))
    slow_drop = sharpe(ls_returns(slow_w, ret)) - sharpe(
        net_of_cost(ls_returns(slow_w, ret), slow_w, 10.0))
    assert fast_drop > slow_drop


def test_combination_weights_zero_out_a_noise_alpha_and_discount_an_expensive_one(panel):
    fwd = forward_return(panel["returns"], 1)
    sub = {k: panel["alphas"][k] for k in ("idio_fast", "idio_slow", "pure_noise")}
    icw = ic_weights(sub, fwd)
    caw = cost_aware_weights(sub, fwd, panel["returns"], 10.0)
    assert all(v >= 0.0 for v in icw.values()) and all(v >= 0.0 for v in caw.values())
    assert icw["pure_noise"] < 0.2 * icw["idio_slow"]
    # the cost-aware scheme discounts the fast alpha far harder than the slow one
    assert caw["idio_fast"] / icw["idio_fast"] < caw["idio_slow"] / icw["idio_slow"]


def test_combiners_are_weighted_sums_of_the_transformed_alphas(panel):
    sub = {k: panel["alphas"][k] for k in ("idio_fast", "idio_slow")}
    z = combine_zscore(sub, {"idio_fast": 2.0, "idio_slow": 0.0})
    pd.testing.assert_frame_equal(z, 2.0 * cs_zscore(sub["idio_fast"]))
    r = combine_rank(sub, {"idio_fast": 0.0, "idio_slow": 1.0})
    pd.testing.assert_frame_equal(r, cs_rank(sub["idio_slow"]))


def test_smoothing_trades_ic_for_turnover_monotonically(panel):
    sig = combine_zscore({k: panel["alphas"][k] for k in ("idio_fast", "idio_slow")})
    tos = [turnover(to_weights(smooth(sig, hl))) for hl in (1.0, 5.0, 20.0)]
    assert tos == sorted(tos, reverse=True)
    assert turnover(to_weights(sig)) > tos[0]
    with pytest.raises(ValueError, match="halflife"):
        smooth(sig, 0.0)


def test_icir_and_sharpe_conventions():
    assert icir(pd.Series([0.02] * 100 + [0.0] * 100)) == pytest.approx(
        pd.Series([0.02] * 100 + [0.0] * 100).mean()
        / pd.Series([0.02] * 100 + [0.0] * 100).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    assert np.isnan(icir(pd.Series([0.01] * 50)))
    assert np.isnan(sharpe(pd.Series([0.01] * 50)))


# ------------------------------------------------------------------------------- panel and demo
def test_panel_is_deterministic_and_the_planted_structure_is_there(panel):
    again = simulate_panel(n_names=60, n_days=500, n_sectors=4, seed=5)
    pd.testing.assert_frame_equal(panel["returns"], again["returns"])
    assert panel["sector"].nunique() == 4
    assert panel["beta_t"].std(axis=1).mean() > 0.1        # betas are dispersed
    assert panel["beta_t"].diff().abs().mean().mean() > 0  # and they drift
    with pytest.raises(ValueError, match="at least 20 names"):
        simulate_panel(n_names=5)


@pytest.mark.slow
def test_demo_reproduces_the_three_traps_and_prints_the_rule(run_main):
    out = run_main("fin_skills.strategies.alpha_combine")
    assert "TRAP 1" in out and "TRAP 2" in out and "TRAP 3" in out
    assert "AC(1) of IC" in out and "realized book beta" in out
    assert "0.00%" in out                                   # the clip that clips nothing
    assert "Rule: correct the IC t-stat for overlap" in out
    assert out.isascii()
