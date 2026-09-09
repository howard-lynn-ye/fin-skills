"""fin_skills.strategies.trend_models - TSMOM as MOP (2012) define it, and its two look-aheads."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.strategies.trend_models import (COM_DAYS, LOOKBACK, PERIODS_PER_YEAR,
                                                VOL_TARGET, backtest, close_from_returns,
                                                donchian_signal, eq1_variance, ewma_center_of_mass,
                                                ex_ante_vol, ma_crossover_signal, max_drawdown,
                                                SHOCKS, month_end_mask, monthly_tsmom,
                                                position_is_causal,
                                                position_of, rolling_vol, sharpe, simulate_panel,
                                                summary, tsmom_backtest, tsmom_signal)


@pytest.fixture(scope="module")
def panel():
    return simulate_panel(n_assets=4, n_years=6, seed=3)


# ------------------------------------------------------- MOP eq. (1), against its own longhand
def test_pandas_ewm_is_literally_mop_equation_1():
    """The one-line ewm(com=60).var(bias=True) equals eq. (1) written out with explicit weights."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.011, 900))
    ours = float(ex_ante_vol(r.to_frame("x"), lag=1)["x"].iloc[-1] ** 2)
    paper = eq1_variance(r.to_numpy())
    assert ours == pytest.approx(paper, rel=1e-12)


def test_centre_of_mass_is_the_papers_60_days():
    """MOP eq. (1): d chosen so sum_i (1-d) d^i i = d/(1-d) = 60. pandas' com IS that quantity."""
    delta = COM_DAYS / (1.0 + COM_DAYS)
    assert ewma_center_of_mass(delta) == pytest.approx(COM_DAYS, abs=1e-3)
    assert delta / (1.0 - delta) == pytest.approx(COM_DAYS)
    assert ewma_center_of_mass(10.0 / 11.0) == pytest.approx(10.0, abs=1e-3)


def test_unbiased_var_is_not_equation_1():
    """bias=True is not cosmetic - the default unbiased form is a different number."""
    rng = np.random.default_rng(1)
    r = pd.DataFrame({"x": rng.normal(0, 0.01, 400)})
    paper = eq1_variance(r["x"].to_numpy())
    biased = float(ex_ante_vol(r, lag=1)["x"].iloc[-1] ** 2)
    unbiased = float((np.sqrt(r.ewm(com=COM_DAYS).var() * PERIODS_PER_YEAR)
                      .shift(1)["x"].iloc[-1]) ** 2)
    assert biased == pytest.approx(paper, rel=1e-12)
    assert unbiased != pytest.approx(paper, rel=1e-6)


def test_lagged_sigma_cannot_see_its_own_return(panel):
    """The documented property of lag=1: sigma at row t does not move when r_t moves."""
    ret = panel["returns"]
    t = 900
    bumped = ret.copy()
    bumped.iloc[t] = bumped.iloc[t] + 0.5
    for lag, should_move in ((1, False), (0, True)):
        a = ex_ante_vol(ret, lag=lag).iloc[t]
        b = ex_ante_vol(bumped, lag=lag).iloc[t]
        moved = bool((a - b).abs().max() > 1e-12)
        assert moved is should_move


def test_eq1_and_vol_estimators_validate_their_inputs(panel):
    with pytest.raises(ValueError, match="lag must be"):
        ex_ante_vol(panel["returns"], lag=-1)
    with pytest.raises(ValueError, match="lag must be"):
        rolling_vol(panel["returns"], lag=-1)
    with pytest.raises(ValueError, match="at least two"):
        eq1_variance([0.01])


# ----------------------------------------------------------------------------- the signals
def test_tsmom_signal_skip_shifts_the_window_and_is_never_the_default():
    """MOP eq. (5) skips nothing; skip=k is footnote 10's cross-sectional convention."""
    r = pd.DataFrame({"a": np.r_[np.full(30, 0.01), np.full(30, -0.05)]})
    base = tsmom_signal(r, lookback=20, skip=0)
    skipped = tsmom_signal(r, lookback=20, skip=5)
    assert skipped.iloc[30:].equals(base.shift(5).iloc[30:])
    # the skip delays the turn: base flips short 5 bars before the skipped version does
    flip_base = int(base["a"].eq(-1.0).idxmax())
    flip_skip = int(skipped["a"].eq(-1.0).idxmax())
    assert flip_skip == flip_base + 5
    with pytest.raises(ValueError):
        tsmom_signal(r, lookback=0)
    with pytest.raises(ValueError):
        tsmom_signal(r, skip=-1)


def test_donchian_channel_is_lagged_so_the_breakout_bar_is_not_in_its_own_channel():
    """A monotonically rising close must go long; with an unlagged channel it never could."""
    close = pd.DataFrame({"a": np.arange(1, 121, dtype=float)})
    sig = donchian_signal(close, entry=20, exit=10)
    assert set(np.unique(sig.to_numpy())) <= {-1.0, 0.0, 1.0}
    assert sig["a"].iloc[-1] == 1.0
    assert sig["a"].iloc[:20].eq(0.0).all()               # warm-up emits flat, not a guess
    falling = donchian_signal(pd.DataFrame({"a": np.arange(120, 0, -1.0)}), 20, 10)
    assert falling["a"].iloc[-1] == -1.0
    with pytest.raises(ValueError):
        donchian_signal(close, entry=1)


def test_ma_crossover_is_the_sign_of_the_gap():
    close = pd.DataFrame({"a": np.arange(1, 301, dtype=float)})
    sig = ma_crossover_signal(close, fast=5, slow=20)
    assert sig["a"].iloc[-1] == 1.0
    assert sig["a"].iloc[:19].isna().all()
    with pytest.raises(ValueError, match="1 <= fast < slow"):
        ma_crossover_signal(close, fast=20, slow=20)


# ---------------------------------------------------------------- the traps: fire and clear
def test_trap1_the_scaling_probe_finds_the_sizing_leak_and_clears_the_fix(panel):
    """vol_lag=0 puts r_t in the sigma that sizes it; vol_lag=1 is MOP. The probe separates them."""
    ret = panel["returns"]
    k = len(ret) // 2
    leaky = position_is_causal(lambda f: tsmom_backtest(f, vol_lag=0)[1], ret, k)
    clean = position_is_causal(lambda f: tsmom_backtest(f, vol_lag=1)[1], ret, k)
    assert leaky is False and clean is True


def test_trap1_the_leak_pays_and_pays_more_at_a_shorter_window(panel):
    """Documented: the size of the leak is set by the estimator's window, not the strategy."""
    ret = panel["returns"]

    def gain(vol_fn):
        honest = sharpe(tsmom_backtest(ret, vol_fn=vol_fn, vol_lag=1)[0].mean(axis=1).dropna())
        leaky = sharpe(tsmom_backtest(ret, vol_fn=vol_fn, vol_lag=0)[0].mean(axis=1).dropna())
        return leaky - honest

    long_win = gain(lambda r, lag: ex_ante_vol(r, com=60.0, lag=lag))
    short_win = gain(lambda r, lag: rolling_vol(r, 5, lag=lag))
    assert long_win > 0.0 and short_win > long_win


def test_trap2_the_scaling_probe_misses_a_sign_signal_and_the_shock_finds_it(panel):
    """Section 5's headline: assert_causal's x2 perturbation cannot see through np.sign()."""
    ret = panel["returns"]
    k = len(ret) // 2
    for kind in ("tsmom", "ma", "donchian"):
        leaky = (lambda f, kd=kind: position_of(f, kd, signal_lag=0))
        clean = (lambda f, kd=kind: position_of(f, kd, signal_lag=1))
        # the two-sided shock finds the leak on every model, and clears the correct one
        assert position_is_causal(leaky, ret, k, shock=SHOCKS) is False
        assert position_is_causal(clean, ret, k, shock=SHOCKS) is True
        assert position_is_causal(clean, ret, k) is True
    # x2 cannot see through np.sign() on a 12-month sum or a 50/200 gap. (On a price-level
    # breakout it sometimes can, by luck - which is the point: it is not a reliable probe.)
    for kind in ("tsmom", "ma"):
        leaky = (lambda f, kd=kind: position_of(f, kd, signal_lag=0))
        assert position_is_causal(leaky, ret, k) is True


def test_trap2_the_missing_shift_pays_most_on_the_breakout(panel):
    """Donchian fires on the bar that moved, so crediting that bar's return is worth the most."""
    ret = panel["returns"]
    gains = {}
    for kind in ("tsmom", "ma", "donchian"):
        no_shift = sharpe((position_of(ret, kind, 0) * ret).mean(axis=1).dropna())
        shifted = sharpe((position_of(ret, kind, 1) * ret).mean(axis=1).dropna())
        gains[kind] = no_shift - shifted
    assert all(g > 0 for g in gains.values())
    assert gains["donchian"] == max(gains.values())


def test_position_is_causal_rejects_a_bad_k_and_an_impossible_shock(panel):
    ret = panel["returns"]
    with pytest.raises(ValueError, match="inside the frame"):
        position_is_causal(lambda f: f, ret, 0)
    with pytest.raises(ValueError, match="stay positive"):
        position_is_causal(lambda f: f, ret, 10, shock=(-1.0, 9.0))


# --------------------------------------------------------------- monthly TSMOM and plumbing
def test_month_end_mask_marks_the_last_row_of_each_calendar_month():
    idx = pd.bdate_range("2020-01-01", periods=70)
    me = month_end_mask(idx)
    assert me[-1] and me.sum() == idx.to_period("M").nunique()
    assert list(idx[me].month) == sorted(set(idx.month))


def test_monthly_tsmom_holds_one_month_and_is_causal(panel):
    """The position set at month end t earns month t -> t+1 and nothing earlier."""
    ret = panel["returns"]
    strat, pos = monthly_tsmom(ret)
    assert len(strat) == month_end_mask(ret.index).sum()
    assert strat.iloc[0].isna().all()                     # nothing is earned in the first month
    assert set(np.unique(np.sign(pos.dropna().to_numpy()))) <= {-1.0, 0.0, 1.0}
    skipped, _ = monthly_tsmom(ret, skip_m=1)
    assert not skipped.dropna(how="all").equals(strat.dropna(how="all"))
    with pytest.raises(ValueError):
        monthly_tsmom(ret, lookback_m=0)


def test_vol_target_sets_the_scale_not_the_ranking(panel):
    """MOP: 'The choice of 40% is inconsequential'. Doubling it doubles vol, not Sharpe."""
    ret = panel["returns"]
    a = tsmom_backtest(ret)[0].mean(axis=1).dropna()
    b = tsmom_backtest(ret, target=2 * VOL_TARGET)[0].mean(axis=1).dropna()
    assert sharpe(b) == pytest.approx(sharpe(a), rel=1e-9)
    assert b.std() == pytest.approx(2 * a.std(), rel=1e-9)


def test_backtest_validates_lags_and_can_run_unsized(panel):
    ret = panel["returns"]
    with pytest.raises(ValueError, match="lags must be"):
        backtest(tsmom_signal(ret), ret, signal_lag=-1)
    unsized, pos = tsmom_backtest(ret, size=False)
    assert set(np.unique(pos.dropna().to_numpy())) <= {-1.0, 0.0, 1.0}


def test_summary_and_drawdown_are_the_documented_conventions():
    r = pd.Series([0.1, -0.5, 0.2])
    assert max_drawdown(r) == pytest.approx(-0.5)
    assert max_drawdown(pd.Series([0.01, 0.02])) == 0.0
    s = summary(pd.Series([0.001] * 261), PERIODS_PER_YEAR)
    assert s["ann_return"] == pytest.approx(0.261) and s["ann_vol"] == pytest.approx(0.0)
    assert sharpe(pd.Series([0.01, 0.02, 0.03])) > 0


# --------------------------------------------------------------------------- panel and demo
def test_panel_is_deterministic_and_internally_consistent(panel):
    again = simulate_panel(n_assets=4, n_years=6, seed=3)
    pd.testing.assert_frame_equal(panel["returns"], again["returns"])
    # close = open * exp(intraday), and close-to-close = overnight + intraday in logs
    assert np.allclose(panel["close"], panel["open"] * np.exp(panel["intraday"]))
    recon = np.expm1(panel["overnight"] + panel["intraday"])
    pd.testing.assert_frame_equal(panel["returns"], recon)
    assert close_from_returns(panel["returns"]).iloc[-1].gt(0).all()
    with pytest.raises(ValueError, match="at least one asset"):
        simulate_panel(n_assets=0)
    with pytest.raises(ValueError, match="overnight_share"):
        simulate_panel(overnight_share=1.0)


def test_demo_reproduces_the_papers_parameters_and_prints_the_rule(run_main):
    out = run_main("fin_skills.strategies.trend_models")
    assert "0.983607" in out and "60.0000 days" in out          # eq. (1) delta and centre of mass
    assert "variance scaled by 261" in out
    assert "nothing skipped (eq. 5)" in out and "fn. 10" in out
    assert "TRAP 1" in out and "TRAP 2" in out and "TRAP 3" in out
    assert "Rule: the position that earns r_t must be known at t-1" in out
    assert out.isascii()
