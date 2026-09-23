"""fin_skills.models.factor_regression - portfolio formation in calendar order, and HAC alphas.

The properties asserted here are the ones the SKILL.md states: the look-ahead guard fires on
lag=0 and clears on lag=1; the leaked sort produces a huge alpha t on a panel whose true alpha
is 0 while the honest sort does not; the numpy Newey-West reproduces statsmodels' three
different HAC conventions; and the Data Library date stamps join on the period, not on a
timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import requires
from fin_skills.models.factor_regression import (align_monthly, fama_french_2x3, fama_macbeth,
                                                 french_period_index, market_return, momentum,
                                                 newey_west, nw_lags, simulate_panel,
                                                 sort_portfolio, statsmodels_hac_check, ts_alpha)


@pytest.fixture(scope="module")
def panel():
    return simulate_panel(n_stocks=120, n_months=180, seed=0)


# --------------------------------------------------------------------------- the look-ahead guard
def test_the_guard_fires_on_lag_zero_and_clears_on_lag_one(panel):
    ret, mcap, bm = panel["ret"], panel["mcap"], panel["bm"]
    with pytest.raises(ValueError, match="look-ahead"):
        sort_portfolio(bm, ret, mcap, lag=0)
    with pytest.raises(ValueError, match="look-ahead"):
        fama_french_2x3(mcap, bm, ret, mcap, lag=0)
    with pytest.raises(ValueError, match="look-ahead"):
        fama_macbeth(ret, {"bm": bm}, lag=0)
    ok = sort_portfolio(bm, ret, mcap, lag=1)
    assert ok.attrs["lag"] == 1 and len(ok) == ret.shape[0] - 1
    # ... and the escape hatch exists, because measuring the leak is the point of the skill
    assert len(sort_portfolio(bm, ret, mcap, lag=0, allow_lookahead=True)) == ret.shape[0]


def test_the_leak_manufactures_an_alpha_on_a_zero_alpha_panel(panel):
    ret, mcap, bm = panel["ret"], panel["mcap"], panel["bm"]
    mkt = market_return(ret, mcap)
    lags = nw_lags(len(mkt))

    def capm(ls):
        both = pd.concat([ls, mkt], axis=1, join="inner").dropna().to_numpy()
        return ts_alpha(both[:, 0], both[:, 1], lags)

    honest = capm(sort_portfolio(bm, ret, mcap, lag=1)["ls"])
    leaked = capm(sort_portfolio(bm, ret, mcap, lag=0, allow_lookahead=True)["ls"])
    assert abs(honest["alpha_t"]) < 2.0                  # true alpha is 0 and the test says so
    assert leaked["alpha_t"] < -4.0                      # the same panel, one subscript wrong
    assert abs(leaked["alpha"]) > 5 * abs(honest["alpha"])
    assert abs(leaked["alpha_t"]) > 4 * abs(honest["alpha_t"])

    # a momentum window that ends in the month being predicted leaks the other way
    mom = momentum(ret, 11)
    mom_honest = capm(sort_portfolio(mom, ret, mcap, lag=1)["ls"])
    mom_leaked = capm(sort_portfolio(mom, ret, mcap, lag=0, allow_lookahead=True)["ls"])
    assert abs(mom_honest["alpha_t"]) < 2.0
    assert mom_leaked["alpha_t"] > 8.0


def test_simulate_panel_is_deterministic_and_alpha_free(panel):
    again = simulate_panel(n_stocks=120, n_months=180, seed=0)
    for key in ("ret", "mcap", "bm", "beta", "market"):
        assert np.array_equal(panel[key], again[key])
    assert not np.array_equal(panel["ret"], simulate_panel(n_stocks=120, n_months=180, seed=1)["ret"])
    # by construction ret = beta * market + noise: the cross-sectional mean tracks the market
    fitted = np.outer(panel["market"], panel["beta"])
    assert np.corrcoef(panel["ret"].ravel(), fitted.ravel())[0, 1] > 0.4


# --------------------------------------------------------------------------- portfolio mechanics
def test_value_weighting_uses_the_caps_known_at_formation():
    ret = np.array([[0.0] * 4, [0.10, 0.00, -0.10, -0.10], [0.0] * 4])
    char = np.array([[4.0, 3.0, 2.0, 1.0], [0.0] * 4, [0.0] * 4])
    mcap = np.array([[9.0, 1.0, 4.0, 4.0], [0.0] * 4, [0.0] * 4])
    out = sort_portfolio(char, ret, mcap, lag=1, quantiles=(0.5, 0.5), weighting="value")
    # period 1: top half is stocks 0 and 1 (char 4, 3), caps 9 and 1 -> 0.9*0.10 + 0.1*0.0
    assert out.loc[1, "long"] == pytest.approx(0.09)
    assert out.loc[1, "short"] == pytest.approx(-0.10)   # bottom half, both at -10 %
    assert out.loc[1, "ls"] == pytest.approx(0.19)
    eq = sort_portfolio(char, ret, mcap, lag=1, quantiles=(0.5, 0.5), weighting="equal")
    assert eq.loc[1, "long"] == pytest.approx(0.05)      # equal weight ignores the 9:1 caps


def test_inputs_are_validated(panel):
    ret, mcap, bm = panel["ret"], panel["mcap"], panel["bm"]
    with pytest.raises(ValueError, match="n_periods, n_stocks"):
        sort_portfolio(bm[:, :5], ret, mcap)
    with pytest.raises(ValueError, match="needs mcap"):
        sort_portfolio(bm, ret, None, weighting="value")
    with pytest.raises(ValueError, match="'value' or 'equal'"):
        sort_portfolio(bm, ret, mcap, weighting="cap-squared")
    with pytest.raises(ValueError, match="rebalance"):
        fama_french_2x3(mcap, bm, ret, mcap, rebalance=0)
    with pytest.raises(ValueError, match="lookback"):
        momentum(ret, 0)


def test_momentum_is_the_cumulative_return_over_the_window_ending_at_t():
    r = np.array([[0.10], [0.20], [-0.05], [0.00]])
    m = momentum(r, 2)
    assert np.isnan(m[0, 0])                                     # warm-up
    assert m[1, 0] == pytest.approx(1.10 * 1.20 - 1.0)
    assert m[2, 0] == pytest.approx(1.20 * 0.95 - 1.0)
    assert m[3, 0] == pytest.approx(0.95 * 1.00 - 1.0)


def test_the_2x3_formulas_are_the_data_library_averages(panel):
    ret, mcap, bm = panel["ret"], panel["mcap"], panel["bm"]
    ff = fama_french_2x3(mcap, bm, ret, mcap, lag=1)
    assert set("SG SN SV BG BN BV SMB HML".split()) <= set(ff.columns)
    assert ff.SMB.to_numpy() == pytest.approx(
        ((ff.SV + ff.SN + ff.SG) / 3.0 - (ff.BV + ff.BN + ff.BG) / 3.0).to_numpy())
    assert ff.HML.to_numpy() == pytest.approx(
        ((ff.SV + ff.BV) / 2.0 - (ff.SG + ff.BG) / 2.0).to_numpy())


def test_annual_rebalance_holds_the_assignment_for_twelve_periods(panel):
    ret, mcap, bm = panel["ret"], panel["mcap"], panel["bm"]
    monthly = fama_french_2x3(mcap, bm, ret, mcap, lag=1, rebalance=1)
    annual = fama_french_2x3(mcap, bm, ret, mcap, lag=1, rebalance=12)
    assert annual.attrs["rebalance"] == 12
    # the first period after each formation date uses the same groups in both
    assert annual.HML.iloc[0] == pytest.approx(monthly.HML.iloc[0])
    assert annual.HML.iloc[12] == pytest.approx(monthly.HML.iloc[12])
    # and in between they diverge, because the annual portfolio is stale on purpose
    assert annual.HML.iloc[5] != pytest.approx(monthly.HML.iloc[5])


# --------------------------------------------------------------------------- Newey-West / HAC
def test_nw_lags_is_the_floor_rule():
    assert nw_lags(100) == 4                       # floor(4 * 1) = 4
    assert nw_lags(240) == int(np.floor(4.0 * (240 / 100.0) ** (2.0 / 9.0)))
    assert nw_lags(1) == 1                         # floor(4 * 0.01**(2/9)) = 1


def test_lag_zero_hac_is_white_hc0_by_closed_form():
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = X @ np.array([0.01, 0.7]) + rng.normal(scale=0.05, size=200)
    res = newey_west(y, X, lags=0)
    xtx_inv = np.linalg.inv(X.T @ X)
    meat = (X * res["resid"][:, None]).T @ (X * res["resid"][:, None])
    hc0 = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
    assert res["se"] == pytest.approx(hc0, rel=1e-12)
    assert newey_west(y, X, lags=0, use_correction=True)["se"] == pytest.approx(
        hc0 * np.sqrt(200 / 198.0), rel=1e-12)


def test_hac_uses_bartlett_weights_one_minus_j_over_l_plus_one():
    rng = np.random.default_rng(4)
    X = np.column_stack([np.ones(300), rng.normal(size=300)])
    y = rng.normal(size=300)
    res = newey_west(y, X, lags=3)
    xu = X * res["resid"][:, None]
    S = xu.T @ xu
    for j in range(1, 4):
        g = xu[j:].T @ xu[:-j]
        S += (1.0 - j / 4.0) * (g + g.T)           # 1 - j/(L+1) with L = 3
    xtx_inv = np.linalg.inv(X.T @ X)
    assert res["se"] == pytest.approx(np.sqrt(np.diag(xtx_inv @ S @ xtx_inv)), rel=1e-12)


@requires("statsmodels")
def test_matches_statsmodels_hac_and_its_two_defaults():
    rng = np.random.default_rng(5)
    n = 400
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.6 * e[t - 1] + rng.normal(scale=0.02)      # serially correlated residuals
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = X @ np.array([0.001, 0.5]) + e
    chk = statsmodels_hac_check(y, X, 4)
    assert chk is not None
    assert chk["max_abs_se_diff"] < 1e-12
    assert chk["max_abs_beta_diff"] < 1e-12
    assert "without small sample correction" in chk["description"]
    # fit(cov_type="HAC") has NO default lag length: maxlags is required
    assert chk["fit_without_maxlags"].startswith("KeyError")
    # cov_hac_simple() corrects by sqrt(T/(T-k)) and fit() does not
    assert chk["direct_over_fit_se_ratio"] == pytest.approx(chk["sqrt_T_over_T_minus_k"], rel=1e-9)
    assert chk["direct_over_fit_se_ratio"] > 1.0


@requires("statsmodels")
def test_bartlett_weight_convention_matches_the_statsmodels_source():
    from statsmodels.stats.sandwich_covariance import weights_bartlett
    for L in (0, 1, 4, 12):
        assert weights_bartlett(L) == pytest.approx(1.0 - np.arange(L + 1) / (L + 1.0))


def test_ts_alpha_recovers_a_planted_alpha_and_beta():
    rng = np.random.default_rng(6)
    f = rng.normal(0.005, 0.04, 600)
    y = 0.002 + 1.3 * f + rng.normal(0, 0.01, 600)
    out = ts_alpha(y, f, lags=4)
    assert out["alpha"] == pytest.approx(0.002, abs=0.001)
    assert out["betas"][0] == pytest.approx(1.3, abs=0.02)
    assert out["alpha_t"] > 3.0 and out["nobs"] == 600


def test_fama_macbeth_slopes_are_zero_when_honest_and_huge_when_leaked(panel):
    ret, bm, mcap = panel["ret"], panel["bm"], panel["mcap"]
    chars = {"log_bm": np.log(bm), "log_cap": np.log(mcap)}
    ok = fama_macbeth(ret, chars, lag=1)
    leak = fama_macbeth(ret, chars, lag=0, allow_lookahead=True)
    assert list(ok.index) == ["const", "log_bm", "log_cap"]
    assert ok.attrs["n_periods"] == ret.shape[0] - 1
    assert abs(ok.loc["log_bm", "t_nw"]) < 2.0
    assert leak.loc["log_bm", "t_nw"] < -5.0
    # the plain Fama-MacBeth t ignores serial correlation in the slope series; the HAC t is
    # smaller exactly where the slopes are persistent
    assert abs(leak.loc["log_bm", "t_nw"]) < abs(leak.loc["log_bm", "t_plain"])


# --------------------------------------------------------------------------- Data Library dates
def test_french_stamps_join_on_the_period_not_on_a_timestamp():
    stamps = [202601, 202602, 202603, 202604, 202605, 202606]
    fac = pd.DataFrame({"Mkt-RF": [1.5, -0.7, 2.1, 0.3, -1.2, 0.8]}, index=stamps)
    strat = pd.Series(np.arange(6) / 100.0,
                      index=pd.date_range("2026-01-31", periods=6, freq="ME"))
    naive = fac.copy()
    naive.index = pd.to_datetime(pd.Index(stamps).astype(str), format="%Y%m")
    assert naive.index[0] == pd.Timestamp("2026-01-01")          # first of month, not month end
    assert len(pd.concat([strat, naive], axis=1, join="inner")) == 0
    fac.index = french_period_index(stamps, "monthly")
    assert isinstance(fac.index, pd.PeriodIndex) and fac.index.freqstr == "M"
    assert len(align_monthly(strat, fac)) == 6


def test_french_daily_stamps_and_a_bad_frequency():
    idx = french_period_index([20260102, 20260105], "daily")
    assert isinstance(idx, pd.DatetimeIndex)
    assert list(idx) == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")]
    with pytest.raises(ValueError, match="'monthly' or 'daily'"):
        french_period_index([202601], "weekly")


# --------------------------------------------------------------------------- the demo
def test_demo_prints_the_rule_and_the_measured_leak(run_main):
    out = run_main("fin_skills.models.factor_regression")
    assert "THE RULE:" in out and "characteristic at t, portfolio at t, return t->t+1" in out
    assert "LEAK lag 0, value-weight" in out and "The truth is 0." in out
    assert "maxlags" in out and "FAMA-MACBETH" in out
    assert out.isascii()
