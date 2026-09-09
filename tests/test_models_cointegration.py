"""fin_skills.models.cointegration - the wrong table, the trial count, and the traded window.

The properties the SKILL.md claims:

  * the hand-written Engle-Granger statistic IS statsmodels' `coint` statistic;
  * the single-series ADF table over-rejects on a FITTED residual, and the fitted-residual null
    distribution is shifted;
  * among independent random walks some pairs pass, they trade beautifully in the window they
    were screened on, and out of sample the frozen z-score never comes back inside the band;
  * a random walk's residual reports a finite half-life every time;
  * a hedge ratio and z-score fitted on the traded window beat a causal rolling version;
  * `coint`'s own defaults (autolag, trend="n", the collinearity short circuit) change or void
    the answer.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from conftest import requires
from fin_skills.models import cointegration as co

SIMS = 300          # the demo uses 2000; 300 is enough for the qualitative assertions


@pytest.fixture(scope="module")
def pair():
    return co.cointegrated_pair(co.T_PAIR, co.SEED)


@pytest.fixture(scope="module")
def walks():
    return co.random_walks(co.T_SCREEN + co.T_OOS, co.N_SERIES, co.SEED)


def test_the_generators_are_seeded_and_build_what_they_say(pair):
    y, x, s = pair
    y2, x2, s2 = co.cointegrated_pair(co.T_PAIR, co.SEED)
    assert np.array_equal(y, y2) and np.array_equal(x, x2) and np.array_equal(s, s2)
    assert np.allclose(y, 10.0 + co.BETA_TRUE * x + s)
    assert not np.array_equal(y, co.cointegrated_pair(co.T_PAIR, co.SEED + 1)[0])
    # the spread is stationary, the legs are not
    assert abs(np.corrcoef(s[:-1], s[1:])[0, 1] - co.PHI_TRUE) < 0.06
    w = co.random_walks(200, 3, co.SEED)
    assert w.shape == (200, 3) and np.array_equal(w, co.random_walks(200, 3, co.SEED))
    assert np.all(np.abs(w[0] - 100.0) < 5.0)          # starts at 100 plus one increment
    assert abs(np.diff(w, axis=0).std(ddof=1) - 1.0) < 0.1


def test_ols_hedge_residual_is_orthogonal_to_the_regressor(pair):
    y, x, _ = pair
    a, b, u = co.ols_hedge(y, x)
    assert abs(u.mean()) < 1e-10
    assert abs(float(np.dot(u, x - x.mean()))) < 1e-6
    assert abs(b - co.BETA_TRUE) < 0.2
    assert np.allclose(y, a + b * x + u)


def test_adf_tstat_rejects_a_bad_trend_and_needs_data():
    u = co.cointegrated_pair(300, co.SEED)[2]
    with pytest.raises(ValueError):
        co.adf_tstat(u, trend="ct")
    with pytest.raises(ValueError):
        co.adf_tstat(u[:8])
    assert co.adf_tstat(u, trend="n") != co.adf_tstat(u, trend="c")


@requires("statsmodels")
def test_adf_tstat_is_statsmodels_adfuller_in_both_regressions(pair):
    from statsmodels.tsa.stattools import adfuller
    _, _, u = co.ols_hedge(*pair[:2])
    for trend in ("n", "c"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = adfuller(u, maxlag=co.EG_LAGS, autolag=None, regression=trend)[0]
        assert co.adf_tstat(u, co.EG_LAGS, trend) == pytest.approx(float(ref), abs=1e-9)


@requires("statsmodels")
def test_engle_granger_is_statsmodels_coint(pair):
    y, x, _ = pair
    t_own, beta = co.engle_granger_tstat(y[:co.T_SCREEN], x[:co.T_SCREEN])
    sm = co.statsmodels_coint(y[:co.T_SCREEN], x[:co.T_SCREEN])
    assert sm is not None
    assert abs(t_own - sm["t"]) < 1e-9
    assert sm["p"] < 0.05                       # the genuine pair is found
    assert abs(beta - co.BETA_TRUE) < 0.2
    assert sm["crit"].shape == (3,) and sm["crit"][0] < sm["crit"][1] < sm["crit"][2]


def test_the_simulated_null_is_a_null_and_the_fitted_residual_shifts_it():
    mc = co.mc_critical_values(200, co.EG_LAGS, SIMS, co.SEED)
    cv1, cv5, cv10 = mc["critical_values"]
    assert cv1 < cv5 < cv10 < 0
    assert np.mean(mc["stats"] < cv5) == pytest.approx(0.05, abs=0.02)
    # the ADF-with-constant statistic on a FITTED residual is shifted well below -2.87
    assert mc["critical_values_c"][1] < -3.1
    assert mc["stats_with_constant"].shape == mc["stats"].shape


@requires("statsmodels")
def test_the_single_series_adf_table_over_rejects_on_fitted_residuals():
    from statsmodels.tsa.adfvalues import mackinnoncrit
    mc = co.mc_critical_values(200, co.EG_LAGS, SIMS, co.SEED)
    adf_cv5 = float(mackinnoncrit(N=1, regression="c", nobs=199)[1])
    assert adf_cv5 == pytest.approx(-2.87, abs=0.02)
    size = float(np.mean(mc["stats_with_constant"] < adf_cv5))
    assert size > 0.10                          # 16.5 % at T = 500 in the demo


def test_screening_random_walks_produces_false_positives(walks):
    mc = co.mc_critical_values(co.T_SCREEN, co.EG_LAGS, SIMS, co.SEED)
    recs = co.screen_pairs(walks[:co.T_SCREEN], mc["critical_values"][1])
    assert len(recs) == co.N_SERIES * (co.N_SERIES - 1) // 2 == 190
    n_pass = sum(r["pass_mc"] for r in recs)
    assert 1 <= n_pass <= 25                    # 6 with the demo's 2000-sim critical value
    if "pass_adf_wrong" in recs[0]:
        assert sum(r["pass_adf_wrong"] for r in recs) > n_pass
        assert max(abs(r["t"] - r["t_sm"]) for r in recs) < 1e-9


def test_every_random_walk_residual_reports_a_finite_half_life(walks):
    hls = []
    for i in range(co.N_SERIES):
        for j in range(i + 1, co.N_SERIES):
            _, _, u = co.ols_hedge(walks[:co.T_SCREEN, i], walks[:co.T_SCREEN, j])
            hls.append(co.half_life(u)["half_life"])
    hls = np.array(hls)
    assert len(hls) == 190
    assert np.isfinite(hls).all()               # b < 0 in every single fitted residual
    assert 10 < np.median(hls) < 200


def test_half_life_recovers_a_known_ar1(pair):
    _, _, s = pair
    hl = co.half_life(s)
    assert hl["phi"] == pytest.approx(co.PHI_TRUE, abs=0.03)
    assert hl["half_life"] == pytest.approx(-math.log(2) / math.log(co.PHI_TRUE), rel=0.25)
    assert hl["t_stat"] < -3.0
    ramp = co.half_life(np.cumsum(np.ones(50)))          # a deterministic trend: b == 0
    assert ramp["half_life"] == float("inf") and math.isnan(ramp["t_stat"])


def test_zscore_positions_enter_and_exit_where_documented_and_do_not_look_ahead():
    z = np.array([0.0, 2.5, 1.0, 0.4, -2.1, -1.0, -0.4, np.nan, 3.0])
    pos = co.zscore_positions(z, entry=2.0, exit_=0.5)
    assert list(pos) == [0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0]
    shocked = z.copy()
    shocked[5] = 9.9
    assert np.array_equal(co.zscore_positions(shocked)[:5], pos[:5])   # the past is unchanged
    assert co.n_trades(pos) == int((np.diff(np.r_[0.0, pos]) != 0).sum())


def test_spread_pnl_earns_the_next_days_move(pair):
    y, x, _ = pair
    y, x = y[:100], x[:100]
    beta = np.full(100, 0.8)
    pos = np.r_[np.ones(50), -np.ones(50)]
    pnl = co.spread_pnl(y, x, beta, pos)
    assert pnl[0] == 0.0
    assert pnl[1] == pytest.approx(pos[0] * ((y[1] - y[0]) - beta[0] * (x[1] - x[0])))
    shocked = y.copy()
    shocked[-1] += 100.0
    assert np.array_equal(co.spread_pnl(shocked, x, beta, pos)[:-1], pnl[:-1])
    assert np.isnan(co.sharpe(np.zeros(10)))


def test_the_rolling_strategy_uses_nothing_after_t(pair):
    y, x, _ = pair
    r = co.rolling_strategy(y[:600], x[:600], hedge_window=100, z_window=30)
    assert r["start"] == 99
    assert np.isnan(r["beta"][:99]).all() and np.isfinite(r["beta"][99:]).all()
    shocked = y[:600].copy()
    shocked[400] += 50.0
    r2 = co.rolling_strategy(shocked, x[:600], hedge_window=100, z_window=30)
    assert np.array_equal(r2["beta"][:400], r["beta"][:400], equal_nan=True)
    assert np.array_equal(r2["z"][:400], r["z"][:400], equal_nan=True)
    assert r2["beta"][400] != r["beta"][400]         # day t IS in the window that ends at t
    assert np.allclose(r["pnl"][:400], r2["pnl"][:400])


def test_a_hedge_ratio_fitted_on_the_traded_window_beats_a_causal_one(pair):
    y, x, _ = pair
    B = slice(co.IN_SAMPLE, co.T_PAIR)
    ins = co.in_sample_strategy(y[B], x[B])
    roll = co.rolling_strategy(y, x)
    sh_roll = co.sharpe(roll["pnl"][co.IN_SAMPLE:])
    assert ins["sharpe"] - sh_roll > 0.5              # +1.29 in the demo
    assert sh_roll > 0                                # the pair is real, both make money
    # freezing the spread's mean and sd as well is the version that breaks
    a = co.in_sample_strategy(y[:co.IN_SAMPLE], x[:co.IN_SAMPLE])
    frozen = co.frozen_strategy(y[B], x[B], a["beta"], a["mu"], a["sd"])
    assert frozen["sharpe"] < sh_roll
    assert 0.0 <= frozen["in_market"] <= 1.0


def test_the_frozen_zscore_on_a_false_pair_enters_once_and_never_exits(walks):
    mc = co.mc_critical_values(co.T_SCREEN, co.EG_LAGS, SIMS, co.SEED)
    recs = co.screen_pairs(walks[:co.T_SCREEN], mc["critical_values"][1])
    passing = [r for r in recs if r["pass_mc"]]
    assert passing, "the screen found no false positives on this seed"
    ins, stuck = [], []
    for r in passing:
        a = co.in_sample_strategy(walks[:co.T_SCREEN, r["i"]], walks[:co.T_SCREEN, r["j"]])
        b = co.frozen_strategy(walks[co.T_SCREEN:, r["i"]], walks[co.T_SCREEN:, r["j"]],
                               a["beta"], a["mu"], a["sd"])
        ins.append(a["sharpe"])
        stuck.append(b["in_market"] == 1.0 and b["n_trades"] == 1 and b["z_abs_min"] > 2.0)
    assert np.nanmean(ins) > 1.0                      # 1.76 in the demo, on pure noise
    assert all(stuck)


@requires("statsmodels")
def test_johansen_returns_the_90_95_99_table_and_a_usable_hedge_ratio(pair):
    y, x, _ = pair
    jo = co.johansen(y[:co.T_SCREEN], x[:co.T_SCREEN])
    assert jo is not None
    assert jo["trace_cv"].shape == (2, 3) and jo["max_eig_cv"].shape == (2, 3)
    # columns are 90 %, 95 %, 99 % - increasing, the OPPOSITE orientation to coint's 1/5/10 %
    assert (np.diff(jo["trace_cv"], axis=1) > 0).all()
    assert jo["trace_cv"][0, 1] == pytest.approx(15.4943, abs=1e-3)
    assert jo["trace"][0] > jo["trace"][1] and jo["rank_at_95"] in (0, 1, 2)
    ols_beta = co.ols_hedge(y[:co.T_SCREEN], x[:co.T_SCREEN])[1]
    assert abs(jo["hedge_ratio"] - ols_beta) < 0.1


@requires("statsmodels")
def test_coint_defaults_change_or_void_the_answer(pair):
    y, x, _ = pair
    p = co.coint_defaults_probe(y[:co.T_SCREEN], x[:co.T_SCREEN])
    assert p is not None
    assert p["t_autolag"] != p["t_fixed"]             # autolag="aic" is a lag search
    assert p["trend_n_crit_all_nan"] and math.isfinite(p["p_trend_n"])
    assert math.isinf(p["twin_t"]) and p["twin_t"] < 0
    assert p["twin_p"] == 0.0
    assert p["twin_warning"] == "CollinearityWarning"


def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.models.cointegration")
    for head in ("=== 1. Engle-Granger critical values", "=== 2. 20 independent random walks",
                 "=== 3. The", "=== 4. A genuinely cointegrated pair",
                 "=== 5. Johansen and the OU half-life"):
        assert head in out
    assert out.isascii()
    rule = [ln for ln in out.splitlines() if ln.startswith("Rule: ")]
    assert len(rule) == 1 and "trials" in rule[0] and "half-life" in rule[0]
    assert out.strip().splitlines()[-1].startswith("total runtime")
