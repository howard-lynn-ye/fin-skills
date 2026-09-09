"""The analytics bridges: optimizers, QuantLib and reporting.

PyPortfolioOpt, QuantLib, quantstats and empyrical are installed on this checkout, so the
optimizer, option-pricing and tear-sheet round trips here are real rather than stubbed.
The engine bridges (vectorbt, backtesting.py, zipline, qlib) are in
test_bridges_enforcement.py.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from conftest import has_module, requires

from fin_skills.api import Bundle, conventions, get
from fin_skills.bridges import _lazy
from fin_skills.bridges import optimizers as O
from fin_skills.bridges import quantlib as QB
from fin_skills.bridges import reporting as R
from fin_skills.libraries.rf_convention import _sharpe_annual_rf, _sharpe_per_period_rf
from fin_skills.libraries.weight_traps import hrp_weights, make_panel

PERIODS = 252


# ------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def panel():
    """The measured fixture: seed 0, 3 years, 6 names, one $8 high-vol penny stock."""
    return make_panel(0)


@pytest.fixture(scope="module")
def blocks():
    """Three correlated blocks plus a bridge name - where single and Ward disagree."""
    rng = np.random.default_rng(0)
    n, n_assets = 504, 8
    f = rng.normal(0, 0.01, (n, 3))
    idio = rng.normal(0, 1, (n, n_assets)) * np.linspace(0.004, 0.02, n_assets)
    load = np.zeros((3, n_assets))
    for j, b in enumerate([0, 0, 0, 1, 1, 2, 2, 2]):
        load[b, j] = 0.8 + 0.2 * ((j % 3) / 3)
    load[:, -1] = 0.4
    return pd.DataFrame(f @ load + idio, columns=[f"A{i}" for i in range(n_assets)],
                        index=pd.bdate_range("2022-01-03", periods=n))


@pytest.fixture(scope="module")
def bars():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2022-01-03", periods=300)
    close = pd.Series(100 * (1 + rng.normal(0.0004, 0.01, 300)).cumprod(), index=idx)
    return pd.DataFrame({"open": close.shift(1).bfill(), "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 1e6}, index=idx)


@pytest.fixture(scope="module")
def strategy_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.01, PERIODS * 4),
                     index=pd.bdate_range("2021-01-01", periods=PERIODS * 4))


# ================================================================= optimizers
def test_optimizer_rejects_a_price_matrix_with_the_guards_own_message(panel):
    rets, prices = panel
    guard = get("weight_traps").run(asset_returns=prices)
    guard_msg = str(guard.errors[0])
    with pytest.raises(O.PricesWhereReturnsError) as exc:
        O.optimize(O.OptimizerInput(returns=prices, linkage="single"),
                   backend="pypfopt", model="HRP")
    assert guard_msg in str(exc.value), "the bridge must re-raise the guard's own message"
    assert "PRICE matrix" in str(exc.value)


def test_optimizer_reproduces_the_measured_damage_on_the_unguarded_path(panel):
    """If this ever stops holding, the trap stopped being real and the guard is theatre."""
    rets, prices = panel
    w_ret, w_prc = hrp_weights(rets), hrp_weights(prices)
    assert w_prc["PENNY"] == pytest.approx(0.8518, abs=5e-5)
    assert float((w_ret - w_prc).abs().sum()) == pytest.approx(1.5648, abs=5e-5)
    ann = rets.std(ddof=1) * np.sqrt(PERIODS)
    assert ann.idxmax() == "PENNY", "PENNY must still be the highest-vol name"
    cov = rets.cov()

    def vol(w):
        v = w.reindex(cov.columns).to_numpy()
        return float(np.sqrt(v @ cov.to_numpy() @ v) * np.sqrt(PERIODS))

    assert vol(w_prc) / vol(w_ret) - 1.0 == pytest.approx(1.414, abs=5e-4)


def test_optimizer_rejects_percent_scaled_returns(panel):
    rets, _ = panel
    with pytest.raises(O.PricesWhereReturnsError, match="Divide by 100"):
        O.prove_returns(rets * 100.0)


def test_optimizer_accepts_a_real_returns_matrix(panel):
    rets, _ = panel
    O.prove_returns(rets)                    # does not raise


def test_optimizer_linkage_is_required_for_hierarchical_models(panel):
    rets, _ = panel
    with pytest.raises(O.LinkageRequiredError) as exc:
        O.optimize(O.OptimizerInput(returns=rets), backend="pypfopt", model="HRP")
    assert "'single'" in str(exc.value) and "Ward" in str(exc.value)
    assert O.require_linkage("min_vol", None) == ""


@requires("pypfopt")
def test_optimizer_linkage_actually_changes_the_weights(blocks):
    single = O.optimize(O.OptimizerInput(returns=blocks, linkage="single"),
                        backend="pypfopt", model="HRP")
    ward = O.optimize(O.OptimizerInput(returns=blocks, linkage="ward"),
                      backend="pypfopt", model="HRP")
    assert float((single - ward).abs().sum()) > 0.05, \
        "the two library defaults must disagree, or the requirement is theatre"
    assert single.attrs["linkage"] == "single" and ward.attrs["linkage"] == "ward"


def test_optimizer_never_hands_back_a_mutable_riskfolio_handle(panel):
    """rp.Portfolio is imperative: assets_stats() before optimization(), re-called after
    the data changes. The bridge does both in one step and lets no handle escape."""
    src = inspect.getsource(O._riskfolio)
    assert src.index("assets_stats") < src.index("port.optimization"), \
        "assets_stats must be called before optimization"
    assert "return port" not in src, "no mutable Portfolio may escape this function"
    for name, obj in vars(O).items():
        if callable(obj) and not name.startswith("_"):
            ann = getattr(obj, "__annotations__", {}).get("return", "")
            assert "Portfolio" not in str(ann)


@requires("pypfopt")
def test_optimizer_result_is_independent_of_later_mutation(panel):
    rets, _ = panel
    work = rets.copy()
    inp = O.OptimizerInput(returns=work, linkage="single")
    first = O.optimize(inp, backend="pypfopt", model="HRP")
    work *= 3.0                                    # the data changes underneath
    second = O.optimize(O.OptimizerInput(returns=rets, linkage="single"),
                        backend="pypfopt", model="HRP")
    pd.testing.assert_series_equal(first, second, check_names=False)


def test_optimizer_rf_is_taken_once_and_converted_per_backend():
    inp = O.OptimizerInput(returns=pd.DataFrame({"a": [0.01, -0.01, 0.02],
                                                 "b": [-0.01, 0.01, 0.0]}),
                           rf_annual=0.05, periods_per_year=252)
    assert inp.rf_per_period_simple == pytest.approx(0.05 / 252)
    assert inp.rf_per_period_geometric == pytest.approx(1.05 ** (1 / 252) - 1)
    assert inp.rf_per_period_simple != inp.rf_per_period_geometric
    with pytest.raises(ValueError, match="percentage"):
        O.OptimizerInput(returns=inp.returns, rf_annual=5.0)


@requires("pypfopt")
def test_optimizer_records_the_rf_conversion_it_used(panel):
    rets, _ = panel
    w = O.optimize(O.OptimizerInput(returns=rets, rf_annual=0.05), backend="pypfopt",
                   model="max_sharpe")
    assert w.attrs["rf_annual"] == 0.05
    assert w.attrs["rf_used"] == 0.05                   # PyPortfolioOpt's rf is ANNUAL
    assert "ANNUAL" in w.attrs["rf_convention"]
    assert w.attrs["backend"] == "pypfopt" and w.attrs["library_version"]


@requires("pypfopt")
def test_optimizer_rf_units_agree_with_the_conventions_module(panel):
    """One rf_annual, two conventions, and the Sharpe each one implies is reproducible."""
    rets, _ = panel
    w = O.optimize(O.OptimizerInput(returns=rets, linkage="single"), backend="pypfopt",
                   model="HRP")
    port = (rets * w).sum(axis=1)
    geometric = 1.05 ** (1 / 252) - 1
    assert _sharpe_annual_rf(port, 0.05, 252) == pytest.approx(
        conventions.annualize_sharpe(port - geometric, 252), abs=1e-12)
    assert _sharpe_per_period_rf(port, 0.05 / 252, 252) == pytest.approx(
        conventions.annualize_sharpe(port - 0.05 / 252, 252), abs=1e-12)
    assert _sharpe_annual_rf(port, 0.05, 252) != _sharpe_per_period_rf(port, 0.05, 252)


def test_optimizer_solver_class_is_named_not_left_to_the_backend():
    kw = {"rm": "EVaR"}
    try:
        got = O.require_solver_class("min_vol", "EVaR", kw)
    except O.SolverClassError as exc:
        assert "exponential- or power-cone" in str(exc)
        assert "pip install clarabel" in str(exc)
    else:
        assert got in O.CONE_SOLVERS
    assert O.require_solver_class("min_vol", "MV", {}) == ""


def test_optimizer_rejects_an_unknown_backend(panel):
    rets, _ = panel
    with pytest.raises(ValueError, match="backend must be one of"):
        O.optimize(O.OptimizerInput(returns=rets), backend="pyfolio", model="min_vol")


@pytest.mark.parametrize("backend,pip", [("skfolio", "skfolio"),
                                         ("riskfolio", "Riskfolio-Lib")])
def test_optimizer_names_the_pip_install_when_a_backend_is_absent(panel, backend, pip):
    if has_module({"skfolio": "skfolio", "riskfolio": "riskfolio"}[backend]):
        pytest.skip(f"{pip} is installed here")
    rets, _ = panel
    with pytest.raises(_lazy.MissingLibrary, match=f"pip install {pip}"):
        O.optimize(O.OptimizerInput(returns=rets, linkage="single"), backend=backend,
                   model="HRP")


@requires("pypfopt")
def test_weights_to_bundle_unlocks_the_guard_that_checks_the_input(panel):
    rets, _ = panel
    w = O.optimize(O.OptimizerInput(returns=rets, linkage="single", rf_annual=0.02),
                   backend="pypfopt", model="HRP")
    b = O.weights_to_bundle(w, rets)
    assert b.has("asset_returns", "returns", "turnover", "rf", "periods_per_year")
    report = b.check(guards=["weight_traps", "cost_curve", "rf_convention"])
    assert set(report.ran) == {"weight_traps", "cost_curve", "rf_convention"}
    assert float(b.turnover.iloc[0]) == pytest.approx(float(w.abs().sum()))
    assert float(b.turnover.iloc[1:].sum()) == 0.0


# =================================================================== QuantLib
def test_quantlib_refuses_an_evaluation_date_past_expiry():
    with pytest.raises(QB.EvaluationDateError) as exc:
        with QB.evaluation_date("2026-12-31", curve_ref="2026-12-31",
                                expiry="2026-06-30"):
            pytest.fail("the body must never run")
    assert "returns exactly 0.0" in str(exc.value)


def test_quantlib_refuses_a_curve_reference_that_is_not_the_evaluation_date():
    with pytest.raises(QB.EvaluationDateError, match="curve reference"):
        with QB.evaluation_date("2026-06-30", curve_ref="2026-01-02",
                                expiry="2026-12-18"):
            pytest.fail("the body must never run")


def test_quantlib_warns_when_the_check_cannot_run():
    if has_module("QuantLib"):
        import QuantLib as ql
        before = ql.Settings.instance().evaluationDate
    with pytest.warns(UserWarning, match="npv_zero could not run"):
        try:
            with QB.evaluation_date("2026-06-30"):
                pass
        except _lazy.MissingLibrary:
            pytest.skip("QuantLib is not installed here")
    if has_module("QuantLib"):
        assert ql.Settings.instance().evaluationDate == before


@requires("QuantLib")
def test_quantlib_restores_the_global_on_both_paths():
    import QuantLib as ql

    before = ql.Settings.instance().evaluationDate
    with QB.evaluation_date("2026-06-30", curve_ref="2026-06-30", expiry="2026-12-18"):
        assert ql.Settings.instance().evaluationDate == ql.Date(30, 6, 2026)
    assert ql.Settings.instance().evaluationDate == before

    with pytest.raises(RuntimeError):
        with QB.evaluation_date("2026-06-30", curve_ref="2026-06-30",
                                expiry="2026-12-18"):
            raise RuntimeError("boom")
    assert ql.Settings.instance().evaluationDate == before, \
        "the global must be restored on the exception path too"


def _ql_option(ql, expiry_year: int = 2027):
    cal = ql.TARGET()
    spot = ql.QuoteHandle(ql.SimpleQuote(100.0))
    day = ql.Actual365Fixed()
    r = ql.YieldTermStructureHandle(ql.FlatForward(0, cal, 0.05, day))
    q = ql.YieldTermStructureHandle(ql.FlatForward(0, cal, 0.0, day))
    v = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, cal, 0.20, day))
    proc = ql.BlackScholesMertonProcess(spot, q, r, v)
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
                           ql.EuropeanExercise(ql.Date(30, 6, expiry_year)))
    return opt, ql.AnalyticEuropeanEngine(proc)


@requires("QuantLib")
def test_quantlib_price_round_trip_matches_the_repos_reference():
    import QuantLib as ql

    from _helpers import QUANTLIB_CALL

    ql.Settings.instance().evaluationDate = ql.Date(30, 6, 2026)
    opt, engine = _ql_option(ql)
    res = QB.price(opt, engine, eval_date="2026-06-30", curve_ref="2026-06-30",
                   expiry="2027-06-30", flag="c")
    assert res.npv == pytest.approx(QUANTLIB_CALL["price"], abs=1e-6)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert res.greeks[name] == pytest.approx(QUANTLIB_CALL[name], rel=1e-5)
    b = QB.to_bundle(res)
    report = b.check(guards=["greeks_convention"])
    assert report.ran == ["greeks_convention"] and report.passed


@requires("QuantLib")
def test_quantlib_refuses_a_bare_zero_npv():
    import QuantLib as ql

    ql.Settings.instance().evaluationDate = ql.Date(30, 6, 2026)
    opt, engine = _ql_option(ql)

    class _Zero:
        def setPricingEngine(self, _e):
            pass

        def NPV(self):
            return 0.0

    with pytest.raises(QB.ZeroNpvError, match="allow_zero=True"):
        QB.price(_Zero(), None, eval_date="2026-06-30", curve_ref="2026-06-30",
                 expiry="2027-06-30")
    res = QB.price(_Zero(), None, eval_date="2026-06-30", curve_ref="2026-06-30",
                   expiry="2027-06-30", allow_zero=True)
    assert res.npv == 0.0
    _ = opt, engine


def test_quantlib_names_its_pip_install_when_absent(monkeypatch):
    monkeypatch.setattr(_lazy, "find", lambda _n: False)
    with pytest.raises(_lazy.MissingLibrary, match="pip install QuantLib"):
        with QB.evaluation_date("2026-06-30", curve_ref="2026-06-30",
                                expiry="2026-12-18"):
            pass


# ================================================================== reporting
def _oracle_factor(n: int = 200, names=("A", "B", "C", "D")):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2022-01-03", periods=n)
    px = pd.DataFrame(100 * (1 + rng.normal(0, 0.01, (n, len(names)))).cumprod(axis=0),
                      index=idx, columns=list(names))
    fwd = px.pct_change().shift(-1)
    factor = fwd.stack(future_stack=True)
    factor.index.names = ["date", "asset"]
    return px, fwd, factor


def _ic(wide: pd.DataFrame, fwd: pd.DataFrame) -> float:
    common = wide.index.intersection(fwd.index)
    return float(np.nanmean(wide.loc[common].corrwith(fwd.loc[common], axis=1)))


def test_alphalens_bridge_lags_the_factor():
    """The oracle factor IS the next bar's return: IC 1.0 raw, ~0 through the bridge."""
    _, fwd, factor = _oracle_factor()
    assert _ic(factor.unstack(), fwd) == pytest.approx(1.0, abs=1e-9)
    lagged = R.lag_factor(factor, already_lagged=False)
    assert abs(_ic(lagged.unstack(), fwd)) < 0.1


def test_alphalens_bridge_refuses_an_untestable_lag_claim():
    _, _, factor = _oracle_factor()
    with pytest.raises(R.FactorNotLaggedError, match="nothing was supplied to test it"):
        R.lag_factor(factor, already_lagged=True)


def test_alphalens_bridge_tests_the_lag_claim_with_assert_causal(bars):
    _, _, factor = _oracle_factor()
    with pytest.raises(R.FactorNotLaggedError, match="already_lagged=True is FALSE"):
        R.lag_factor(factor, already_lagged=True, bars=bars,
                     signal_fn=lambda d: d["close"].pct_change())
    kept = R.lag_factor(factor, already_lagged=True, bars=bars,
                        signal_fn=lambda d: d["close"].pct_change().shift(1))
    assert kept is factor


def test_alphalens_names_its_pip_install_when_absent():
    if has_module("alphalens"):
        pytest.skip("alphalens-reloaded is installed here")
    px, _, factor = _oracle_factor()
    with pytest.raises(_lazy.MissingLibrary, match="pip install alphalens-reloaded"):
        R.to_alphalens(factor, px)


def test_tearsheet_refuses_a_percentage_rf(strategy_returns):
    with pytest.raises(ValueError, match="percentage"):
        R.tearsheet(Bundle(returns=strategy_returns), rf_annual=5.0)


def test_quantstats_path_converts_rf_and_never_hands_it_to_cagr(strategy_returns):
    t = R.tearsheet(Bundle(returns=strategy_returns), backend="quantstats",
                    rf_annual=0.05)
    assert t.rf_handed_to_backend == pytest.approx(1.05 ** (1 / 252) - 1)
    assert "GEOMETRICALLY" in t.rf_conversion
    assert t.stats["sharpe"] == pytest.approx(
        _sharpe_annual_rf(strategy_returns, 0.05, 252), abs=1e-12)
    # the excess CAGR is this bridge's, computed from the excess series
    excess = strategy_returns - t.rf_handed_to_backend
    expected = float((1 + excess).prod()) ** (252 / len(excess)) - 1
    assert t.stats["cagr_excess"] == pytest.approx(expected, abs=1e-12)


@requires("quantstats")
def test_quantstats_cagr_really_does_discard_rf(strategy_returns):
    """If this ever stops holding, quantstats fixed the bug and the workaround can go."""
    import quantstats as qs

    with_rf = float(qs.stats.cagr(strategy_returns, rf=0.05))
    without = float(qs.stats.cagr(strategy_returns, rf=0.0))
    assert with_rf == pytest.approx(without, abs=1e-15), "cagr(rf=) stopped being ignored"
    t = R.tearsheet(Bundle(returns=strategy_returns), backend="quantstats", rf_annual=0.05)
    assert t.stats["cagr_excess"] != pytest.approx(with_rf, abs=1e-6)


@requires("quantstats")
def test_quantstats_render_reproduces_the_skills_measured_sharpe(strategy_returns):
    """-0.285948 is the number lib-quantstats reports for qs.stats.sharpe(r, rf=0.05)."""
    t = R.tearsheet(Bundle(returns=strategy_returns), backend="quantstats",
                    rf_annual=0.05, render=True)
    assert t.stats["sharpe"] == pytest.approx(-0.285948, abs=5e-6)
    assert t.rendered["sharpe"] == pytest.approx(t.stats["sharpe"], abs=1e-9)


def test_pyfolio_path_converts_annual_rf_to_per_period(strategy_returns):
    t = R.tearsheet(Bundle(returns=strategy_returns), backend="pyfolio", rf_annual=0.05)
    assert t.rf_handed_to_backend == pytest.approx(0.05 / 252), \
        "empyrical's risk_free= is PER-PERIOD; handing it the annual rate is the bug"
    assert t.rf_handed_to_backend != 0.05
    assert "PER-PERIOD" in t.rf_conversion
    assert t.stats["sharpe"] == pytest.approx(
        _sharpe_per_period_rf(strategy_returns, 0.05 / 252, 252), abs=1e-12)


@requires("empyrical")
def test_pyfolio_conversion_matches_empyrical_exactly(strategy_returns):
    """The cross-check the bridge does not do itself: empyrical is not a bridged library."""
    import empyrical as ep

    t = R.tearsheet(Bundle(returns=strategy_returns), backend="pyfolio", rf_annual=0.05)
    assert float(ep.sharpe_ratio(strategy_returns,
                                 risk_free=t.rf_handed_to_backend)) == pytest.approx(
        t.stats["sharpe"], abs=1e-9)
    wrong = float(ep.sharpe_ratio(strategy_returns, risk_free=0.05))
    assert wrong < -60, "the annual-rate-as-per-period signature stopped being < -60"


def test_tearsheet_runs_the_rf_guard_on_the_same_number_it_reports(strategy_returns):
    t = R.tearsheet(Bundle(returns=strategy_returns), backend="quantstats",
                    rf_annual=0.05)
    assert t.guard.guard == "rf_convention" and t.guard.passed
    assert t.guard.evidence["rf"] == 0.05
    assert t.guard.evidence["sharpe_annual_rf_geometric"] == pytest.approx(
        t.stats["sharpe"], abs=1e-12)


def test_tearsheet_rejects_an_unknown_backend(strategy_returns):
    with pytest.raises(ValueError, match="backend must be one of"):
        R.tearsheet(Bundle(returns=strategy_returns), backend="ffn")


def test_pyfolio_render_names_its_pip_install_when_absent(strategy_returns):
    if has_module("pyfolio"):
        pytest.skip("pyfolio-reloaded is installed here")
    with pytest.raises(_lazy.MissingLibrary, match="pip install pyfolio-reloaded"):
        R.tearsheet(Bundle(returns=strategy_returns), backend="pyfolio", render=True)
