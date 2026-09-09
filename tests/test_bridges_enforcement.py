"""Each bridge's enforcement fires, and the round trip runs for real where it can.

Every test here either (a) proves a refusal that happens BEFORE the vendor library is
imported - so it runs on a bare machine - or (b) is marked `@requires(...)` and does the
real round trip. On this checkout PyPortfolioOpt, QuantLib and quantstats are installed,
so the optimizer, QuantLib and reporting round trips are real.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import has_module, requires

from fin_skills.api import Bundle, conventions, get
from fin_skills.bridges import _lazy
from fin_skills.bridges import backtesting_py as BP
from fin_skills.bridges import execution as X
from fin_skills.bridges import optimizers as O
from fin_skills.bridges import qlib as QL
from fin_skills.bridges import quantlib as QB
from fin_skills.bridges import reporting as R
from fin_skills.bridges import vectorbt as V
from fin_skills.bridges import zipline as Z
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


# ================================================================== vectorbt
def test_vectorbt_refuses_close_with_unlagged_signals(bars):
    b = Bundle(bars=bars, close=bars["close"])
    entries = (bars[["close"]] > bars[["close"]].rolling(20).mean())
    with pytest.raises(V.SameBarFillError) as exc:
        V.to_vectorbt(b, entries=entries, exits=~entries, price="close",
                      already_lagged=False)
    msg = str(exc.value)
    assert "same-bar" in msg or "signal bar" in msg
    assert "not knowable until the bar is over" in msg


def test_vectorbt_already_lagged_is_verified_not_believed(bars):
    """An unlagged signal claimed as lagged must be caught by assert_causal, not trusted."""
    b = Bundle(bars=bars, close=bars["close"])
    entries = (bars[["close"]] > bars[["close"]].rolling(20).mean())
    with pytest.raises(_lazy.NotLaggedError) as exc:
        V.to_vectorbt(b, entries=entries, exits=~entries, price="open",
                      already_lagged=True,
                      signal_fn=lambda d: d["close"].rolling(20).mean())
    assert "already_lagged=True is FALSE" in str(exc.value)
    assert "LOOK-AHEAD" in str(exc.value)


def test_vectorbt_already_lagged_accepts_a_genuinely_lagged_signal(bars):
    """The same claim, honestly made, gets past the proof and fails only on the import."""
    b = Bundle(bars=bars, close=bars["close"])
    entries = (bars[["close"]] > bars[["close"]].rolling(20).mean()).shift(1, fill_value=False)
    with pytest.raises(_lazy.MissingLibrary if not has_module("vectorbt")
                       else AssertionError):
        V.to_vectorbt(b, entries=entries, exits=~entries, price="open",
                      already_lagged=True,
                      signal_fn=lambda d: d["close"].rolling(20).mean().shift(1))


def test_vectorbt_refuses_an_untestable_lag_claim(bars):
    b = Bundle(close=bars["close"])           # no `bars`, so nothing to test the claim on
    entries = pd.DataFrame({"x": [True] * len(bars)}, index=bars.index)
    with pytest.raises(_lazy.NotLaggedError) as exc:
        V.to_vectorbt(b, entries=entries, exits=~entries, price="close",
                      already_lagged=True)
    assert "nothing was supplied to test it with" in str(exc.value)


def test_vectorbt_refuses_a_bare_numeric_price(bars):
    b = Bundle(bars=bars)
    entries = pd.DataFrame({"x": [True] * len(bars)}, index=bars.index)
    with pytest.raises(V.SameBarFillError, match="bare number"):
        V.to_vectorbt(b, entries=entries, exits=~entries, price=np.inf)


def test_vectorbt_names_its_pip_install_when_absent(bars):
    if has_module("vectorbt"):
        pytest.skip("vectorbt is installed here")
    b = Bundle(bars=bars)
    entries = pd.DataFrame({"x": [True] * len(bars)}, index=bars.index)
    with pytest.raises(_lazy.MissingLibrary, match="pip install vectorbt"):
        V.to_vectorbt(b, entries=entries, exits=~entries, price="open")


class _FakePortfolio:
    """The detectable signature of price=np.inf: every fill priced at its own bar's close."""

    def __init__(self, close: pd.Series, same_bar: bool):
        self.close = close
        px = close.to_numpy() if same_bar else close.shift(1).bfill().to_numpy()
        self.order_records = pd.DataFrame({"idx": [3, 7, 11], "col": [0, 0, 0],
                                           "price": [px[3], px[7], px[11]]})

    def returns(self):
        return self.close.pct_change().fillna(0.0)

    def asset_flow(self):
        return pd.Series(0.0, index=self.close.index)

    def value(self):
        return pd.Series(1.0, index=self.close.index)


def test_from_vectorbt_reports_the_same_bar_signature(bars):
    with pytest.raises(V.SameBarFillError, match="price=np.inf"):
        V.from_vectorbt(_FakePortfolio(bars["close"], same_bar=True))


def test_from_vectorbt_accepts_a_next_bar_portfolio(bars):
    b = V.from_vectorbt(_FakePortfolio(bars["close"], same_bar=False), sessions=252)
    assert "returns" in b and b.periods_per_year == 252


@requires("vectorbt")
def test_vectorbt_shifts_by_default_and_the_two_sharpes_differ(bars):
    import vectorbt as vbt

    b = Bundle(bars=bars, close=bars["close"])
    entries = (bars[["close"]] > bars[["close"]].rolling(20).mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        guarded = V.to_vectorbt(b, entries=entries, exits=~entries, price="open")
        raw = vbt.Portfolio.from_signals(bars["close"], entries.iloc[:, 0],
                                         ~entries.iloc[:, 0])
    g = V.from_vectorbt(guarded, sessions=252).returns
    r = raw.returns()
    assert conventions.annualize_sharpe(g, 252) != conventions.annualize_sharpe(r, 252)


# ============================================================== backtesting.py
def _centred(a):
    return pd.Series(a).rolling(11, center=True).mean().to_numpy()


def _causal_sma(a):
    return pd.Series(a).rolling(11).mean().to_numpy()


def test_backtesting_rejects_a_noncausal_indicator(bars):
    b = Bundle(bars=bars.rename(columns=str.title), indicator={"centred": _centred})
    with pytest.raises(_lazy.NonCausalError) as exc:
        BP.to_backtesting(b, object)
    assert "LOOK-AHEAD" in str(exc.value)
    assert "centred" in str(exc.value)


def test_backtesting_accepts_a_causal_indicator_and_then_needs_the_library(bars):
    b = Bundle(bars=bars.rename(columns=str.title), indicator={"sma": _causal_sma})
    if has_module("backtesting"):
        pytest.skip("backtesting.py is installed here")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _lazy.LicenceWarning)
        with pytest.raises(_lazy.MissingLibrary, match="pip install backtesting"):
            BP.to_backtesting(b, object)


def test_backtesting_trade_on_close_requires_the_acknowledgement(bars):
    b = Bundle(bars=bars.rename(columns=str.title))
    with pytest.raises(BP.TradeOnCloseError) as exc:
        BP.to_backtesting(b, object, trade_on_close=True)
    assert "Close[-2]" in str(exc.value)
    # with the acknowledgement it gets past the refusal and stops at the import
    if not has_module("backtesting"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _lazy.LicenceWarning)
            with pytest.raises(_lazy.MissingLibrary):
                BP.to_backtesting(b, object, trade_on_close=True,
                                  acknowledge_close_minus_2=True)


def test_backtesting_needs_bars_to_test_anything():
    with pytest.raises(TypeError, match="single-asset"):
        BP.to_backtesting(Bundle(returns=pd.Series([0.1, 0.2])), object)


def test_from_backtesting_rebuilds_a_bundle():
    idx = pd.bdate_range("2022-01-03", periods=50)
    curve = pd.DataFrame({"Equity": np.linspace(100.0, 120.0, 50)}, index=idx)
    trades = pd.DataFrame({"EntryBar": [1], "ExitBar": [10], "Size": [10.0],
                           "EntryPrice": [100.0], "ExitPrice": [110.0]})
    stats = pd.Series({"Return [%]": 20.0})
    stats._equity_curve, stats._trades = curve, trades
    b = BP.from_backtesting(stats, sessions=252)
    assert len(b.returns) == 49
    assert float(b.turnover.sum()) > 0


# ==================================================================== zipline
def test_zipline_refuses_a_survivor_only_panel():
    idx = pd.bdate_range("2015-01-02", periods=2600)
    survivors = pd.DataFrame(100.0, index=idx, columns=[f"T{i}" for i in range(20)])
    with pytest.raises(Z.SurvivorOnlyUniverseError) as exc:
        Z.to_zipline_assets(Bundle(prices=survivors))
    assert "survivorship_audit" in str(exc.value) or "SURVIVOR" in str(exc.value)
    assert "today's ticker list" in str(exc.value)


def test_zipline_accepts_a_panel_with_delistings():
    idx = pd.bdate_range("2015-01-02", periods=2600)
    cols = [f"T{i}" for i in range(20)]
    prices = pd.DataFrame(100.0, index=idx, columns=cols)
    listings = []
    for i, c in enumerate(cols):
        end = idx[-1] if i >= 6 else idx[300 + 200 * i]
        prices.loc[prices.index > end, c] = np.nan
        listings.append({"ticker": c, "start_date": idx[0], "end_date": end})
    assets = Z.to_zipline_assets(Bundle(prices=prices,
                                        listings=pd.DataFrame(listings)))
    assert list(assets.columns) == ["symbol", "start_date", "end_date", "first_traded",
                                    "auto_close_date", "exchange"]
    assert assets.index.name == "sid"
    assert (assets["auto_close_date"] > assets["end_date"]).all()


def test_zipline_commission_min_trade_cost_has_no_default():
    with pytest.raises(TypeError, match="min_trade_cost"):
        Z.to_zipline_commission(cost_per_share=0.001)
    if not has_module("zipline"):
        with pytest.raises(_lazy.MissingLibrary, match="pip install zipline-reloaded"):
            Z.to_zipline_commission(min_trade_cost=1.0)


def test_from_zipline_adds_costs_back_so_returns_are_gross():
    idx = pd.bdate_range("2022-01-03", periods=20)
    perf = pd.DataFrame({"returns": np.full(20, 0.001),
                         "portfolio_value": np.full(20, 1e6),
                         "commission": np.full(20, 100.0),
                         "slippage": np.full(20, 50.0),
                         "transactions": [[{"amount": 10, "price": 100.0}]] * 20},
                        index=idx)
    with pytest.warns(UserWarning, match="GROSS"):
        b = Z.from_zipline(perf, sessions=252)
    assert np.allclose(b.returns.to_numpy(), 0.001 + 150.0 / 1e6)
    assert np.allclose(b.turnover.to_numpy(), 1000.0 / 1e6)


# ======================================================================= qlib
def test_qlib_overlapping_fit_window_is_rejected():
    with pytest.raises(QL.NormalizerLeakError) as exc:
        QL.to_qlib_handler(Bundle(), fit_start="2020-01-01", fit_end="2023-01-01",
                           infer_start="2022-06-01", infer_end="2023-06-01")
    assert "ZScoreNorm" in str(exc.value)
    assert "test-set" in str(exc.value)


def test_qlib_disjoint_window_passes_the_check_and_needs_the_library():
    if has_module("qlib"):
        pytest.skip("pyqlib is installed here")
    with pytest.raises(_lazy.MissingLibrary, match="pip install pyqlib"):
        QL.to_qlib_handler(Bundle(), fit_start="2018-01-01", fit_end="2020-12-31",
                           infer_start="2021-01-01", infer_end="2021-12-31")


def test_qlib_processor_fit_window_is_checked_too():
    proc = [{"class": "ZScoreNorm", "kwargs": {"fit_start_time": "2018-01-01",
                                               "fit_end_time": "2021-06-30"}}]
    with pytest.raises(QL.NormalizerLeakError, match="reaches the inference window"):
        QL.check_processors(proc, infer_start=pd.Timestamp("2021-01-01"),
                            infer_end=pd.Timestamp("2021-12-31"), where="infer_processors")


def test_qlib_processor_without_a_declared_window_is_refused():
    with pytest.raises(QL.NormalizerLeakError, match="declares no fit_start_time"):
        QL.check_processors([{"class": "ZScoreNorm"}],
                            infer_start=pd.Timestamp("2021-01-01"),
                            infer_end=pd.Timestamp("2021-12-31"), where="infer_processors")


def test_qlib_requires_both_ends_of_the_fit_window():
    with pytest.raises(TypeError, match="fit_start is required"):
        QL.to_qlib_handler(Bundle(), fit_start=None, fit_end="2020-12-31",
                           infer_start="2021-01-01", infer_end="2021-12-31")


def test_from_qlib_rebuilds_a_bundle():
    idx = pd.bdate_range("2021-01-04", periods=60)
    report = pd.DataFrame({"return": np.full(60, 0.001), "bench": np.full(60, 0.0005),
                           "turnover": np.full(60, 0.05), "cost": np.full(60, 0.0001)},
                          index=idx)
    b = QL.from_qlib(report, sessions=252)
    assert b.has("returns", "benchmark_returns", "turnover", "periods_per_year")


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


def test_execution_module_is_reachable_from_the_package():
    assert X.read_execution.__doc__ and "paper" in X.read_execution.__doc__
