"""The engine bridges refuse before they import: vectorbt, backtesting.py, zipline, qlib.

Every test here proves a refusal that happens BEFORE the vendor library is imported, so it
runs on a bare numpy/pandas machine - which is the point of the AGPL/Commons-Clause
firewall. The two `@requires("vectorbt")` round trips skip where it is absent.
The optimizer, QuantLib and reporting bridges are in test_bridges_analytics.py.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import has_module, requires

from fin_skills.api import Bundle, conventions
from fin_skills.bridges import _lazy
from fin_skills.bridges import backtesting_py as BP
from fin_skills.bridges import qlib as QL
from fin_skills.bridges import vectorbt as V
from fin_skills.bridges import zipline as Z


# ------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def bars():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2022-01-03", periods=300)
    close = pd.Series(100 * (1 + rng.normal(0.0004, 0.01, 300)).cumprod(), index=idx)
    return pd.DataFrame({"open": close.shift(1).bfill(), "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 1e6}, index=idx)


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
    fills = pd.DataFrame(guarded.order_records.records)
    close = bars["close"].to_numpy()
    assert not any(np.isclose(float(row.price), close[int(row.idx)], atol=1e-12)
                   for row in fills.itertuples(index=False)), \
        "a guarded fill must never be priced at the close of its own bar"


@requires("vectorbt")
def test_vectorbt_roundtrip_reproduces_the_returns(bars):
    """from_vectorbt(to_vectorbt(b)) reproduces `returns` on a fixture with no costs."""
    b = Bundle(bars=bars, close=bars["close"])
    entries = (bars[["close"]] > bars[["close"]].rolling(20).mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = V.to_vectorbt(b, entries=entries, exits=~entries, price="open",
                           fees=0.0, slippage=0.0)
        back = V.from_vectorbt(pf, sessions=252)
    direct = pf.returns()
    if isinstance(direct, pd.DataFrame):
        direct = direct.iloc[:, 0]
    pd.testing.assert_series_equal(back.returns, direct, check_names=False, atol=1e-10)


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
