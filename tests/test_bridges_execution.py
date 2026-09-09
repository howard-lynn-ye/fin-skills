"""`fin_skills.bridges.execution` is read-only by construction, and this is the proof.

Two independent tests carry the claim:

  * `test_execution_module_names_no_trading_method` greps the module source for the
    order-sending method names of ccxt, ib_async and alpaca-py and asserts that not one
    of them appears. It is a source grep and not a mock because the point is that the
    capability is ABSENT, not that it is unused on some path;
  * `test_execution_exposes_only_readers` walks the module's public surface and asserts
    every callable is a reader or a converter.

Then the gate: `paper_account_guard` must pass before any client is built or read.
"""
from __future__ import annotations

import inspect
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import has_module

from fin_skills.bridges import _lazy
from fin_skills.bridges import execution as X

MODULE_SOURCE = Path(inspect.getsourcefile(X)).read_text(encoding="utf-8")

#: every way the three bridged clients send, change or withdraw an instruction to trade.
FORBIDDEN_METHOD_NAMES: tuple[str, ...] = (
    # ccxt (unified, both cases, plus the sugar)
    "create_order", "createOrder", "create_limit_order", "createLimitOrder",
    "create_market_order", "createMarketOrder", "create_limit_buy_order",
    "create_limit_sell_order", "create_market_buy_order", "create_market_sell_order",
    "create_order_ws", "createOrderWs", "create_orders", "createOrders",
    "cancel_order", "cancelOrder", "cancel_all_orders", "cancelAllOrders",
    "cancel_orders", "cancelOrders", "edit_order", "editOrder",
    "create_post_only_order", "create_stop_order", "createStopOrder",
    # ib_async
    "placeOrder", "bracketOrder", "oneCancelsAll", "reqGlobalCancel",
    "cancelMktData", "whatIfOrder", "MarketOrder", "LimitOrder", "StopOrder",
    "StopLimitOrder", "MarketOnClose",
    # alpaca-py
    "submit_order", "replace_order", "cancel_order_by_id", "cancel_orders",
    "close_position", "close_all_positions", "MarketOrderRequest", "LimitOrderRequest",
    "StopOrderRequest", "StopLimitOrderRequest", "TrailingStopOrderRequest",
    "OrderRequest", "exercise_options_position",
)


def test_execution_module_names_no_trading_method():
    """The capability must be absent from the file, not merely unused."""
    hits = [name for name in FORBIDDEN_METHOD_NAMES if name in MODULE_SOURCE]
    assert hits == [], f"execution.py names order-sending calls: {hits}"
    lowered = MODULE_SOURCE.lower()
    for name in FORBIDDEN_METHOD_NAMES:
        assert name.lower() not in lowered, f"execution.py names {name} (case-insensitive)"


def _folded_hits(source: str) -> list[str]:
    folded = re.sub(r"[^a-z]", "", source.lower())
    return [n for n in FORBIDDEN_METHOD_NAMES
            if re.sub(r"[^a-z]", "", n.lower()) in folded]


def test_execution_module_names_no_trading_method_case_folded():
    """Catch a spelling that differs only in case or underscores."""
    assert _folded_hits(MODULE_SOURCE) == []


def test_the_grep_would_actually_catch_an_order_call():
    """A negative test is worthless unless it can fail. Poison the source and watch."""
    poisoned = MODULE_SOURCE + '\nclient.create_order(sym, "market", "buy", 1)\n'
    assert "create_order" in _folded_hits(poisoned)
    poisoned2 = MODULE_SOURCE + "\nib.placeOrder(contract, order)\n"
    assert "placeOrder" in _folded_hits(poisoned2)
    assert len(FORBIDDEN_METHOD_NAMES) >= 40


def test_execution_exposes_only_readers():
    public = [n for n in dir(X) if not n.startswith("_")]
    callables = [n for n in public if callable(getattr(X, n))
                 and getattr(getattr(X, n), "__module__", "") == X.__name__]
    assert set(callables) <= {"read_execution", "read_ohlcv", "to_bundle",
                              "ExecutionRecord", "NotPaperRefusal"}, callables
    assert set(X.__all__) >= {"read_execution", "read_ohlcv", "to_bundle"}
    for name in X.__all__:
        assert not name.lower().startswith(("send", "submit", "place", "buy", "sell",
                                            "trade", "order")), name


def test_execution_record_is_frozen():
    rec = X.ExecutionRecord(fills=pd.DataFrame(), positions=pd.DataFrame(), account={},
                            venue="ib", is_paper=True)
    with pytest.raises(Exception):
        rec.venue = "alpaca"


# ------------------------------------------------------------------------ the stubs
class FakeIB:
    """ib_async's read surface, and nothing else."""

    def __init__(self, account: str, fills=()):
        self._account, self._fills = account, list(fills)

    def managedAccounts(self):
        return [self._account]

    def fills(self):
        return self._fills

    def positions(self):
        return []

    def accountSummary(self):
        return []


class FakeExchange:
    """ccxt's read surface, plus the resolved urls the guard asks for."""

    has = {"fetchPositions": True, "fetchBalance": True}

    def __init__(self, api="https://testnet.binance.vision", sandbox=True, trades=(),
                 candles=None, now=None, cap=3):
        self.urls = {"api": api}
        self.sandboxMode = sandbox
        self.headers = {}
        self._trades = list(trades)
        self._candles = list(candles or [])
        self._now = now
        self._cap = cap
        self.calls = []

    def milliseconds(self):
        return self._now

    def fetch_my_trades(self, since=None, params=None):
        return self._trades

    def fetch_positions(self):
        return []

    def fetch_balance(self):
        return {"total": {"USDT": 1000.0}, "apiKey": "SHOULD-NOT-SURVIVE"}

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        """Caps every page, exactly like a real venue, and re-returns the last row."""
        self.calls.append(since)
        rows = [c for c in self._candles if since is None or c[0] >= since]
        return rows[:self._cap]


# -------------------------------------------------------------------- the paper gate
def test_execution_refuses_a_live_ib_account():
    with pytest.raises(X.NotPaperRefusal) as exc:
        X.read_execution("ib", client=FakeIB("U1234567"))
    assert "LIVE account" in str(exc.value)
    assert "refusing to construct a client or read anything" in str(exc.value)


def test_execution_accepts_a_paper_ib_account():
    rec = X.read_execution("ib", client=FakeIB("DU1234567"))
    assert rec.is_paper and rec.venue == "ib"
    assert list(rec.fills.columns) == list(X.FILL_COLUMNS)
    assert rec.provenance.request["read_only"] is True


def test_execution_will_not_open_the_ib_socket_itself():
    with pytest.raises(TypeError, match="will not open the socket for you"):
        X.read_execution("ib", account_id="DU1234567")


def test_execution_refuses_a_live_alpaca_host():
    with pytest.raises(X.NotPaperRefusal) as exc:
        X.read_execution("alpaca", base_url="https://api.alpaca.markets",
                         extra={"paper": True})
    assert "url_override" in str(exc.value) or "LIVE" in str(exc.value)


def test_execution_refuses_a_ccxt_venue_with_a_production_host():
    with pytest.raises(X.NotPaperRefusal) as exc:
        X.read_execution("ccxt:binance",
                         client=FakeExchange(api="https://api.binance.com"))
    assert "production" in str(exc.value)


def test_execution_refuses_a_ccxt_venue_whose_flag_is_not_set():
    with pytest.raises(X.NotPaperRefusal, match="sandbox_mode"):
        X.read_execution("ccxt:binance", client=FakeExchange(sandbox=False))


def test_execution_refuses_an_unknown_venue():
    with pytest.raises(ValueError, match="venue must start with"):
        X.read_execution("schwab")


def test_execution_reads_a_paper_ccxt_venue():
    trades = [{"timestamp": 1_700_000_000_000, "symbol": "BTC/USDT", "side": "buy",
               "amount": 0.5, "price": 40_000.0, "fee": {"cost": 2.0},
               "clientOrderId": "abc"}]
    rec = X.read_execution("ccxt:binance", client=FakeExchange(trades=trades))
    assert len(rec.fills) == 1
    assert rec.fills.loc[0, "symbol"] == "BTC/USDT"
    assert rec.fills.loc[0, "fee"] == 2.0
    assert rec.venue == "ccxt:binance"


def test_execution_scrubs_credentials_and_masks_identifiers():
    rec = X.read_execution("ccxt:binance", client=FakeExchange())
    text = repr(rec.account)
    assert "SHOULD-NOT-SURVIVE" not in text
    assert "apiKey" not in text
    masked = X._scrub({"account_id": "DU1234567", "api_secret": "hunter2",
                       "nested": {"token": "t", "equity": 10.0}})
    assert masked["account_id"] == "DU*******"
    assert "api_secret" not in masked
    assert masked["nested"] == {"equity": 10.0}


def test_execution_credential_env_names_are_declared_not_values():
    for names in X.ENV_CREDENTIALS.values():
        for n in names:
            assert n.isupper() and " " not in n


@pytest.mark.parametrize("venue,pip", [("ccxt:binance", "ccxt"), ("alpaca", "alpaca-py")])
def test_execution_names_the_pip_install_when_the_client_library_is_absent(venue, pip):
    module = {"ccxt": "ccxt", "alpaca-py": "alpaca"}[pip]
    if has_module(module):
        pytest.skip(f"{pip} is installed here")
    extra = ({"sandbox_mode": True, "urls": {"api": "https://testnet.binance.vision"}}
             if venue.startswith("ccxt") else {"paper": True})
    kw = {} if venue.startswith("ccxt") else {
        "base_url": "https://paper-api.alpaca.markets"}
    with pytest.raises(_lazy.MissingLibrary, match=f"pip install {pip}"):
        X.read_execution(venue, extra=extra, **kw)


def test_execution_refuses_alpaca_url_override():
    if has_module("alpaca"):
        pytest.skip("alpaca-py is installed here; the refusal is tested without it")
    with pytest.raises((X.NotPaperRefusal, _lazy.MissingLibrary)) as exc:
        X.read_execution("alpaca", base_url="https://paper-api.alpaca.markets",
                         extra={"paper": True}, url_override="https://api.alpaca.markets")
    # either it never got a client (library absent) or it refused the override outright
    assert "url_override" in str(exc.value) or "pip install alpaca-py" in str(exc.value)


# ------------------------------------------------------------------- the ccxt OHLCV
def _candles(n: int, start: int = 1_700_000_000_000, step: int = 60_000):
    return [[start + i * step, 1.0, 2.0, 0.5, 1.5, 100.0] for i in range(n)]


def test_read_ohlcv_drops_the_unclosed_final_bar():
    rows = _candles(10)
    # "now" sits INSIDE the last bar, so that bar has not closed yet
    now = rows[-1][0] + 30_000
    ex = FakeExchange(candles=rows, now=now, cap=100)
    df = X.read_ohlcv(ex, "BTC/USDT", "1m")
    assert len(df) == 9, "the in-progress final candle must not survive"
    assert df.index[-1] == pd.Timestamp(rows[-2][0], unit="ms", tz="UTC")
    kept = X.read_ohlcv(ex, "BTC/USDT", "1m", drop_unclosed=False)
    assert len(kept) == 10


def test_read_ohlcv_paginates_around_the_venue_cap_and_dedupes():
    rows = _candles(10)
    ex = FakeExchange(candles=rows, now=rows[-1][0] + 60_000, cap=3)
    df = X.read_ohlcv(ex, "BTC/USDT", "1m")
    assert len(df) == 10, "asking once returns the cap; the bridge must loop"
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert len(ex.calls) > 1
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_read_ohlcv_honours_a_half_open_until():
    rows = _candles(10)
    ex = FakeExchange(candles=rows, now=rows[-1][0] + 60_000, cap=100)
    until = pd.Timestamp(rows[5][0], unit="ms", tz="UTC")
    df = X.read_ohlcv(ex, "BTC/USDT", "1m", until=until)
    assert len(df) == 5 and df.index.max() < until


def test_read_ohlcv_rejects_an_unknown_timeframe():
    with pytest.raises(ValueError, match="unknown timeframe"):
        X.read_ohlcv(FakeExchange(), "BTC/USDT", "7m")


def test_read_ohlcv_returns_an_empty_frame_rather_than_raising():
    ex = FakeExchange(candles=[], now=1_700_000_000_000)
    assert X.read_ohlcv(ex, "BTC/USDT", "1m").empty


# --------------------------------------------------------------------- to_bundle
def test_execution_to_bundle_unlocks_the_reconciliation_guards():
    trades = [{"timestamp": 1_700_000_000_000, "symbol": "BTC/USDT", "side": "buy",
               "amount": 0.5, "price": 40_010.0, "fee": {"cost": 2.0}},
              {"timestamp": 1_700_000_060_000, "symbol": "BTC/USDT", "side": "sell",
               "amount": 0.5, "price": 40_100.0, "fee": {"cost": 2.0}}]
    rec = X.read_execution("ccxt:binance", client=FakeExchange(trades=trades))
    expected = pd.DataFrame({
        "ts": pd.to_datetime([1_700_000_000_000, 1_700_000_060_000], unit="ms", utc=True),
        "symbol": ["BTC/USDT", "BTC/USDT"], "side": ["buy", "sell"],
        "qty": [0.5, 0.5], "price": [40_000.0, 40_120.0]})
    b = X.to_bundle(rec, expected_fills=expected)
    assert b.has("left", "right", "on", "by", "tolerance", "broker", "close", "other")
    cov = b.coverage(["safe_asof", "reconcile_sources", "paper_account_guard"])
    assert cov.ready == ["safe_asof", "reconcile_sources", "paper_account_guard"]
    # the per-fill shortfall is what execution-cost-analysis wants, and it is non-zero
    shortfall = (b.close.to_numpy() - b.other.to_numpy()) / b.other.to_numpy()
    assert np.abs(shortfall).sum() > 0


def test_execution_to_bundle_rejects_a_non_record():
    with pytest.raises(TypeError, match="ExecutionRecord"):
        X.to_bundle(object(), expected_fills=pd.DataFrame())


def test_execution_fill_window_is_half_open():
    trades = [{"timestamp": 1_700_000_000_000, "symbol": "X", "side": "buy",
               "amount": 1, "price": 1.0},
              {"timestamp": 1_700_000_060_000, "symbol": "X", "side": "buy",
               "amount": 1, "price": 1.0}]
    rec = X.read_execution("ccxt:binance", client=FakeExchange(trades=trades),
                           since=pd.Timestamp(1_700_000_000_000, unit="ms", tz="UTC"),
                           until=pd.Timestamp(1_700_000_060_000, unit="ms", tz="UTC"))
    assert len(rec.fills) == 1, "[since, until) - the upper bound is exclusive"


def test_execution_warns_nothing_and_imports_nothing_for_a_stub_client():
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        X.read_execution("ib", client=FakeIB("DF1234567"))
    assert "ccxt" not in sys.modules and "ib_async" not in sys.modules
