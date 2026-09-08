"""fin_skills.core.paper_account_guard - prove paper from a server-returned fact, or refuse."""
from __future__ import annotations

import pytest

from fin_skills.core.paper_account_guard import (GateClosedError, KillSwitchTripped,
                                                 LiveTradingGate, NotPaperError, PaperVerdict,
                                                 _host_of, assert_paper)


# ------------------------------------------------------------------------------- IB
def test_ib_account_prefix_is_the_evidence():
    v = assert_paper("ib", account_id="DU1234567")
    assert isinstance(v, PaperVerdict) and v.is_paper and "DU" in v.evidence
    assert assert_paper("ibkr", account_id="df9876543").is_paper       # advisor paper, any case
    with pytest.raises(NotPaperError, match="LIVE account"):
        assert_paper("tws", account_id="U1234567", extra={"port": 7497})
    with pytest.raises(NotPaperError, match="no known prefix"):
        assert_paper("ib", account_id="X1")


def test_ib_port_is_not_evidence():
    with pytest.raises(NotPaperError, match="port is a convention"):
        assert_paper("ib", extra={"port": 7497})
    with pytest.raises(NotPaperError, match="account_id is required"):
        assert_paper("ib")


# --------------------------------------------------------------------------- Alpaca
def test_alpaca_resolved_host_decides():
    assert assert_paper("alpaca", base_url="https://paper-api.alpaca.markets/v2",
                        extra={"paper": True}).is_paper
    assert assert_paper("alpaca-py", base_url="broker-api.sandbox.alpaca.markets").is_paper
    with pytest.raises(NotPaperError, match="LIVE trading"):
        assert_paper("alpaca", base_url="https://api.alpaca.markets/v2", extra={"paper": False})
    with pytest.raises(NotPaperError, match="url_override trap"):
        assert_paper("alpaca", base_url="https://api.alpaca.markets/v2", extra={"paper": True})
    with pytest.raises(NotPaperError, match="disagree"):
        assert_paper("alpaca", base_url="https://paper-api.alpaca.markets", extra={"paper": False})


def test_alpaca_data_host_unknown_host_and_missing_url():
    with pytest.raises(NotPaperError, match="MARKET DATA host"):
        assert_paper("alpaca", base_url="https://data.alpaca.markets/v2")
    with pytest.raises(NotPaperError, match="unrecognised host"):
        assert_paper("alpaca", base_url="https://proxy.example.com")
    with pytest.raises(NotPaperError, match="base_url is required"):
        assert_paper("alpaca")
    assert _host_of("PAPER-API.alpaca.markets:443") == "paper-api.alpaca.markets"
    with pytest.raises(NotPaperError):
        _host_of("")


# --------------------------------------------------------------------------- Schwab
def test_schwab_has_no_sandbox():
    with pytest.raises(NotPaperError, match="NO sandbox"):
        assert_paper("schwab", account_id="12345678")
    with pytest.raises(NotPaperError, match="NO sandbox"):
        assert_paper("td_ameritrade", account_id="DU1")


# ----------------------------------------------------------------------------- ccxt
def test_ccxt_needs_the_flag_and_testnet_hosts():
    ok = assert_paper("ccxt:binance", extra={
        "sandbox_mode": True,
        "urls": {"api": {"public": "https://testnet.binance.vision/api",
                         "private": "https://testnet.binance.vision/api"}}})
    assert ok.is_paper and ok.broker == "ccxt:binance"
    with pytest.raises(NotPaperError, match="still points at production"):
        assert_paper("ccxt:binance", extra={"sandbox_mode": True,
                                            "urls": {"api": {"public": "https://api.binance.com"}}})
    with pytest.raises(NotPaperError, match="sandbox_mode is None"):
        assert_paper("ccxt:bybit", extra={"sandbox_mode": None,
                                          "urls": {"api": {"public": "https://api.bybit.com"}}})
    with pytest.raises(NotPaperError, match="missing"):
        assert_paper("ccxt:bybit", extra={"sandbox_mode": True})
    assert assert_paper("ccxt:kraken", extra={"sandbox_mode": True,
                                              "urls": {"api": "https://demo.kraken.com"}}).is_paper


def test_okx_is_header_switched():
    ok = assert_paper("ccxt:okx", extra={"sandbox_mode": True,
                                         "headers": {"X-Simulated-Trading": "1"},
                                         "urls": {"api": {"rest": "https://www.okx.com"}}})
    assert ok.is_paper and "header" in ok.evidence
    with pytest.raises(NotPaperError, match="HEADER-switched"):
        assert_paper("ccxt:okx", extra={"sandbox_mode": True, "headers": {},
                                        "urls": {"api": {"rest": "https://www.okx.com"}}})


def test_unknown_broker_is_refused_and_ccxt_is_inferred_from_the_facts():
    with pytest.raises(NotPaperError, match="unknown broker"):
        assert_paper("robinhood", account_id="DU1")
    assert assert_paper("kraken", extra={"sandbox_mode": True,
                                         "urls": {"api": "https://sandbox.kraken.com"}}).is_paper


# ----------------------------------------------------------------------------- gate
@pytest.fixture
def gate(monkeypatch):
    monkeypatch.delenv("TRADING_LIVE", raising=False)
    clock = [1_700_000_000.0]
    g = LiveTradingGate(max_orders_per_minute=3, max_notional_per_day=100_000,
                        clock=lambda: clock[0])
    return g, clock


def test_gate_is_closed_unless_explicitly_truthy(gate, monkeypatch):
    g, _ = gate
    with pytest.raises(GateClosedError):
        g.check_order(1_000)
    for value in ("0", "", "maybe", "TRUE "):
        monkeypatch.setenv("TRADING_LIVE", value)
        if value.strip().lower() in LiveTradingGate.TRUTHY:
            assert g.is_live_enabled()
        else:
            with pytest.raises(GateClosedError):
                g.check_order(1_000)
    monkeypatch.setenv("TRADING_LIVE", "yes")
    assert g.check_order(1_000)["orders_last_minute"] == 1.0


def test_rate_trip_is_one_way_until_a_reasoned_reset(gate, monkeypatch):
    g, clock = gate
    monkeypatch.setenv("TRADING_LIVE", "1")
    for _ in range(3):
        clock[0] += 5.0
        g.check_order(10_000)
    clock[0] += 5.0
    with pytest.raises(KillSwitchTripped, match="order rate 4 > 3/min"):
        g.check_order(10_000)
    assert g.should_flatten and g.status()["halted"]
    with pytest.raises(KillSwitchTripped, match="HALTED"):
        g.check_order(1.0)
    with pytest.raises(ValueError, match="requires a reason"):
        g.reset("")
    g.reset("understood")
    assert not g.should_flatten and g.status()["notional_today"] == 0.0


def test_rate_window_slides_and_daily_notional_trips(gate, monkeypatch):
    g, clock = gate
    monkeypatch.setenv("TRADING_LIVE", "on")
    for _ in range(3):
        g.check_order(10_000)
    clock[0] += 61.0                                    # the minute has passed
    st = g.check_order(10_000)
    assert st["orders_last_minute"] == 1.0 and st["notional_today"] == 40_000
    with pytest.raises(KillSwitchTripped, match="daily notional"):
        g.check_order(70_000)
    g.reset("daily cap understood")
    clock[0] += 86_400.0                                # a new UTC day resets the counter
    assert g.check_order(90_000)["notional_remaining"] == pytest.approx(10_000)
    with pytest.raises(ValueError):
        g.check_order(-1.0)


def test_demo_runs_and_restores_the_environment(run_main, monkeypatch):
    monkeypatch.delenv("TRADING_LIVE", raising=False)
    out = run_main("fin_skills.core.paper_account_guard")
    assert "Prove paper from a server-returned fact" in out
    assert "KILL SWITCH" in out
