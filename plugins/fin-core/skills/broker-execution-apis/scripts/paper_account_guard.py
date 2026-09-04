"""Refuse to trade unless a SERVER-RETURNED fact proves the account is paper.

WHY this exists — every "am I in paper?" check people actually write is a local belief,
not evidence, and each of these has fired for real money:

  * **IB**: the port is not proof. 7497/4002 are *conventions* for the paper TWS/Gateway;
    TWS lets you change every port number, and a live login on the "paper port" is one
    dropdown away. The only server-returned fact is the account id: **paper accounts start
    with `DU` (or `DF` for advisor/family paper), live accounts start with `U`**. Assert on
    `ib.managedAccounts()`, never on the socket you dialled.
  * **Alpaca**: `alpaca-py`'s `TradingClient` defaults to **`paper=True`**, which reads as
    safe — but `url_override` bypasses the flag, so the flag and the host can disagree and
    the HOST is what routes the order. Check the RESOLVED base URL: `paper-api.alpaca.markets`
    (or `broker-api.sandbox.alpaca.markets`) is paper, `api.alpaca.markets` is real money.
  * **Schwab**: there is **no sandbox**. The Trader API has one environment and it is
    production; the old TD Ameritrade sandbox died with the migration. Any code path that
    believes it is "testing against Schwab" is testing against a live account.
  * **ccxt venues**: `exchange.sandbox = True` set as an attribute does nothing. Only
    `set_sandbox_mode(True)` rewrites `urls['api']`, so the proof is BOTH the flag and a
    testnet host actually being in the resolved URLs. (OKX is the documented exception:
    its demo trading is header-switched — `x-simulated-trading: 1` — on the same host, so
    the host test can never pass and the header is what must be checked.)

This module refuses by default. Every unknown broker, missing account id, ambiguous host
or absent flag raises `NotPaperError`. "I could not prove it is paper" and "it is live"
are treated as the same outcome, because they have the same consequence.

`LiveTradingGate` is the second half: when you DO mean to trade live, an env-var gate that
is closed unless explicitly opened, plus order-rate and daily-notional counters that trip
once and stay tripped until a human resets them.

Usage:
    from paper_account_guard import assert_paper, LiveTradingGate

    assert_paper("ib", account_id=ib.managedAccounts()[0])
    assert_paper("alpaca", base_url=client._get_base_url(), extra={"paper": client._paper})

    gate = LiveTradingGate(max_orders_per_minute=30, max_notional_per_day=250_000)
    gate.check_order(notional=12_500)     # raises unless TRADING_LIVE is explicitly set
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, ClassVar, Mapping
from urllib.parse import urlsplit


class NotPaperError(RuntimeError):
    """Could not PROVE the account is paper. Fail closed; never downgrade to a warning."""


class GateClosedError(RuntimeError):
    """Live trading was not explicitly enabled."""


class KillSwitchTripped(RuntimeError):
    """A rate or notional limit was breached. Flatten and halt; do not auto-resume."""


@dataclass(frozen=True)
class PaperVerdict:
    broker: str
    is_paper: bool
    evidence: str
    rule: str

    def __str__(self) -> str:
        return f"[{self.broker}] PAPER confirmed -- {self.evidence}"


# --------------------------------------------------------------------------- IB
IB_PAPER_PREFIXES: tuple[str, ...] = ("DU", "DF")   # DU individual, DF advisor/family
IB_LIVE_PREFIXES: tuple[str, ...] = ("U", "F", "I")

# ------------------------------------------------------------------------ Alpaca
ALPACA_PAPER_HOSTS: frozenset[str] = frozenset({
    "paper-api.alpaca.markets",
    "broker-api.sandbox.alpaca.markets",
})
ALPACA_LIVE_HOSTS: frozenset[str] = frozenset({
    "api.alpaca.markets",
    "broker-api.alpaca.markets",
})
# Market-data hosts carry no account at all -- seeing one means the caller checked the
# wrong client and has proved nothing about where orders go.
ALPACA_DATA_HOSTS: frozenset[str] = frozenset({
    "data.alpaca.markets", "stream.data.alpaca.markets",
})

# -------------------------------------------------------------------------- ccxt
CCXT_TESTNET_MARKERS: tuple[str, ...] = (
    "testnet", "sandbox", "-test.", "test.", "demo", "paper", "uat", "stage",
)
# Venues whose demo environment shares the production hostname and is selected by a
# header instead. For these the host test CANNOT pass, so the header is the evidence.
CCXT_HEADER_SWITCHED: dict[str, tuple[str, str]] = {
    "okx": ("x-simulated-trading", "1"),
    "okex": ("x-simulated-trading", "1"),
}


def _host_of(url: str) -> str:
    """Hostname, lowercased, port stripped. Accepts bare hosts as well as full URLs."""
    u = url.strip()
    if "//" not in u:
        u = "//" + u
    host = (urlsplit(u).hostname or "").lower()
    if not host:
        raise NotPaperError(f"could not parse a hostname out of base_url={url!r}")
    return host


def _assert_paper_ib(account_id: str | None, extra: Mapping[str, Any]) -> PaperVerdict:
    if not account_id:
        hint = ""
        if any(k in extra for k in ("port", "socket_port")):
            hint = (" You passed a port. The port is a convention, not evidence: TWS lets "
                    "you set any port on either login, so 7497 can be a LIVE session.")
        raise NotPaperError(
            "IB: account_id is required. Read it from the server with "
            "`ib.managedAccounts()` (ib_insync) or the managedAccounts callback "
            f"(ibapi) -- do not hardcode it.{hint}"
        )
    acct = account_id.strip().upper()
    if acct.startswith(IB_PAPER_PREFIXES):
        return PaperVerdict("ib", True, f"account id {acct!r} starts with "
                            f"{acct[:2]!r} (IB paper)",
                            "IB paper ids start with DU/DF; live ids start with U")
    if acct.startswith(IB_LIVE_PREFIXES):
        raise NotPaperError(
            f"IB: account {acct!r} is a LIVE account (paper ids start with DU/DF). "
            f"If you believed this was paper because of the port, that is the bug."
        )
    raise NotPaperError(
        f"IB: account {acct!r} matches no known prefix (paper DU/DF, live U/F/I). "
        f"Unrecognised means unproven, and unproven is refused."
    )


def _assert_paper_alpaca(base_url: str | None, extra: Mapping[str, Any]) -> PaperVerdict:
    flag = extra.get("paper", extra.get("is_paper"))
    if not base_url:
        raise NotPaperError(
            "Alpaca: base_url is required and must be the RESOLVED url the client will "
            "actually call (alpaca-py: `client._get_base_url()`; alpaca-trade-api: "
            "`api._base_url`). The `paper=True` constructor flag is the default and is "
            "overridden by `url_override`, so the flag alone proves nothing."
        )
    host = _host_of(base_url)
    if host in ALPACA_DATA_HOSTS:
        raise NotPaperError(
            f"Alpaca: {host!r} is a MARKET DATA host, not a trading host. It says nothing "
            f"about where orders route. Check the TradingClient's base url."
        )
    if host in ALPACA_PAPER_HOSTS:
        if flag is False:
            # Harmless direction (paper host, live flag) but still a lie in the config:
            # the next refactor may trust the flag.
            raise NotPaperError(
                f"Alpaca: host {host!r} is paper but the client's paper flag is False. "
                f"The flag and the host disagree; fix the config rather than picking a "
                f"winner."
            )
        return PaperVerdict("alpaca", True, f"resolved host {host!r} is a paper endpoint",
                            "Alpaca paper is proven by the resolved host, not the flag")
    if host in ALPACA_LIVE_HOSTS:
        extra_msg = ""
        if flag is True:
            extra_msg = (" The client reports paper=True while pointing at the LIVE host "
                         "-- this is the url_override trap: the flag is ignored, the host "
                         "routes the order.")
        raise NotPaperError(
            f"Alpaca: resolved host {host!r} is LIVE trading (real money).{extra_msg}"
        )
    raise NotPaperError(
        f"Alpaca: unrecognised host {host!r}. Known paper: {sorted(ALPACA_PAPER_HOSTS)}; "
        f"known live: {sorted(ALPACA_LIVE_HOSTS)}. A proxy or a typo'd override is not "
        f"evidence of paper."
    )


def _assert_paper_schwab(account_id: str | None, extra: Mapping[str, Any]) -> PaperVerdict:
    raise NotPaperError(
        "Schwab: there is NO sandbox. The Trader API has a single production "
        "environment, and the TD Ameritrade sandbox was retired with the migration. "
        "Every Schwab account -- including one you have never funded -- must be treated "
        "as live. Test against a simulator you own, or against a different broker's paper "
        "endpoint, and gate the Schwab path behind LiveTradingGate."
    )


def _assert_paper_ccxt(broker: str, extra: Mapping[str, Any]) -> PaperVerdict:
    venue = broker.split(":", 1)[-1].strip().lower() if ":" in broker else broker.lower()
    sandbox = extra.get("sandbox_mode", extra.get("sandbox"))
    urls = extra.get("urls") or {}
    api = urls.get("api") if isinstance(urls, Mapping) else None
    headers = {str(k).lower(): str(v) for k, v in (extra.get("headers") or {}).items()}

    if sandbox is not True:
        raise NotPaperError(
            f"ccxt/{venue}: sandbox_mode is {sandbox!r}. Pass "
            f"`exchange.set_sandbox_mode(True)` BEFORE the first request and report "
            f"`exchange.urls` afterwards. Setting `exchange.sandbox = True` as a plain "
            f"attribute does not rewrite urls['api'] and does not switch anything."
        )

    if venue in CCXT_HEADER_SWITCHED:
        hdr, want = CCXT_HEADER_SWITCHED[venue]
        got = headers.get(hdr)
        if got != want:
            raise NotPaperError(
                f"ccxt/{venue}: demo trading is HEADER-switched, not host-switched -- the "
                f"hostname stays production. Required header {hdr}: {want!r}, got {got!r}."
            )
        return PaperVerdict(f"ccxt:{venue}", True,
                            f"sandbox_mode=True and header {hdr}={want!r}",
                            f"{venue} demo trading shares the production host; the header "
                            f"is the only evidence")

    hosts: list[str] = []
    if isinstance(api, str):
        hosts = [api]
    elif isinstance(api, Mapping):
        hosts = [str(v) for v in api.values() if isinstance(v, str)]
    if not hosts:
        raise NotPaperError(
            f"ccxt/{venue}: extra['urls']['api'] is missing. Report the exchange's "
            f"RESOLVED urls after set_sandbox_mode(True); the flag alone is a local "
            f"variable."
        )
    lowered = [h.lower() for h in hosts]
    matched = [h for h in lowered if any(m in h for m in CCXT_TESTNET_MARKERS)]
    if len(matched) != len(lowered):
        live = [h for h in lowered if h not in matched]
        raise NotPaperError(
            f"ccxt/{venue}: sandbox_mode=True but urls['api'] still points at production: "
            f"{live}. The flag was set after the urls were resolved, or this venue has no "
            f"testnet. Orders would hit the live book."
        )
    return PaperVerdict(f"ccxt:{venue}", True,
                        f"sandbox_mode=True and every urls['api'] host is a testnet "
                        f"({matched[0]})",
                        "ccxt needs BOTH set_sandbox_mode(True) and testnet hosts")


_IB_ALIASES = {"ib", "ibkr", "interactive_brokers", "ib_insync", "ibapi", "tws"}
_ALPACA_ALIASES = {"alpaca", "alpaca-py", "alpaca_py", "alpaca-trade-api"}
_SCHWAB_ALIASES = {"schwab", "charles_schwab", "schwabdev", "tda", "td_ameritrade"}


def assert_paper(broker: str, account_id: str | None = None,
                 base_url: str | None = None,
                 extra: Mapping[str, Any] | None = None) -> PaperVerdict:
    """Prove the session is paper, or raise `NotPaperError`. There is no third outcome.

    Call this once at startup AND again after any reconnect — a reconnect can land on a
    different account, and the object you checked five minutes ago is not the object
    sending the order.

    `extra` carries the server-returned facts a given broker needs: `paper` for Alpaca,
    `sandbox_mode` / `urls` / `headers` for ccxt.
    """
    ex = dict(extra or {})
    key = broker.strip().lower()
    base = key.split(":", 1)[0]
    if base in _IB_ALIASES:
        return _assert_paper_ib(account_id, ex)
    if base in _ALPACA_ALIASES:
        return _assert_paper_alpaca(base_url, ex)
    if base in _SCHWAB_ALIASES:
        return _assert_paper_schwab(account_id, ex)
    if base == "ccxt" or "sandbox_mode" in ex or "urls" in ex:
        return _assert_paper_ccxt(key, ex)
    raise NotPaperError(
        f"unknown broker {broker!r}: no rule to prove paper, so trading is refused. Add a "
        f"rule that reads a SERVER-RETURNED fact (account id, resolved host) -- not a "
        f"constructor flag, not a port, not an env var."
    )


def _utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


@dataclass
class LiveTradingGate:
    """Closed-by-default live gate with a one-way kill switch.

    Two independent jobs:
      1. `TRADING_LIVE` must be explicitly set to a truthy value. Unset, empty, "0" and
         anything unrecognised all mean CLOSED — the default state of a fresh shell, a CI
         runner and a container is "cannot trade".
      2. Rate and notional counters that TRIP rather than throttle. A throttle hides a
         runaway loop; a trip stops it and demands a human. Once tripped, `should_flatten`
         is True and every subsequent `check_order` raises until `reset()` is called by
         hand.

    `clock` and `day_key` are injected so this is testable without sleeping and so the
    day boundary can follow the VENUE's session rather than UTC midnight (which lands
    mid-session for Asian markets).
    """

    max_orders_per_minute: int = 60
    max_notional_per_day: float = 1_000_000.0
    env_var: str = "TRADING_LIVE"
    clock: Callable[[], float] = time.time
    day_key: Callable[[float], str] = _utc_day
    halted: bool = False
    halt_reason: str = ""
    _order_times: deque[float] = field(default_factory=deque, repr=False)
    _day: str = field(default="", repr=False)
    _day_notional: float = field(default=0.0, repr=False)

    # ClassVar, not a field: the truthy set is policy, not per-instance configuration.
    TRUTHY: ClassVar[frozenset[str]] = frozenset({"1", "true", "yes", "on", "enabled"})

    def is_live_enabled(self) -> bool:
        return os.environ.get(self.env_var, "").strip().lower() in self.TRUTHY

    @property
    def should_flatten(self) -> bool:
        """True once the kill switch has tripped: cancel working orders, go flat, stop."""
        return self.halted

    def check_order(self, notional: float, now: float | None = None) -> dict[str, float]:
        """Authorise ONE order and count it. Call immediately before sending — this both
        checks and records, so a call that is not followed by a send under-counts.

        Raises GateClosedError if live is not enabled, KillSwitchTripped on a breach.
        """
        if notional < 0:
            raise ValueError("notional must be non-negative (use absolute value)")
        ts = self.clock() if now is None else now

        if self.halted:
            raise KillSwitchTripped(
                f"gate is HALTED: {self.halt_reason}. Flatten, investigate, then call "
                f"reset() by hand. Auto-resume is how one bad loop becomes two."
            )
        if not self.is_live_enabled():
            raise GateClosedError(
                f"{self.env_var} is not set to a truthy value "
                f"({sorted(self.TRUTHY)}); live trading is closed by default. Set it "
                f"deliberately, in the process that trades, and never in a dotfile."
            )

        day = self.day_key(ts)
        if day != self._day:
            self._day, self._day_notional = day, 0.0

        cutoff = ts - 60.0
        while self._order_times and self._order_times[0] <= cutoff:
            self._order_times.popleft()

        if len(self._order_times) + 1 > self.max_orders_per_minute:
            self._trip(f"order rate {len(self._order_times) + 1} > "
                       f"{self.max_orders_per_minute}/min")
        if self._day_notional + notional > self.max_notional_per_day:
            self._trip(f"daily notional {self._day_notional + notional:,.2f} > "
                       f"{self.max_notional_per_day:,.2f} on {day}")

        self._order_times.append(ts)
        self._day_notional += notional
        return {
            "orders_last_minute": float(len(self._order_times)),
            "orders_remaining": float(self.max_orders_per_minute - len(self._order_times)),
            "notional_today": self._day_notional,
            "notional_remaining": self.max_notional_per_day - self._day_notional,
        }

    def _trip(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason
        raise KillSwitchTripped(f"KILL SWITCH: {reason}. FLATTEN AND HALT.")

    def reset(self, reason: str) -> None:
        """Manual, deliberate, and logged. Deliberately not callable from a retry loop."""
        if not reason.strip():
            raise ValueError("reset() requires a reason; an unexplained reset is the "
                             "incident happening twice")
        self.halted, self.halt_reason = False, ""
        self._order_times.clear()
        self._day_notional = 0.0

    def status(self) -> dict[str, Any]:
        return {"live_enabled": self.is_live_enabled(), "halted": self.halted,
                "halt_reason": self.halt_reason, "day": self._day,
                "notional_today": self._day_notional,
                "orders_in_window": len(self._order_times)}


if __name__ == "__main__":
    def show(label: str, fn: Callable[[], PaperVerdict]) -> None:
        try:
            print(f"  PASS  {label}\n        {fn()}")
        except NotPaperError as e:
            print(f"  BLOCK {label}\n        NotPaperError: {e}")

    print("=" * 88)
    print("1. INTERACTIVE BROKERS -- the account id is the evidence, the port is not")
    print("=" * 88)
    show("managedAccounts() -> 'DU1234567'",
         lambda: assert_paper("ib", account_id="DU1234567"))
    show("managedAccounts() -> 'DF9876543' (advisor paper)",
         lambda: assert_paper("ib", account_id="DF9876543"))
    show("managedAccounts() -> 'U1234567' on port 7497 ('the paper port')",
         lambda: assert_paper("ib", account_id="U1234567", extra={"port": 7497}))
    show("no account id, just the port",
         lambda: assert_paper("ib", extra={"port": 7497}))

    print("\n" + "=" * 88)
    print("2. ALPACA -- the RESOLVED host is the evidence, `paper=True` is a default")
    print("=" * 88)
    show("TradingClient(paper=True) -> https://paper-api.alpaca.markets",
         lambda: assert_paper("alpaca", base_url="https://paper-api.alpaca.markets/v2",
                              extra={"paper": True}))
    show("TradingClient(paper=False) -> https://api.alpaca.markets",
         lambda: assert_paper("alpaca", base_url="https://api.alpaca.markets/v2",
                              extra={"paper": False}))
    print("\n  ...and the trap: the flag and the host DISAGREE")
    show("TradingClient(paper=True, url_override='https://api.alpaca.markets')",
         lambda: assert_paper("alpaca", base_url="https://api.alpaca.markets/v2",
                              extra={"paper": True}))
    show("checked the data client instead of the trading client",
         lambda: assert_paper("alpaca", base_url="https://data.alpaca.markets/v2",
                              extra={"paper": True}))

    print("\n" + "=" * 88)
    print("3. SCHWAB -- no sandbox exists, so no Schwab account can be proven paper")
    print("=" * 88)
    show("a brand new, unfunded Schwab account",
         lambda: assert_paper("schwab", account_id="12345678"))

    print("\n" + "=" * 88)
    print("4. CCXT -- needs set_sandbox_mode(True) AND testnet hosts in urls['api']")
    print("=" * 88)
    show("binance after set_sandbox_mode(True)",
         lambda: assert_paper("ccxt:binance", extra={
             "sandbox_mode": True,
             "urls": {"api": {"public": "https://testnet.binance.vision/api",
                              "private": "https://testnet.binance.vision/api"}}}))
    show("binance: sandbox flag set, urls still production",
         lambda: assert_paper("ccxt:binance", extra={
             "sandbox_mode": True,
             "urls": {"api": {"public": "https://api.binance.com/api",
                              "private": "https://api.binance.com/api"}}}))
    show("bybit: `exchange.sandbox = True` set as a plain attribute (no-op)",
         lambda: assert_paper("ccxt:bybit", extra={
             "sandbox_mode": None,
             "urls": {"api": {"public": "https://api.bybit.com"}}}))
    show("okx demo: header-switched, production host, header present",
         lambda: assert_paper("ccxt:okx", extra={
             "sandbox_mode": True, "headers": {"x-simulated-trading": "1"},
             "urls": {"api": {"rest": "https://www.okx.com"}}}))
    show("okx demo: sandbox flag set but the header never made it onto the session",
         lambda: assert_paper("ccxt:okx", extra={
             "sandbox_mode": True, "headers": {},
             "urls": {"api": {"rest": "https://www.okx.com"}}}))

    print("\n" + "=" * 88)
    print("5. LIVE TRADING GATE -- closed by default, trips once, stays tripped")
    print("=" * 88)
    fake_now = [1_700_000_000.0]                      # deterministic clock; no sleeping
    gate = LiveTradingGate(max_orders_per_minute=3, max_notional_per_day=100_000,
                           clock=lambda: fake_now[0])

    os.environ.pop("TRADING_LIVE", None)
    try:
        gate.check_order(notional=1_000)
    except GateClosedError as e:
        print(f"  BLOCK TRADING_LIVE unset\n        GateClosedError: {e}")
    os.environ["TRADING_LIVE"] = "0"
    try:
        gate.check_order(notional=1_000)
    except GateClosedError:
        print("  BLOCK TRADING_LIVE='0' -- only an explicit truthy value opens the gate")

    os.environ["TRADING_LIVE"] = "1"
    print("\n  TRADING_LIVE='1' -- gate open, 3 orders/min, 100,000/day:")
    for i in range(1, 5):
        fake_now[0] += 5.0
        try:
            st = gate.check_order(notional=10_000)
            print(f"    order {i}: OK   orders_in_window={st['orders_last_minute']:.0f} "
                  f"notional_today={st['notional_today']:,.0f}")
        except KillSwitchTripped as e:
            print(f"    order {i}: {e}")
    print(f"    should_flatten = {gate.should_flatten}")
    try:
        gate.check_order(notional=1.0)
    except KillSwitchTripped as e:
        print(f"    next order after the trip: {str(e).split('.')[0]}.")

    gate.reset(reason="demo: rate limit understood and raised deliberately")
    fake_now[0] += 120.0
    gate.max_orders_per_minute = 100
    print("\n  after a manual reset, the DAILY NOTIONAL cap is the second, slower trip:")
    for i in range(1, 4):
        fake_now[0] += 1.0
        try:
            st = gate.check_order(notional=40_000)
            print(f"    order {i}: OK   notional_today={st['notional_today']:,.0f} "
                  f"remaining={st['notional_remaining']:,.0f}")
        except KillSwitchTripped as e:
            print(f"    order {i}: {e}")
    print(f"    status = {gate.status()}")
    os.environ.pop("TRADING_LIVE", None)

    print("\n" + "=" * 88)
    print("Prove paper from a server-returned fact. Gate live behind an explicit switch.")
    print("=" * 88)
