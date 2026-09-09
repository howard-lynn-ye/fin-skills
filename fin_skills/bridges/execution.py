"""Read fills and positions back from a broker. READ-ONLY BY CONSTRUCTION.

This module ingests what already happened and emits records. It cannot build, sign, send,
change or withdraw an instruction to trade, and it exposes no callable that could -
matching the repo's own scope line, *"no skill in this repo places an order"*.
That is not a convention here, it is tested: `tests/test_bridges_execution.py` greps this
file for the trading method names of ccxt, ib_async and alpaca-py and asserts that not one
of them appears, and separately asserts that every public callable is a reader.

Everything below is a read: ccxt's `fetch_my_trades` / `fetch_positions` / `fetch_balance`
/ `fetch_ohlcv`, ib_async's `fills()` / `positions()` / `accountSummary()` /
`managedAccounts()`, alpaca-py's `get_account()` / `get_all_positions()` /
`get_account_activities()`.

THE GATE (fin_skills.load('broker-execution-apis')): `paper_account_guard` must PASS before
any client is constructed or touched. It exists because

  * one character separates IB's live 7496/4001 from paper 7497/4002, and the port is a
    local config value you can typo - the account id is a SERVER-RETURNED fact, `DU`/`DF`
    for paper and `U` for live (fin_skills.load('lib-ib-async'));
  * alpaca-py's `url_override` replaces the base URL outright while `sandbox=paper` is
    still set from the flag, so a client that believes it is in the sandbox can be pointed
    at the live host (fin_skills.load('lib-alpaca-py'));
  * ccxt's `set_sandbox_mode(True)` succeeds on venues that have no testnet at all, so the
    proof is the RESOLVED host, not the call (fin_skills.load('lib-ccxt')).

Credentials are read from the environment variable named for the venue, are never stored,
logged, printed or written into `Provenance`, and `ExecutionRecord.account` is scrubbed of
identifiers before it is returned.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fin_skills.api import Bundle, get
from fin_skills.bridges import _lazy

#: venue prefix -> the audited library that reads it.
VENUE_LIBRARY = {"ccxt": "ccxt", "ib": "ib_async", "alpaca": "alpaca-py"}
LIBRARY = "ccxt"
LICENCE = _lazy.info(LIBRARY).licence
VERIFIED_ON = _lazy.info(LIBRARY).verified_on

#: environment variables the venues read their credentials from. Names only, never values.
ENV_CREDENTIALS: dict[str, tuple[str, ...]] = {
    "ccxt": ("CCXT_API_KEY", "CCXT_SECRET"),
    "alpaca": ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
    "ib": (),                       # IB authenticates in TWS/Gateway, not in the client
}
#: anything matching these is stripped out of `account` before it is returned.
_SECRETISH = re.compile(r"(key|secret|token|password|passphrase|auth|cookie|session)",
                        re.IGNORECASE)
_ACCOUNTISH = re.compile(r"(account|acct|id)$", re.IGNORECASE)

_TIMEFRAME_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
                 "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
                 "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
                 "1d": 86_400_000, "1w": 604_800_000}


class NotPaperRefusal(RuntimeError):
    """The account could not be PROVEN to be paper. Read nothing, construct nothing."""


@dataclass(frozen=True)
class ExecutionRecord:
    """What happened, as read back from the venue. Nothing here can be sent anywhere."""

    fills: pd.DataFrame          # ts, symbol, side, qty, price, fee, venue, client_ref
    positions: pd.DataFrame      # ts, symbol, qty, avg_price
    account: dict                # SERVER-RETURNED facts, scrubbed of identifiers
    venue: str
    is_paper: bool               # from a server-returned fact, never a local flag
    provenance: _lazy.Provenance | None = None

    def summary(self) -> str:
        return (f"{self.venue}: {len(self.fills)} fill(s), {len(self.positions)} position "
                f"row(s), paper={self.is_paper}")


FILL_COLUMNS = ("ts", "symbol", "side", "qty", "price", "fee", "venue", "client_ref")
POSITION_COLUMNS = ("ts", "symbol", "qty", "avg_price")


def _empty(cols: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def _scrub(facts: Mapping[str, Any]) -> dict:
    """Drop anything credential-shaped; mask anything identifier-shaped to its prefix."""
    out: dict[str, Any] = {}
    for k, v in (facts or {}).items():
        key = str(k)
        if _SECRETISH.search(key):
            continue
        if isinstance(v, Mapping):
            out[key] = _scrub(v)
        elif _ACCOUNTISH.search(key) and isinstance(v, str) and len(v) > 2:
            out[key] = v[:2] + "*" * (len(v) - 2)
        else:
            out[key] = v
    return out


def _split_venue(venue: str) -> tuple[str, str]:
    if not isinstance(venue, str) or not venue.strip():
        raise TypeError("venue must be 'ccxt:<exchange>', 'ib' or 'alpaca'")
    base, _, rest = venue.strip().lower().partition(":")
    if base not in VENUE_LIBRARY:
        raise ValueError(f"venue must start with one of {sorted(VENUE_LIBRARY)}, "
                         f"got {venue!r}")
    return base, rest


# ------------------------------------------------------------------- the paper gate
def _facts_from_client(base: str, client: Any) -> dict[str, Any]:
    """Read the SERVER-RETURNED facts off an already-built client. Reads only."""
    facts: dict[str, Any] = {}
    if client is None:
        return facts
    if base == "ccxt":
        facts["sandbox_mode"] = bool(getattr(client, "sandboxMode",
                                             getattr(client, "sandbox", False)))
        facts["urls"] = getattr(client, "urls", {}) or {}
        facts["headers"] = dict(getattr(client, "headers", {}) or {})
    elif base == "ib":
        managed = getattr(client, "managedAccounts", None)
        if callable(managed):
            got = managed() or []
            facts["account_id"] = str(got[0]) if got else ""
    elif base == "alpaca":
        acct = getattr(client, "get_account", None)
        if callable(acct):
            a = acct()
            facts["account_id"] = str(getattr(a, "account_number", "") or "")
            facts["paper"] = bool(getattr(client, "_paper", getattr(client, "paper", True)))
        base_url = getattr(client, "_base_url", getattr(client, "base_url", None))
        if base_url is not None:
            facts["base_url"] = str(base_url)
    return facts


def _assert_paper(base: str, venue: str, facts: Mapping[str, Any]) -> Any:
    """Run `paper_account_guard`. No pass, no read - and no client, either."""
    broker = "ccxt:" + venue.partition(":")[2] if base == "ccxt" else base
    inputs: dict[str, Any] = {"broker": broker}
    if facts.get("account_id"):
        inputs["account_id"] = str(facts["account_id"])
    if facts.get("base_url"):
        inputs["base_url"] = str(facts["base_url"])
    extra = {k: v for k, v in facts.items()
             if k in ("paper", "sandbox_mode", "urls", "headers")}
    if extra:
        inputs["extra"] = extra
    res = get("paper_account_guard").run(**inputs)
    if not res.passed:
        raise NotPaperRefusal(
            "; ".join(str(f) for f in res.errors)
            + " -- refusing to construct a client or read anything. 'Could not prove it is "
              "paper' and 'it is live' have the same consequence. Supply the SERVER-RETURNED "
              "fact: account_id from ib.managedAccounts() (DU/DF), the RESOLVED alpaca "
              "base_url, or extra={'sandbox_mode': ..., 'urls': ...} for a ccxt venue.")
    return res


# --------------------------------------------------------------------- normalisation
def _fills_frame(rows: Sequence[Mapping[str, Any]], venue: str) -> pd.DataFrame:
    out = []
    for r in rows or ():
        ts = r.get("timestamp", r.get("ts"))
        fee = r.get("fee") or {}
        out.append({
            "ts": pd.to_datetime(ts, unit="ms", utc=True) if isinstance(ts, (int, float))
            else pd.to_datetime(ts, utc=True, errors="coerce"),
            "symbol": r.get("symbol", r.get("contract", "")),
            "side": str(r.get("side", "")).lower(),
            "qty": float(r.get("amount", r.get("qty", r.get("shares", np.nan))) or np.nan),
            "price": float(r.get("price", np.nan) or np.nan),
            "fee": float((fee.get("cost") if isinstance(fee, Mapping) else fee) or 0.0),
            "venue": venue,
            "client_ref": str(r.get("clientOrderId", r.get("client_ref", "")) or ""),
        })
    df = pd.DataFrame(out, columns=list(FILL_COLUMNS)) if out else _empty(FILL_COLUMNS)
    return df.sort_values("ts").reset_index(drop=True) if len(df) else df


def _positions_frame(rows: Sequence[Mapping[str, Any]], ts: Any) -> pd.DataFrame:
    out = []
    for r in rows or ():
        out.append({"ts": ts, "symbol": r.get("symbol", ""),
                    "qty": float(r.get("contracts", r.get("qty", r.get("amount", 0.0)))
                                 or 0.0),
                    "avg_price": float(r.get("entryPrice",
                                             r.get("avg_price", np.nan)) or np.nan)})
    return (pd.DataFrame(out, columns=list(POSITION_COLUMNS)) if out
            else _empty(POSITION_COLUMNS))


# ------------------------------------------------------------------------ the readers
def _read_ccxt(exchange_id: str, client: Any, since: Any, until: Any,
               client_kw: dict) -> tuple[list, list, dict, Any]:
    if client is None:
        ccxt = _lazy.need("ccxt", why="to read fills back from a venue")
        cls = getattr(ccxt, exchange_id, None)
        if cls is None:
            raise ValueError(f"ccxt has no exchange {exchange_id!r}")
        key, secret = (os.environ.get(n, "") for n in ENV_CREDENTIALS["ccxt"])
        client = cls({"apiKey": key, "secret": secret,
                      # NOT optional: the built-in limiter is the only thing between you
                      # and a ban, and it is a per-INSTANCE leaky bucket.
                      "enableRateLimit": True, **client_kw})
        client.set_sandbox_mode(True)
        urls = getattr(client, "urls", {}) or {}
        flat = str(urls.get("api", ""))
        if "test" not in flat and "sandbox" not in flat and "demo" not in flat:
            raise NotPaperRefusal(
                f"set_sandbox_mode(True) succeeded on {exchange_id!r} but the resolved "
                f"host is not a testnet: urls['api']={flat[:120]!r}. The call succeeds "
                f"whether or not the venue has a testnet - verify the host, never the flag.")
        _assert_paper("ccxt", f"ccxt:{exchange_id}", _facts_from_client("ccxt", client))
    trades = client.fetch_my_trades(since=_ms(since), params={})
    positions = client.fetch_positions() if getattr(
        client, "has", {}).get("fetchPositions", True) else []
    balance = client.fetch_balance() if getattr(
        client, "has", {}).get("fetchBalance", True) else {}
    account = {"total": (balance or {}).get("total", {}),
               "sandbox_mode": bool(getattr(client, "sandboxMode",
                                            getattr(client, "sandbox", False)))}
    return list(trades or []), list(positions or []), account, client


def _read_ib(client: Any, since: Any, until: Any) -> tuple[list, list, dict, Any]:
    if client is None:
        raise TypeError(
            "the IB path needs an already-connected ib_async.IB(): this bridge will not "
            "open the socket for you, because the port is the one thing that separates "
            "paper from live and it is a LOCAL value. Connect yourself, then hand the "
            "client here - the account id is checked before anything is read.")
    fills = []
    for f in (client.fills() or []):
        ex, comm = getattr(f, "execution", None), getattr(f, "commissionReport", None)
        fills.append({"timestamp": getattr(ex, "time", None),
                      "symbol": getattr(getattr(f, "contract", None), "symbol", ""),
                      "side": "buy" if str(getattr(ex, "side", "")).upper().startswith("B")
                      else "sell",
                      "amount": getattr(ex, "shares", np.nan),
                      "price": getattr(ex, "price", np.nan),
                      "fee": getattr(comm, "commission", 0.0) if comm else 0.0,
                      "client_ref": str(getattr(ex, "permId", ""))})
    positions = [{"symbol": getattr(getattr(p, "contract", None), "symbol", ""),
                  "qty": getattr(p, "position", 0.0),
                  "avg_price": getattr(p, "avgCost", np.nan)}
                 for p in (client.positions() or [])]
    summary = {str(getattr(v, "tag", "")): getattr(v, "value", "")
               for v in (client.accountSummary() or [])}
    return fills, positions, summary, client


def _read_alpaca(client: Any, since: Any, until: Any,
                 client_kw: dict) -> tuple[list, list, dict, Any]:
    if client is None:
        alpaca = _lazy.need("alpaca-py", why="to read fills back from Alpaca")
        from alpaca.trading.client import TradingClient          # noqa: PLC0415 - lazy
        key, secret = (os.environ.get(n, "") for n in ENV_CREDENTIALS["alpaca"])
        if "url_override" in client_kw:
            raise NotPaperRefusal(
                "url_override replaces the base URL outright while the paper flag stays "
                "set, so the client can believe it is in the sandbox while pointing at the "
                "live host. This bridge does not accept it.")
        client = TradingClient(key, secret, paper=True, **client_kw)
        _ = alpaca
        _assert_paper("alpaca", "alpaca", _facts_from_client("alpaca", client))
    acct = client.get_account()
    activities = []
    getter = getattr(client, "get_account_activities", None)
    if callable(getter):
        for a in (getter() or []):
            activities.append({
                "timestamp": getattr(a, "transaction_time", None),
                "symbol": getattr(a, "symbol", ""),
                "side": str(getattr(a, "side", "")).lower(),
                "amount": getattr(a, "qty", np.nan),
                "price": getattr(a, "price", np.nan),
                "fee": 0.0,
                "client_ref": str(getattr(a, "id", "") or "")})
    positions = [{"symbol": getattr(p, "symbol", ""),
                  "qty": getattr(p, "qty", 0.0),
                  "avg_price": getattr(p, "avg_entry_price", np.nan)}
                 for p in (client.get_all_positions() or [])]
    account = {"equity": getattr(acct, "equity", None),
               "cash": getattr(acct, "cash", None),
               "status": str(getattr(acct, "status", "")),
               "account_number": str(getattr(acct, "account_number", ""))}
    return activities, positions, account, client


def _utc(x: Any) -> pd.Timestamp:
    """Any timestamp -> tz-aware UTC. Naive input is READ as UTC and said to be."""
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _ms(x: Any) -> int | None:
    return None if x is None else int(_utc(x).value // 1_000_000)


def read_execution(venue: str, *, since: Any = None, until: Any = None,
                   client: Any = None, account_id: str | None = None,
                   base_url: str | None = None, extra: Mapping[str, Any] | None = None,
                   **client_kw: Any) -> ExecutionRecord:
    """Read fills and positions back from a paper venue. Reads only, and never otherwise.

    venue       'ccxt:<exchange>' | 'ib' | 'alpaca'
    client      an already-built read-only client. Required for 'ib' (see below), optional
                elsewhere; when it is given, its own server-returned facts are what the
                guard is run against.
    account_id  IB's `managedAccounts()[0]`; `DU`/`DF` is paper, `U` is live.
    base_url    Alpaca's RESOLVED base url - `paper-api.alpaca.markets` is paper.
    extra       ccxt's `{'sandbox_mode': ..., 'urls': ..., 'headers': ...}`.

    `paper_account_guard` runs FIRST, on facts supplied by you or read off `client`. It has
    to pass before any client is constructed and before a single read happens; a failure
    raises `NotPaperRefusal` and nothing else runs.
    """
    base, exchange_id = _split_venue(venue)
    facts: dict[str, Any] = dict(extra or {})
    if account_id:
        facts["account_id"] = account_id
    if base_url:
        facts["base_url"] = base_url
    facts.update({k: v for k, v in _facts_from_client(base, client).items()
                  if k not in facts})
    guard = _assert_paper(base, venue, facts)

    if base == "ccxt":
        raw_fills, raw_pos, account, used = _read_ccxt(exchange_id, client, since, until,
                                                       dict(client_kw))
    elif base == "ib":
        raw_fills, raw_pos, account, used = _read_ib(client, since, until)
    else:
        raw_fills, raw_pos, account, used = _read_alpaca(client, since, until,
                                                         dict(client_kw))

    fills = _fills_frame(raw_fills, venue)
    if len(fills) and (since is not None or until is not None):
        keep = pd.Series(True, index=fills.index)
        if since is not None:
            keep &= fills["ts"] >= _utc(since)
        if until is not None:
            keep &= fills["ts"] < _utc(until)   # half-open, like the rest of the library
        fills = fills[keep].reset_index(drop=True)
    positions = _positions_frame(
        raw_pos, _utc(until) if until is not None else pd.Timestamp.now(tz="UTC"))
    lib = VENUE_LIBRARY[base]
    prov = _lazy.provenance(lib, source=venue, request={
        "since": str(since), "until": str(until), "read_only": True,
        "credentials_env": list(ENV_CREDENTIALS.get(base, ())),
        "paper_rule": guard.evidence.get("rule", "")})
    return ExecutionRecord(fills=fills, positions=positions, account=_scrub(account),
                           venue=venue, is_paper=True, provenance=prov)


# ------------------------------------------------------------------------- ccxt OHLCV
def read_ohlcv(exchange: Any, symbol: str, timeframe: str = "1m", *, since: Any = None,
               until: Any = None, limit: int | None = None,
               drop_unclosed: bool = True) -> pd.DataFrame:
    """Paginated OHLCV with the UNCLOSED FINAL BAR removed. A read, and only a read.

    Three silent ccxt data bugs, all in `fetch_ohlcv`
    (fin_skills.load('lib-ccxt')):

      1. it returns FEWER candles than you asked for - every venue caps `limit` differently
         (commonly 500 or 1000) with no error, so this loops on `since` and dedupes on
         timestamp, because several venues re-return overlapping windows and duplicate bars
         both inflate the bar count and break `pct_change`;
      2. **the last candle is unclosed** - the in-progress bar comes back as the final row,
         and using it in a signal is a live-only look-ahead that never shows in a backtest.
         `drop_unclosed=True` removes any bar whose close time is still in the future;
      3. perp funding is not in OHLCV at all, and is frequently larger than the alpha being
         measured. Fetch it separately.
    """
    tf_ms = _TIMEFRAME_MS.get(timeframe)
    if tf_ms is None:
        raise ValueError(f"unknown timeframe {timeframe!r}; known: "
                         f"{sorted(_TIMEFRAME_MS)}")
    start, stop = _ms(since), _ms(until)
    rows: list[list] = []
    seen: set[int] = set()
    cursor = start
    for _ in range(1000):                       # a hard stop; no unbounded venue loop
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        batch = [b for b in (batch or []) if int(b[0]) not in seen]
        if not batch:
            break
        for b in batch:
            seen.add(int(b[0]))
        rows.extend(batch)
        last = int(batch[-1][0])
        if stop is not None and last >= stop:
            break
        if len(batch) < 2:
            break
        cursor = last + tf_ms

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    rows.sort(key=lambda b: int(b[0]))
    now_fn = getattr(exchange, "milliseconds", None)
    now = int(now_fn()) if callable(now_fn) else int(time.time() * 1000)
    if drop_unclosed:
        rows = [b for b in rows if int(b[0]) + tf_ms <= now]
    if stop is not None:
        rows = [b for b in rows if int(b[0]) < stop]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    df.index.name = "ts"
    return df.astype(float)


# ---------------------------------------------------------------------------- Bundle
def to_bundle(rec: ExecutionRecord, *, expected_fills: pd.DataFrame,
              on: str = "ts", by: str = "symbol", tolerance: Any = "5min",
              **extra: Any) -> Bundle:
    """Realised vs assumed, in the shape `execution-cost-analysis` wants.

    Fills `left` (what you expected) and `right` (what the venue says happened) plus `on`,
    `by` and `tolerance`, which unlocks `safe_asof`; `close`/`other` (the two price series
    for the same fills) which unlocks `reconcile_sources`; and `broker`, which unlocks
    `paper_account_guard`. Aggregate P&L reconciles while every fill is wrong, so the
    comparison has to be per fill.
    """
    if not isinstance(rec, ExecutionRecord):
        raise TypeError("rec must be an ExecutionRecord")
    if not isinstance(expected_fills, pd.DataFrame):
        raise TypeError("expected_fills must be a DataFrame of the fills you assumed")
    slots: dict[str, Any] = {"left": expected_fills, "right": rec.fills, "on": on,
                             "by": by, "tolerance": tolerance, "broker": rec.venue}
    if "price" in expected_fills.columns and "price" in rec.fills.columns and len(rec.fills):
        idx = pd.DatetimeIndex(pd.to_datetime(rec.fills[on], utc=True, errors="coerce"))
        actual = pd.Series(rec.fills["price"].to_numpy(dtype=float), index=idx).sort_index()
        exp_idx = pd.DatetimeIndex(pd.to_datetime(expected_fills[on], utc=True,
                                                  errors="coerce"))
        expected = pd.Series(expected_fills["price"].to_numpy(dtype=float),
                             index=exp_idx).sort_index()
        slots["close"] = actual
        slots["other"] = expected
    slots.update(extra)
    return Bundle(**slots)


__all__ = ["ENV_CREDENTIALS", "ExecutionRecord", "FILL_COLUMNS", "LIBRARY", "LICENCE",
           "NotPaperRefusal", "POSITION_COLUMNS", "VENUE_LIBRARY", "VERIFIED_ON",
           "read_execution", "read_ohlcv", "to_bundle"]
