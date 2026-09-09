"""yfinance - the default free equity source, and the one whose defaults move the most.

What this adapter pins, and why each pin exists (verified 2026-09-09):

  * `auto_adjust` is stated, never inherited. It has defaulted to True since 1.0, so the
    series you get by asking for nothing is FORWARD-adjusted - anchored at the present, so
    every new dividend rewrites the whole history and the same query next month returns
    different numbers.
  * `progress=False`, `ignore_tz` and the frame shape are pinned, because the returned
    column layout has changed between releases and a silently reshaped frame becomes a
    silently misaligned panel.
  * **The adapter owns its own backoff.** `retries` defaults to 0 and 429 is excluded from
    yfinance's transient-retry path, so nothing under this call will wait for you.
  * **No session is ever passed.** yfinance now RAISES on a caching session
    (`YFDataException: Caching sessions (e.g. requests_cache) are not supported`), which is
    why this layer's cache sits above the client rather than inside the transport.
  * Sub-daily intervals are refused without an explicit `tz`, because 1.4.0 moved intraday
    stamps between UTC and exchange-local and an unstated zone silently shifts every bar.
  * yfinance's `end` is EXCLUSIVE, which happens to be this layer's half-open contract -
    so it passes through unchanged and the request records that it did.

There is no published rate limit anywhere in yfinance or Yahoo's documentation - zero hits
for 429, limiter or throttl in the whole documentation index - so `rate_limit` is
`Unpublished()` and this module contains no rate constant of its own.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from fin_skills.data import declare
from fin_skills.data.adapters import Base, library_version, require
from fin_skills.data.provenance import make as make_provenance
from fin_skills.data.ratelimit import Unpublished
from fin_skills.data.schema import Adjustment, Bars, stack_fields

TERMS = "https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html"

DECL = declare.Declaration(
    name="yfinance",
    library="yfinance",
    library_license="Apache-2.0",
    licence_source="pypi",                      # info.license == 'Apache-2.0' + classifier
    adjustment_default=Adjustment.FORWARD,      # auto_adjust=True since 1.0
    adjustment_supported=(Adjustment.RAW, Adjustment.FORWARD),
    calendar="XNYS",
    tz="America/New_York",
    bar_label="close",
    interval_support=("1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d", "5d", "1wk",
                      "1mo", "3mo"),
    includes_delisted=False,                    # Yahoo returns today's names
    point_in_time=False,
    rate_limit=Unpublished(),
    free_tier="free, no key; no numeric limit is published anywhere (verified 2026-09-09)",
    requires_key=False,
    key_env_var="",
    key_sharing="unstated",
    cache_policy="persist-ok",
    non_display_use="unstated",
    terms_url=TERMS,
    redistribution="prohibited",
    verified_on="2026-09-09",
    notes=("Apache-2.0 is the CODE licence; Yahoo's own terms describe the data as "
           "intended for personal use, so installing this adapter grants nothing.\n"
           "includes_delisted=False is not a limitation of this adapter - Yahoo returns "
           "today's names, so any universe built here is survivor-only and every result "
           "from it is an upper bound.\n"
           "retries defaults to 0 and 429 is excluded from the transient-retry path, so "
           "backoff is this adapter's job, not the library's."),
)


def _normalise(raw: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Whatever shape yfinance returned -> the (field, ticker) frame `Bars` wants.

    Pure and offline, so the reshaping can be tested without a network: yfinance has
    returned a flat frame for one symbol, a (field, ticker) MultiIndex for many, and a
    (ticker, field) MultiIndex under `group_by='ticker'`, in different releases.
    """
    if raw is None or not len(raw):
        raise ValueError("yfinance returned no rows; an empty frame is not a price series")
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = {str(c).lower() for c in df.columns.get_level_values(0)}
        if not (lvl0 & {"open", "high", "low", "close", "adj close", "volume"}):
            df.columns = df.columns.swaplevel(0, 1)          # (ticker, field) layout
        df.columns = pd.MultiIndex.from_tuples(
            [(str(a).lower().replace(" ", "_"), str(b)) for a, b in df.columns])
    else:
        if len(symbols) != 1:
            raise ValueError(f"a flat frame came back for {len(symbols)} symbols; the "
                             f"column layout cannot be attributed to a ticker")
        t = str(symbols[0])
        df.columns = pd.MultiIndex.from_tuples(
            [(str(c).lower().replace(" ", "_"), t) for c in df.columns])
    df.columns = df.columns.set_names(["field", "ticker"])
    keep = [c for c in df.columns if c[0] in ("open", "high", "low", "close", "volume")]
    df = df[keep].sort_index(axis=1)
    for col in df.columns:
        if col[0] == "volume":
            v = pd.to_numeric(df[col], errors="coerce")
            # int64, not float32: 2**24 is 16,777,216, below any large-cap's daily volume
            df[col] = v.fillna(0).round().astype("int64") if v.notna().all() else v
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df.sort_index()


def _actions_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """yfinance's actions frame -> (date, ratio, kind), the shape every guard reads."""
    rows = []
    if raw is None or not len(raw):
        return pd.DataFrame(columns=["date", "ratio", "kind"])
    for ts, r in raw.iterrows():
        split = float(r.get("Stock Splits", 0.0) or 0.0)
        if split:
            rows.append({"date": pd.Timestamp(ts), "ratio": split, "kind": "split"})
        div = float(r.get("Dividends", 0.0) or 0.0)
        if div:
            rows.append({"date": pd.Timestamp(ts), "ratio": np.nan, "kind": "dividend",
                         "amount": div})
    return pd.DataFrame(rows, columns=["date", "ratio", "kind", "amount"])


class YFinanceAdapter(Base):
    decl = DECL

    def __init__(self, **client_kw: Any) -> None:
        if "session" in client_kw:
            raise TypeError(
                "yfinance raises on a caching session (YFDataException: 'Caching sessions "
                "(e.g. requests_cache) are not supported'), so this adapter never passes "
                "one. Use fin_skills.data.Cache, which caches the normalised RESULT above "
                "the client instead of the HTTP response inside it.")
        super().__init__(**client_kw)

    def bars(self, symbols, start, end, *, interval: str = "1d",
             adjustment: Adjustment | None = None, tz: str | None = None,
             max_attempts: int = 5, **kw) -> Bars:
        # arguments are checked BEFORE the vendor import, so a bad interval is a bad
        # interval rather than "yfinance is not installed"
        syms = [symbols] if isinstance(symbols, str) else list(symbols)
        adj = adjustment or self.decl.adjustment_default
        if adj not in self.decl.adjustment_supported:
            raise ValueError(f"yfinance serves {[a.value for a in self.decl.adjustment_supported]}; "
                             f"for {adj.value} fetch RAW plus actions and Bars.readjust()")
        if interval.lower() not in self.decl.interval_support:
            raise ValueError(f"interval {interval!r} is not one of "
                             f"{list(self.decl.interval_support)}")
        if not interval.lower().endswith(("d", "wk", "mo")) and tz is None:
            raise ValueError(
                "a sub-daily interval needs an explicit tz: yfinance 1.4.0 moved intraday "
                "timestamps between UTC and exchange-local, so an unstated zone silently "
                "shifts every bar. Pass tz='America/New_York' (or the venue's zone).")

        request = {"method": "bars", "symbols": syms, "start": _iso(start),
                   "end": _iso(end), "interval": interval,
                   "adjustment": adj.value, "auto_adjust": adj is Adjustment.FORWARD,
                   "end_is_exclusive": True, "half_open": True,
                   "actions": True, "progress": False, "ignore_tz": True}

        yf = require("yfinance")
        raw = None
        for attempt in range(1, int(max_attempts) + 1):
            self.limiter.acquire()
            try:
                raw = yf.download(syms, start=start, end=end, interval=interval,
                                  auto_adjust=adj is Adjustment.FORWARD, actions=True,
                                  progress=False, ignore_tz=True, threads=False, **kw)
                break
            except Exception as exc:                     # noqa: BLE001 - vendor-specific
                if not _is_rate_limited(exc) or attempt == int(max_attempts):
                    raise
                self.limiter.backoff(attempt)

        frame = _normalise(raw, syms)
        actions = _actions_frame(raw) if isinstance(raw, pd.DataFrame) else None
        prov = make_provenance("yfinance", library_version=library_version(yf),
                               request=request, content=frame, terms_url=TERMS,
                               redistributable=False)
        return Bars(frame=frame, adjustment=adj, calendar=self.decl.calendar,
                    tz=tz or self.decl.tz, interval=interval, bar_label="close",
                    currency="USD", provenance=prov, actions=actions, half_open=True)

    def actions(self, symbols, start, end) -> pd.DataFrame:
        yf = require("yfinance")
        syms = [symbols] if isinstance(symbols, str) else list(symbols)
        out = []
        for t in syms:
            self.limiter.acquire()
            a = _actions_frame(yf.Ticker(t).actions)
            a["ticker"] = t
            out.append(a)
        acts = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
            columns=["date", "ratio", "kind", "amount", "ticker"])
        if len(acts):
            acts = acts[(acts["date"] >= pd.Timestamp(start))
                        & (acts["date"] < pd.Timestamp(end))]
        return acts.reset_index(drop=True)


def _iso(value: Any) -> str | None:
    return None if value is None else str(pd.Timestamp(value).date())


def _is_rate_limited(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


declare.register(YFinanceAdapter, DECL)

__all__ = ["DECL", "YFinanceAdapter"]
