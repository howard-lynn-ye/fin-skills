"""Alpha Vantage - 25 requests a day and 100 bars a call, and the only free source in
this layer that can name the stocks that DIED.

Verified at alphavantage.co on 2026-09-10.

  * **The free tier is 25 requests a day.** alphavantage.co/support: "We are pleased to
    provide free stock API service covering the majority of our datasets for 25 API
    requests per day and unlimited API requests for verified open-source or educational
    projects." No per-minute number is published for the free key at all; the premium
    plans are quoted per minute (75/min at $49.99/month up to 1,200/min at $249.99/month)
    and say "No daily limits", so the two tiers are not the same SHAPE and pacing per
    minute tells a free user nothing about when they get cut off. Hence `PerDay(25)`.

  * **A free daily call returns 100 bars.** TIME_SERIES_DAILY: "compact returns only the
    latest 100 data points; full returns the full-length time series of 25+ years of
    historical data ... The 'compact' outputsize is available to both free and premium API
    keys. The 'full' outputsize is available to premium keys." A free key therefore cannot
    reach 25 years of history at any request rate, and asking for a window it cannot cover
    is refused here rather than silently answered with the last 100 sessions.

  * **The free daily series is UNADJUSTED.** TIME_SERIES_DAILY returns "raw (as-traded)
    daily time series"; TIME_SERIES_DAILY_ADJUSTED and TIME_SERIES_INTRADAY are premium.
    So `adjustment_supported` is `(RAW,)`: on a free key there is no adjusted daily series
    to ask for, and a backtest run on raw prices across a split is wrong by the split.

  * **LISTING_STATUS is free, dated, and includes the dead.** "This API returns a list of
    active or delisted US stocks and ETFs, either as of the latest trading day or at a
    specific time in history. The endpoint is positioned to facilitate equity research on
    asset lifecycle and survivorship." `date` accepts "Any YYYY-MM-DD date later than
    2010-01-01" and `state` accepts `active` (default) or `delisted`. That is a
    point-in-time MEMBERSHIP table, and it is the only one available for free anywhere in
    this layer - so `universe()` is implemented and `bars()` is still survivor-only.
    `includes_delisted` stays False because it is the axis for PRICE history: Alpha
    Vantage documents no daily series for a name that has stopped trading, and the rule
    here is the one akshare is held to - if it cannot be answered, it is False.

Two mechanical notes. The `alpha_vantage` client (3.0.0, MIT, released 2024-07-18) has no
`listing_status` method - `alpha_vantage.timeseries` and `alpha_vantage.fundamentaldata`
between them expose none - so `universe()` calls the documented REST endpoint directly.
And Alpha Vantage authenticates with the key as a QUERY PARAMETER
(`_ALPHA_VANTAGE_API_URL = "https://www.alphavantage.co/query?"` plus `&apikey={}`): the
transport is HTTPS, but a key in a URL is the kind that lands in a proxy log, a traceback
or a copied-and-pasted command. Nothing in this module ever puts a URL, a params mapping
or a key into a request dict or a Provenance.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from fin_skills.data import declare
from fin_skills.data.adapters import Base, library_version, require
from fin_skills.data.provenance import make as make_provenance
from fin_skills.data.ratelimit import PerDay
from fin_skills.data.schema import Adjustment, Bars

TERMS = "https://www.alphavantage.co/terms_of_service/"
QUERY_URL = "https://www.alphavantage.co/query"

#: rows a free key gets from outputsize=compact - the vendor's number, not a pace
COMPACT_ROWS = 100

#: the earliest date LISTING_STATUS accepts: "Any YYYY-MM-DD date later than 2010-01-01"
LISTING_STATUS_FLOOR = "2010-01-01"

#: this layer's interval -> the client method that serves it on a FREE key
METHODS = {"1d": "get_daily", "1wk": "get_weekly", "1mo": "get_monthly"}

_FIELDS = {"1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close",
           "5. volume": "volume", "open": "open", "high": "high", "low": "low",
           "close": "close", "volume": "volume"}

DECL = declare.Declaration(
    name="alphavantage",
    library="alpha-vantage",
    library_license="MIT",
    licence_source="pypi",            # info.license 'MIT' + classifier, 2026-09-10
    adjustment_default=Adjustment.RAW,        # TIME_SERIES_DAILY is "raw (as-traded)"
    adjustment_supported=(Adjustment.RAW,),   # DAILY_ADJUSTED is premium-only
    calendar="XNYS",
    tz="America/New_York",
    bar_label="close",
    interval_support=tuple(METHODS),
    includes_delisted=False,          # no price history for a dead name is documented;
                                      # LISTING_STATUS gives the universe, not the bars
    point_in_time=False,              # for prices. universe() IS as-of - see notes
    rate_limit=PerDay(25),
    free_tier=("free key: 25 API requests/day, 'and unlimited API requests for verified "
               "open-source or educational projects' (alphavantage.co/support, "
               "2026-09-10). No per-minute number is published for the free key. Each "
               "daily call returns outputsize=compact, 'only the latest 100 data points'; "
               "outputsize=full (25+ years), TIME_SERIES_DAILY_ADJUSTED and "
               "TIME_SERIES_INTRADAY are premium, from $49.99/month for 75 requests/min"),
    requires_key=True,
    key_env_var="ALPHAVANTAGE_API_KEY",
    key_sharing="byok-required",      # ToS 3: a "non-sublicensable, non-transferable,
                                      # non-assignable" licence effective when the User
                                      # clicks "Get Free API Key" - it is not yours to lend
    cache_policy="persist-ok",        # nothing in the ToS restricts retention
    non_display_use="unstated",
    terms_url=TERMS,
    redistribution="prohibited",      # ToS 2a: providing access to others is commercial use
    verified_on="2026-09-10",
    notes=("The free daily series is UNADJUSTED. TIME_SERIES_DAILY returns 'raw "
           "(as-traded)' prices and TIME_SERIES_DAILY_ADJUSTED is premium, so on a free "
           "key there is no adjusted daily series to ask for and every split is still a "
           "jump in the numbers.\n"
           f"outputsize=compact returns {COMPACT_ROWS} points and is the free ceiling; "
           "outputsize=full is premium. bars() refuses a window it cannot cover rather "
           "than returning the last 100 sessions as if they were the answer.\n"
           "LISTING_STATUS is free and is the only point-in-time delisted UNIVERSE in "
           "this layer: state=delisted, and any date after " + LISTING_STATUS_FLOOR + ". "
           "It fixes the membership half of survivorship; the price half stays broken, "
           "because the bars of a delisted name are not served.\n"
           "ToS 2a licenses the platform 'for personal, non-commercial use' and counts "
           "'any type of commercial activity that allows individuals or entities other "
           "than User to access information' as commercial use - that is redistribution.\n"
           "The key travels as a QUERY PARAMETER (&apikey=...), so it lands in URLs. "
           "Nothing in this module writes a URL or a params mapping into a request or a "
           "Provenance.\n"
           "alpha_vantage 3.0.0 was released 2024-07-18 and exposes no listing_status "
           "method, so universe() calls the documented REST endpoint directly."),
)


def normalise(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """One Alpha Vantage time-series frame -> the (field, ticker) frame `Bars` wants.

    Pure and offline. The client's `output_format='pandas'` frame carries the vendor's own
    numbered column names ("1. open"), which is the shape that changes between the JSON
    and CSV paths, so both spellings are accepted and anything else is a loud failure.
    """
    if raw is None or not len(raw):
        raise ValueError(f"alphavantage returned no rows for {symbol!r}; an empty frame "
                         f"is not a price series")
    df = pd.DataFrame(raw).copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None)
    df = df.rename(columns={c: _FIELDS[c] for c in df.columns if c in _FIELDS})
    missing = [f for f in ("open", "high", "low", "close") if f not in df.columns]
    if missing:
        raise ValueError(f"alphavantage frame for {symbol!r} is missing {missing}; the "
                         f"returned column names moved (columns: {list(raw.columns)})")
    keep = [f for f in ("open", "high", "low", "close", "volume") if f in df.columns]
    out = pd.DataFrame(index=df.index)
    for f in keep:
        out[(f, symbol)] = pd.to_numeric(df[f], errors="coerce").astype("float64")
    out.columns = pd.MultiIndex.from_tuples(list(out.columns), names=["field", "ticker"])
    out.index.name = "date"
    return out[~out.index.duplicated(keep="first")].sort_index(axis=1).sort_index()


def listing_to_frame(rows: pd.DataFrame, as_of: Any, state: str) -> pd.DataFrame:
    """A LISTING_STATUS CSV -> (ticker, start_date, end_date, reason, as_of, status).

    Pure and offline. The vendor's columns are symbol, name, exchange, assetType,
    ipoDate, delistingDate, status; `delistingDate` is the literal string "null" for a
    live name rather than an empty cell, which `pd.to_datetime` turns into NaT only
    because `errors="coerce"` is passed.
    """
    df = pd.DataFrame(rows).copy()
    for want in ("symbol", "status"):
        if want not in df.columns:
            raise ValueError(f"expected a LISTING_STATUS frame with a {want!r} column, "
                             f"got {list(df.columns)}")
    out = pd.DataFrame({
        "ticker": df["symbol"].astype(str),
        "name": df.get("name", pd.Series(index=df.index, dtype=object)),
        "exchange": df.get("exchange", pd.Series(index=df.index, dtype=object)),
        "asset_type": df.get("assetType", pd.Series(index=df.index, dtype=object)),
        "start_date": pd.to_datetime(df.get("ipoDate"), errors="coerce"),
        "end_date": pd.to_datetime(df.get("delistingDate"), errors="coerce"),
        "status": df["status"].astype(str),
    })
    out["reason"] = out["status"].map(lambda s: "delisted" if s.lower() == "delisted"
                                      else "")
    out["as_of"] = pd.Timestamp(as_of) if as_of is not None else pd.NaT
    out["requested_state"] = str(state)
    return out.sort_values("ticker").reset_index(drop=True)


class AlphaVantageAdapter(Base):
    decl = DECL

    def bars(self, symbols, start, end, *, interval: str = "1d",
             adjustment: Adjustment | None = None, outputsize: str = "compact",
             **kw) -> Bars:
        syms = [symbols] if isinstance(symbols, str) else list(symbols)
        adj = adjustment or self.decl.adjustment_default
        if adj is not Adjustment.RAW:
            raise ValueError(
                "alphavantage serves RAW only on a free key: TIME_SERIES_DAILY is 'raw "
                "(as-traded)' and TIME_SERIES_DAILY_ADJUSTED is a premium function. Fetch "
                "RAW and readjust from an actions table, or use a source that serves the "
                "convention you need.")
        if interval.lower() not in METHODS:
            raise ValueError(f"interval {interval!r} is not one of {list(METHODS)}; "
                             f"TIME_SERIES_INTRADAY is a premium function")
        if outputsize not in ("compact", "full"):
            raise ValueError("outputsize must be 'compact' or 'full'")

        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if outputsize == "compact" and interval.lower() == "1d" \
                and not covered_by_compact(s, e):
            raise ValueError(
                f"outputsize='compact' returns only the latest {COMPACT_ROWS} data "
                f"points, which cannot cover {_iso(s)}..{_iso(e)}. outputsize='full' is a "
                f"PREMIUM capability on this vendor, so a free key cannot answer this "
                f"request at all - it would return the last {COMPACT_ROWS} sessions and "
                f"they would look like the answer. Shorten the window or use another "
                f"source (python -m fin_skills.data advise).")

        request = {"method": "bars", "symbols": syms, "start": _iso(s), "end": _iso(e),
                   "interval": interval, "adjustment": adj.value,
                   "function": {"1d": "TIME_SERIES_DAILY", "1wk": "TIME_SERIES_WEEKLY",
                                "1mo": "TIME_SERIES_MONTHLY"}[interval.lower()],
                   "outputsize": outputsize,
                   # the vendor returns a whole series and this layer slices it, so the
                   # boundary belongs to this adapter and is recorded as such
                   "end_is_exclusive": False, "half_open": True}

        av = require("alpha_vantage", pip_name="alpha-vantage")
        ts_mod = require("alpha_vantage.timeseries", pip_name="alpha-vantage")
        # the client reads $ALPHAVANTAGE_API_KEY itself; credential() is called only so an
        # unset key fails with THIS library's message. No key is bound in this module.
        declare.credential(self.decl)
        client = ts_mod.TimeSeries(output_format="pandas")

        # only the daily function takes outputsize; weekly and monthly return the whole
        # series, so forwarding it would be a TypeError against the vendor's own signature
        extra = {"outputsize": outputsize} if interval.lower() == "1d" else {}
        parts = []
        for sym in syms:
            self.limiter.acquire()
            raw, _meta = getattr(client, METHODS[interval.lower()])(
                symbol=str(sym), **extra, **kw)
            parts.append(normalise(raw, str(sym)))

        frame = pd.concat(parts, axis=1).sort_index(axis=1)
        frame = frame[(frame.index >= s) & (frame.index < e)]      # [start, end)
        if not len(frame):
            raise ValueError(f"alphavantage returned no rows inside "
                             f"[{_iso(s)}, {_iso(e)}) for {syms}")
        prov = make_provenance("alphavantage",
                               library_version=library_version(av),
                               request=request, content=frame, terms_url=TERMS,
                               redistributable=False)
        return Bars(frame=frame, adjustment=adj, calendar=self.decl.calendar,
                    tz=self.decl.tz, interval=interval, bar_label="close",
                    currency="USD", provenance=prov, half_open=True)

    def universe(self, market: str, as_of, *,
                 include_delisted: bool = False) -> pd.DataFrame:
        """The LISTING_STATUS table as of a date - active, delisted, or both.

        This is the one point-in-time membership answer available for free in this layer.
        It does NOT make `includes_delisted` True: it names the dead, it does not price
        them.
        """
        if str(market).lower() not in ("us", "usa", "united-states", ""):
            raise self._unsupported(
                "universe", f"LISTING_STATUS covers 'active or delisted US stocks and "
                            f"ETFs'; {market!r} is not a market it lists")
        stamp = pd.Timestamp(as_of) if as_of is not None else None
        if stamp is not None and stamp <= pd.Timestamp(LISTING_STATUS_FLOOR):
            raise ValueError(
                f"LISTING_STATUS accepts 'Any YYYY-MM-DD date later than "
                f"{LISTING_STATUS_FLOOR}'; {_iso(stamp)} is not, so a universe as of that "
                f"date cannot be fetched - it is not a limitation this adapter can work "
                f"around by asking for the latest and filtering")

        requests_ = require("requests")
        key = declare.credential(self.decl)
        states = ("active", "delisted") if include_delisted else ("active",)
        parts = []
        for state in states:
            self.limiter.acquire()
            params = {"function": "LISTING_STATUS", "state": state, "apikey": key}
            if stamp is not None:
                params["date"] = _iso(stamp)
            resp = requests_.get(QUERY_URL, params=params, timeout=60)
            self.limiter.observe(getattr(resp, "headers", {}) or {})
            resp.raise_for_status()
            parts.append(listing_to_frame(_read_csv(resp.text), stamp, state))
        out = pd.concat(parts, ignore_index=True).sort_values("ticker")
        return out.reset_index(drop=True)


def covered_by_compact(start: Any, end: Any, *, sessions_per_week: float = 5.0) -> bool:
    """Can `outputsize='compact'` reach `start`? Pure, so the refusal above is testable.

    100 trading sessions is about 140 calendar days. The test is deliberately generous -
    it refuses only windows that certainly do not fit - because a false refusal would be
    this adapter inventing a limit, and the point is to carry the vendor's.
    """
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    span_days = (e - s).days
    return span_days <= COMPACT_ROWS * (7.0 / sessions_per_week)


def _read_csv(text: str) -> pd.DataFrame:
    from io import StringIO                                          # noqa: PLC0415
    return pd.read_csv(StringIO(text))


def _iso(value: Any) -> str | None:
    return None if value is None else str(pd.Timestamp(value).date())


declare.register(AlphaVantageAdapter, DECL)

__all__ = ["AlphaVantageAdapter", "COMPACT_ROWS", "DECL", "LISTING_STATUS_FLOOR",
           "METHODS", "QUERY_URL", "covered_by_compact", "listing_to_frame", "normalise"]
