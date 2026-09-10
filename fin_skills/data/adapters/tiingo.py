"""Tiingo - the only source here that returns raw prices, adjusted prices and the
adjustment factors in the SAME row, and the only one whose terms forbid a cache.

Everything below was re-verified at Tiingo's own pages on 2026-09-10.

  * **Both conventions arrive together, so nothing has to be reconstructed.** The
    end-of-day response carries `open/high/low/close/volume`, `adjOpen/adjHigh/adjLow/
    adjClose/adjVolume`, and `divCash` and `splitFactor` for that date
    (tiingo.com/documentation/end-of-day). That is exactly `Adjustment.RAW_PLUS_FACTORS`,
    and it is the reason this adapter's default does NOT rewrite history: a new dividend
    appends a factor row, it does not restate the raw quotes already on file. The adjusted
    columns follow "the standard method set forth by 'The Center for Research in Security
    Prices' (CRSP)" and are anchored at the present, so asking for
    `Adjustment.ANCHORED_PRESENT` gets you a series a cached copy will drift from.

  * **The terms forbid keeping it.** ToS 1.6(a), Starter and Trial plans: "You may not
    write, save, archive, back up, or otherwise retain Tiingo Data in any persistent or
    durable storage", and "You may process Tiingo Data only transiently in volatile memory
    or in a temporary, non-persistent cache". So `cache_policy="no-persist"` and
    `Cache.put()` REFUSES unless the caller passes `acknowledge_paid_tier=True`. Note that
    even the $30 Power and $50 Commercial plans are licensed "Internal Use Only"
    (tiingo.com/products/end-of-day-stock-price-data); redistribution is a separate plan
    at $250/month for startups and $500/month for enterprise, and ToS 7.3 requires the
    phrase "Data sourced by Tiingo" with a link on anything redistributed under it.

  * **The free quota is four counters on three clocks.** Starter, $0/month: 50 requests an
    hour, 1,000 a day, 500 UNIQUE SYMBOLS a month, 1 GB a month (tiingo.com/about/pricing).
    The symbol counter is the one that bites - nothing in a response tells you how close
    you are, and a 600-name universe cannot be fetched at all in one calendar month, at any
    request rate.

  * **A list of tickers collapses to one column.** In tiingo 0.16.1,
    `get_dataframe(tickers=[...])` raises `MissingRequiredArgumentError` unless
    `metric_name` is given, so a panel request returns ONE metric - not OHLCV. This adapter
    therefore loops one symbol per call, which is also what the 500-symbols-a-month counter
    is measured in.

Two things the vendor documents ambiguously, which are surfaced rather than guessed:
`splitFactor` is described only as "The factor used to adjust prices when a company
splits, reverse splits, or pays a distribution" with no statement of direction, so run
`fin_skills.api.check(close=..., actions=...)` before trusting `Bars.readjust()`; and the
`endDate` in the meta endpoint is "The latest date we have price data available for the
asset", which is a per-ticker hint and not a delisting archive - `includes_delisted` is
therefore False, by the same rule that makes it False for akshare.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from fin_skills.data import declare
from fin_skills.data.adapters import Base, library_version, require
from fin_skills.data.provenance import make as make_provenance
from fin_skills.data.ratelimit import PerHourDayMonth
from fin_skills.data.schema import Adjustment, Bars

TERMS = "https://app.tiingo.com/tos/"

#: the exact phrase ToS 7.3 requires on data redistributed under a redistribution plan.
#: It is NOT an attribution that makes the free tier redistributable - it is the condition
#: attached to the separate paid licence, so it lives here and not in `attribution`.
REDISTRIBUTION_PHRASE = "Data sourced by Tiingo"

#: this layer's interval -> the vendor's `resampleFreq`
FREQUENCY = {"1d": "daily", "1wk": "weekly", "1mo": "monthly", "1y": "annually"}

#: vendor column -> (field, is_adjusted)
_PRICE_COLUMNS = {
    "open": ("open", False), "high": ("high", False), "low": ("low", False),
    "close": ("close", False), "volume": ("volume", False),
    "adjOpen": ("open", True), "adjHigh": ("high", True), "adjLow": ("low", True),
    "adjClose": ("close", True), "adjVolume": ("volume", True),
}

DECL = declare.Declaration(
    name="tiingo",
    library="tiingo",
    library_license="MIT",
    licence_source="pypi",             # info.license 'MIT license' + classifier, 2026-09-10
    adjustment_default=Adjustment.RAW_PLUS_FACTORS,   # raw + adj + divCash + splitFactor
    adjustment_supported=(Adjustment.RAW, Adjustment.RAW_PLUS_FACTORS,
                          Adjustment.ANCHORED_PRESENT),
    calendar="XNYS",
    tz="America/New_York",
    bar_label="close",
    interval_support=tuple(FREQUENCY),
    includes_delisted=False,           # not documented as a delisted archive; the meta
                                       # endpoint's endDate is a per-ticker hint only
    point_in_time=False,
    rate_limit=PerHourDayMonth(50, 1_000, symbols_per_month=500,
                               bytes_per_month=1_000_000_000),
    free_tier=("Starter, $0/month, key required: 50 requests/hour, 1,000/day, 500 unique "
               "symbols/month, 1 GB/month (tiingo.com/about/pricing, 2026-09-10). "
               "Coverage 80,000+ US equities, ETFs, mutual funds and Chinese A-shares; "
               "history '60+ Years; Data going back from 1962'. Every plan including the "
               "$30 Power and $50 Commercial tiers is licensed 'Internal Use Only'"),
    requires_key=True,
    key_env_var="TIINGO_API_KEY",
    key_sharing="byok-required",       # ToS 1.1 licenses one copy to YOU, non-transferable
                                       # and non-sublicensable; a shared key cannot ship
    cache_policy="no-persist",         # ToS 1.6(a)
    non_display_use="unstated",
    terms_url=TERMS,
    redistribution="prohibited",       # ToS 7.3: only on request, for a fee
    verified_on="2026-09-10",
    notes=("ToS 1.6(a): 'You may not write, save, archive, back up, or otherwise retain "
           "Tiingo Data in any persistent or durable storage.' Cache.put() refuses "
           "without acknowledge_paid_tier=True, and even the paid tiers are Internal Use "
           "Only.\n"
           "ToS 7.3: 'Redistribution is only available upon special request and "
           "permission, and comes with additional fees' - the redistribution plan is "
           "$250/month for startups and $500/month for enterprise and requires the phrase "
           f"'{REDISTRIBUTION_PHRASE}' with a link to https://www.tiingo.com.\n"
           "The 500-unique-symbols-a-month counter is on SYMBOLS, not requests: a "
           "600-name universe cannot be fetched in one calendar month at any rate, and no "
           "response field tells you how close you are.\n"
           "tiingo 0.16.1 get_dataframe(tickers=[...]) raises MissingRequiredArgumentError "
           "without metric_name, so a list request returns ONE metric rather than OHLCV; "
           "this adapter loops one symbol per call instead.\n"
           "splitFactor is documented only as 'The factor used to adjust prices' with no "
           "direction stated, so the actions table it produces should be checked with "
           "adjustment_check before Bars.readjust() is trusted."),
)


def _wanted(adjustment: Adjustment) -> dict[str, str]:
    """Vendor column -> this layer's field name, for one convention."""
    want_adjusted = adjustment is Adjustment.ANCHORED_PRESENT
    return {src: field for src, (field, is_adj) in _PRICE_COLUMNS.items()
            if is_adj is want_adjusted}


def normalise(raw: pd.DataFrame, symbol: str, adjustment: Adjustment) -> pd.DataFrame:
    """One ticker's end-of-day frame -> the (field, ticker) frame `Bars` wants.

    Pure and offline. The vendor returns raw and adjusted columns side by side, so which
    set is selected is a decision this function makes explicitly rather than a default it
    inherits.
    """
    if raw is None or not len(raw):
        raise ValueError(f"tiingo returned no rows for {symbol!r}; an empty frame is not "
                         f"a price series")
    df = pd.DataFrame(raw).copy()
    if "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"]))
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None)

    picks = _wanted(adjustment)
    missing = [c for c in picks if c not in df.columns]
    if missing:
        raise ValueError(f"tiingo frame for {symbol!r} is missing {missing}; the "
                         f"end-of-day payload carries raw and adjusted columns together "
                         f"(columns: {list(df.columns)})")
    out = pd.DataFrame(index=df.index)
    for src, field in picks.items():
        values = pd.to_numeric(df[src], errors="coerce")
        out[(field, symbol)] = values.astype("float64")
    out.columns = pd.MultiIndex.from_tuples(list(out.columns), names=["field", "ticker"])
    out.index.name = "date"
    return out[~out.index.duplicated(keep="first")].sort_index(axis=1).sort_index()


def actions_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """`splitFactor` and `divCash` -> the (date, ratio, kind, ticker) table every guard reads.

    `ratio` is this layer's multiplicative PRICE factor of the event. For a dividend it is
    computable from the row before the ex-date - prev_close / (prev_close - div) - and it
    is computed here rather than left NaN, because the raw close is in the same payload.
    For a split it is the vendor's `splitFactor` VERBATIM: Tiingo documents that field only
    as "The factor used to adjust prices" and never states its direction, so passing it
    through unchanged keeps the ambiguity visible instead of encoding a guess.
    """
    cols = ["date", "ratio", "kind", "ticker"]
    if raw is None or not len(raw):
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(raw).copy()
    if "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"]))
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None)
    df = df.sort_index()

    close = pd.to_numeric(df.get("close", pd.Series(index=df.index, dtype=float)),
                          errors="coerce")
    prev = close.shift(1)
    rows = []
    for ts, row in df.iterrows():
        split = float(pd.to_numeric(row.get("splitFactor", 1.0), errors="coerce") or 1.0)
        if np.isfinite(split) and not np.isclose(split, 1.0):
            rows.append({"date": ts, "ratio": split, "kind": "split", "ticker": symbol})
        div = float(pd.to_numeric(row.get("divCash", 0.0), errors="coerce") or 0.0)
        if div:
            p = float(prev.get(ts, np.nan))
            ratio = p / (p - div) if np.isfinite(p) and (p - div) > 0 else np.nan
            rows.append({"date": ts, "ratio": ratio, "kind": "dividend",
                         "ticker": symbol})
    return pd.DataFrame(rows, columns=cols)


class TiingoAdapter(Base):
    decl = DECL

    def bars(self, symbols, start, end, *, interval: str = "1d",
             adjustment: Adjustment | None = None, **kw) -> Bars:
        syms = [symbols] if isinstance(symbols, str) else list(symbols)
        adj = adjustment or self.decl.adjustment_default
        if adj not in self.decl.adjustment_supported:
            raise ValueError(
                f"tiingo serves {[a.value for a in self.decl.adjustment_supported]}; "
                f"{adj.value} is not one of them. The payload carries raw prices, "
                f"adjusted prices and the factors together, so ask for "
                f"'{Adjustment.RAW_PLUS_FACTORS.value}' and readjust locally.")
        if interval.lower() not in FREQUENCY:
            raise ValueError(f"interval {interval!r} is not one of {list(FREQUENCY)}; "
                             f"Tiingo resamples end-of-day with resampleFreq")

        s, e = pd.Timestamp(start), pd.Timestamp(end)
        request = {"method": "bars", "symbols": syms, "start": _iso(s), "end": _iso(e),
                   "interval": interval, "adjustment": adj.value,
                   "resampleFreq": FREQUENCY[interval.lower()],
                   # the vendor's endDate is INCLUSIVE and this layer's contract is not,
                   # so the final day is trimmed here and the decision is written down
                   "end_is_exclusive": False, "half_open": True,
                   "unique_symbols_charged": len(syms)}

        tiingo = require("tiingo")
        # The client reads $TIINGO_API_KEY itself. credential() is called only so an
        # unset key fails with THIS library's message; the value is never bound here,
        # never stored on the adapter and never written into a Provenance.
        declare.credential(self.decl)
        client = tiingo.TiingoClient()

        parts, acts = [], []
        for sym in syms:
            # the monthly quota counts SYMBOLS, so it is charged per symbol, not per call
            self.limiter.acquire(symbols=(str(sym),))
            raw = client.get_dataframe(str(sym), startDate=_iso(s), endDate=_iso(e),
                                       frequency=FREQUENCY[interval.lower()], **kw)
            parts.append(normalise(raw, str(sym), adj))
            acts.append(actions_frame(raw, str(sym)))

        frame = pd.concat(parts, axis=1).sort_index(axis=1)
        frame = frame[frame.index < e]                     # [start, end), trimmed here
        actions = pd.concat(acts, ignore_index=True) if acts else None
        prov = make_provenance("tiingo", library_version=library_version(tiingo),
                               request=request, content=frame, terms_url=TERMS,
                               redistributable=False)
        return Bars(frame=frame, adjustment=adj, calendar=self.decl.calendar,
                    tz=self.decl.tz, interval=interval, bar_label="close",
                    currency="USD", provenance=prov,
                    actions=actions if actions is not None and len(actions) else None,
                    half_open=True)

    def actions(self, symbols, start, end) -> pd.DataFrame:
        bars = self.bars(symbols, start, end, adjustment=Adjustment.RAW_PLUS_FACTORS)
        acts = bars.actions
        if acts is None or not len(acts):
            return pd.DataFrame(columns=["date", "ratio", "kind", "ticker"])
        return acts.reset_index(drop=True)

    def universe(self, market: str, as_of, *,
                 include_delisted: bool = False) -> pd.DataFrame:
        raise self._unsupported(
            "universe",
            "TiingoClient.list_tickers() downloads supported_tickers.zip with a bare "
            "requests.get outside the client's session, headers and rate limiter, and "
            "Tiingo documents that file as 'current coverage as well as symbol "
            "reservations that we have preserved as we expand our data' - reservations "
            "are planned coverage, so it cannot answer what was listed on a past date")


def _iso(value: Any) -> str | None:
    return None if value is None else str(pd.Timestamp(value).date())


declare.register(TiingoAdapter, DECL)

__all__ = ["DECL", "FREQUENCY", "REDISTRIBUTION_PHRASE", "TiingoAdapter",
           "actions_frame", "normalise"]
