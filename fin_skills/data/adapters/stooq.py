"""Stooq - no key, no account, sixty years of daily bars, and a bot wall in front of it.

This adapter exists to test a claim the rest of the ecosystem repeats: that stooq is the
honest keyless default for daily equity bars. Two of the three parts hold and the third
does not, all checked on 2026-09-10.

  * **No key, no account, and the URL is documented by the site's own form.** The CSV
    endpoint is `https://stooq.com/q/d/l/?s=<symbol>&f=YYYYMMDD&t=YYYYMMDD&i=d`, read off
    the download link on stooq.com/q/d/; `i` is one of d, w, m, q, y and `t` is INCLUSIVE.

  * **The history is real and long.** AAPL.US on stooq.com/q/d/ has daily bars in 1984,
    and the whole /db/h/ bulk archive - World 184 MB, U.S. 514 MB, U.K., Japan, Hong Kong,
    Poland, Hungary and macro, daily / hourly / 5-minute - downloads without a login.

  * 🔴 **The endpoint no longer answers a script.** Every stooq.com and stooq.pl URL tried
    on 2026-09-10 - the CSV endpoint, the quote page, /db/h/ - returned HTTP **200** with
    an identical 796-byte HTML body carrying a JavaScript proof-of-work challenge:
    "This site requires JavaScript to verify your browser." A 200 with an HTML body is the
    worst possible failure shape, because `pd.read_csv(url)` does not raise on it; it
    parses the challenge markup and hands back a frame. `fetch()` below therefore checks
    the body BEFORE parsing and raises, and this adapter never returns a frame it did not
    recognise as CSV.

  * 🔴 **And there is no client library any more.** `pandas_datareader.stooq` was the
    ecosystem's reader; pandas-datareader 0.11.0 and 0.11.1 (2026-06-23 and 2026-06-24,
    the first releases since 0.10.0 of 2021-07-13) narrowed the package to "macroeconomic,
    policy, and factor-style data sources such as FRED, Fama/French, Bank of Canada, World
    Bank, OECD, Eurostat". Verified on the installed 0.11.1: the shipped modules are
    bankofcanada, econdb, eurostat, famafrench, fred, macro, oecd and wb, and
    `DataReader(..., 'stooq')` raises `NotImplementedError: data_source='stooq' is not
    implemented`. So the declared library here is `requests`, and the adapter owns the
    request.

This module solves no challenge and ships no workaround. The value it adds is that a
blocked fetch fails loudly with the reason, instead of becoming a DataFrame of HTML.

The default series is adjusted and anchored at the PRESENT: stooq's AAPL.US close for
29 Dec 1989 reads 0.263998, which is today's split-and-dividend-adjusted number, and the
quote page's own "Skip" checkboxes (`o_s` splits, `o_d` dividends, `o_p` preemptive
rights, `o_n` prepurchase rights, `o_o` preaccession rights, `o_m` denominations,
`o_x` others) are how the raw series is requested instead. Every one of those toggles is
off by default, so the series you get by asking for nothing is rewritten by every new
corporate action.
"""
from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd

from fin_skills.data import declare
from fin_skills.data.adapters import Base, library_version, require
from fin_skills.data.provenance import make as make_provenance
from fin_skills.data.ratelimit import Unpublished
from fin_skills.data.schema import Adjustment, Bars

TERMS = "https://stooq.com/terms.html"
CSV_URL = "https://stooq.com/q/d/l/"

#: this layer's interval -> stooq's `i` parameter, read off the form on stooq.com/q/d/
INTERVALS = {"1d": "d", "1wk": "w", "1mo": "m", "3mo": "q", "1y": "y"}

#: the "Skip" toggles on the same form. All seven set to 1 is stooq's unadjusted series.
SKIP_ADJUSTMENTS = ("o_s", "o_d", "o_p", "o_n", "o_o", "o_m", "o_x")

#: substrings that identify a body which is NOT the CSV that was asked for. The first is
#: the JavaScript proof-of-work interstitial served to every scripted client on
#: 2026-09-10; the rest are stooq's own plain-text refusals.
NOT_CSV = ("requires JavaScript to verify your browser", "<!DOCTYPE html", "<html",
           "Exceeded the daily hits limit", "No data")

DECL = declare.Declaration(
    name="stooq",
    library="requests",
    library_license="Apache-2.0",     # info.license 'Apache-2.0' + classifier, 2026-09-10
    licence_source="pypi",
    adjustment_default=Adjustment.ANCHORED_PRESENT,   # every Skip toggle defaults to off
    adjustment_supported=(Adjustment.RAW, Adjustment.ANCHORED_PRESENT),
    calendar="XNYS",                  # the .us suffix; bars() requires an explicit
                                      # calendar for any other market
    tz="America/New_York",
    bar_label="close",
    interval_support=tuple(INTERVALS),
    includes_delisted=False,
    point_in_time=False,
    rate_limit=Unpublished(),
    free_tier=("free, NO KEY and no account - the only source in this layer that needs "
               "neither. No rate limit is published anywhere on the site. VERIFIED "
               "BLOCKED for scripted clients on 2026-09-10: stooq.com and stooq.pl both "
               "answer every URL with HTTP 200 and a 796-byte JavaScript proof-of-work "
               "page instead of the data, so the keyless default is currently keyless and "
               "unreachable"),
    requires_key=False,
    key_env_var="",
    key_sharing="unstated",
    cache_policy="persist-ok",        # the site's own /db/h/ archive is files you save
    non_display_use="unstated",
    terms_url=TERMS,
    redistribution="prohibited",      # terms 5.3
    verified_on="2026-09-10",
    notes=("Terms 5.3: 'Redistribution of data found on the website is not allowed "
           "without the consent of Stooq.' The /db/h/ download page adds 'This data is "
           "intended solely for personal use. Any commercial use is prohibited.'\n"
           "Terms 6.1 and 6.2 carry third-party licences ON TOP of that: S&P Dow Jones "
           "index data 'may be used only for your own personal, non-commercial purposes', "
           "and London Metal Exchange data may not be redistributed or sold. A symbol's "
           "licence is therefore not uniform across the site.\n"
           "Terms 5.2: 'Continuous access to the website cannot be guaranteed.' As of "
           "2026-09-10 it is not available to a script at all - a JavaScript proof-of-work "
           "interstitial answers with HTTP 200, which means pd.read_csv(url) parses the "
           "challenge instead of raising. fetch() checks the body before parsing.\n"
           "pandas-datareader 0.11.x removed the stooq reader; DataReader(..., 'stooq') "
           "raises NotImplementedError on 0.11.1, so there is no client library left and "
           "this adapter owns the HTTP request.\n"
           "The default series is adjusted and anchored at the PRESENT, so a cached copy "
           "drifts from a live pull after every dividend. RAW is requested by setting all "
           "seven Skip toggles (o_s, o_d, o_p, o_n, o_o, o_m, o_x).\n"
           "Only the .us suffix is declared as XNYS. bars() requires an explicit calendar "
           "and tz for any other suffix rather than trading a Warsaw name on a New York "
           "session list."),
)


class StooqBlocked(RuntimeError):
    """The endpoint answered with something that is not the CSV that was requested."""


def url_for(symbol: str, start: Any, end: Any, *, interval: str = "1d",
            adjustment: Adjustment = Adjustment.ANCHORED_PRESENT) -> tuple[str, dict]:
    """The documented CSV URL and its query, built from the site's own form parameters.

    Pure, so the URL this adapter would issue can be asserted without a socket.
    """
    if interval.lower() not in INTERVALS:
        raise ValueError(f"interval {interval!r} is not one of {list(INTERVALS)}")
    params: dict[str, Any] = {"s": str(symbol).lower(), "i": INTERVALS[interval.lower()]}
    if start is not None:
        params["f"] = pd.Timestamp(start).strftime("%Y%m%d")
    if end is not None:
        # stooq's `t` is INCLUSIVE; this layer is half-open, so the frame is trimmed after
        # parsing rather than by shifting the boundary, which would hide the difference
        params["t"] = pd.Timestamp(end).strftime("%Y%m%d")
    if adjustment is Adjustment.RAW:
        params.update({k: 1 for k in SKIP_ADJUSTMENTS})
    return CSV_URL, params


def looks_like_csv(text: str) -> bool:
    """True only for a body that is actually the daily CSV stooq documents."""
    if not text:
        return False
    head = text.lstrip()[:400]
    if any(marker in head for marker in NOT_CSV):
        return False
    return head.split("\n", 1)[0].strip().lower().startswith("date,")


def parse_csv(text: str, symbol: str) -> pd.DataFrame:
    """A stooq daily CSV -> the (field, ticker) frame `Bars` wants.

    Pure and offline, and it refuses a body that is not CSV. That refusal is the whole
    point of this module: the interstitial arrives with HTTP 200, so a parser that trusts
    the status code turns a block into data.
    """
    if not looks_like_csv(text):
        head = (text or "").strip().replace("\n", " ")[:160]
        raise StooqBlocked(
            f"stooq did not return CSV for {symbol!r} - the body starts {head!r}. On "
            f"2026-09-10 every stooq.com and stooq.pl URL answered HTTP 200 with a "
            f"796-byte JavaScript proof-of-work page ('This site requires JavaScript to "
            f"verify your browser'), which pd.read_csv() would have parsed into a frame. "
            f"There is no workaround in this library; use a source that serves scripts "
            f"(python -m fin_skills.data advise).")
    df = pd.read_csv(StringIO(text))
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in ("date", "open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"stooq CSV for {symbol!r} is missing {missing} "
                         f"(columns: {list(df.columns)})")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["date"])).tz_localize(None)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    out = pd.DataFrame(index=df.index)
    for f in keep:
        out[(f, symbol)] = pd.to_numeric(df[f], errors="coerce").astype("float64")
    out.columns = pd.MultiIndex.from_tuples(list(out.columns), names=["field", "ticker"])
    out.index.name = "date"
    return out[~out.index.duplicated(keep="first")].sort_index(axis=1).sort_index()


class StooqAdapter(Base):
    decl = DECL

    def bars(self, symbols, start, end, *, interval: str = "1d",
             adjustment: Adjustment | None = None, calendar: str | None = None,
             tz: str | None = None, timeout: float = 60.0, **kw) -> Bars:
        syms = [symbols] if isinstance(symbols, str) else list(symbols)
        adj = adjustment or self.decl.adjustment_default
        if adj not in self.decl.adjustment_supported:
            raise ValueError(
                f"stooq serves {[a.value for a in self.decl.adjustment_supported]}; "
                f"{adj.value} is not one of them")
        if interval.lower() not in INTERVALS:
            raise ValueError(f"interval {interval!r} is not one of {list(INTERVALS)}")
        non_us = [s for s in syms if not str(s).lower().endswith(".us")]
        if non_us and (calendar is None or tz is None):
            raise ValueError(
                f"{non_us} are not .us symbols, and this adapter declares XNYS / "
                f"America/New_York only for the .us suffix. Pass calendar= and tz= for "
                f"the venue you are actually reading, or a Warsaw name gets aligned onto "
                f"a New York session list without anything failing.")

        s, e = pd.Timestamp(start), pd.Timestamp(end)
        request = {"method": "bars", "symbols": syms, "start": _iso(s), "end": _iso(e),
                   "interval": interval, "adjustment": adj.value,
                   "i": INTERVALS[interval.lower()],
                   "skip_adjustments": adj is Adjustment.RAW,
                   # stooq's `t` is inclusive and this layer's contract is not
                   "end_is_exclusive": False, "half_open": True}

        requests_ = require("requests")
        parts = []
        for sym in syms:
            self.limiter.acquire()
            url, params = url_for(sym, s, e, interval=interval, adjustment=adj)
            resp = requests_.get(url, params=params, timeout=timeout, **kw)
            self.limiter.observe(getattr(resp, "headers", {}) or {})
            resp.raise_for_status()
            parts.append(parse_csv(resp.text, str(sym)))

        frame = pd.concat(parts, axis=1).sort_index(axis=1)
        frame = frame[frame.index < e]                     # [start, end), trimmed here
        prov = make_provenance("stooq", library_version=library_version(requests_),
                               request=request, content=frame, terms_url=TERMS,
                               redistributable=False)
        return Bars(frame=frame, adjustment=adj, calendar=calendar or self.decl.calendar,
                    tz=tz or self.decl.tz, interval=interval, bar_label="close",
                    currency="USD", provenance=prov, half_open=True)


def _iso(value: Any) -> str | None:
    return None if value is None else str(pd.Timestamp(value).date())


declare.register(StooqAdapter, DECL)

__all__ = ["CSV_URL", "DECL", "INTERVALS", "NOT_CSV", "SKIP_ADJUSTMENTS", "StooqAdapter",
           "StooqBlocked", "looks_like_csv", "parse_csv", "url_for"]
