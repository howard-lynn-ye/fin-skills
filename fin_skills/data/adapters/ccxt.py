"""ccxt - one exchange instance per venue, held for the lifetime of the adapter.

The trap here is structural, not numeric. ccxt's limiter is a leaky bucket that lives on
the exchange INSTANCE: `enableRateLimit` is True by default, `rateLimit` is milliseconds
per unit of endpoint cost, and constructing a fresh `ccxt.binance()` inside a loop creates
a brand-new empty bucket every iteration and sends at full speed. That is how people get
banned. This adapter therefore keeps one instance per venue and reuses it, and the
`PerInstanceDelay` limiter mirrors that instance rather than the process.

Three more things it does rather than trusting a default (verified 2026-09-09):

  * **Drops the unclosed final bar.** ccxt's own documentation: the information from the
    last (current) candle may be incomplete until the candle is closed. A backtest that
    keeps it trades on a partial bar.
  * **Paginates explicitly.** `paginate=True` is documented as experimental and "might
    produce unexpected/incorrect results", so the loop is here where it can be read.
  * **`since` is milliseconds.** Passing seconds returns data from 1970 without an error.

Crypto has no corporate actions, so `adjustment` is RAW and there is nothing to readjust;
the calendar is 24/7 and `periods_per_year` is 365, not 252.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from fin_skills.data import declare
from fin_skills.data.adapters import Base, library_version, require
from fin_skills.data.provenance import make as make_provenance
from fin_skills.data.ratelimit import PerInstanceDelay
from fin_skills.data.schema import Adjustment, Bars

TERMS = "https://github.com/ccxt/ccxt#licence"

#: ccxt timeframe -> this layer's interval vocabulary
TIMEFRAMES = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}

DECL = declare.Declaration(
    name="ccxt",
    library="ccxt",
    library_license="MIT",
    licence_source="pypi",                 # license_expression == 'MIT' (info.license null)
    adjustment_default=Adjustment.RAW,     # no corporate actions exist in crypto
    adjustment_supported=(Adjustment.RAW,),
    calendar="24/7",
    tz="UTC",
    bar_label="open",                      # an OHLCV candle is stamped at its OPEN
    interval_support=tuple(TIMEFRAMES),
    includes_delisted=False,               # a delisted pair simply stops being listed
    point_in_time=False,
    rate_limit=PerInstanceDelay(),         # ccxt's own default, per exchange instance
    free_tier="per venue; enableRateLimit=True by default, rateLimit in ms x endpoint cost",
    requires_key=False,                    # public OHLCV needs no key
    key_env_var="",
    key_sharing="unstated",
    cache_policy="persist-ok",
    non_display_use="unstated",
    terms_url=TERMS,
    redistribution="prohibited",
    verified_on="2026-09-09",
    notes=("The limiter is PER EXCHANGE INSTANCE. This adapter holds one instance per "
           "venue for its own lifetime; a fresh ccxt.<venue>() per call resets the bucket "
           "and is how accounts get banned.\n"
           "The last candle may be incomplete until it closes, so it is dropped. "
           "paginate=True is flagged experimental upstream, so pagination is explicit "
           "here. since is in MILLISECONDS.\n"
           "Rate limits are the VENUE's, not ccxt's: the declared shape is the ms-delay "
           "the instance enforces, and each venue publishes its own ceiling."),
)


def _to_frame(rows: Sequence[Sequence[float]], symbol: str) -> pd.DataFrame:
    """ccxt's [ms, o, h, l, c, v] rows -> the (field, ticker) frame `Bars` wants."""
    if not len(rows):
        raise ValueError(f"ccxt returned no candles for {symbol!r}")
    arr = np.asarray(rows, dtype="float64")
    # int64 epochs: float64's step is 256 ns at 2024 dates, so ms must not stay float
    idx = pd.to_datetime(np.asarray([r[0] for r in rows], dtype="int64"), unit="ms",
                         utc=True)
    out = {(f, symbol): arr[:, i].astype("float64")
           for i, f in enumerate(("open", "high", "low", "close"), start=1)}
    vol = arr[:, 5]
    out[("volume", symbol)] = vol.astype("float64")
    frame = pd.DataFrame(out, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(list(frame.columns),
                                              names=["field", "ticker"])
    frame.index.name = "open_time"
    return frame[~frame.index.duplicated(keep="first")].sort_index()


def drop_unclosed(frame: pd.DataFrame, interval: str, now: pd.Timestamp) -> pd.DataFrame:
    """Remove a final candle whose interval has not finished.

    Separated out and pure so the rule can be tested against a fixed clock rather than
    against whatever the exchange happened to be doing.
    """
    if not len(frame):
        return frame
    step = pd.Timedelta(interval.replace("m", "min") if interval.endswith("m")
                        else interval)
    last = frame.index[-1]
    return frame.iloc[:-1] if last + step > pd.Timestamp(now).tz_convert("UTC") else frame


class CcxtAdapter(Base):
    decl = DECL

    def __init__(self, venue: str = "binance", **client_kw: Any) -> None:
        super().__init__(**client_kw)
        self.venue = str(venue)
        self._exchange: Any = None
        self.limiter = PerInstanceDelay()        # one bucket, matching the one instance

    # ------------------------------------------------------------------ the instance
    def exchange(self) -> Any:
        """The ONE exchange object this adapter owns. Constructed once, reused forever."""
        if self._exchange is None:
            ccxt = require("ccxt")
            if not hasattr(ccxt, self.venue):
                raise KeyError(f"ccxt has no exchange {self.venue!r}")
            self._exchange = getattr(ccxt, self.venue)({"enableRateLimit": True,
                                                        **self.client_kw})
            ms = getattr(self._exchange, "rateLimit", None)
            if ms:
                self.limiter = PerInstanceDelay(float(ms))
        return self._exchange

    def bars(self, symbols, start, end, *, interval: str = "1d",
             adjustment: Adjustment | None = None, limit: int = 1000, **kw) -> Bars:
        ex = self.exchange()
        syms = [symbols] if isinstance(symbols, str) else list(symbols)
        adj = adjustment or Adjustment.RAW
        if adj is not Adjustment.RAW:
            raise ValueError("crypto has no corporate actions; RAW is the only convention")
        if interval not in TIMEFRAMES:
            raise ValueError(f"interval {interval!r} is not one of {sorted(TIMEFRAMES)}")

        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        parts = []
        for sym in syms:
            rows: list[list[float]] = []
            since = int(s.timestamp() * 1000)        # MILLISECONDS, not seconds
            end_ms = int(e.timestamp() * 1000)
            while since < end_ms:
                self.limiter.acquire()
                page = ex.fetch_ohlcv(sym, timeframe=interval, since=since,
                                      limit=int(limit), **kw)
                if not page:
                    break
                rows.extend(r for r in page if since <= r[0] < end_ms)
                nxt = int(page[-1][0]) + 1
                if nxt <= since:
                    break
                since = nxt
                if len(page) < int(limit):
                    break
            parts.append(_to_frame(rows, str(sym)))
        frame = pd.concat(parts, axis=1).sort_index(axis=1)
        frame = drop_unclosed(frame, interval, pd.Timestamp.now(tz="UTC"))

        request = {"method": "bars", "symbols": [str(x) for x in syms],
                   "start": s.isoformat(), "end": e.isoformat(), "interval": interval,
                   "adjustment": adj.value, "venue": self.venue,
                   "since_units": "milliseconds", "half_open": True,
                   "dropped_unclosed_final_bar": True}
        prov = make_provenance(f"ccxt:{self.venue}",
                               library_version=library_version(require("ccxt")),
                               request=request, content=frame, terms_url=TERMS,
                               redistributable=False)
        return Bars(frame=frame, adjustment=adj, calendar="24/7", tz="UTC",
                    interval=interval, bar_label="open", currency=_quote(syms[0]),
                    provenance=prov, half_open=True)


def _quote(symbol: str) -> str:
    """"BTC/USDT" -> "USDT". The quote currency IS the price's currency."""
    return str(symbol).split("/")[-1].split(":")[0] if "/" in str(symbol) else "unknown"


declare.register(CcxtAdapter, DECL)

__all__ = ["CcxtAdapter", "DECL", "TIMEFRAMES", "drop_unclosed"]
