"""akshare - the widest free Chinese-market source, and a scraper you cannot pin.

Four facts this adapter is built around, all re-verified on 2026-09-09:

  * **The version is read at call time, not from a lockfile.** PyPI lists 219 releases and
    the oldest surviving one is 1.16.72 of 2025-04-05, against 1,437 versions in the
    upstream changelog - roughly 1,200 are gone, and nothing older than about 17 months
    installs. A `requirements.txt` pinning an old akshare no longer resolves, so
    `Provenance.library_version` records what actually ran.
  * **The default is RAW.** `stock_zh_a_hist(..., adjust="")` returns unadjusted prices
    (verified in source), which is the correct base for limit-up, tick-size and lot-size
    logic. The *ecosystem* default is qfq; akshare's own is not, and this adapter declares
    what akshare does.
  * **qfq rewrites history.** Its anchor is the present, so re-running the same query next
    month returns different numbers. Any qfq series is stamped ANCHORED_PRESENT, and
    `Adjustment.ANCHORED_PRESENT.rewrites_history` is True - do not persist it and expect it back.
  * **The code is MIT; the data is not.** akshare's own documentation restricts the data to
    academic research, which is stronger than a plain reading of the repository's MIT
    licence. The adapter surfaces that once, the first time it is used.

akshare publishes no rate limit at all - no number in the docs, only its own docstrings
warning that heavy scraping gets an IP blocked - so `rate_limit` is `Unpublished()` and
this module contains no rate constant.
"""
from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from fin_skills.data import declare
from fin_skills.data.adapters import Base, library_version, require
from fin_skills.data.provenance import make as make_provenance
from fin_skills.data.ratelimit import Unpublished
from fin_skills.data.schema import Adjustment, Bars

TERMS = "https://akshare.akfamily.xyz/introduction.html"

#: akshare's adjust= argument, and what each value means in this layer's vocabulary
ADJUST = {Adjustment.RAW: "", Adjustment.ANCHORED_START: "hfq", Adjustment.ANCHORED_PRESENT: "qfq"}

DATA_USE_NOTICE = (
    "akshare's code is MIT; its documentation restricts the DATA to academic research "
    "(the upstream wording is 'for academic research only'). Installing this adapter "
    "grants no data licence, and redistribution is declared prohibited.")

DECL = declare.Declaration(
    name="akshare",
    library="akshare",
    library_license="MIT",
    licence_source="pypi",                    # info.license == 'MIT' + classifier, 2026-09-09
    adjustment_default=Adjustment.RAW,        # stock_zh_a_hist(adjust="") is unadjusted
    adjustment_supported=(Adjustment.RAW, Adjustment.ANCHORED_START, Adjustment.ANCHORED_PRESENT),
    calendar="XSHG",
    tz="Asia/Shanghai",
    bar_label="close",
    interval_support=("1d", "1wk", "1mo"),
    includes_delisted=False,                  # delisting lists exist, the price history
                                              # does not - and the rule is: if it cannot be
                                              # answered, it is False
    point_in_time=False,
    rate_limit=Unpublished(),
    free_tier="free, no key, no published rate limit of any kind (verified 2026-09-09)",
    requires_key=False,
    key_env_var="",
    key_sharing="unstated",
    cache_policy="persist-ok",
    non_display_use="unstated",
    terms_url=TERMS,
    redistribution="prohibited",
    verified_on="2026-09-09",
    notes=("PyPI keeps only 219 releases, oldest 1.16.72 (2025-04-05), against 1,437 in "
           "the upstream changelog - a version pin is not evidence of what ran, so the "
           "adapter records akshare.__version__ at call time.\n"
           "index_stock_cons_csindex() has no date parameter: it returns TODAY's HS300 / "
           "CSI500 membership, so an index backtest built on it holds names that were "
           "promoted later. universe() refuses rather than returning it as history.\n"
           + DATA_USE_NOTICE),
)

_notified = False


def _notify_once() -> None:
    """Surface the academic-research-only framing the first time this adapter is used."""
    global _notified
    if not _notified:
        _notified = True
        warnings.warn(DATA_USE_NOTICE, stacklevel=3)


_COLUMNS = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
            "收盘": "close", "成交量": "volume", "date": "date", "open": "open",
            "high": "high", "low": "low", "close": "close", "volume": "volume"}


def _normalise(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """One akshare daily frame -> the (field, ticker) columns `Bars` wants.

    Pure and offline: akshare renames its returned columns between releases, and the
    Chinese headers are the stable ones.
    """
    if raw is None or not len(raw):
        raise ValueError(f"akshare returned no rows for {symbol!r}")
    df = raw.rename(columns=_COLUMNS)
    missing = [c for c in ("date", "open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"akshare frame for {symbol!r} is missing {missing}; the "
                         f"returned column names moved again (columns: {list(raw.columns)})")
    df = df.set_index(pd.to_datetime(df["date"])).sort_index()
    out = {}
    for f in ("open", "high", "low", "close"):
        out[(f, symbol)] = pd.to_numeric(df[f], errors="coerce").astype("float64")
    if "volume" in df.columns:
        v = pd.to_numeric(df["volume"], errors="coerce")
        out[("volume", symbol)] = (v.fillna(0).round().astype("int64")
                                   if v.notna().all() else v)
    frame = pd.DataFrame(out)
    frame.columns = pd.MultiIndex.from_tuples(list(frame.columns),
                                              names=["field", "ticker"])
    frame.index.name = "date"
    return frame.sort_index(axis=1)


class AkshareAdapter(Base):
    decl = DECL

    def bars(self, symbols, start, end, *, interval: str = "1d",
             adjustment: Adjustment | None = None, **kw) -> Bars:
        _notify_once()
        syms = [symbols] if isinstance(symbols, str) else list(symbols)
        adj = adjustment or self.decl.adjustment_default
        if adj not in ADJUST:
            raise ValueError(f"akshare serves {[a.value for a in ADJUST]}, not {adj.value}")
        if interval != "1d":
            raise ValueError("this adapter maps only the daily endpoint "
                             "(stock_zh_a_hist period='daily')")

        ak = require("akshare")
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        parts = []
        for sym in syms:
            self.limiter.acquire()
            raw = ak.stock_zh_a_hist(symbol=str(sym), period="daily",
                                     start_date=s.strftime("%Y%m%d"),
                                     end_date=e.strftime("%Y%m%d"),
                                     adjust=ADJUST[adj], **kw)
            parts.append(_normalise(raw, str(sym)))
        frame = pd.concat(parts, axis=1).sort_index(axis=1)
        # the layer's contract is half-open [start, end); akshare's end_date is inclusive
        frame = frame[frame.index < e]

        request = {"method": "bars", "symbols": [str(x) for x in syms],
                   "start": s.strftime("%Y-%m-%d"), "end": e.strftime("%Y-%m-%d"),
                   "interval": "1d", "adjustment": adj.value,
                   "akshare_adjust": ADJUST[adj], "period": "daily",
                   "end_is_exclusive": False, "half_open": True}
        prov = make_provenance("akshare", library_version=library_version(ak),
                               request=request, content=frame, terms_url=TERMS,
                               redistributable=False)
        return Bars(frame=frame, adjustment=adj, calendar=self.decl.calendar,
                    tz=self.decl.tz, interval="1d", bar_label="close", currency="CNY",
                    provenance=prov, half_open=True)

    def universe(self, market: str, as_of, *,
                 include_delisted: bool = False) -> pd.DataFrame:
        raise self._unsupported(
            "a point-in-time universe",
            "index_stock_cons_csindex() has no date parameter - it returns TODAY's "
            "membership and nothing else, so returning it as history would silently "
            "produce inclusion bias. Use tushare index_weight / index_member, baostock's "
            "dated lists, or a licensed vendor")

    def fundamentals(self, entities, tags=(), **kw):
        raise self._unsupported(
            "point-in-time fundamentals",
            "akshare's statements are keyed on the reporting period with no usable "
            "announcement date, so joining on them leaks the future by 30-90 days")


declare.register(AkshareAdapter, DECL)

__all__ = ["ADJUST", "DATA_USE_NOTICE", "DECL", "AkshareAdapter"]
