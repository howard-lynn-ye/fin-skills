"""`to_bundle()` - fill every slot the data layer can fill, and no others.

    bars         -> prices, close, bars, actions, listings, liquidity, periods_per_year
    fundamentals -> facts, as_of
    macro        -> no direct slot today; a Macro is accepted and validated, and its
                    series go into the caller's own slots via **extra

Four guards then run with no backtest, no strategy and no engine:

    survivorship_audit   prices + listings
    adjustment_check     close + actions
    pit_fundamentals     facts + as_of
    reconcile_sources    close + other, from Cache.refetch().to_bundle()

That is the point of the module. The five gates in research-integrity-guards are DATA
gates: they should fail before a strategy is ever written, not after a Sharpe has been
computed and believed.

`periods_per_year` is filled only where the calendar and interval make it unambiguous -
252 for a daily bar on an exchange calendar, 365 for a daily bar on a 24/7 calendar.
Anything else is left empty rather than guessed, because an annualisation factor invented
by a data layer is the quietest way to be wrong by sqrt(365/252) = 1.20x.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from fin_skills.data.schema import Bars, Fundamentals, Macro

#: exactly the slots this module fills, and the source each comes from
FILLS: dict[str, str] = {
    "prices": "bars", "close": "bars", "bars": "bars", "actions": "bars",
    "listings": "bars", "liquidity": "bars", "periods_per_year": "bars",
    "facts": "fundamentals", "as_of": "fundamentals",
}

#: daily periods per year, by calendar. Nothing else is annualised here.
_PERIODS_PER_YEAR = {"24/7": 365}
_DEFAULT_DAILY = 252


def periods_per_year(interval: str, calendar: str) -> int | None:
    """252 or 365 for a daily bar, None for anything this layer cannot state."""
    if str(interval).lower() not in ("1d", "d", "1day", "daily"):
        return None
    return _PERIODS_PER_YEAR.get(str(calendar), _DEFAULT_DAILY)


#: Adjustment -> the vocabulary `api.guards.adjustment_check` uses for `expected`.
#:
#: READ THIS BEFORE USING IT. The two vocabularies are INVERTED, and passing an
#: Adjustment value straight through would silently invert every convention check:
#:
#:   Adjustment.ANCHORED_START    is anchored at the START (A-share hfq, history never rewritten)
#:                      and `detect_convention` calls that shape "forward-adjusted",
#:                      because history was scaled ANCHORED_PRESENT from the anchor;
#:   Adjustment.ANCHORED_PRESENT is anchored at the PRESENT (A-share qfq, Yahoo auto_adjust) and
#:                      `detect_convention` calls that shape "back-adjusted", because
#:                      history was rewritten BACKWARD from today's anchor.
#:
#: Both namings are in common use and neither is wrong; they name opposite ends of the
#: same operation. The enum follows the A-share anchor convention the design specifies,
#: the guard follows the US/Yahoo one, and this table is the only place the two meet.
#: Verified empirically against `core.adjustment_check.detect_convention` and its own
#: `_synthetic()` builder.
_GUARD_CONVENTION: dict[str, str] = {
    "raw": "raw",
    "raw+factors": "raw",
    "anchored_start": "forward-adjusted",
    "anchored_present": "back-adjusted",
    "unknown": "unknown",
}


def guard_convention(adjustment) -> str:
    """The `expected=` string `adjustment_check` wants for this Adjustment.

    Use it, never `adjustment.value` - see `_GUARD_CONVENTION` above for why the two
    vocabularies are inverted.
    """
    from fin_skills.data.schema import Adjustment as _A                # noqa: PLC0415
    return _GUARD_CONVENTION[_A(adjustment).value]


def pit_used(fundamentals: Fundamentals, ts: Any, *, tag: str | None = None,
             entity: str | None = None, naive: bool = False) -> pd.Series:
    """The per-period values a pipeline consumed, keyed the way `pit_fundamentals` keys them.

    `naive=True` returns what `drop_duplicates(keep="last")` would have selected - the
    latest vintage of every period regardless of when it was filed - so a test can show
    the two differ instead of asserting that they should.
    """
    f = fundamentals.frame
    if tag is not None:
        f = f[f["tag"] == tag]
    if entity is not None:
        f = f[f["entity_id"] == entity]
    if naive:
        rows = (f.sort_values(["available_at", "accn"], kind="mergesort")
                 .drop_duplicates(["entity_id", "tag", "period_start", "period_end"],
                                  keep="last"))
    else:
        rows = fundamentals.as_of(ts, entities=[entity] if entity else None)
        if tag is not None:
            rows = rows[rows["tag"] == tag]
    keys = [_period_key(r) for _, r in rows.iterrows()]
    return pd.Series(rows["value"].astype(float).to_numpy(), index=keys)


def _period_key(row) -> str:
    """"2022-07-01..2022-09-30", or "instant..2022-09-30" for a balance-sheet fact.

    The same key `core.pit_fundamentals.facts_to_frame` builds, so the guard can match.
    """
    start = row["period_start"]
    head = "instant" if pd.isna(start) else pd.Timestamp(start).strftime("%Y-%m-%d")
    return f"{head}..{pd.Timestamp(row['period_end']).strftime('%Y-%m-%d')}"


#: intervals whose label is a SESSION DATE rather than an instant
_SESSION_INTERVALS = frozenset({"1d", "d", "daily", "1day", "5d", "1wk", "1w", "1mo",
                                "1month", "3mo", "1y"})


def _to_guard_index(bars: Bars, obj):
    """Put a bars-derived object on the index the guards compare against.

    For a DAILY-or-coarser bar the label names a session, not an instant, so the zone is
    metadata: it stays on `Bars.tz` and in the cache sidecar, and the guard input carries
    the exchange-local session date. Intraday keeps its zone, because there the instant is
    the datum.

    This is a boundary conversion, not a repair - no value moves. It exists because
    `core.survivorship_audit` compares the panel index against listing dates with
    `pd.Timestamp(t) > start`, and `Series.values` on a tz-aware column yields naive
    numpy datetimes, so a tz-aware panel raises "Cannot compare tz-naive and tz-aware
    timestamps" before the audit can run.
    """
    idx = obj.index
    if str(bars.interval).lower() in _SESSION_INTERVALS and getattr(idx, "tz", None):
        out = obj.copy()
        out.index = idx.tz_convert(bars.tz).tz_localize(None)
        return out
    return obj


def prices_panel(bars: Bars) -> pd.DataFrame:
    """The wide close panel `survivorship_audit` reads: dates x tickers, NaN after a name
    stops trading. Never forward-filled - a filled hole hides the delisting."""
    return _to_guard_index(bars, bars.field("close").copy())


def liquidity_panel(bars: Bars) -> pd.DataFrame | None:
    """close * volume, the dollar-volume panel a trailing-ADV screen needs."""
    if "volume" not in bars.fields or "close" not in bars.fields:
        return None
    close, vol = bars.field("close"), bars.field("volume")
    cols = [c for c in close.columns if c in vol.columns]
    if not cols:
        return None
    return _to_guard_index(bars, close[cols].astype(float) * vol[cols].astype(float))


def actions_table(bars: Bars, ticker: str | None = None) -> pd.DataFrame | None:
    """(date, ratio, kind) on the same index type as the price slots.

    Dividend rows with no multiplicative ratio are dropped: `adjustment_check` reads the
    ratio as the price factor of the event, and a NaN there is not information it can use.
    """
    if bars.actions is None or not len(bars.actions):
        return None
    a = bars.actions.copy()
    if ticker is not None and "ticker" in a.columns and a["ticker"].notna().any():
        a = a[(a["ticker"] == ticker) | a["ticker"].isna()]
    a = a[pd.to_numeric(a["ratio"], errors="coerce").notna()]
    if not len(a):
        return None
    tz = prices_panel(bars).index.tz
    d = pd.to_datetime(a["date"])
    if tz is None and d.dt.tz is not None:
        d = d.dt.tz_convert(bars.tz).dt.tz_localize(None)
    elif tz is not None and d.dt.tz is None:
        d = d.dt.tz_localize(tz)
    a = a.assign(date=d)
    return a[["date", "ratio", "kind"]].reset_index(drop=True)


def listings_table(bars: Bars) -> pd.DataFrame | None:
    """The listings table under the column names `survivorship_audit` expects.

    The dates are put in the PANEL's timezone, because the guard compares them against
    the panel's own index and pandas raises rather than guessing when one side is naive.
    Aligning zones is normalisation, which is this layer's job; guessing a zone is not.
    """
    if bars.listings is None or not len(bars.listings):
        return None
    lt = bars.listings.copy()
    rename = {"start_date": "listing_date", "end_date": "delisting_date"}
    lt = lt.rename(columns={k: v for k, v in rename.items() if k in lt.columns})
    tz = prices_panel(bars).index.tz
    for col in ("listing_date", "delisting_date"):
        if col in lt.columns:
            d = pd.to_datetime(lt[col], errors="coerce")
            if tz is not None:
                d = d.dt.tz_localize(tz) if d.dt.tz is None else d.dt.tz_convert(tz)
            elif d.dt.tz is not None:
                d = d.dt.tz_localize(None)
            lt[col] = d
    return lt


def to_bundle(bars: Bars | None = None,
              fundamentals: Fundamentals | None = None,
              macro: Macro | None = None,
              *, ticker: str | None = None, as_of: Any = None,
              tag: str | None = None, entity: str | None = None,
              **extra: Any):
    """Build the `api.Bundle` the guards read, from schema objects alone.

    `ticker` picks the single instrument for the `close` and `bars` slots; with one
    instrument in the panel it is inferred, and with several it is required for those two
    slots (the panel-wide slots are filled either way). `as_of` is the decision date
    `pit_fundamentals` checks against; without it, the latest `available_at` in the
    fundamentals is used and recorded as such.
    """
    from fin_skills.api.bundle import Bundle                          # noqa: PLC0415

    slots: dict[str, Any] = {}

    if bars is not None:
        if not isinstance(bars, Bars):
            raise TypeError(f"bars must be a Bars, got {type(bars).__name__}")
        slots["prices"] = prices_panel(bars)
        if bars.actions is not None and len(bars.actions):
            slots["actions"] = actions_table(bars)
        lt = listings_table(bars)
        if lt is not None:
            slots["listings"] = lt
        liq = liquidity_panel(bars)
        if liq is not None:
            slots["liquidity"] = liq
        ppy = periods_per_year(bars.interval, bars.calendar)
        if ppy is not None:
            slots["periods_per_year"] = ppy
        names = bars.tickers
        pick = ticker if ticker is not None else (names[0] if len(names) == 1 else None)
        if pick is not None:
            if pick not in names:
                raise KeyError(f"no ticker {pick!r} in this panel; have {names}")
            slots["close"] = _to_guard_index(bars, bars.close(pick))
            slots["bars"] = _to_guard_index(bars, bars.ohlcv(pick))

    if fundamentals is not None:
        if not isinstance(fundamentals, Fundamentals):
            raise TypeError(f"fundamentals must be a Fundamentals, got "
                            f"{type(fundamentals).__name__}")
        stamp = (pd.Timestamp(as_of) if as_of is not None
                 else fundamentals.frame["available_at"].max())
        facts = fundamentals.to_facts(entity=entity, tag=tag)
        if facts:
            slots["facts"] = facts
        slots["as_of"] = stamp
    elif as_of is not None:
        slots["as_of"] = pd.Timestamp(as_of)

    if macro is not None and not isinstance(macro, Macro):
        raise TypeError(f"macro must be a Macro, got {type(macro).__name__}")

    unexpected = set(slots) - set(FILLS)
    if unexpected:                                             # pragma: no cover
        raise AssertionError(f"to_bundle filled a slot it does not document: "
                             f"{sorted(unexpected)}")
    slots.update(extra)
    return Bundle(**slots)


# ----------------------------------------------------------------------- long <-> wide
def to_long(bars: Bars, *, value_name: str = "value") -> pd.DataFrame:
    """(date, ticker, field, value) - the shape a columnar store wants."""
    out = bars.frame.stack(level=["field", "ticker"], future_stack=True)
    out = out.rename(value_name).reset_index()
    out.columns = ["date", "field", "ticker", value_name]
    return out[["date", "ticker", "field", value_name]].dropna(subset=[value_name])


def from_long(long: pd.DataFrame, *, value_name: str = "value") -> pd.DataFrame:
    """The inverse: back to the (field, ticker) MultiIndex frame `Bars` wants."""
    wide = long.pivot(index="date", columns=["field", "ticker"], values=value_name)
    wide.columns = pd.MultiIndex.from_tuples(list(wide.columns),
                                             names=["field", "ticker"])
    return wide.sort_index(axis=1)


def fills(source: str | None = None) -> list[str]:
    """The slots `to_bundle()` fills, optionally for one source only."""
    return sorted(k for k, v in FILLS.items() if source is None or v == source)


__all__ = ["FILLS", "fills", "from_long", "guard_convention", "liquidity_panel",
           "listings_table", "periods_per_year", "pit_used", "prices_panel", "to_bundle",
           "to_long"]
