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

from typing import Any, Iterable

import numpy as np
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


def prices_panel(bars: Bars) -> pd.DataFrame:
    """The wide close panel `survivorship_audit` reads: dates x tickers, NaN after a name
    stops trading. Never forward-filled - a filled hole hides the delisting."""
    return bars.field("close").copy()


def liquidity_panel(bars: Bars) -> pd.DataFrame | None:
    """close * volume, the dollar-volume panel a trailing-ADV screen needs."""
    if "volume" not in bars.fields or "close" not in bars.fields:
        return None
    close, vol = bars.field("close"), bars.field("volume")
    cols = [c for c in close.columns if c in vol.columns]
    return (close[cols].astype(float) * vol[cols].astype(float)) if cols else None


def listings_table(bars: Bars) -> pd.DataFrame | None:
    """The listings table under the column names `survivorship_audit` expects."""
    if bars.listings is None or not len(bars.listings):
        return None
    lt = bars.listings.copy()
    rename = {"start_date": "listing_date", "end_date": "delisting_date"}
    return lt.rename(columns={k: v for k, v in rename.items() if k in lt.columns})


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
            slots["actions"] = bars.actions[["date", "ratio", "kind"]].copy()
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
            slots["close"] = bars.close(pick)
            slots["bars"] = bars.ohlcv(pick)

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


__all__ = ["FILLS", "fills", "from_long", "liquidity_panel", "listings_table",
           "periods_per_year", "prices_panel", "to_bundle", "to_long"]
