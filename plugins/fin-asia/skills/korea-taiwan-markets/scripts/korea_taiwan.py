"""Dated Korea/Taiwan equity rules and a synthetic, explicit queue-capacity audit.

Offline numpy/pandas only. A limit-price OHLC bar does NOT prove zero liquidity.
The demo supplies synthetic bid/ask capacity at the order time; this is a modelling
assumption, not evidence about either market. Unknown capacity stays unknown.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

KRX_LIMIT_CHANGE = date(2015, 6, 15)
TWSE_LIMIT_CHANGE = date(2015, 6, 1)
KRX_PARTIAL_RESUMPTION = date(2021, 5, 3)
KRX_FULL_RESUMPTION = date(2025, 3, 31)
IRC_ABOLISHED_ON = date(2023, 12, 14)
# Ordinary investors; market-maker exemptions and per-stock restrictions are separate.
KRX_SHORT_BANS = (
    (date(2008, 10, 1), date(2009, 5, 31)),
    (date(2011, 8, 10), date(2011, 11, 9)),
    (date(2020, 3, 16), date(2021, 5, 2)),
    (date(2023, 11, 6), date(2025, 3, 30)),
)
TWSE_TICKS = ((10., .01), (50., .05), (100., .1), (500., .5),
              (1000., 1.), (math.inf, 5.))


def _date(value) -> date:
    if value is None:
        raise ValueError("date is required")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def daily_limit_pct(market: str, when, *, ordinary_share: bool = True) -> float:
    """Ordinary seasoned KOSPI/KOSDAQ or TWSE shares, from 2010 onward.

    Use the exchange's published daily upper/lower price for orders. IPOs, KONEX,
    leveraged products and corporate-action reference prices are outside this helper.
    """
    d = _date(when)
    if d < date(2010, 1, 1) or not ordinary_share:
        raise ValueError("outside ordinary-share rule coverage; use security master")
    m = market.upper()
    if m in ("KRX", "KOSPI", "KOSDAQ"):
        return .30 if d >= KRX_LIMIT_CHANGE else .15
    if m == "TWSE":
        return .10 if d >= TWSE_LIMIT_CHANGE else .07
    raise ValueError("market must be KRX, KOSPI, KOSDAQ or TWSE")


def twse_tick(price: float) -> float:
    if not math.isfinite(price) or price <= 0:
        raise ValueError("price must be positive and finite")
    return next(tick for bound, tick in TWSE_TICKS if price < bound)


def short_selling_allowed(when, index_member: bool = False,
                          financial_stock: bool = False) -> tuple[bool, str]:
    """Historical broad-ban gate, not borrow availability or an order approval.

    Membership must be as of the trade date. Borrow, account controls, market-maker
    exceptions and an individual security's overheated-short designation are not inferred.
    """
    d = _date(when)
    if d < date(2008, 10, 1):
        raise ValueError("ban history starts on 2008-10-01")
    if financial_stock and date(2008, 10, 1) <= d <= date(2013, 11, 13):
        return False, "financial-stock ban"
    for start, end in KRX_SHORT_BANS:
        if start <= d <= end:
            return False, "broad short-selling ban"
    if KRX_PARTIAL_RESUMPTION <= d < date(2023, 11, 6) and not index_member:
        return False, "partial resumption: only KOSPI200 / KOSDAQ150 members"
    return True, "broad-ban gate passed; borrow and per-stock restrictions still required"


def settlement_date(trade_date, holidays: Iterable = ()) -> date:
    """T+2 weekday arithmetic for the covered modern equity regime.

    Pass every applicable exchange/custodian settlement holiday. No calendar is downloaded.
    """
    current = _date(trade_date)
    if current < date(2010, 1, 1):
        raise ValueError("historical settlement regime not covered")
    closed = {_date(day) for day in holidays}
    moved = 0
    while moved < 2:
        current += timedelta(days=1)
        if current.weekday() < 5 and current not in closed:
            moved += 1
    return current


def reachable(bar: Mapping, side: str, quantity: float = 1.) -> tuple[bool | None, str]:
    """Whether recorded immediate opposite-side capacity covers a marketable order.

    Required inputs: price, lower_limit, upper_limit, volume. Optional ask_qty/bid_qty
    MUST describe executable capacity at the order timestamp and price, after queues ahead.
    OHLC or daily volume alone cannot supply that fact. True means capacity in this model,
    not a guarantee against latency/cancellations. None means unknown, never a free fill.
    """
    if side not in ("buy", "sell") or not math.isfinite(quantity) or quantity <= 0:
        raise ValueError("positive finite quantity and buy/sell side required")
    price, low, high, volume = [float(bar[key]) for key in
                                ("price", "lower_limit", "upper_limit", "volume")]
    if not all(math.isfinite(v) for v in (price, low, high, volume)):
        raise ValueError("price, limits and volume must be finite")
    if low < 0 or high < low or volume < 0:
        raise ValueError("invalid price limits or volume")
    if not low <= price <= high:
        return False, "price outside published band"
    if bar.get("suspended", False):
        return False, "suspended at order time"
    key = "ask_qty" if side == "buy" else "bid_qty"
    if key not in bar or bar[key] is None:
        return None, "OHLC/volume does not establish executable opposite-side capacity"
    capacity = float(bar[key])
    if not math.isfinite(capacity) or capacity < 0:
        raise ValueError("capacity must be finite and non-negative")
    return (capacity >= quantity,
            "capacity covers order" if capacity >= quantity else "insufficient capacity")


def synthetic_limited_series(n_days: int = 1250, limit_pct: float = .10,
                             seed: int = 20260910) -> pd.DataFrame:
    """Same latent shocks across bands, with residual shocks carried forward.

    Assumption: a bound latent move creates zero capacity on its pressured side;
    other bars have capacity 100. Prices are abstract (no venue tick rounding).
    This example isolates queue assumptions; it estimates no real-world fill rate.
    """
    if n_days < 2 or not 0 < limit_pct < 1:
        raise ValueError("n_days >= 2 and 0 < limit_pct < 1 required")
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, .035, n_days)
    shocks += rng.normal(0, .18, n_days) * (rng.random(n_days) < .1)
    rows, price, pending = [], 100., 0.
    for shock in shocks:
        want = shock + pending
        got = float(np.clip(want, math.log1p(-limit_pct), math.log1p(limit_pct)))
        up, down = price * (1 + limit_pct), price * (1 - limit_pct)
        up_locked = want >= math.log1p(limit_pct)
        down_locked = want <= math.log1p(-limit_pct)
        close = up if up_locked else down if down_locked else price * math.exp(got)
        rows.append(dict(price=close, lower_limit=down, upper_limit=up, volume=1000.,
                         ask_qty=0. if up_locked else 100.,
                         bid_qty=0. if down_locked else 100.,
                         locked=up_locked or down_locked))
        price, pending = close, want - got
    return pd.DataFrame(rows)


def fill_rate(df: pd.DataFrame, signal_threshold: float = .04) -> dict:
    signal = df.price.pct_change().shift(1).gt(signal_threshold)
    eligible = [reachable(row, "buy")[0] is True for row in df.to_dict("records")]
    signals = int(signal.sum())
    filled = int((signal & np.asarray(eligible)).sum())
    return dict(signals=signals, filled=filled, rejected=signals-filled,
                naive_fill_rate=1. if signals else math.nan,
                model_fill_rate=filled/signals if signals else math.nan,
                locked_bars=int(df.locked.sum()))


def main():
    print("Synthetic queue assumptions, not measured exchange liquidity.")
    for market, pct in (("KRX", .30), ("TWSE", .10)):
        out = fill_rate(synthetic_limited_series(limit_pct=pct))
        print(f"{market} band={pct:.0%}: signals={out['signals']}, "
              f"filled={out['filled']}, rejected={out['rejected']}, "
              f"naive=100.0%, model={out['model_fill_rate']:.1%}, "
              f"bound bars={out['locked_bars']}")
    bar = dict(price=110., lower_limit=90., upper_limit=110., volume=50.)
    print("Limit bar without depth:", reachable(bar, "buy"))
    print("Same bar with offers:", reachable(dict(bar, ask_qty=10.), "buy"))
    for day in ("2021-05-02", "2021-05-03", "2023-11-06", "2025-03-31"):
        print(day, "nonmember=", short_selling_allowed(day)[0],
              "index member=", short_selling_allowed(day, index_member=True)[0])
    print("RULE: a limit-price bar does not prove a fill or no liquidity; "
          "use dated bands and observed queue capacity.")


if __name__ == "__main__":
    main()
