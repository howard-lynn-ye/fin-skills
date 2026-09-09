"""Panel/listings -> zipline assets; perf -> Bundle. The asset lifetime is the whole point.

zipline-reloaded is the US-equity correctness benchmark for one reason above all others
(fin_skills.load('backtesting-engines'), `references/zipline-reloaded.md`): its asset
database carries **`start_date`, `end_date` and `auto_close_date`** per asset, so a
position in a name that delists is closed automatically and the asset stops being
tradeable. Neither vectorbt nor backtesting.py has any concept of an asset ceasing to
exist.

THE TRAP THIS BRIDGE ENFORCES: that machinery is worth nothing if the universe you ingest
was built from today's ticker list. Delisted names are simply absent, every `end_date` is
the end of the sample, and the engine dutifully never closes anything - a survivor-only
backtest with the most careful lifetime handling in the ecosystem bolted to it. So
`to_zipline_assets` runs `survivorship_audit` FIRST and refuses on a clean sweep;
"could not prove the universe is clean" fails the same way as "it is biased", which is the
guard's own posture.

SECOND ENFORCEMENT: `to_zipline_commission` has **no default** for `min_trade_cost`.
zipline's `PerShare` defaults to `$0.001/share with a $0.00 minimum` (verified in
`references/zipline-reloaded.md`), and a $0.00 minimum is fiction at every retail US
broker: a strategy trading 3-share lots is free under it.

Licence: Apache-2.0 - safe as an extra. Note the NOTICE obligation, and that wheels are
cp310-cp313 only with no Linux aarch64.
"""
from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd

from fin_skills.api import Bundle, get
from fin_skills.bridges import _lazy

LIBRARY = "zipline-reloaded"
LICENCE = _lazy.info(LIBRARY).licence
VERIFIED_ON = _lazy.info(LIBRARY).verified_on

_LISTING_ALIASES = {"listing_date": "start_date", "start": "start_date",
                    "delisting_date": "end_date", "end": "end_date",
                    "symbol": "ticker", "asset": "ticker"}


class SurvivorOnlyUniverseError(ValueError):
    """The panel handed to zipline has no delistings, so its asset lifetimes are fiction."""


def _panel_parts(panel: Any) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """(wide close panel, listings) from a Bundle, an engine Panel, or a mapping."""
    if isinstance(panel, Bundle):
        prices = panel.get("prices")
        listings = panel.get("listings", panel.get("members"))
    elif isinstance(panel, Mapping):
        prices = panel.get("prices", panel.get("close"))
        listings = panel.get("listings", panel.get("members"))
    else:
        prices = getattr(panel, "close", None)
        if prices is None:
            prices = getattr(panel, "prices", None)
        listings = getattr(panel, "listings", None)
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(
            "expected a wide close panel (dates x tickers) on the `prices` slot / the "
            "Panel's .close, with NaN after a name stops trading - that NaN pattern is "
            "survivorship_audit's entire contract")
    if listings is not None and not isinstance(listings, pd.DataFrame):
        raise TypeError("listings must be a DataFrame with ticker/start_date/end_date")
    return prices, listings


def _normalise_listings(listings: pd.DataFrame | None) -> pd.DataFrame | None:
    if listings is None:
        return None
    out = listings.rename(columns=_LISTING_ALIASES).copy()
    if "ticker" not in out.columns:
        raise TypeError(f"listings needs a ticker column; has {list(listings.columns)}")
    return out


def to_zipline_assets(panel: Any, *, exchange: str = "XNYS",
                      auto_close_offset_days: int = 1) -> pd.DataFrame:
    """Panel.listings -> zipline asset metadata, refusing a survivor-only universe.

    Returns the frame `AssetDBWriter.write(equities=...)` wants: an integer `sid` index and
    columns symbol, start_date, end_date, first_traded, auto_close_date, exchange.

    Runs `survivorship_audit` on the panel first. `auto_close_date` is `end_date` plus
    `auto_close_offset_days` sessions, because zipline closes the position ON that date and
    a same-day auto-close silently drops the final bar.
    """
    prices, listings = _panel_parts(panel)
    listings = _normalise_listings(listings)

    inputs: dict[str, Any] = {"prices": prices}
    if listings is not None:
        inputs["listings"] = listings.rename(columns={"start_date": "listing_date",
                                                      "end_date": "delisting_date"})
    res = get("survivorship_audit").run(**inputs)
    if not res.passed:
        raise SurvivorOnlyUniverseError(
            "refusing to build zipline assets from this panel - "
            + "; ".join(str(f) for f in res.errors)
            + ". zipline's start_date/end_date/auto_close_date machinery is the best in the "
              "ecosystem and it cannot invent the names your screen already dropped: a "
              "universe built from today's ticker list has no delistings to model.")

    if listings is None:
        first = prices.apply(lambda c: c.first_valid_index())
        last = prices.apply(lambda c: c.last_valid_index())
        listings = pd.DataFrame({"ticker": prices.columns, "start_date": first.to_numpy(),
                                 "end_date": last.to_numpy()})
    start = pd.to_datetime(listings["start_date"])
    end = pd.to_datetime(listings["end_date"]).fillna(prices.index[-1])
    out = pd.DataFrame({
        "symbol": listings["ticker"].astype(str).to_numpy(),
        "start_date": start.to_numpy(),
        "end_date": end.to_numpy(),
        "first_traded": start.to_numpy(),
        "auto_close_date": (end + pd.Timedelta(days=int(auto_close_offset_days))).to_numpy(),
        "exchange": exchange,
    })
    out.index = pd.RangeIndex(len(out), name="sid")
    return out


def to_zipline_commission(*, cost_per_share: float = 0.001, min_trade_cost: float):
    """`zipline.finance.commission.PerShare` with a minimum you had to state.

    `min_trade_cost` is keyword-only and has NO default here, so omitting it is a TypeError
    before anything is imported. zipline's own default is $0.00, under which a strategy
    trading 3-share lots pays nothing - the one number in the commission model that is pure
    fiction for a retail US broker. IBKR's tiered minimum is $0.35/order; pass what you
    actually pay.
    """
    if min_trade_cost is None or float(min_trade_cost) < 0:
        raise ValueError("min_trade_cost must be a non-negative number of dollars per order")
    commission = _lazy.need(LIBRARY, why="to build a commission model").finance.commission
    return commission.PerShare(cost=float(cost_per_share),
                               min_trade_cost=float(min_trade_cost))


def from_zipline(perf: pd.DataFrame, *, sessions: Any = None, panel: Any = None,
                 **extra: Any) -> Bundle:
    """zipline's perf frame -> Bundle, keeping returns GROSS.

    `perf.returns` is NET of the slippage and commission zipline applied. The Bundle's
    `returns` slot is documented as GROSS and `cost_curve` applies bps itself, so this
    bridge adds the per-period `slippage` and `commission` columns back onto the return
    before filling the slot, and warns naming the columns it added - silently handing
    `cost_curve` a net series would price execution twice. Turnover comes from
    `perf.transactions`, one-way traded notional over the period's portfolio value.
    """
    if not isinstance(perf, pd.DataFrame) or "returns" not in perf.columns:
        raise TypeError("perf must be the DataFrame zipline's run_algorithm returns "
                        "(it has a 'returns' column)")
    net = perf["returns"].astype(float)
    value = perf.get("portfolio_value")
    gross = net.copy()
    added = []
    if value is not None:
        for col in ("slippage", "commission"):
            if col in perf.columns:
                gross = gross + perf[col].astype(float) / value.astype(float)
                added.append(col)
    slots: dict[str, Any] = {"returns": gross}

    tx = perf.get("transactions")
    if tx is not None and value is not None:
        notional = tx.map(lambda day: sum(abs(float(t.get("amount", 0.0)))
                                          * float(t.get("price", 0.0))
                                          for t in (day or [])))
        slots["turnover"] = (notional.astype(float) / value.astype(float)).fillna(0.0)

    pos = perf.get("positions")
    if pos is not None:
        px = {}
        for day, entries in pos.items():
            for p in (entries or []):
                px.setdefault(str(getattr(p.get("sid", ""), "symbol", p.get("sid", ""))),
                              {})[day] = float(p.get("last_sale_price", np.nan))
        if px:
            slots["prices"] = pd.DataFrame(px).sort_index()

    if "benchmark_return" in perf.columns:
        slots["benchmark_returns"] = perf["benchmark_return"].astype(float)
    ppy = _lazy.periods_per_year(sessions)
    if ppy is not None:
        slots["periods_per_year"] = ppy
    if panel is not None:
        _, listings = _panel_parts(panel)
        if listings is not None:
            slots["listings"] = _normalise_listings(listings)
    slots.update(extra)
    if added:
        warnings.warn(
            f"from_zipline: added {added} back onto perf.returns so the Bundle's `returns` "
            f"slot is GROSS, which is what it documents and what cost_curve assumes. Do not "
            f"also configure a cost model here.", stacklevel=2)
    return Bundle(**slots)


__all__ = ["LIBRARY", "LICENCE", "SurvivorOnlyUniverseError", "VERIFIED_ON",
           "from_zipline", "to_zipline_assets", "to_zipline_commission"]
