"""fin_skills.engine.costs - four slippage models and the three charges a fill actually pays.

A SlippageModel is a callable, so a cost assumption is an argument rather than a constant
buried in the loop, and `cost_curve` can be run against the same number the engine charged.
Slippage is returned in BPS and charged as CASH: the fill price stays the fill bar's own
price, which is what makes "no same-bar fill" testable by reading `fills.price`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class SlippageContext:
    """Everything a slippage model may look at. `history` is the whole panel; a model that
    reads rows after `date` is a look-ahead the engine's own causality test will catch."""

    date: pd.Timestamp
    tickers: pd.Index
    side: pd.Series          # +1 buy, -1 sell
    shares: pd.Series        # ABS shares filled this session
    price: pd.Series         # the un-slipped fill price
    volume: pd.Series        # the FILL bar's volume
    history: "object"        # engine.Panel


SlippageModel = Callable[[SlippageContext], pd.Series]


def _named(fn, label: str, bps_hint: float = 0.0):
    fn.__name__ = label
    fn.bps_hint = float(bps_hint)      # what round_trip_bps() can state without a context
    return fn


def _participation(ctx: SlippageContext) -> pd.Series:
    return (ctx.shares / ctx.volume.replace(0.0, np.nan)).abs().fillna(0.0)


def fixed_bps(bps: float) -> SlippageModel:
    """A flat assumption. Honest only as a placeholder you then vary in a cost curve."""
    def slip(ctx: SlippageContext) -> pd.Series:
        return pd.Series(float(bps), index=ctx.tickers)
    return _named(slip, f"fixed_bps({bps:g})", bps)


def spread_share(fraction: float = 0.5) -> SlippageModel:
    """A share of the spread, proxied by the fill bar's own high-low range in bps."""
    def slip(ctx: SlippageContext) -> pd.Series:
        h = ctx.history.high.loc[ctx.date].reindex(ctx.tickers)
        lo = ctx.history.low.loc[ctx.date].reindex(ctx.tickers)
        return (float(fraction) * ((h - lo) / ctx.price).abs() * 1e4).fillna(0.0)
    return _named(slip, f"spread_share({fraction:g})")


def volume_share(price_impact: float = 0.1) -> SlippageModel:
    """zipline's quadratic model: bps = 1e4 * price_impact * participation**2."""
    def slip(ctx: SlippageContext) -> pd.Series:
        return 1e4 * float(price_impact) * _participation(ctx).pow(2)
    return _named(slip, f"volume_share({price_impact:g})")


def square_root_impact(coef: float = 0.6, sigma_lookback: int = 21) -> SlippageModel:
    """Almgren-style: bps = 1e4 * coef * sigma_daily * sqrt(participation)."""
    def slip(ctx: SlippageContext) -> pd.Series:
        hist = ctx.history.close.loc[:ctx.date].reindex(columns=ctx.tickers)
        sigma = hist.pct_change(fill_method=None).tail(int(sigma_lookback)).std(ddof=1)
        return (1e4 * float(coef) * sigma.fillna(0.0)
                * np.sqrt(_participation(ctx))).fillna(0.0)
    return _named(slip, f"square_root_impact({coef:g})")


def trade_charge(notional: pd.Series, slip_bps: pd.Series, costs) -> pd.Series:
    """Cash cost of one session's fills: commission (with its per-ORDER minimum), the
    half-spread paid in both directions, and the slippage model's bps."""
    comm = (notional * costs.commission_bps / 1e4).clip(lower=float(costs.min_commission))
    comm = comm.where(notional > 0.0, 0.0)
    return comm + notional * (float(costs.spread_bps) + slip_bps.clip(lower=0.0)) / 1e4


def carry_charge(short_notional: float, gross_notional: float, equity: float, costs,
                 periods_per_year: int) -> float:
    """Per-session borrow on the short leg plus financing on gross leverage above 1.0."""
    borrow = short_notional * float(costs.borrow_bps_annual) / 1e4 / periods_per_year
    excess = max(gross_notional - max(equity, 0.0), 0.0)
    return borrow + excess * float(costs.financing_bps_annual) / 1e4 / periods_per_year


__all__ = ["SlippageContext", "SlippageModel", "carry_charge", "fixed_bps", "spread_share",
           "square_root_impact", "trade_charge", "volume_share"]
