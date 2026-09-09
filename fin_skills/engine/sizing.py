"""fin_skills.engine.sizing - target weights from a signal, as five callables.

A Sizer never sees the future: `SizingContext.history` is the panel already sliced to the
decision bar by the engine, so slicing is not something a strategy author can get wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class SizingContext:
    date: pd.Timestamp
    signal: pd.Series        # the signal AS OF the decision bar, universe-restricted
    universe: list
    history: "object"        # engine.Panel, sliced to <= date
    prev_weights: pd.Series
    equity: float
    sessions: "object"       # engine.Sessions


Sizer = Callable[[SizingContext], pd.Series]
_EMPTY = pd.Series(dtype=float)


def _named(fn, label: str) -> Sizer:
    fn.__name__ = label
    return fn


def _live(ctx: SizingContext) -> pd.Series:
    return ctx.signal.reindex(ctx.universe).dropna().astype(float)


def _returns(ctx: SizingContext, names, lookback: int) -> pd.DataFrame:
    return ctx.history.close[list(names)].pct_change(fill_method=None).tail(int(lookback))


def equal_weight(long_only: bool = True, max_names: int | None = None) -> Sizer:
    """1/N over the names the signal likes. The baseline every other sizer has to beat."""
    def size(ctx: SizingContext) -> pd.Series:
        s = _live(ctx)
        if long_only:
            s = s[s > 0]
        if max_names is not None:
            s = s.nlargest(int(max_names))
        return pd.Series(1.0 / len(s), index=s.index) if len(s) else _EMPTY
    return _named(size, f"equal_weight(long_only={long_only}, max_names={max_names})")


def rank_long_short(quantile: float = 0.2, gross: float = 1.0) -> Sizer:
    """Top and bottom quantile, each leg carrying half the gross. Dollar-neutral."""
    def size(ctx: SizingContext) -> pd.Series:
        s = _live(ctx).sort_values()
        k = max(int(len(s) * float(quantile)), 1) if len(s) >= 2 else 0
        if k == 0:
            return _EMPTY
        w = pd.Series(0.0, index=s.index)
        w[s.index[-k:]] = float(gross) / 2.0 / k
        w[s.index[:k]] = -float(gross) / 2.0 / k
        return w
    return _named(size, f"rank_long_short(quantile={quantile:g}, gross={gross:g})")


def vol_target(annual_vol: float = 0.10, lookback: int = 63, cap: float = 3.0) -> Sizer:
    """Scale an equal-weight book to a target ANNUAL vol, levered no more than `cap`.

    Realised vol is measured on the book's own trailing returns, not name by name, so the
    correlation the portfolio actually has is in the number.
    """
    def size(ctx: SizingContext) -> pd.Series:
        base = equal_weight()(ctx)
        if base.empty:
            return base
        r = _returns(ctx, base.index, lookback).fillna(0.0)
        realised = float((r @ base).std(ddof=1)) * np.sqrt(ctx.sessions.periods_per_year)
        return base * (min(float(annual_vol) / realised, float(cap)) if realised > 0 else 0.0)
    return _named(size, f"vol_target(annual_vol={annual_vol:g}, lookback={lookback})")


def fractional_kelly(fraction: float = 0.25, lookback: int = 252, cap: float = 1.0) -> Sizer:
    """f* = mu / sigma^2 per name, times `fraction`, and only where the signal agrees.

    Full Kelly is a theoretical maximum, not a target: estimation error in mu makes it a
    ruin machine, which is why the fraction defaults to a quarter.
    """
    def size(ctx: SizingContext) -> pd.Series:
        s = _live(ctx)
        if s.empty:
            return _EMPTY
        r = _returns(ctx, s.index, lookback)
        f = (float(fraction) * r.mean() / r.var(ddof=1))
        f = f.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-float(cap), float(cap))
        w = f.where(np.sign(f) == np.sign(s), 0.0)
        gross = float(w.abs().sum())
        return w / gross if gross > 1.0 else w
    return _named(size, f"fractional_kelly(fraction={fraction:g}, lookback={lookback})")


def from_weights(weights: pd.DataFrame) -> Sizer:
    """Bring your own optimizer output (PyPortfolioOpt / skfolio / Riskfolio).

    The engine still applies the lag, the participation cap, the costs and the closeouts -
    an optimizer's weights are a target, not a fill.
    """
    frame = pd.DataFrame(weights).sort_index()
    def size(ctx: SizingContext) -> pd.Series:
        rows = frame.loc[:ctx.date]
        if rows.empty:
            return _EMPTY
        return rows.iloc[-1].reindex(ctx.universe).dropna().astype(float)
    return _named(size, f"from_weights({frame.shape[0]}x{frame.shape[1]} frame)")


__all__ = ["Sizer", "SizingContext", "equal_weight", "fractional_kelly", "from_weights",
           "rank_long_short", "vol_target"]
