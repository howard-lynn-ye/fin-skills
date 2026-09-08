"""Guard: cost-sensitivity curve (backtest-validation / cost_curve.py)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_number
from fin_skills.core.cost_curve import breakeven_bps, cost_curve, render

DEFAULT_COST_BPS = 10.0   # the script's own "marginal at 10" - a placeholder, override it


@register
class CostCurveGuard(Guard):
    """A Sharpe is a result AT ONE COST. Report the curve and test survival at a stated cost.

    Inputs
        returns          : per-period GROSS strategy returns (Series or array).
        turnover         : one-way traded notional per period as a fraction of the book
                          (sum |w_t - w_{t-1}|), or a scalar for constant turnover.
        bps              : ROUND-TRIP cost levels for the curve. Default (0, 5, 10, 20, 50).
        cost_bps         : the all-in round-trip cost you claim to pay. Default 10 bps -
                          a placeholder taken from the script's own rule of thumb;
                          pass your measured number.
        benchmark        : hurdle for the breakeven: None (zero), an annualised return,
                          or a per-period return series.
        periods_per_year : annualisation. Default 252.

    Fails when the breakeven cost is at or below `cost_bps`, i.e. the edge does not
    survive the cost you say you pay. Net Sharpe below 0.5 at that cost is a warning
    (the render() verdict scale). Evidence carries the curve and the fixed-width table.
    """

    name = "cost_curve"
    skill = "backtest-validation"
    summary = "Reports Sharpe/return/drawdown across round-trip costs and fails if the edge dies below your stated cost."
    wraps = ("fin_skills.core.cost_curve.cost_curve",
             "fin_skills.core.cost_curve.breakeven_bps",
             "fin_skills.core.cost_curve.render")
    required = ("returns", "turnover")
    optional = ("bps", "cost_bps", "benchmark", "periods_per_year")

    def check(self, returns: pd.Series | np.ndarray, turnover: pd.Series | np.ndarray | float,
              bps: Sequence[float] = (0, 5, 10, 20, 50), cost_bps: float = DEFAULT_COST_BPS,
              benchmark: float | pd.Series | np.ndarray | None = None,
              periods_per_year: int = 252) -> Outcome:
        out = Outcome()
        cost_bps = require_number(cost_bps, "cost_bps")
        if cost_bps < 0:
            raise TypeError("cost_bps must be non-negative")
        if not isinstance(returns, (pd.Series, np.ndarray, list, tuple)):
            raise TypeError("returns must be a Series or array")
        levels = sorted(set(float(b) for b in bps) | {0.0, cost_bps})
        curve = cost_curve(returns, turnover, bps=levels, periods_per_year=periods_per_year)
        be = breakeven_bps(returns, turnover, benchmark=benchmark,
                           periods_per_year=periods_per_year)
        at_cost = curve.loc[cost_bps]
        sharpe_at_cost = float(at_cost["sharpe"])
        out.note(curve=curve, breakeven_bps=be, cost_bps=cost_bps,
                 sharpe_at_cost=sharpe_at_cost, sharpe_gross=float(curve.loc[0.0, "sharpe"]),
                 avg_turnover=curve.attrs.get("avg_turnover"),
                 table=render(curve, be))

        hurdle = "zero" if benchmark is None else "the benchmark hurdle"
        if be <= cost_bps:
            out.error(f"breakeven {be:.1f} bps round-trip is at or below the {cost_bps:.1f} bps "
                      f"you pay: the strategy does not beat {hurdle} after costs "
                      f"(gross Sharpe {curve.loc[0.0, 'sharpe']:.2f}, net {sharpe_at_cost:.2f})",
                      where="breakeven")
        else:
            out.info(f"breakeven {be:.1f} bps vs {cost_bps:.1f} bps paid; net Sharpe "
                     f"{sharpe_at_cost:.2f} at that cost", where="breakeven")
        if np.isfinite(sharpe_at_cost) and 0.0 < sharpe_at_cost < 0.5 and be > cost_bps:
            out.warning(f"net Sharpe {sharpe_at_cost:.2f} at {cost_bps:.1f} bps is marginal",
                        where="sharpe")
        return out
