"""Guard: two price sources reconciled (market-data-sourcing / adjustment_check.py)."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_series
from fin_skills.market_data.adjustment_check import reconcile, reconcile_report


@register
class ReconcileSourcesGuard(Guard):
    """Say WHY two sources for the same instrument differ: convention, action, or bad data.

    Inputs
        close, other : the two price Series (DatetimeIndex) to compare.
        actions      : optional corporate actions, DataFrame(date, ratio[, kind]) or
                       (date, ratio) pairs, used to attribute each step in the ratio.
        tol_bps      : level tolerance in basis points. Default 10.
        window_days  : how close an action must be to explain a step. Default 3.

    Fails on a DATA ERROR (a step in close/other with no corporate action near it) or
    when the two series do not overlap. A pure convention difference (constant level
    offset, identical returns) is a warning: safe for returns, wrong for any
    price-level rule.
    """

    name = "reconcile_sources"
    skill = "market-data-sourcing"
    summary = "Attributes every divergence between two price sources to a convention, an action, or a data error."
    wraps = ("fin_skills.market_data.adjustment_check.reconcile",
             "fin_skills.market_data.adjustment_check.reconcile_report")
    required = ("close", "other")
    optional = ("actions", "tol_bps", "window_days")

    def check(self, close: pd.Series, other: pd.Series,
              actions: pd.DataFrame | Sequence[tuple] | None = None,
              tol_bps: float = 10.0, window_days: int = 3) -> Outcome:
        out = Outcome()
        close = require_series(close, "close")
        other = require_series(other, "other")
        res = reconcile(close, other, actions, tol_bps=tol_bps, window_days=window_days)
        verdict = str(res["verdict"])
        out.note(verdict=verdict, n_common=res.get("n_common"),
                 n_divergent_dates=res.get("n_divergent_dates"),
                 max_abs_rel_diff=res.get("max_abs_rel_diff"),
                 max_abs_return_diff=res.get("max_abs_return_diff"),
                 steps=res.get("steps"))
        if res.get("n_common", 0):
            out.note(report=reconcile_report(res))

        if verdict.startswith("NO OVERLAP"):
            out.error("the two series share no dates; nothing can be reconciled", where="overlap")
        elif verdict.startswith("DATA ERROR"):
            out.error(verdict, where="steps")
        elif verdict.startswith("ADJUSTMENT CONVENTION ONLY"):
            out.warning(verdict, where="levels")
        elif verdict.startswith("ADJUSTMENT DIFFERENCE"):
            out.warning(verdict, where="steps")
        else:
            out.info(verdict)
        return out
