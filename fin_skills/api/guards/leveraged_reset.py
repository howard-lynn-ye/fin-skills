"""Guard: daily-reset leveraged products (etf-mechanics / leveraged_reset.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import as_1d, require_number
from fin_skills.core.leveraged_reset import analytic_lev_return, leveraged_wealth


@register
class LeveragedResetGuard(Guard):
    """'k times the index' is a one-day statement; over any path the product compounds daily.

    Inputs
        index_returns   : daily simple returns of the index over the holding period.
        lev             : the product's leverage (3, -1, -3, ...).
        modelled_return : optional - the product return your backtest assumed over the
                          same days (typically lev x the index's total return).
        financing       : annual rate on the borrowed (lev - 1) x NAV. Default 0.
        expense         : annual expense ratio. Default 0.
        return_tol      : absolute tolerance on the return gap. Default 1e-4.

    Fails when `modelled_return` differs from the daily-reset product's exact return
    by more than `return_tol`. Without it, the guard reports the gap between lev x index and
    the exact path result as information, together with the analytic drag estimate.
    """

    name = "leveraged_reset"
    skill = "etf-mechanics"
    summary = "Fails a backtest that models a daily-reset leveraged product as lev x the index return."
    wraps = ("fin_skills.core.leveraged_reset.leveraged_wealth",
             "fin_skills.core.leveraged_reset.analytic_lev_return")
    required = ("index_returns", "lev")
    optional = ("modelled_return", "financing", "expense", "return_tol")

    def check(self, index_returns: pd.Series | np.ndarray, lev: float,
              modelled_return: float | None = None, financing: float = 0.0,
              expense: float = 0.0, return_tol: float = 1e-4) -> Outcome:
        out = Outcome()
        r = as_1d(index_returns, "index_returns")
        if len(r) < 1 or not np.isfinite(r).all():
            raise TypeError("index_returns needs at least one finite value")
        lev = require_number(lev, "lev")
        if lev == 0:
            raise TypeError("lev must be non-zero")
        exact = float(leveraged_wealth(r, lev, financing=financing, expense=expense)[-1] - 1.0)
        index_total = float(np.prod(1.0 + r) - 1.0)
        naive = lev * index_total
        approx = float(analytic_lev_return(index_total, float(np.sum(r ** 2)), lev))
        out.note(exact_product_return=exact, index_total_return=index_total,
                 lev_times_index=naive, analytic_approximation=approx, n_days=int(len(r)),
                 gap_vs_lev_times_index=exact - naive)
        if modelled_return is not None:
            modelled_return = require_number(modelled_return, "modelled_return")
            gap = modelled_return - exact
            out.note(modelled_return=modelled_return, gap_vs_exact=gap)
            if abs(gap) > return_tol:
                out.error(f"modelled {modelled_return:+.4%} vs daily-reset exact {exact:+.4%} "
                          f"over {len(r)} days (gap {gap:+.4%}); lev x index is "
                          f"{naive:+.4%}", where="modelled_return")
            else:
                out.info(f"modelled return matches the daily-reset product within {return_tol:g}",
                         where="modelled_return")
        else:
            out.info(f"{lev:g}x product over {len(r)} days: exact {exact:+.4%}, lev x index "
                     f"{naive:+.4%}, analytic approximation {approx:+.4%}", where="reset")
        return out
