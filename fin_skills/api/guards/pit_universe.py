"""Guard: point-in-time universe stability (research-integrity-guards / pit_universe.py)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.core.pit_universe import audit_universe_stability, rebalance_universe


@register
class PitUniverseGuard(Guard):
    """A point-in-time universe LOSES names. One that never does was built from today's list.

    Inputs (either `universe`, or `members` + `rebalance_dates`)
        universe              : {date: [tickers]} as produced at each rebalance.
        members               : membership table with ticker / start_date / end_date,
                                used with `rebalance_dates` to build the universe via
                                rebalance_universe (trailing-ADV screen if `liquidity`
                                and `min_adv` are given).
        rebalance_dates       : the dates to rebalance on.
        liquidity, min_adv, adv_lookback : optional trailing liquidity screen.
        min_expected_turnover : one-way turnover floor per rebalance. Default 0.005.

    Fails when no name is ever removed across the rebalances (current-snapshot screen)
    or when turnover sits below the floor (suspiciously static). Evidence carries the
    per-rebalance sizes, additions, removals and turnover.
    """

    name = "pit_universe"
    skill = "research-integrity-guards"
    summary = "Flags a universe that never deletes a name: the signature of a current-snapshot screen."
    wraps = ("fin_skills.core.pit_universe.audit_universe_stability",
             "fin_skills.core.pit_universe.rebalance_universe")
    required = ()
    optional = ("universe", "members", "rebalance_dates", "liquidity", "min_adv",
                "adv_lookback", "min_expected_turnover")

    def missing(self, inputs: Mapping[str, Any]) -> list[str]:
        if "universe" in inputs or ("members" in inputs and "rebalance_dates" in inputs):
            return []
        return ["universe or (members + rebalance_dates)"]

    def check(self, universe: Mapping[Any, Sequence[str]] | None = None,
              members: pd.DataFrame | None = None,
              rebalance_dates: Sequence | pd.DatetimeIndex | None = None,
              liquidity: pd.DataFrame | None = None, min_adv: float | None = None,
              adv_lookback: int = 21, min_expected_turnover: float = 0.005) -> Outcome:
        out = Outcome()
        if universe is None:
            if not isinstance(members, pd.DataFrame):
                raise TypeError("members must be a DataFrame with ticker/start_date/end_date")
            if rebalance_dates is None or len(rebalance_dates) < 2:
                raise TypeError("rebalance_dates needs at least two dates")
            if liquidity is not None and not isinstance(liquidity, pd.DataFrame):
                raise TypeError("liquidity must be a DataFrame (dates x tickers)")
            universe = rebalance_universe(rebalance_dates, members, liquidity=liquidity,
                                          min_adv=min_adv, adv_lookback=adv_lookback)
            out.note(built_from_members=True)
            if liquidity is not None and min_adv is None:
                out.warning("liquidity was supplied without min_adv, so no screen was applied",
                            where="liquidity")
        elif not isinstance(universe, Mapping):
            raise TypeError("universe must be a mapping {date: [tickers]}")
        if len(universe) < 2:
            raise TypeError("need at least two rebalance dates to measure turnover")

        audit = audit_universe_stability(universe, min_expected_turnover=min_expected_turnover)
        out.note(verdict=audit.verdict, n_dates=audit.n_dates, sizes=audit.sizes,
                 additions=audit.additions, removals=audit.removals, turnover=audit.turnover,
                 mean_turnover=audit.mean_turnover, total_additions=audit.total_additions,
                 total_removals=audit.total_removals, n_ever=audit.n_ever,
                 never_loses_a_name=audit.never_loses_a_name, report=audit.report())

        if audit.never_loses_a_name:
            out.error(audit.verdict, where="removals")
        elif audit.verdict.startswith("SUSPICIOUSLY"):
            out.error(audit.verdict, where="turnover")
        else:
            out.info(audit.verdict, where="turnover")
        for n in audit.notes:
            out.warning(n, where="notes")
        return out
