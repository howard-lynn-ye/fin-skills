"""Guard: Brinson attribution must reconcile (portfolio-and-risk / brinson_attribution.py)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.core.brinson_attribution import (ReconciliationError, brinson_single_period,
                                                 multi_period_attribution, render)


@register
class BrinsonGuard(Guard):
    """The effects must sum to Rp - Rb exactly; when they do not, the DATA is wrong.

    Inputs (either `panel` or `panels`)
        panel      : DataFrame indexed by sector with wp, wb, rp, rb for one period.
        panels     : a sequence of such frames, Carino-linked across periods.
        method     : 'fachler' (default) or 'bhb'.
        tol        : reconciliation tolerance. Default 1e-10.
        weight_tol : tolerance on sum(wp) == sum(wb) == 1. Default 1e-8.

    Fails when the weights do not sum to one (a missing cash line, weights and returns
    from different timestamps) or when the effects do not reconcile to the active
    return. Evidence carries the effects table and the rendered report.
    """

    name = "brinson_attribution"
    skill = "portfolio-and-risk"
    summary = "Fails an attribution whose effects do not reconcile to the active return (a data error, never rounding)."
    wraps = ("fin_skills.core.brinson_attribution.brinson_single_period",
             "fin_skills.core.brinson_attribution.multi_period_attribution",
             "fin_skills.core.brinson_attribution.render")
    required = ()
    optional = ("panel", "panels", "method", "tol", "weight_tol")

    def missing(self, inputs: Mapping[str, Any]) -> list[str]:
        return [] if ("panel" in inputs or "panels" in inputs) else ["panel or panels"]

    def check(self, panel: pd.DataFrame | None = None,
              panels: Sequence[pd.DataFrame] | None = None, method: str = "fachler",
              tol: float = 1e-10, weight_tol: float = 1e-8) -> Outcome:
        out = Outcome()
        if panel is None and panels is None:
            raise TypeError("give panel (one period) or panels (several)")
        if method not in ("fachler", "bhb"):
            raise TypeError("method must be 'fachler' or 'bhb'")
        try:
            if panels is not None:
                frames = list(panels)
                if not frames or not all(isinstance(p, pd.DataFrame) for p in frames):
                    raise TypeError("panels must be a non-empty sequence of DataFrames")
                eff = multi_period_attribution(frames, method=method, tol=tol)
                scope = f"{len(frames)} periods, Carino-linked"
            else:
                if not isinstance(panel, pd.DataFrame):
                    raise TypeError("panel must be a DataFrame with wp, wb, rp, rb")
                eff = brinson_single_period(panel, method=method, tol=tol, weight_tol=weight_tol)
                scope = "single period"
        except ReconciliationError as exc:
            out.error(str(exc), where="reconciliation")
            return out
        out.note(effects=eff, report=render(eff), active=eff.attrs.get("active"),
                 rp=eff.attrs.get("rp"), rb=eff.attrs.get("rb"), method=method)
        out.info(f"{scope}: effects reconcile to active return "
                 f"{eff.attrs.get('active', float('nan')) * 1e4:+.1f} bps", where="reconciliation")
        return out
