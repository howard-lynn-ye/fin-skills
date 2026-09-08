"""Guard: the risk-free argument's units (lib-quantstats / rf_convention.py).

rf_convention.py's public entry point is a demo; its two computations are the module
functions `_sharpe_annual_rf` (quantstats: annual rate, de-annualised geometrically) and
`_sharpe_per_period_rf` (empyrical: per-period rate, subtracted raw). The guard wraps
those two directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import as_1d, require_int, require_number
from fin_skills.libraries.rf_convention import PERIODS, _sharpe_annual_rf, _sharpe_per_period_rf


@register
class RfConventionGuard(Guard):
    """quantstats' rf= is ANNUAL; empyrical's risk_free= is PER PERIOD. Same 0.05, different Sharpe.

    Inputs
        returns         : per-period strategy returns.
        rf              : the risk-free rate you pass, as an ANNUAL decimal (0.05 = 5%).
        periods         : periods per year. Default 252.
        reported_sharpe : optional - the Sharpe a library gave you; the guard says
                          which convention produced it.

    Computes the Sharpe under both conventions plus the common mis-call (the annual
    rate subtracted from every period). Fails when `reported_sharpe` matches that
    mis-call, or when `rf` looks like a percentage (>= 0.5) rather than a decimal.
    """

    name = "rf_convention"
    skill = "lib-quantstats"
    summary = "Identifies a Sharpe computed with an annual risk-free rate subtracted per period (the < -10 symptom)."
    wraps = ("fin_skills.libraries.rf_convention._sharpe_annual_rf",
             "fin_skills.libraries.rf_convention._sharpe_per_period_rf")
    required = ("returns", "rf")
    optional = ("periods", "reported_sharpe")

    def check(self, returns: pd.Series | np.ndarray, rf: float, periods: int = PERIODS,
              reported_sharpe: float | None = None) -> Outcome:
        out = Outcome()
        r = pd.Series(as_1d(returns, "returns"))
        if len(r) < 3 or not np.isfinite(r).all():
            raise TypeError("returns needs at least 3 finite values")
        rf = require_number(rf, "rf")
        periods = require_int(periods, "periods", minimum=1)
        annual = _sharpe_annual_rf(r, rf, periods)
        per_period_raw = _sharpe_per_period_rf(r, rf, periods)
        per_period_scaled = _sharpe_per_period_rf(r, rf / periods, periods)
        out.note(sharpe_annual_rf_geometric=annual, sharpe_annual_rf_subtracted_raw=per_period_raw,
                 sharpe_rf_over_periods=per_period_scaled, rf=rf, periods=periods)
        if abs(rf) >= 0.5:
            out.error(f"rf={rf:g} looks like a percentage, not a decimal rate (0.05 = 5%)",
                      where="rf")
        if reported_sharpe is None:
            out.info(f"quantstats-style (annual, geometric) {annual:.4f}; empyrical with rf/periods "
                     f"{per_period_scaled:.4f}; the annual rate subtracted raw gives "
                     f"{per_period_raw:.4f}", where="conventions")
            return out
        reported_sharpe = require_number(reported_sharpe, "reported_sharpe")
        cands = {"annual rate, de-annualised geometrically (quantstats rf=)": annual,
                 "per-period rate = rf/periods (empyrical risk_free= done right)": per_period_scaled,
                 "ANNUAL rate subtracted from every period (empyrical risk_free=rf mis-call)":
                     per_period_raw}
        label, val = min(cands.items(), key=lambda kv: abs(kv[1] - reported_sharpe))
        out.note(reported_sharpe=reported_sharpe, closest_convention=label, closest_value=val)
        if label.startswith("ANNUAL"):
            out.error(f"reported Sharpe {reported_sharpe:.4f} matches {label} ({val:.4f}); "
                      f"the correct number is about {annual:.4f}", where="reported_sharpe")
        else:
            out.info(f"reported Sharpe {reported_sharpe:.4f} matches {label} ({val:.4f})",
                     where="reported_sharpe")
        return out
