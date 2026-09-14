"""Guard: validate delivered daily OHLCV bars without repairing the input."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_frame
from fin_skills.market_data.data_quality import validate_panel


@register
class DataQualityGuard(Guard):
    """Check one instrument's daily bars against a declared session calendar.

    Inputs
        bars            : daily OHLCV DataFrame, indexed by local session dates.
        dates           : optional complete session index for the requested sample.
        max_stale_run   : maximum accepted consecutive repeated closes. Default 3.
        max_stale_frac  : maximum accepted repeated-close fraction. Default 0.10.
        max_abs_return  : absolute simple-return review threshold. Default 0.5.
        outlier_sigma   : robust log-return deviation threshold. Default 8.
        require_volume : whether the OHLCV contract requires volume. Default True.
        periods_per_year: annualization periods for volatility diagnostics. Default 252.

    ``bars`` contains open/high/low/close and volume. ``dates`` is the optional full
    calendar for the requested sample; absence produces a coverage warning. Dates are
    local session labels, so convert closing instants to the exchange timezone first.
    Threshold breaches are review findings, not proof that an economic move is false.
    No row is filled, deleted, clipped or reordered by this guard.
    """

    name = "data_quality"
    skill = "data-quality-validation"
    summary = "Reports invalid OHLCV, stale closes and missing sessions without changing data."
    wraps = ("fin_skills.market_data.data_quality.validate_panel",)
    required = ("bars",)
    optional = ("dates", "max_stale_run", "max_stale_frac", "max_abs_return",
                "outlier_sigma", "require_volume", "periods_per_year")

    def check(self, bars: pd.DataFrame, dates: Sequence | pd.DatetimeIndex | None = None,
              max_stale_run: int = 3, max_stale_frac: float = 0.10,
              max_abs_return: float = 0.5, outlier_sigma: float = 8.0,
              require_volume: bool = True, periods_per_year: float = 252.0) -> Outcome:
        bars = require_frame(bars, "bars")
        try:
            report = validate_panel(bars, sessions=dates, max_stale_run=max_stale_run,
                                    max_stale_frac=max_stale_frac,
                                    max_abs_return=max_abs_return, outlier_sigma=outlier_sigma,
                                    require_volume=require_volume,
                                    periods_per_year=periods_per_year)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"invalid data_quality input: {exc}") from exc
        out = Outcome()
        out.note(n_rows=report.n_rows, counts={c.name: c.count for c in report.checks},
                 **report.stats)
        for check in report.checks:
            message = f"{check.count}: {check.detail}"
            if check.where:
                message += "; examples " + ", ".join(check.where)
            getattr(out, check.severity)(message, where=check.name)
        return out
