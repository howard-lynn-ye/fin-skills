"""Guard: price-adjustment convention (market-data-sourcing / adjustment_check.py)."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_series
from fin_skills.market_data.adjustment_check import detect_convention

_CONVENTIONS = ("raw", "back-adjusted", "forward-adjusted", "unknown")


@register
class AdjustmentGuard(Guard):
    """Classify a close series as raw / back-adjusted / forward-adjusted against its actions.

    Inputs
        close         : price Series with a DatetimeIndex (one instrument).
        actions       : DataFrame(date, ratio[, kind]) or a list of (date, ratio) pairs.
        raw_reference : optional unadjusted quotes overlapping the index; upgrades the
                        anchor test from inference to measurement.
        expected      : optional convention you believe you hold; a mismatch fails.
        jump_tol      : tolerance on the log-jump test. Default 0.15.

    Fails when the series is RAW - it jumps by the split ratio on the action date, and
    every return-based rule will trade that jump - or when `expected` disagrees with
    what was detected. An ambiguous series is a warning; no action inside the sample
    is information (nothing to test against).
    """

    name = "adjustment_check"
    skill = "market-data-sourcing"
    summary = "Detects an unadjusted (raw) price series, or one whose convention is not the one you expect."
    wraps = ("fin_skills.market_data.adjustment_check.detect_convention",)
    required = ("close", "actions")
    optional = ("raw_reference", "expected", "jump_tol")

    def check(self, close: pd.Series, actions: pd.DataFrame | Sequence[tuple],
              raw_reference: pd.Series | None = None, expected: str | None = None,
              jump_tol: float = 0.15) -> Outcome:
        out = Outcome()
        close = require_series(close, "close")
        if not isinstance(close.index, pd.DatetimeIndex):
            raise TypeError("close must have a DatetimeIndex")
        if not isinstance(actions, pd.DataFrame) and not isinstance(actions, (list, tuple)):
            raise TypeError("actions must be a DataFrame(date, ratio) or a list of (date, ratio)")
        if raw_reference is not None:
            raw_reference = require_series(raw_reference, "raw_reference")
        if expected is not None and expected not in _CONVENTIONS:
            raise TypeError(f"expected must be one of {_CONVENTIONS}, got {expected!r}")

        res = detect_convention(close, actions, raw_reference=raw_reference, jump_tol=jump_tol)
        conv = res["convention"]
        out.note(**{k: v for k, v in res.items()})

        if conv == "raw":
            out.error(f"series is RAW ({res['confidence']} confidence): {res['reason']}. "
                      f"pct_change() books the split as a return; use it for price LEVELS "
                      f"only, or adjust before computing anything return-based.",
                      where="convention")
        elif conv == "unknown":
            if res.get("confidence") == "none" and "no corporate action" in res.get("reason", ""):
                out.info(res["reason"], where="convention")
            else:
                out.warning(f"convention could not be classified: {res['reason']}",
                            where="convention")
        else:
            out.info(f"{conv} ({res['confidence']} confidence): {res['reason']}",
                     where="convention")
            if conv == "back-adjusted":
                out.info("back-adjusted: every new corporate action rewrites the whole "
                         "history, so cached copies drift from the live series",
                         where="convention")

        if expected is not None and conv != expected:
            out.error(f"expected {expected!r} but detected {conv!r}: any price-level rule "
                      f"(share sizing, penny filters, live reconciliation) is on the wrong scale",
                      where="expected")
        return out
