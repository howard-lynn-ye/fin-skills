"""Guard: the as-of join audit (market-data-engineering / safe_asof.py)."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_frame, require_str
from fin_skills.core.safe_asof import (RIGHT_TS_COL, assert_row_count_preserved,
                                       assert_strictly_prior, check_key_dtypes, safe_merge_asof)


@register
class AsOfJoinGuard(Guard):
    """Run the as-of join with the caller's settings and audit the four silent leaks.

    Inputs
        left, right         : the two frames (signals on the left, quotes on the right).
        on                  : timestamp column present in both.
        by                  : optional group key(s).
        tolerance           : staleness budget ('5min', '3D', or a float for numeric
                              keys). REQUIRED by the skill; None is reported as an error.
        allow_exact_matches : pandas' default is True; that is the same-stamp leak.
        direction           : 'backward' is the only causal choice.

    Fails when: a matched row used right-hand data stamped at or after the left
    timestamp; the row count changed; a key dtype differs between the sides; or no
    tolerance was stated. Evidence carries the joined frame (with the right-hand
    stamp in `_asof_right_ts`) so the staleness of every match is auditable.
    """

    name = "safe_asof"
    skill = "market-data-engineering"
    summary = "Audits an as-of join: strictly-prior matches, bounded staleness, row count, key dtypes."
    wraps = ("fin_skills.core.safe_asof.safe_merge_asof",
             "fin_skills.core.safe_asof.assert_strictly_prior",
             "fin_skills.core.safe_asof.assert_row_count_preserved",
             "fin_skills.core.safe_asof.check_key_dtypes")
    required = ("left", "right", "on")
    optional = ("by", "tolerance", "allow_exact_matches", "direction")

    def check(self, left: pd.DataFrame, right: pd.DataFrame, on: str,
              by: str | Sequence[str] | None = None, tolerance: Any = None,
              allow_exact_matches: bool = False, direction: str = "backward") -> Outcome:
        out = Outcome()
        left = require_frame(left, "left")
        right = require_frame(right, "right")
        on = require_str(on, "on")
        if on not in left.columns or on not in right.columns:
            raise TypeError(f"on={on!r} must be a column of both frames")
        by_keys = [by] if isinstance(by, str) else list(by or [])
        for k in by_keys:
            if k not in left.columns or k not in right.columns:
                raise TypeError(f"by key {k!r} must be a column of both frames")
        if direction not in ("backward", "forward", "nearest"):
            raise TypeError("direction must be 'backward', 'forward' or 'nearest'")
        out.note(on=on, by=by_keys, tolerance=tolerance, allow_exact_matches=allow_exact_matches,
                 direction=direction, n_left=len(left))

        # 4. key dtype drift: category on one side, object on the other, matches nothing
        try:
            check_key_dtypes(left, right, [on], kind="on")
            check_key_dtypes(left, right, by_keys, kind="by")
        except TypeError as exc:
            out.error(str(exc), where="keys")
            return out

        # 2. unbounded reach-back: no staleness budget stated
        tol = tolerance
        if tol is None:
            out.error("tolerance is REQUIRED: an unbounded as-of join reaches back across "
                      "weekends, halts and delistings and prices a position off a quote that "
                      "may be months old. State the staleness you accept, e.g. '5min' or '3D'.",
                      where="tolerance")
            # continue with an effectively unbounded budget so the other checks still run
            tol = (pd.Timedelta(days=36_500)
                   if pd.api.types.is_datetime64_any_dtype(left[on]) else float("inf"))

        if allow_exact_matches:
            out.warning("allow_exact_matches=True (pandas' default): a quote stamped at the "
                        "signal's own timestamp is information you did not have yet.",
                        where="allow_exact_matches")
        if direction != "backward":
            out.warning(f"direction={direction!r} looks into the future by design.",
                        where="direction")

        try:
            joined = safe_merge_asof(left, right, on=on, by=by_keys or None, tolerance=tol,
                                     allow_exact_matches=allow_exact_matches,
                                     require_prior=False, direction=direction)
        except AssertionError as exc:            # row count changed
            out.error(str(exc), where="rows")
            return out

        # 3. the row-count invariant, asserted again on the returned frame
        try:
            assert_row_count_preserved(left, joined, context=self.name)
        except AssertionError as exc:
            out.error(str(exc), where="rows")

        # 1. strictly prior: the check the caller's settings may have disabled
        try:
            assert_strictly_prior(joined, on, RIGHT_TS_COL)
        except AssertionError as exc:
            out.error(str(exc), where="look-ahead")

        matched = joined[RIGHT_TS_COL].notna()
        n_matched = int(matched.sum())
        staleness = None
        if n_matched:
            gap = joined.loc[matched, on] - joined.loc[matched, RIGHT_TS_COL]
            staleness = gap.max()
            if isinstance(staleness, (np.integer, np.floating)):
                staleness = float(staleness)
        out.note(joined=joined, n_matched=n_matched, n_unmatched=int(len(joined) - n_matched),
                 max_staleness=staleness)
        if out.passed:
            out.info(f"{n_matched}/{len(joined)} rows matched strictly prior within {tolerance!r}; "
                     f"{len(joined) - n_matched} honestly unmatched")
        return out
