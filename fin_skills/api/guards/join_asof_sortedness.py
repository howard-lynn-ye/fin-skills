"""Guard: unsorted input into a cursor-based as-of join (lib-polars / join_asof_sortedness.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_frame, require_str
from fin_skills.libraries.join_asof_sortedness import asof_backward_correct, asof_backward_cursor


@register
class AsOfSortednessGuard(Guard):
    """polars' join_asof(by=) cannot check sortedness; unsorted input matches FUTURE quotes.

    Inputs
        left, right : signals and quotes, as you would hand them to join_asof.
        on          : timestamp column.
        by          : group key column.
        val         : right-hand value column to compare. Default: the first column of
                      `right` that is neither `on` nor `by`.

    The script's two reference implementations are run on your frames: the forward-only
    per-group cursor (what a sorted-merge join does) and an order-independent
    searchsorted join. Fails when they disagree on any row - the rows a real join_asof
    would silently match to a quote from the future. Row counts do not catch this.
    """

    name = "join_asof_sortedness"
    skill = "lib-polars"
    summary = "Fails frames whose order would make a cursor-based join_asof (polars with by=) return future quotes."
    wraps = ("fin_skills.libraries.join_asof_sortedness.asof_backward_cursor",
             "fin_skills.libraries.join_asof_sortedness.asof_backward_correct")
    required = ("left", "right", "on", "by")
    optional = ("val",)

    def check(self, left: pd.DataFrame, right: pd.DataFrame, on: str, by: str,
              val: str | None = None) -> Outcome:
        out = Outcome()
        left = require_frame(left, "left")
        right = require_frame(right, "right")
        on, by = require_str(on, "on"), require_str(by, "by")
        for col in (on, by):
            if col not in left.columns or col not in right.columns:
                raise TypeError(f"{col!r} must be a column of both frames")
        if val is None:
            rest = [c for c in right.columns if c not in (on, by)]
            if not rest:
                raise TypeError("right has no value column besides on/by; pass val=")
            val = rest[0]
        if val not in right.columns:
            raise TypeError(f"val={val!r} is not a column of right")
        if len(left) == 0:
            raise TypeError("left is empty")

        cursor = asof_backward_cursor(left, right, on, by, val).to_numpy(dtype=float)
        correct = asof_backward_correct(left, right, on, by, val).to_numpy(dtype=float)
        same = (np.isnan(cursor) & np.isnan(correct)) | (np.abs(cursor - correct) < 1e-12)
        n_bad = int((~same).sum())
        left_sorted = bool(left.sort_values([by, on], kind="mergesort").index.equals(left.index))
        right_sorted = bool(right.sort_values([by, on], kind="mergesort").index.equals(right.index))
        out.note(n_rows=int(len(left)), n_wrong=n_bad, left_sorted_by_group_then_time=left_sorted,
                 right_sorted_by_group_then_time=right_sorted, val=val)
        if n_bad:
            out.error(f"{n_bad} of {len(left)} rows would be matched to a different (later) "
                      f"quote by a forward-only cursor join: sort BOTH frames by [{by!r}, "
                      f"{on!r}] before join_asof, carry the quote timestamp through, and "
                      f"assert quote_time <= time", where="order")
        else:
            out.info(f"cursor join and order-independent join agree on all {len(left)} rows",
                     where="order")
        if not left_sorted or not right_sorted:
            out.warning(f"left sorted by [{by}, {on}]: {left_sorted}; right: {right_sorted} - "
                        f"polars warns on every by= call and raises on none of them",
                        where="order")
        return out
