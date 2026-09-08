"""Guard: look-ahead in a feature function (signal-construction / assert_causal.py)."""
from __future__ import annotations

from typing import Any, Callable, Mapping

import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_frame, require_int
from fin_skills.core.assert_causal import assert_causal, scan_indicators


@register
class CausalityGuard(Guard):
    """Perturb only rows >= k and assert that nothing the function produced before k moved.

    Inputs
        fn   : callable df -> Series/DataFrame indexed like df, OR a {name: callable}
               mapping to scan several indicators at once.
        df   : the input frame (numeric columns are the ones perturbed).
        k    : split index; must sit well past any warm-up. Default len(df) // 2.
        tol  : absolute tolerance on a cell moving. Default 1e-9.
        name : label for a single fn in messages.

    Fails when any output cell before k changed when only rows >= k were perturbed.
    A function that raises on the perturbed frame is reported as a warning, not a
    leak - a broken indicator is not a leakage result.
    """

    name = "assert_causal"
    skill = "signal-construction"
    summary = "Detects look-ahead: perturb rows >= k, assert output before k is unchanged."
    wraps = ("fin_skills.core.assert_causal.assert_causal",
             "fin_skills.core.assert_causal.scan_indicators")
    required = ("fn", "df")
    optional = ("k", "tol", "name")

    def check(self, fn: Callable[[pd.DataFrame], Any] | Mapping[str, Callable[..., Any]],
              df: pd.DataFrame, k: int | None = None, tol: float = 1e-9,
              name: str = "") -> Outcome:
        out = Outcome()
        df = require_frame(df, "df")
        if len(df) < 4:
            raise TypeError("df needs at least 4 rows to split into before/after k")
        k = len(df) // 2 if k is None else require_int(k, "k", minimum=1)
        if k >= len(df):
            raise TypeError(f"k={k} must be < len(df)={len(df)}")
        out.note(k=k, n_rows=len(df), tol=tol)

        if isinstance(fn, Mapping):
            if not fn:
                raise TypeError("fn mapping is empty")
            table = scan_indicators(dict(fn), df, k=k)
            out.note(table=table)
            for row in table.itertuples(index=False):
                if row.causal is True:
                    out.info("causal: no cell before k depends on rows >= k", where=row.indicator)
                elif row.causal is False:
                    out.error(row.error, where=row.indicator)
                else:
                    out.warning(f"indicator raised, not a leakage result: {row.error}",
                                where=row.indicator)
            return out

        if not callable(fn):
            raise TypeError("fn must be callable or a mapping of callables")
        label = name or getattr(fn, "__name__", "fn")
        try:
            assert_causal(fn, df, k=k, tol=tol, name=label)
        except AssertionError as exc:
            out.error(str(exc), where=label)
        else:
            out.info(f"causal: no cell before index {k} depends on rows >= {k}", where=label)
        return out
