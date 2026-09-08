"""Guard: which regimes the test period contained (regime-detection / regime_coverage.py)."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import as_1d, require_int
from fin_skills.core.regime_coverage import regime_table


@register
class RegimeCoverageGuard(Guard):
    """A single bull quarter is not a backtest: count regimes AND episodes in the test window.

    Inputs
        dates    : DatetimeIndex (or dates) of the test period.
        asset    : per-period asset returns.
        strategy : per-period strategy returns.
        position : per-period position (0 = flat).
        labels   : regime label per period (strings).
        how      : how the label was defined - an EX-ANTE rule fixed before the test
                   period, or an EX-POST label read off it. Default "unstated", which
                   is reported as a warning.
        min_regimes  : distinct labels required. Default 2.
        min_episodes : episodes per regime below which the per-regime number rests on
                       one observation. Default 2.

    Fails when the test period contains fewer than `min_regimes` distinct labels. A
    regime seen in fewer than `min_episodes` episodes is a warning. Evidence carries
    the per-regime table and the strings to paste into ResultCard.regimes_covered.
    """

    name = "regime_coverage"
    skill = "regime-detection"
    summary = "Fails a test period that contains a single regime; reports episodes, not just days, per regime."
    wraps = ("fin_skills.core.regime_coverage.regime_table",)
    required = ("dates", "asset", "strategy", "position", "labels")
    optional = ("how", "min_regimes", "min_episodes")

    def check(self, dates: Sequence[Any] | pd.DatetimeIndex, asset: Any, strategy: Any,
              position: Any, labels: Sequence[Any] | pd.Series, how: str = "unstated",
              min_regimes: int = 2, min_episodes: int = 2) -> Outcome:
        out = Outcome()
        try:
            idx = pd.DatetimeIndex(dates)
        except (TypeError, ValueError) as exc:
            raise TypeError("dates must be convertible to a DatetimeIndex") from exc
        a, s, p = as_1d(asset, "asset"), as_1d(strategy, "strategy"), as_1d(position, "position")
        lab = pd.Series(np.asarray(labels).astype(object)).reset_index(drop=True)
        n = len(idx)
        if not (len(a) == len(s) == len(p) == len(lab) == n):
            raise TypeError(f"dates ({n}), asset ({len(a)}), strategy ({len(s)}), position "
                            f"({len(p)}) and labels ({len(lab)}) must have equal length")
        if n < 2:
            raise TypeError("need at least two periods")
        min_regimes = require_int(min_regimes, "min_regimes", minimum=1)
        min_episodes = require_int(min_episodes, "min_episodes", minimum=1)
        if not isinstance(how, str) or not how:
            raise TypeError("how must be a non-empty string")

        table, strings = regime_table(idx, a, s, p, lab, how)
        out.note(table=table, regimes_covered=strings, n_regimes=int(len(table)), how=how)
        if how == "unstated":
            out.warning("say whether each label is an EX-ANTE rule (threshold fixed before "
                        "the test period) or an EX-POST label read off it", where="how")
        if len(table) < min_regimes:
            out.error(f"test period {idx[0].date()}..{idx[-1].date()} contains "
                      f"{len(table)} regime label(s) (need {min_regimes}); a single regime "
                      f"is not a backtest", where="regimes")
        else:
            out.info(f"{len(table)} regimes over {n} periods", where="regimes")
        for lab_name, row in table.iterrows():
            if int(row["n_episodes"]) < min_episodes:
                out.warning(f"{int(row['n_episodes'])} episode(s) of {int(row['n_obs'])} obs: "
                            f"one observation of this regime, whatever the day count",
                            where=str(lab_name))
        return out
