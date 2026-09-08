"""Guard: point-in-time fundamentals (fundamental-and-macro-data / pit_fundamentals.py)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.core.pit_fundamentals import naive_latest, pit_facts, restatement_report


def _as_period_values(used: Any) -> pd.Series:
    if isinstance(used, pd.Series):
        return used.astype(float)
    if isinstance(used, pd.DataFrame):
        if "period" not in used.columns or "val" not in used.columns:
            raise TypeError("used must carry 'period' and 'val' columns (as pit_facts returns)")
        return used.set_index("period")["val"].astype(float)
    if isinstance(used, Mapping):
        return pd.Series({str(k): float(v) for k, v in used.items()})
    raise TypeError("used must be a Series indexed by period, a DataFrame with period/val, "
                    "or a {period: value} mapping")


@register
class PitFundamentalsGuard(Guard):
    """Only the vintage that was ON FILE at `as_of` may be used; restatements arrive later.

    Inputs
        facts : the `units.USD` list from SEC companyfacts (dicts with start, end, val,
                accn, fy, fp, form, filed).
        as_of : the decision date.
        used  : optional - the per-period values your pipeline actually consumed
                (Series indexed by period 'start..end', a DataFrame with period/val, or
                a mapping). Each is checked against the point-in-time vintage.
        forms : optional tuple of forms to keep, e.g. ("10-K", "10-Q").

    Fails when a used value is a vintage filed AFTER as_of (the keep='last' trap) or
    belongs to a period not yet on file. Without `used`, the guard reports how many
    periods the naive path would get wrong as a warning - it cannot know what you did.
    """

    name = "pit_fundamentals"
    skill = "fundamental-and-macro-data"
    summary = "Catches fundamentals taken from a restatement filed after the as-of date."
    wraps = ("fin_skills.core.pit_fundamentals.pit_facts",
             "fin_skills.core.pit_fundamentals.naive_latest",
             "fin_skills.core.pit_fundamentals.restatement_report")
    required = ("facts", "as_of")
    optional = ("used", "forms")

    def check(self, facts: Sequence[Mapping[str, Any]], as_of: str | pd.Timestamp,
              used: Any = None, forms: tuple[str, ...] | None = None) -> Outcome:
        out = Outcome()
        if not isinstance(facts, (list, tuple)) or not facts \
                or not all(isinstance(f, Mapping) for f in facts):
            raise TypeError("facts must be a non-empty list of dicts (companyfacts units list)")
        try:
            as_of_ts = pd.Timestamp(as_of)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"as_of must be a date, got {as_of!r}") from exc
        if forms is not None:
            facts = [f for f in facts if f.get("form") in tuple(forms)]
            if not facts:
                raise TypeError(f"no facts left after filtering forms={forms!r}")

        pit = pit_facts(list(facts), as_of_ts)
        naive = naive_latest(list(facts))
        restated = restatement_report(list(facts), by=("start", "end"))
        pit_vals = pit.set_index("period")["val"].astype(float)
        naive_vals = naive.set_index("period")["val"].astype(float)
        common = pit_vals.index.intersection(naive_vals.index)
        leaky = [p for p in common
                 if not np.isclose(pit_vals[p], naive_vals[p], rtol=1e-12, atol=0.0)]
        unfiled = [p for p in naive_vals.index if p not in pit_vals.index]
        out.note(as_of=str(as_of_ts.date()), pit=pit, naive=naive, restatements=restated,
                 n_periods_on_file=int(len(pit)), leaky_periods=leaky, unfiled_periods=unfiled)

        if used is None:
            msg = (f"{len(leaky)} period(s) on file at {as_of_ts.date()} carry a later "
                   f"restatement and {len(unfiled)} period(s) were not filed yet: "
                   f"drop_duplicates(keep='last') would use all of them")
            (out.warning if (leaky or unfiled) else out.info)(msg, where="naive")
            return out

        vals = _as_period_values(used)
        later = naive.set_index("period")
        for period, v in vals.items():
            p = str(period)
            if p not in pit_vals.index:
                filed = later["filed"].get(p)
                stamp = f" (first filed {pd.Timestamp(filed).date()})" if filed is not None \
                    and pd.notna(filed) else ""
                out.error(f"period {p} was not on file at {as_of_ts.date()}{stamp}", where=p)
                continue
            if np.isclose(float(v), float(pit_vals[p]), rtol=1e-12, atol=0.0):
                continue
            if p in naive_vals.index and np.isclose(float(v), float(naive_vals[p]),
                                                    rtol=1e-12, atol=0.0):
                filed = pd.Timestamp(later.loc[p, "filed"]).date()
                out.error(f"value {v:,.0f} for {p} is the vintage filed {filed}, after "
                          f"as_of {as_of_ts.date()}; the point-in-time figure was "
                          f"{float(pit_vals[p]):,.0f}", where=p)
            else:
                out.error(f"value {v:,.0f} for {p} matches no vintage; point-in-time is "
                          f"{float(pit_vals[p]):,.0f}", where=p)
        if out.passed:
            out.info(f"all {len(vals)} used value(s) are the vintage on file at "
                     f"{as_of_ts.date()}")
        return out
