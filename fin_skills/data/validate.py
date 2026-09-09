"""The schema invariants, as `api.base.Finding` objects rather than exceptions.

A construction error is a TypeError or a ValueError - the object cannot exist. Everything
here is about an object that CAN exist and is still suspicious: an UNKNOWN adjustment, a
float32 volume, a hole big enough to be a delisting, a vintage table whose realtime stamps
never move. Those are findings, and they travel with the data instead of stopping it.

Severity follows the rest of the library: `error` fails, `warning` and `info` do not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fin_skills.api.base import Finding
from fin_skills.data.schema import Adjustment, Bars, Fundamentals, Macro

#: 2**24. float32 counts integers exactly only up to here, and any large-cap's daily share
#: volume is above it - so a float32 volume column is silently rounded data.
FLOAT32_EXACT_INT = 16_777_216


def _f(sev: str, msg: str, where: str = "") -> Finding:
    return Finding(sev, msg, where)


# --------------------------------------------------------------------------------- Bars
def validate_bars(bars: Bars) -> list[Finding]:
    out: list[Finding] = []
    f = bars.frame

    if bars.adjustment is Adjustment.UNKNOWN:
        out.append(_f("error", "adjustment is UNKNOWN: the convention was never "
                               "established, so every return-based number computed from "
                               "this panel is unverifiable. Downstream guards downgrade "
                               "to warnings and the result card cannot claim a "
                               "convention.", "adjustment"))
    elif bars.adjustment.rewrites_history:
        out.append(_f("warning", f"adjustment={bars.adjustment.value} is anchored at the "
                                 f"present: a new split or dividend rewrites this whole "
                                 f"series, so a cached copy drifts from the live one",
                      "adjustment"))

    if bars.adjustment is not Adjustment.RAW and (bars.actions is None
                                                  or not len(bars.actions)):
        out.append(_f("warning", "an adjusted series with no actions table cannot be "
                                 "checked against its own corporate actions; "
                                 "adjustment_check has nothing to test", "actions"))

    if f.index.tz is None:
        out.append(_f("info", f"index labels are naive and declared to be in {bars.tz}; "
                              f"Parquet does not store timezones and CSV degrades a named "
                              f"zone to a fixed offset, so the zone must come from the "
                              f"sidecar, never from the file", "tz"))

    # dtypes - D11
    for col in f.columns:
        name = col[0] if isinstance(col, tuple) else col
        dtype = f[col].dtype
        if name == "volume" and dtype == np.float32:
            out.append(_f("error", f"volume for {col[1]!r} is float32: integers above "
                                   f"{FLOAT32_EXACT_INT:,} are rounded, and that is below "
                                   f"any large-cap's daily volume", f"dtype:{col}"))
        elif dtype == np.float32:
            out.append(_f("warning", f"{col} is float32; this layer stores float64",
                          f"dtype:{col}"))

    close = f["close"] if "close" in bars.fields else None
    if close is not None:
        for t in close.columns:
            s = close[t]
            if (s.dropna() <= 0).any():
                out.append(_f("error", f"{t}: non-positive close price", f"close:{t}"))
            gaps = _internal_gaps(s)
            if gaps:
                out.append(_f("warning", f"{t}: {len(gaps)} internal gap(s) of >5 "
                                         f"consecutive missing bars (first "
                                         f"{gaps[0].date()}) - left as NaN, never filled",
                              f"gaps:{t}"))
        alive = bars.alive()
        years = (f.index.max() - f.index.min()).days / 365.25 if len(f.index) > 1 else 0.0
        if years >= 1.0 and len(alive) > 1 and not alive["ends_early"].any():
            out.append(_f("warning", f"not one of {len(alive)} names ends early in "
                                     f"{years:.1f} yr: that is a current-snapshot screen, "
                                     f"and results from it are an UPPER BOUND",
                          "survivorship"))

    if bars.listings is None:
        out.append(_f("info", "no listings table: survivorship_audit can run on the panel "
                              "alone but cannot turn the statistical tell into evidence",
                      "listings"))
    if not bars.half_open:
        out.append(_f("info", "half_open=False: this panel's [start, end] is INCLUSIVE at "
                              "the right edge, unlike the layer's default contract",
                      "boundaries"))
    return out


def _internal_gaps(s: pd.Series, run: int = 5) -> list[pd.Timestamp]:
    """Starts of runs of >`run` missing values BETWEEN the first and last observation.

    Missing before the first or after the last observation is a listing or a delisting,
    not a gap.
    """
    v = s.dropna()
    if len(v) < 2:
        return []
    inner = s.loc[v.index.min():v.index.max()]
    isna = inner.isna().to_numpy()
    out, streak = [], 0
    for i, na in enumerate(isna):
        if na:
            streak += 1
        else:
            if streak > run:
                out.append(inner.index[i - streak])
            streak = 0
    if streak > run:
        out.append(inner.index[len(isna) - streak])
    return out


# ------------------------------------------------------------------------- Fundamentals
def validate_fundamentals(fun: Fundamentals) -> list[Finding]:
    out: list[Finding] = []
    f = fun.frame

    late = f[f["available_at"] < f["period_end"]]
    if len(late):
        out.append(_f("error", f"{len(late)} row(s) claim to have been knowable BEFORE "
                               f"the period they report ended - a guaranteed look-ahead",
                      "available_at"))
    early = f[f["available_at"] < f["filed_at"]]
    if len(early):
        out.append(_f("error", f"{len(early)} row(s) have available_at before filed_at",
                      "available_at"))
    if (f["acceptance_at"].isna()).any():
        n = int(f["acceptance_at"].isna().sum())
        out.append(_f("warning", f"{n} row(s) carry no acceptance_at, so available_at was "
                                 f"derived from filed_at - a filing date admits a "
                                 f"post-close leak that an acceptance timestamp does not",
                      "acceptance_at"))
    if fun.tz_of_record.upper() != "UTC":
        out.append(_f("warning", f"tz_of_record={fun.tz_of_record!r}: the SEC's own two "
                                 f"APIs disagree - the Submissions API's "
                                 f"acceptanceDateTime is UTC and the Financial Statement "
                                 f"Data Sets' sub.txt.accepted is Eastern. Reading one as "
                                 f"the other moves post-close filings back into the "
                                 f"session", "tz_of_record"))
    rest = fun.restatements()
    if len(rest):
        out.append(_f("info", f"{len(rest)} period(s) were restated: a point-in-time and "
                              f"a naive study will disagree for every one of them",
                      "restatements"))
    amend = f[f["is_amendment"].astype(bool)] if "is_amendment" in f.columns else f.iloc[:0]
    if len(amend):
        out.append(_f("info", f"{len(amend)} amendment row(s) present; as_of() ranks them "
                              f"by available_at like any other vintage", "amendments"))
    return out


# -------------------------------------------------------------------------------- Macro
def validate_macro(macro: Macro) -> list[Finding]:
    out: list[Finding] = []
    f = macro.frame

    bad = f[f["realtime_start"] < f["obs_date"]]
    if len(bad):
        out.append(_f("error", f"{len(bad)} row(s) have realtime_start before obs_date - "
                               f"a figure published before the period it measures",
                      "realtime_start"))
    for sid, g in f.groupby("series_id", sort=True):
        if g["realtime_start"].nunique() == 1:
            out.append(_f("warning", f"{sid}: one realtime_start for the whole series - "
                                     f"this is the current, fully-revised line, not a "
                                     f"vintage matrix. Indexing a January figure at "
                                     f"January is a one-to-three-month look-ahead",
                          f"vintages:{sid}"))
        dup = g.duplicated(["obs_date", "realtime_start"]).sum()
        if dup:
            out.append(_f("error", f"{sid}: {int(dup)} duplicate (obs_date, "
                                   f"realtime_start) row(s)", f"duplicates:{sid}"))
    if "sa_flag" in f.columns and f["sa_flag"].isna().all():
        out.append(_f("info", "no seasonal-adjustment flag on any row; SA and NSA "
                              "versions of the same series are different series",
                      "sa_flag"))
    return out


def summary(findings: list[Finding]) -> str:
    """One line per finding, ASCII, worst first."""
    order = {"error": 0, "warning": 1, "info": 2}
    lines = [str(x) for x in sorted(findings, key=lambda x: order[x.severity])]
    n_err = sum(1 for x in findings if x.severity == "error")
    head = f"{'FAIL' if n_err else 'OK'}  {len(findings)} finding(s), {n_err} error(s)"
    return "\n".join([head] + [f"  {ln}" for ln in lines])


__all__ = ["FLOAT32_EXACT_INT", "summary", "validate_bars", "validate_fundamentals",
           "validate_macro"]
