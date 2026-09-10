#!/usr/bin/env python3
"""Point-in-time SEC fundamentals: the value that was actually KNOWN on a past date.

WHY: `companyfacts` returns EVERY vintage of every period -- the original 10-Q figure and
each later restatement, one row per accession, each stamped with `filed`. The natural
one-liner, `drop_duplicates(subset=['start','end'], keep='last')`, silently selects the
restated number, which nobody could have seen at the time. The backtest then trades on a
figure published months into the future. Nothing errors; the alpha is just fake.

Two separate look-aheads live here and both are handled:
  1. VINTAGE  -- filter to `filed <= as_of`, then take the last vintage per period.
  2. INTRADAY -- `filed` is EDGAR's assigned date, not the wall clock. Use
     `acceptanceDateTime`; a filing accepted at or after 16:00 exchange-local is first
     tradeable on the NEXT session. Periodic reports cluster just after the 17:30 ET
     cutoff, so this is systematic, not an edge case.

Usage:
    from pit_fundamentals import pit_facts, restatement_report, available_at

    facts = requests.get(COMPANYFACTS_URL).json()["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    known = pit_facts(facts, as_of="2023-01-15")       # only what existed on that date
    print(restatement_report(facts))                   # which periods were rewritten
    print(available_at("2026-07-30T20:30:28Z").first_tradeable_session)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

FACT_FIELDS = ("start", "end", "val", "accn", "fy", "fp", "form", "filed")


# --------------------------------------------------------------------------------------
# vintage handling
# --------------------------------------------------------------------------------------
def facts_to_frame(facts: list[dict]) -> pd.DataFrame:
    """Normalise a `units.USD` fact list into a typed frame.

    Balance-sheet tags are instantaneous and carry no `start`; the period key has to
    tolerate that or those facts vanish from every groupby.
    """
    df = pd.DataFrame(list(facts))
    for col in FACT_FIELDS:
        if col not in df.columns:
            df[col] = pd.NA
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["period"] = (df["start"].dt.strftime("%Y-%m-%d").fillna("instant")
                    + ".." + df["end"].dt.strftime("%Y-%m-%d").fillna("?"))
    return df


def pit_facts(facts: list[dict], as_of: str | pd.Timestamp,
              forms: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Latest vintage of each period that was on file at `as_of`. One row per period.

    tail(1) rather than groupby().last(): GroupBy.last() takes the last NON-NULL value
    per column independently, so a row with a missing `fp` can inherit `fp` from an older
    vintage and produce a period that was never filed in that shape.
    """
    df = facts_to_frame(facts)
    if forms is not None:
        df = df[df["form"].isin(forms)]
    df = df[df["filed"] <= pd.Timestamp(as_of)]
    out = (df.sort_values(["filed", "accn"], kind="mergesort")
             .groupby("period", sort=False).tail(1))
    return out.sort_values("end").reset_index(drop=True)


def naive_latest(facts: list[dict]) -> pd.DataFrame:
    """The WRONG answer, kept here so tests can assert the two differ.

    This is `keep='last'` with no vintage filter: for every period it returns the most
    recently filed value, including restatements published after the as-of date.
    """
    df = facts_to_frame(facts)
    out = (df.sort_values(["filed", "accn"], kind="mergesort")
             .groupby("period", sort=False).tail(1))
    return out.sort_values("end").reset_index(drop=True)


def restatement_report(facts: list[dict],
                       by: tuple[str, ...] = ("start", "end", "form"),
                       rel_tol: float = 1e-9) -> pd.DataFrame:
    """Periods reported with different values across vintages.

    Default key is (start, end, form) -- the same period re-reported under the same form
    in a later filing, e.g. a prior-year comparative inside next year's 10-Q. Pass
    by=("start","end") to also catch cross-form rewrites (10-Q original vs 10-K/A).

    Any row here means a PIT study and a naive study will disagree for that period.
    """
    df = facts_to_frame(facts)
    keys = [k for k in by if k in df.columns]
    rows = []
    for key, g in df.groupby(keys, dropna=False, sort=False):
        g = g.sort_values("filed", kind="mergesort")
        vals = g["val"].dropna().to_numpy()
        if len(vals) < 2:
            continue
        spread = float(vals.max() - vals.min())
        if abs(spread) <= rel_tol * max(abs(float(vals.max())), 1.0):
            continue
        first, last = float(vals[0]), float(vals[-1])
        rows.append({
            "period": g["period"].iloc[0],
            "form": g["form"].iloc[0] if "form" in keys else "|".join(sorted(set(g["form"]))),
            "n_vintages": int(len(g)),
            "first_filed": g["filed"].iloc[0].date(),
            "last_filed": g["filed"].iloc[-1].date(),
            "original_val": first,
            "restated_val": last,
            "pct_change": (last - first) / abs(first) if first else float("nan"),
            "accns": ", ".join(g["accn"].astype(str)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# availability timestamp
# --------------------------------------------------------------------------------------
@dataclass
class Availability:
    acceptance_utc: pd.Timestamp
    exchange_local: pd.Timestamp
    post_close: bool
    first_tradeable_session: pd.Timestamp

    def __str__(self) -> str:
        return (f"{self.acceptance_utc}  ->  {self.exchange_local:%Y-%m-%d %H:%M %Z} "
                f"({'POST-CLOSE' if self.post_close else 'intraday'})  ->  first tradeable "
                f"session {self.first_tradeable_session:%Y-%m-%d} "
                f"({self.first_tradeable_session:%a})")


def available_at(acceptance_dt_utc: str | pd.Timestamp,
                 exchange_tz: str = "America/New_York",
                 sessions: pd.DatetimeIndex | None = None,
                 close: time = time(16, 0),
                 tz_in: str = "UTC") -> Availability:
    """Earliest session a filing accepted at `acceptance_dt_utc` could be traded on.

    Rule: convert to exchange-local; if the local time is at or after the close, the
    earliest tradeable bar is the NEXT session.

    tz_in exists because of a verified, undocumented inconsistency: the Submissions API's
    `acceptanceDateTime` is genuine UTC, while the Financial Statement Data Sets'
    `sub.txt.accepted` is Eastern. Feeding an Eastern stamp in as UTC shifts every filing
    four or five hours early and quietly moves post-close filings back into the session.

    Without `sessions` this only skips weekends -- it has no holiday calendar, so pass the
    exchange's real session index whenever you have one.
    """
    ts = pd.Timestamp(acceptance_dt_utc)
    ts = ts.tz_localize(tz_in) if ts.tz is None else ts
    local = ts.tz_convert(exchange_tz)
    post_close = local.time() >= close

    candidate = local.normalize().tz_localize(None) + pd.Timedelta(days=1 if post_close else 0)
    if sessions is not None and len(sessions):
        idx = pd.DatetimeIndex(sessions).sort_values()
        pos = idx.searchsorted(candidate, side="left")
        if pos >= len(idx):
            raise ValueError(f"no session on or after {candidate.date()} in the supplied calendar")
        session = idx[pos]
    else:
        session = candidate
        while session.weekday() >= 5:      # Sat/Sun: not a session anywhere in the US
            session += pd.Timedelta(days=1)
    return Availability(acceptance_utc=ts.tz_convert("UTC"), exchange_local=local,
                        post_close=post_close, first_tradeable_session=session)


# --------------------------------------------------------------------------------------
# a hand-written companyfacts fragment with a real restatement shape
# --------------------------------------------------------------------------------------
DEMO_FACTS: list[dict] = [
    # FY2022 Q3 revenue, as originally reported in the Q3 10-Q
    {"start": "2022-07-01", "end": "2022-09-30", "val": 1_000_000_000, "accn": "0000000-22-001",
     "fy": 2022, "fp": "Q3", "form": "10-Q", "filed": "2022-11-03"},
    # ...restated DOWN in an amended 10-Q filed three months later
    {"start": "2022-07-01", "end": "2022-09-30", "val": 940_000_000, "accn": "0000000-23-001",
     "fy": 2022, "fp": "Q3", "form": "10-Q/A", "filed": "2023-02-14"},
    # ...and carried at the restated level as next year's comparative
    {"start": "2022-07-01", "end": "2022-09-30", "val": 940_000_000, "accn": "0000000-23-014",
     "fy": 2023, "fp": "Q3", "form": "10-Q", "filed": "2023-11-02"},
    # a clean period, never restated
    {"start": "2022-10-01", "end": "2022-12-31", "val": 1_120_000_000, "accn": "0000000-23-002",
     "fy": 2022, "fp": "Q4", "form": "10-K", "filed": "2023-02-03"},
    # same form re-reporting the same period at a different value: caught by the default key
    {"start": "2022-10-01", "end": "2022-12-31", "val": 1_090_000_000, "accn": "0000000-24-002",
     "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2024-02-02"},
    # instantaneous balance-sheet fact: no `start`, must survive the period grouping
    {"start": None, "end": "2022-12-31", "val": 48_300_000_000, "accn": "0000000-23-002",
     "fy": 2022, "fp": "FY", "form": "10-K", "filed": "2023-02-03"},
]


if __name__ == "__main__":
    pd.set_option("display.width", 130)
    AS_OF = "2023-01-15"

    print("=== restatements across vintages ===")
    print(restatement_report(DEMO_FACTS, by=("start", "end")).to_string(index=False))

    print(f"\n=== FY2022 Q3 revenue as of {AS_OF} ===")
    per = "2022-07-01..2022-09-30"
    naive = naive_latest(DEMO_FACTS).set_index("period").loc[per]
    pit = pit_facts(DEMO_FACTS, AS_OF).set_index("period").loc[per]
    print(f"  naive keep='last' : {naive['val']:>15,.0f}   "
          f"from {naive['form']:<6} filed {naive['filed']:%Y-%m-%d}")
    print(f"  point-in-time     : {pit['val']:>15,.0f}   "
          f"from {pit['form']:<6} filed {pit['filed']:%Y-%m-%d}")
    gap = (naive["val"] - pit["val"]) / pit["val"]
    print(f"  -> the naive path reports a {gap:+.1%} different figure, sourced from a filing "
          f"made {(naive['filed'] - pd.Timestamp(AS_OF)).days} days AFTER the as-of date.")

    later = "2023-06-30"
    print(f"\n  every period on file at {later} (the start-less balance-sheet fact survives "
          f"the grouping, and Q3 still shows its restated-by-then value):")
    print(pit_facts(DEMO_FACTS, later)[["period", "form", "filed", "val"]].to_string(index=False))

    print("\n=== availability: acceptanceDateTime -> first tradeable session ===")
    for stamp, label in [("2026-07-30T20:30:28Z", "verified AAPL 8-K, 16:30 ET"),
                         ("2026-07-30T19:30:00Z", "same day, 15:30 ET"),
                         ("2026-07-31T21:05:00Z", "Friday 17:05 ET, rolls the weekend")]:
        print(f"  {label:<32} {available_at(stamp)}")
    print("\n  Same calendar date, one hour apart, different tradeable session. Using `filed`")
    print("  as the timestamp collapses that distinction and buys the news for free.")
