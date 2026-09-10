#!/usr/bin/env python3
"""Join fundamentals to prices the wrong way and the right way, and price the difference.

A fundamentals vendor hands you a table of quarterly numbers keyed by the period they
describe. Two things are true about that table and both are easy to forget:

  * the number was not PUBLIC on the day the quarter ended - it was filed weeks later;
  * the number in the table today is the LATEST vintage, restatements included, not the
    one that was on the wire at the time.

Join on `period_end` with an exact match and you have made both mistakes at once. This
builds a seeded world where the quarterly surprise drives a 60-day drift, joins it both
ways, and reports what each join is worth in Sharpe.

    python examples/point_in_time_fundamentals.py     (offline, seeded, a few seconds)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fin_skills.api as api                                                # noqa: E402
from fin_skills.market_data.safe_asof import safe_merge_asof                       # noqa: E402

SEED = 20260909
N_NAMES = 24
DRIFT_DAYS = 60          # bars over which a quarter's surprise plays out in the price
FILING_LAG = 45          # calendar days from period end to the 10-Q
AMEND_LAG = 150          # calendar days to the 10-Q/A, when there is one
AMEND_RATE = 0.40        # share of name-quarters later restated
KAPPA = 0.0006           # daily drift per unit of surprise
PERIODS = 252


def world(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prices whose drift is driven by each quarter's TRUE surprise, and the filings table.

    The filings table is what a vendor actually stores: one row per filing, each with the
    period it describes, the date it was filed, and the value as filed that day.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=756)
    names = [f"N{i:02d}" for i in range(N_NAMES)]
    quarter_ends = [d for d in pd.date_range(dates[0], dates[-1], freq="QE")
                    if d <= dates[-1] - pd.Timedelta(days=AMEND_LAG + 5)]

    rows, drift = [], pd.DataFrame(0.0, index=dates, columns=names)
    for q in quarter_ends:
        anchor = dates[dates.searchsorted(q)]
        window = dates[dates.searchsorted(anchor):dates.searchsorted(anchor) + DRIFT_DAYS]
        true = rng.normal(0.0, 1.0, N_NAMES)
        drift.loc[window, names] += KAPPA * true                    # the alpha, post-period
        reported = true + rng.normal(0.0, 0.8, N_NAMES)             # the 10-Q, as filed
        amended = rng.random(N_NAMES) < AMEND_RATE
        for j, n in enumerate(names):
            rows.append({"ticker": n, "period_end": q, "value": reported[j],
                         "filed": q + pd.Timedelta(days=FILING_LAG), "form": "10-Q"})
            if amended[j]:                                          # the restatement
                rows.append({"ticker": n, "period_end": q, "value": true[j],
                             "filed": q + pd.Timedelta(days=AMEND_LAG), "form": "10-Q/A"})

    noise = rng.normal(0.0, 0.012, (len(dates), N_NAMES))
    returns = pd.DataFrame(noise, index=dates, columns=names) + drift
    prices = 50.0 * (1.0 + returns).cumprod()
    return prices, pd.DataFrame(rows)


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(PERIODS)) if r.std(ddof=1) > 0 else np.nan


def backtest(signal: pd.DataFrame, returns: pd.DataFrame) -> tuple[pd.Series, float]:
    """Dollar-neutral cross-sectional weights, executed on the NEXT bar, both ways alike."""
    z = signal.sub(signal.mean(axis=1), axis=0)
    w = z.div(z.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    r = (w.shift(1) * returns).sum(axis=1)
    return r, sharpe(r)


def wide(joined: pd.DataFrame, dates: pd.DatetimeIndex, names: list[str]) -> pd.DataFrame:
    return (joined.pivot(index="date", columns="ticker", values="value")
            .reindex(index=dates, columns=names).ffill())


def rule(title: str) -> None:
    print("\n" + title)
    print("=" * 78)


def main() -> int:
    prices, filings = world()
    dates, names = prices.index, list(prices.columns)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    left = pd.DataFrame([(d, n) for d in dates for n in names], columns=["date", "ticker"])

    # ---------------------------------------------------------------- the wrong join
    # Latest vintage: one row per (ticker, period), the value as it stands TODAY.
    latest = (filings.sort_values("filed").groupby(["ticker", "period_end"], as_index=False)
              .last()[["ticker", "period_end", "value"]])
    wrong = pd.merge_asof(left.sort_values("date"), latest.sort_values("period_end"),
                          left_on="date", right_on="period_end", by="ticker",
                          direction="backward", allow_exact_matches=True)
    sig_wrong = wide(wrong, dates, names)

    # ---------------------------------------------------------------- the right join
    # Filed-date vintage: every filing is its own row, and it exists only from `filed` on.
    as_filed = filings[["ticker", "filed", "value"]].rename(columns={"filed": "date"})
    right = safe_merge_asof(left, as_filed, on="date", by="ticker",
                            tolerance=pd.Timedelta("200D"), right_ts_col="filed_used")
    sig_right = wide(right, dates, names)

    rule("1. THE TWO JOINS")
    print(f"  world: {len(dates)} days, {len(names)} names, {filings.period_end.nunique()} "
          f"quarters, {len(filings)} filings")
    print(f"         of which {int((filings.form == '10-Q/A').sum())} are restatements")
    print(f"  WRONG: merge_asof on period_end, allow_exact_matches=True, latest vintage")
    print(f"         -> the number is in the panel {FILING_LAG} calendar days before it was "
          f"filed,")
    print(f"            and it is the restated number, not the one that was on the wire")
    print(f"  RIGHT: safe_merge_asof on filed, allow_exact_matches=False, 200D tolerance")
    print(f"         -> each day sees the newest filing STRICTLY before it, and nothing else")

    rule("2. HOW DIFFERENT ARE THE SIGNALS?")
    both = sig_wrong.notna() & sig_right.notna()
    disagree = float((np.sign(sig_wrong[both]) != np.sign(sig_right[both])).sum().sum()
                     / both.sum().sum())
    corr = float(sig_wrong[both].stack().corr(sig_right[both].stack()))
    print(f"  cells where both signals exist : {int(both.sum().sum()):,}")
    print(f"  correlation between them       : {corr:.3f}")
    print(f"  fraction with the OPPOSITE sign: {disagree:.1%}")
    print("  They are not two versions of one number. They are two different numbers, and")
    print("  one of them did not exist yet.")

    rule("3. WHAT THE DIFFERENCE IS WORTH")
    r_wrong, s_wrong = backtest(sig_wrong, returns)
    r_right, s_right = backtest(sig_right, returns)
    print(f"  {'join':<34}{'Sharpe':>9}{'ann. return':>14}")
    print("  " + "-" * 57)
    for label, r, s in (("latest vintage, exact match", r_wrong, s_wrong),
                        ("filed-date vintage, backward as-of", r_right, s_right)):
        print(f"  {label:<34}{s:>9.2f}{r.mean() * PERIODS:>13.1%}")
    print("  " + "-" * 57)
    print(f"  {'the leak':<34}{s_wrong - s_right:>9.2f}")

    rule("4. THE GUARD SAYS SO WITHOUT THE BACKTEST")
    as_period = latest.rename(columns={"period_end": "date"})
    verdict = api.get("safe_asof").run(left=left, right=as_period, on="date", by="ticker",
                                       tolerance="200D", allow_exact_matches=True)
    print(f"  {verdict.summary().splitlines()[0]}")
    for f in verdict.errors:
        print(f"    -> {f}")

    rule("TAKEAWAY")
    print(f"  The same fundamentals, the same prices, the same weights. Joining on the period")
    print(f"  a number DESCRIBES instead of the day it was FILED moved the reported Sharpe from")
    print(f"  {s_right:.2f} to {s_wrong:.2f} - and the signals only correlate {corr:.2f}, so it "
          f"is not a better")
    print("  version of the same idea, it is a different one you could not have traded. Join")
    print("  on the filing date, backward, with a tolerance, and never on an exact stamp.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
