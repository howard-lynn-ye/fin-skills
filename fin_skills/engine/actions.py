"""fin_skills.engine.actions - corporate actions as a factor series, one arithmetic for every kind.

`ratio` is the number the RAW quote divides by on the action date: 2.0 for a 2:1 split,
close/(close - dividend) for a cash dividend. Splits and dividends then share one
arithmetic, and the adjustment convention becomes a choice of ANCHOR, not of formula:

    back     anchored at the PRESENT - history before each action is divided by its ratio,
             so every new action REWRITES the stored history (yfinance's Adj Close).
    forward  anchored at the START - prices at and after each action are multiplied up,
             so stored history never changes (A-share qfq).

Those are the behavioural definitions `fin_skills.core.adjustment_check` tests for, and
what it names "back-adjusted" and "forward-adjusted". The two differ by a per-name
constant, so RETURNS are identical and price LEVELS are not - which is why the convention
has to be declared rather than inferred from a return series.
"""
from __future__ import annotations

import pandas as pd

ACTION_COLUMNS = ("date", "ticker", "ratio", "kind")
LISTING_COLUMNS = ("ticker", "start_date", "end_date")


def normalize_actions(actions: pd.DataFrame) -> pd.DataFrame:
    """(date, ticker, ratio[, kind]) -> a sorted frame carrying all four columns."""
    a = pd.DataFrame(actions).copy()
    missing = [c for c in ("date", "ticker", "ratio") if c not in a.columns]
    if missing:
        raise ValueError(f"actions is missing column(s) {missing}; needs date, ticker, ratio")
    a["date"] = pd.to_datetime(a["date"])
    a["ratio"] = pd.to_numeric(a["ratio"]).astype(float)
    if not (a["ratio"] > 0).all():
        raise ValueError("every action ratio must be positive")
    if "kind" not in a.columns:
        a["kind"] = "split"
    return (a[list(ACTION_COLUMNS)].sort_values(["date", "ticker"]).reset_index(drop=True))


def normalize_listings(listings: pd.DataFrame) -> pd.DataFrame:
    """(ticker, start_date, end_date) or (ticker, listing_date, delisting_date) -> the first."""
    t = pd.DataFrame(listings).copy().rename(columns={"listing_date": "start_date",
                                                      "delisting_date": "end_date"})
    if "ticker" not in t.columns:
        raise ValueError("listings needs a 'ticker' column")
    for c in ("start_date", "end_date"):
        t[c] = pd.to_datetime(t[c], errors="coerce") if c in t.columns else pd.NaT
    bad = t["end_date"].notna() & t["start_date"].notna() & (t["end_date"] < t["start_date"])
    if bad.any():
        raise ValueError(f"end_date precedes start_date for {t.loc[bad, 'ticker'].tolist()[:8]}")
    return t[list(LISTING_COLUMNS)].reset_index(drop=True)


def as_delisting_table(listings: pd.DataFrame) -> pd.DataFrame:
    """The same table under the column names survivorship_audit reads."""
    return listings.rename(columns={"start_date": "listing_date", "end_date": "delisting_date"})


def cum_factor(actions: pd.DataFrame | None, index: pd.DatetimeIndex,
               tickers, convention: str) -> pd.DataFrame:
    """The multiplier taking a RAW quote to `convention`, as a dates x tickers frame.

    raw and raw+factors are the identity: raw+factors keeps the quotes AND this table, so
    the two are the same prices with the events kept alongside instead of folded in.
    """
    f = pd.DataFrame(1.0, index=index, columns=list(tickers))
    if actions is None or actions.empty or convention in ("raw", "raw+factors"):
        return f
    for _, a in actions.iterrows():
        t = a["ticker"]
        if t not in f.columns:
            continue
        if convention == "back":
            f.loc[f.index < a["date"], t] /= float(a["ratio"])
        else:
            f.loc[f.index >= a["date"], t] *= float(a["ratio"])
    return f


__all__ = ["ACTION_COLUMNS", "LISTING_COLUMNS", "as_delisting_table", "cum_factor",
           "normalize_actions", "normalize_listings"]
