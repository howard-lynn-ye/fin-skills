"""Cross-sectional equity signal for the 60-name synthetic market.

Design notes (see report.json for the headline number):

*  Prices are quoted, splits unremoved -> a cumulative split factor built from
   corporate_actions.csv is applied before any return or price-level is taken.
*  The universe is point-in-time by construction: a name enters when it first
   quotes and leaves when it stops.  listings.csv is deliberately NOT read -
   its delisting_date column is knowable only after the fact.
*  news_feed.csv rows carry `feed_ts` = the session the commentary is ABOUT and
   are published after that session's close, so a score stamped t-1 is the most
   recent one usable for weights held over session t.
*  fundamentals.csv is consumed point-in-time: only the vintage on file strictly
   before the decision date is used, so 10-Q/A amendments never back-date.
*  llm_score.csv is NOT used.  Its training cutoff is 2020-12-31; measured rank
   IC against next-day returns is +0.79 before that date and +0.002 after it.
   It is a label leak in-sample and noise out-of-sample.

Every field used on row t is observable strictly before the close of session t.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

NEWS_HALFLIVES = (21, 42, 63, 126)
FUND_WEIGHT = 0.3
MIN_NAMES = 8
MAX_ABS_WEIGHT = 0.25
GROSS = 1.0


def _split_adjusted(close: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Back out splits, anchored at the START of the sample (hfq-style).

    Anchoring at the start keeps the series reproducible: a split arriving later
    never rewrites a value that was already published.
    """
    factor = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    if len(actions):
        for _, row in actions.iterrows():
            tic = row["ticker"]
            if tic in factor.columns:
                factor.loc[factor.index >= row["date"], tic] *= float(row["ratio"])
    return close * factor


def _pit_fundamentals(fund: pd.DataFrame, dates: pd.DatetimeIndex,
                      cols: pd.Index) -> pd.DataFrame:
    """Latest vintage of the latest reported quarter, as known before each date.

    Walks filings in `filed` order and keeps a running {period_end: value} map,
    so a 10-Q/A only replaces its quarter from its own filing date onward.
    """
    out = pd.DataFrame(index=dates, columns=cols, dtype=float)
    if not len(fund):
        return out
    fund = fund.sort_values(["filed", "end"])
    for tic, grp in fund.groupby("ticker"):
        if tic not in cols:
            continue
        vintage: dict = {}
        rows = []
        for _, row in grp.iterrows():
            vintage[row["end"]] = float(row["val"])
            rows.append((row["filed"], vintage[max(vintage)]))
        seq = pd.DataFrame(rows, columns=["filed", "val"]).groupby("filed").last()
        merged = seq.reindex(dates.union(seq.index)).ffill().reindex(dates)
        # shift: a filing stamped d is only relied on from the session after d
        out[tic] = merged["val"].shift(1)
    return out


def _cs_rank_z(frame: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank, mapped to roughly [-1, 1]. Outlier-proof by design."""
    frame = frame.where(live)
    ranks = frame.rank(axis=1)
    n = frame.notna().sum(axis=1)
    z = ranks.sub((n + 1) / 2.0, axis=0).div((n.clip(lower=2) / 2.0), axis=0)
    return z.where(live)


def build_positions(data_dir: str) -> pd.DataFrame:
    """Return dates x tickers portfolio weights (row t = held over session t)."""
    d = str(data_dir)
    close = pd.read_csv(os.path.join(d, "close_quoted.csv"),
                        index_col=0, parse_dates=True).sort_index()
    close.index = pd.DatetimeIndex(close.index)
    cols = close.columns

    try:
        actions = pd.read_csv(os.path.join(d, "corporate_actions.csv"),
                              parse_dates=["date"])
    except (FileNotFoundError, ValueError):
        actions = pd.DataFrame(columns=["date", "ticker", "ratio", "kind"])

    news_raw = pd.read_csv(os.path.join(d, "news_feed.csv"), parse_dates=["feed_ts"])
    news = (news_raw.pivot_table(index="feed_ts", columns="ticker", values="score")
            .reindex(index=close.index, columns=cols))

    fund = pd.read_csv(os.path.join(d, "fundamentals.csv"),
                       parse_dates=["start", "end", "filed"])

    return _positions_from_frames(close, news, fund, actions)


def _positions_from_frames(close: pd.DataFrame, news: pd.DataFrame,
                           fund: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """The whole model, on already-loaded frames. Kept separate so the causality
    guard can perturb the inputs directly."""
    cols = close.columns
    adj = _split_adjusted(close, actions)
    live = close.notna()

    # --- signal 1: slow consensus of post-close commentary ------------------
    # A single day's score is ~65% rank-correlated with that day's own return and
    # has no next-day content; the persistent component only shows up once many
    # days are averaged, so an ensemble of half-lives is used rather than one.
    news_z = None
    for hl in NEWS_HALFLIVES:
        z = _cs_rank_z(news.ewm(halflife=hl, min_periods=10).mean(), live)
        news_z = z if news_z is None else news_z.add(z, fill_value=0.0)
    news_z = news_z / float(len(NEWS_HALFLIVES))

    # --- signal 2: reported size, point-in-time -----------------------------
    lat = _pit_fundamentals(fund, close.index, cols)
    fund_z = _cs_rank_z(np.log(lat.clip(lower=1.0)), live)

    score = news_z.add(FUND_WEIGHT * fund_z, fill_value=0.0).where(live)

    # --- score -> dollar-neutral weights ------------------------------------
    enough = score.notna().sum(axis=1) >= MIN_NAMES
    score = score.where(enough, np.nan)
    score = score.sub(score.mean(axis=1), axis=0)
    gross = score.abs().sum(axis=1)
    w = score.div(gross.replace(0.0, np.nan), axis=0) * GROSS
    w = w.clip(-MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)

    # re-impose exact dollar neutrality and the gross budget after clipping
    w = w.sub(w.mean(axis=1), axis=0)
    gross = w.abs().sum(axis=1)
    w = w.div(gross.where(gross > GROSS, GROSS) / GROSS, axis=0)
    w = w * (1.0 - 1e-9)   # keep sum|w| strictly inside the budget under float error

    # the whole frame is built from data up to and including session t; holding
    # it over session t+1 is what makes it causal.
    w = w.shift(1)
    return w.reindex(index=close.index, columns=cols).fillna(0.0)


if __name__ == "__main__":  # pragma: no cover
    import sys
    print(build_positions(sys.argv[1] if len(sys.argv) > 1 else "data").tail())
