"""Cross-sectional equity signal for the 60-name market.

Signal construction (all inputs are strictly lagged one session):

1.  Prices are back-adjusted for the splits in ``corporate_actions.csv``
    before any return is computed - the quoted closes contain raw split
    jumps of about -50%.
2.  ``news_feed.csv`` is published *after* the close of session ``feed_ts``,
    so a score stamped ``t`` may only drive the position held over ``t+1``.
    The raw score is ~0.72 correlated with the same session's return, i.e.
    it is mostly a noisy re-statement of something already known.  Each day
    the score is regressed cross-sectionally on that day's return and only
    the residual - the part of the commentary not explained by the move -
    is kept.  Slow means (60/120/250d) of that residual track a persistent
    per-name drift.
3.  A point-in-time value ratio, log(quarterly fundamental) - log(price),
    built by stepping filings in by their ``filed`` date.  ``10-Q/A`` rows
    are amendments of an already-filed quarter; they are used only from
    their own (later) filed date, never back-stamped onto the original
    10-Q date.
4.  ``llm_score.csv`` is deliberately NOT used: its rank IC against the
    next session's return is 0.79 before its 2020-12-31 training cutoff
    and 0.00 after it, i.e. it is fitted-in-hindsight and carries no
    out-of-sample information.
5.  ``listings.csv`` is not needed - the price panel already stops at each
    name's delisting date - but every name that ever listed is traded, so
    there is no survivorship selection.

Weights are cross-sectionally demeaned (dollar neutral) and scaled so that
the sum of absolute weights is 1.
"""

import os

import numpy as np
import pandas as pd

_NEWS_WINDOWS = (60, 120, 250)
_NEWS_WEIGHT = 2.0
_VALUE_WEIGHT = 1.0
_CLIP = 3.0


def _adjusted_prices(data_dir):
    px = pd.read_csv(os.path.join(data_dir, "close_quoted.csv"),
                     index_col=0, parse_dates=True).sort_index()
    ca_path = os.path.join(data_dir, "corporate_actions.csv")
    if os.path.exists(ca_path):
        ca = pd.read_csv(ca_path, parse_dates=["date"])
        adj = pd.DataFrame(1.0, index=px.index, columns=px.columns)
        for _, r in ca.iterrows():
            if r["ticker"] in adj.columns and r["ratio"]:
                adj.loc[adj.index < r["date"], r["ticker"]] /= float(r["ratio"])
        px = px * adj
    return px


def _news_panel(data_dir, index, columns):
    news = pd.read_csv(os.path.join(data_dir, "news_feed.csv"),
                       parse_dates=["feed_ts"])
    nv = news.pivot_table(index="feed_ts", columns="ticker", values="score",
                          aggfunc="last")
    return nv.reindex(index=index, columns=columns)


def _point_in_time_fundamentals(data_dir, index, columns):
    """Latest fundamental value known at each date, keyed off `filed`."""
    fnd = pd.read_csv(os.path.join(data_dir, "fundamentals.csv"),
                      parse_dates=["filed"])
    out = {}
    for ticker, g in fnd.groupby("ticker"):
        g = g.sort_values("filed")
        rows = g[["filed", "val"]].groupby("filed").last()
        rows = rows.reindex(index.union(rows.index)).ffill().reindex(index)
        out[ticker] = rows["val"]
    if not out:
        return pd.DataFrame(index=index, columns=columns, dtype=float)
    return pd.DataFrame(out).reindex(columns=columns)


def _daily_xs_residual(nv, ret):
    """News score minus its same-session cross-sectional fit on returns."""
    N = nv.values.astype(float)
    R = ret.values.astype(float)
    out = np.full(N.shape, np.nan)
    for i in range(N.shape[0]):
        n = N[i]
        r = R[i]
        m = np.isfinite(n) & np.isfinite(r)
        if m.sum() >= 10:
            x = r[m]
            y = n[m]
            if np.std(x) > 1e-12:
                slope, intercept = np.polyfit(x, y, 1)
                out[i, m] = y - (slope * x + intercept)
            else:
                out[i, m] = y - y.mean()
    return pd.DataFrame(out, index=nv.index, columns=nv.columns)


def _zscore(df, tradable, clip=_CLIP):
    d = df.where(tradable)
    z = d.sub(d.mean(axis=1), axis=0).div(d.std(axis=1).replace(0.0, np.nan), axis=0)
    return z.clip(-clip, clip)


def build_positions(data_dir: str) -> "pd.DataFrame":
    """Return a dates x tickers frame of portfolio weights.

    Row t holds the weights carried over session t and uses only
    information available strictly before the close of session t.
    """
    data_dir = str(data_dir)
    apx = _adjusted_prices(data_dir)
    ret = apx.pct_change()
    nv = _news_panel(data_dir, apx.index, apx.columns)
    fval = _point_in_time_fundamentals(data_dir, apx.index, apx.columns)

    # Tradable on session t iff the name had a close on t-1.
    tradable = apx.notna().shift(1).fillna(False)

    xres = _daily_xs_residual(nv, ret)

    news_z = None
    n_used = 0
    for w in _NEWS_WINDOWS:
        s = xres.rolling(w, min_periods=max(5, w // 4)).mean()
        z = _zscore(s.shift(1), tradable)
        news_z = z if news_z is None else news_z.add(z, fill_value=0.0)
        n_used += 1
    news_z = news_z / float(n_used)

    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.log(fval.where(fval > 0)) - np.log(apx.where(apx > 0))
    value_z = _zscore(value.shift(1), tradable)

    score = (_NEWS_WEIGHT * news_z.fillna(0.0)
             + _VALUE_WEIGHT * value_z.fillna(0.0))
    have = news_z.notna() | value_z.notna()
    score = score.where(have & tradable)

    # dollar neutral, gross exposure 1
    score = score.sub(score.mean(axis=1), axis=0)
    gross = score.abs().sum(axis=1).replace(0.0, np.nan)
    weights = score.div(gross, axis=0).fillna(0.0) * 0.99
    return weights
