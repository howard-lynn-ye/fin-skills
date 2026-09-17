"""Cross-sectional equity strategy for the 60-name synthetic market.

Two orthogonal sleeves, equal risk:

  1. NEWS RESIDUAL.  `news_feed.csv` is published *after* the close of session
     `feed_ts`, so score_t is usable only from session t+1 onward.  Its raw
     cross-sectional rank correlation with the *same-day* return is ~0.66, i.e.
     it is mostly a restatement of a return we already observe.  We strip that
     component out cross-sectionally (regress the z-scored news on the z-scored
     same-day return each day and keep the residual) and average the residual
     over 10/21/42/63-day windows.  The residual carries a genuine, slowly
     decaying forecast of future returns.

  2. PRICE-LEVEL MEAN REVERSION.  Cross-sectional dispersion of log price is
     stationary over the whole sample while single-name vol is ~35% annualised,
     so log prices revert toward the cross-sectional mean.  Signal is the
     negative of the 21-day average deviation of log price from the
     cross-sectional mean.

Both sleeves are divided by trailing 63-day vol, z-scored, added with equal
weight, demeaned cross-sectionally and scaled to gross exposure 1.

Deliberately NOT used:
  * `llm_score.csv` - rank IC 0.79 against next-day returns through its
    2020-12-31 training cutoff and 0.006 afterwards: pure in-sample leakage.
  * same-session news (lookahead: it is published after the close).
  * `delisting_date` from `listings.csv` as a forward-looking filter - a name is
    dropped only once its prices stop printing, never before.
  * `10-Q/A` amendments before their own `filed` date; fundamentals are read
    point-in-time and, in the end, add nothing to the two sleeves above.
"""

import os
import numpy as np
import pandas as pd

_VOL_WIN = 63
_NEWS_WINDOWS = (10, 21, 42, 63)
_MR_WIN = 21


def _xsz(df):
    """Cross-sectional z-score, row by row."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def build_positions(data_dir: str) -> "pd.DataFrame":
    """Return dates x tickers portfolio weights (row t = held over session t)."""
    d = str(data_dir)

    px = pd.read_csv(os.path.join(d, "close_quoted.csv"),
                     index_col=0, parse_dates=True).sort_index()
    px = px.astype(float)
    cols = list(px.columns)

    # ---- undo splits: quoted prices are post-split from the split date on ----
    ca = pd.read_csv(os.path.join(d, "corporate_actions.csv"), parse_dates=["date"])
    factor = pd.DataFrame(1.0, index=px.index, columns=cols)
    for _, row in ca.iterrows():
        tic = row["ticker"]
        if tic in factor.columns and np.isfinite(row["ratio"]) and row["ratio"] > 0:
            factor.loc[factor.index < row["date"], tic] *= float(row["ratio"])
    adj = px / factor

    ret = adj.pct_change()
    live = adj.notna() & adj.shift(1).notna()      # a return actually printed
    ret = ret.where(live)

    # ---- news, aligned to the session it describes -------------------------
    news = pd.read_csv(os.path.join(d, "news_feed.csv"), parse_dates=["feed_ts"])
    nw = (news.pivot_table(index="feed_ts", columns="ticker", values="score",
                           aggfunc="last")
              .reindex(index=adj.index, columns=cols))
    nw = nw.where(adj.notna())

    zn, zr = _xsz(nw), _xsz(ret)
    denom = (zr * zr).mean(axis=1).replace(0.0, np.nan)
    beta = (zn * zr).mean(axis=1) / denom
    resid = _xsz(zn - zr.mul(beta.fillna(0.0), axis=0))

    vol = ret.rolling(_VOL_WIN, min_periods=30).std().replace(0.0, np.nan)

    news_sig = sum(_xsz(resid.rolling(k, min_periods=max(2, k // 2)).mean())
                   for k in _NEWS_WINDOWS) / float(len(_NEWS_WINDOWS))
    sleeve_news = _xsz(news_sig / vol)

    # ---- price-level mean reversion ----------------------------------------
    lp = np.log(adj)
    dev = lp.sub(lp.mean(axis=1), axis=0)
    sleeve_mr = _xsz((-dev.rolling(_MR_WIN, min_periods=10).mean()) / vol)

    signal = (sleeve_news + sleeve_mr).shift(1)    # strictly prior information

    # tradable at t using only information from before the close of t
    mask = adj.shift(1).notna() & adj.shift(2).notna()
    s = signal.where(mask)

    s = s.sub(s.mean(axis=1), axis=0)              # dollar neutral
    gross = s.abs().sum(axis=1).replace(0.0, np.nan)
    w = s.div(gross, axis=0).fillna(0.0)           # sum |w| == 1 (or 0)
    return w
