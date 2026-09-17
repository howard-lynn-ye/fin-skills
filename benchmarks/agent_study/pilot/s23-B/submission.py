"""Cross-sectional equity signal for the 60-name market in `data/`.

Design notes (every choice below was made on history through 2020-12-31 only):

  * Prices are quoted, so a return series can only be built after folding the split
    ex-dates back in from `corporate_actions.csv` (`_split_returns` below).  The
    evaluation of this strategy uses that adjusted series; the traded signal itself
    turned out to need no price history, since every price-based candidate screened
    (1d/5d reversal, 21d/63d/12-1 momentum, volume shock) was insignificant in the
    fit window.  Prices are still read here, for the tradeable-universe mask.
  * `news_feed.csv` is published AFTER the close of session `feed_ts` and is about
    that session.  Used unlagged its rank-IC against the same session's return is
    +0.65 -- that is the session itself, not a forecast.  Everything here uses
    news dated <= t-1 only.
  * `llm_score.csv` has a training cutoff of 2020-12-31.  Its rank-IC against the
    next session's return inside the fit window is +0.78 (t = 405), i.e. a
    standalone Sharpe near 100 -- memorisation, not skill.  Its out-of-sample
    value cannot be estimated from the fit window at all, and its scores have
    ~zero autocorrelation (0.01), so trading them implies >100% turnover per day.
    It carries ZERO weight here.
  * `fundamentals.csv` is consumed point-in-time: at date t only vintages with
    `filed` < t are visible, so a 10-Q/A is known from its own filing date, never
    from the original 10-Q's.
  * The universe is whatever had a quoted price on t-1.  Delisted names are kept
    in history; `delisting_date` is never read, since it is not knowable ex ante.
"""

import numpy as np
import pandas as pd

NEWS_WINDOW = 30      # trading days of news averaged; flat plateau 20-60 in fit window
NEWS_MINP = 10
SMOOTH_SPAN = 10      # EMA on the combined score, chosen on the fit window
MIN_NAMES = 8         # need a cross-section before taking any position


def _load(data_dir):
    d = str(data_dir).rstrip("/") + "/"
    px = pd.read_csv(d + "close_quoted.csv", index_col=0, parse_dates=True).sort_index()
    ca = pd.read_csv(d + "corporate_actions.csv", parse_dates=["date"])
    news = pd.read_csv(d + "news_feed.csv", parse_dates=["feed_ts"])
    fu = pd.read_csv(d + "fundamentals.csv", parse_dates=["start", "end", "filed"])
    nw = (news.pivot_table(index="feed_ts", columns="ticker", values="score")
               .reindex(px.index).reindex(columns=px.columns))
    return px, ca, nw, fu


def _split_returns(px, ca):
    """Split-adjusted simple returns.  Not used by the signal; kept because any
    return computed off `close_quoted.csv` without this books a -50% on every
    split ex-date, and that is the first trap in this market."""
    r = pd.DataFrame(1.0, index=px.index, columns=px.columns)
    for _, x in ca.iterrows():
        if x["date"] in r.index and x["ticker"] in r.columns:
            r.loc[x["date"], x["ticker"]] = float(x["ratio"])
    return ((px * r) / px.shift(1) - 1.0).where(px.notna() & px.shift(1).notna())


def _revision_panel(fu, index, columns):
    """Point-in-time restatement surprise.

    For each (ticker, fiscal period) the filings arrive in `filed` order.  When a
    later vintage of a period already seen arrives, the surprise is
    new/previous - 1.  It is stamped at that vintage's OWN filing date, then
    forward-filled and lagged one session, so nothing is used before it existed.
    """
    out = pd.DataFrame(np.nan, index=index, columns=columns)
    fu = fu.sort_values(["ticker", "filed"])
    for tk, g in fu.groupby("ticker", sort=False):
        if tk not in out.columns:
            continue
        known, recs = {}, []
        for filed, end, val in zip(g["filed"], g["end"], g["val"]):
            prev = known.get(end)
            known[end] = val
            if prev is not None and prev > 0 and np.isfinite(val):
                recs.append((filed, val / prev - 1.0))
        if not recs:
            continue
        s = pd.Series([v for _, v in recs], index=pd.DatetimeIndex([t for t, _ in recs]))
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        out[tk] = (s.reindex(index.union(s.index)).ffill()
                    .reindex(index).shift(1).values)
    return out


def _xs_rank(df, min_names=MIN_NAMES):
    """Cross-sectional rank mapped to [-1, 1], NaN on thin days."""
    r = df.rank(axis=1, pct=True)
    n = df.notna().sum(axis=1)
    return (r - 0.5).mul(2.0).where(n >= min_names)


def build_positions(data_dir: str) -> pd.DataFrame:
    """Return dates x tickers portfolio weights.

    Row t holds the weights you are IN over session t: they may use information
    available strictly before the close of session t.  Weights are dollar-neutral
    and sum of absolute values <= 1 per row.  Missing = 0.
    """
    px, ca, nw, fu = _load(data_dir)

    # ---- tradeable universe: had a quoted price at t-1 (no delisting look-ahead)
    avail = px.shift(1).notna()

    # ---- signal 1: slow drift in post-close commentary, strictly lagged
    news_ma = nw.shift(1).rolling(NEWS_WINDOW, min_periods=NEWS_MINP).mean()

    # ---- signal 2: point-in-time restatement surprise
    rev = _revision_panel(fu, px.index, px.columns)

    # ---- equal-weight rank blend (both carried t ~ 4 in the fit window)
    score = _xs_rank(news_ma.where(avail)) + _xs_rank(rev.where(avail))
    score = score.where(avail)

    # ---- smooth to cut turnover, then dollar-neutralise and budget gross to 1
    score = score.ewm(span=SMOOTH_SPAN, min_periods=1).mean().where(avail)
    z = _xs_rank(score)
    z = z.sub(z.mean(axis=1), axis=0)
    w = z.div(z.abs().sum(axis=1).replace(0.0, np.nan), axis=0)

    w = w.where(avail).fillna(0.0)
    # numerical safety: never exceed the 100% gross budget
    gross = w.abs().sum(axis=1)
    w = w.div(gross.clip(lower=1.0) * (1.0 + 1e-12), axis=0).fillna(0.0)
    return w


if __name__ == "__main__":
    import sys
    W = build_positions(sys.argv[1] if len(sys.argv) > 1 else "data")
    print(W.shape, "max gross", W.abs().sum(axis=1).max(), "max |net|", W.sum(axis=1).abs().max())
