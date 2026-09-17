"""Cross-sectional equity signal: persistent component of post-close commentary.

Design notes (the traps this file is built around):

* `close_quoted.csv` still contains splits.  The adjustment factor is anchored at
  the START of the sample (factor(t) = product of ratios with date <= t), so the
  factor is knowable at t and no past value is rewritten when a later split
  arrives.  The present-anchored convention silently imports future splits into
  price LEVELS and manufactures a "cheap stock" alpha that is pure look-ahead.
* `news_feed.csv` is published AFTER the close of session `feed_ts` and is about
  that session: its rank correlation with the SAME session's return is ~0.68 and
  with the next session's return ~0.00.  It is therefore used only from session
  `feed_ts + 1`, and the contemporaneous-return component is regressed out
  cross-sectionally first.  What is left is a slow, persistent score that does
  predict, and that is the whole signal.
* `llm_score.csv` has rank IC ~0.79 with the NEXT session's return through its
  2020-12-31 training cutoff and ~0.00 after it.  It is leaked label, not a
  feature, and is not read at all.
* The universe is rebuilt every session from `listings.csv` plus observed
  prices, using only information dated strictly before session t, so delisted
  names are traded while they lived and dropped afterwards.
* `fundamentals.csv` is not used: the only point-in-time-safe construction from
  it (revenue level / growth) had no stable in-sample edge once the price-level
  look-ahead above was removed.
"""

import os

import numpy as np
import pandas as pd

WINDOWS = (10, 21, 42, 63)   # residual-news averaging windows, blended
SMOOTH_HALFLIFE = 3.0        # weight smoothing, chosen on the 2017-2020 fit window
VOL_WINDOW = 21
MIN_NAMES = 6                # minimum cross-section for a cross-sectional fit
GROSS = 1.0


def _cs_residual(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional OLS residual of y on x (uses only that date)."""
    Y = np.asarray(y, dtype=float)
    X = np.asarray(x, dtype=float)
    out = np.full(Y.shape, np.nan)
    for i in range(Y.shape[0]):
        yy, xx = Y[i], X[i]
        m = np.isfinite(yy) & np.isfinite(xx)
        if m.sum() < MIN_NAMES:
            continue
        xm, ym = xx[m], yy[m]
        vx = xm.var()
        if vx <= 0:
            out[i, m] = ym - ym.mean()
        else:
            b = ((xm - xm.mean()) * (ym - ym.mean())).mean() / vx
            out[i, m] = ym - (ym.mean() + b * (xm - xm.mean()))
    return pd.DataFrame(out, index=y.index, columns=y.columns)


def _xs_rank(df: pd.DataFrame, ok: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank, centred on zero, only over eligible names."""
    s = df.where(ok)
    r = s.rank(axis=1, pct=True).sub(0.5)
    return r.where(s.notna().sum(axis=1).ge(MIN_NAMES), np.nan)


def build_positions(data_dir: str) -> pd.DataFrame:
    """Return a dates x tickers frame of portfolio weights.

    Row t holds the weights carried over session t and uses only information
    dated strictly before the close of session t.
    """
    d = str(data_dir)
    px = pd.read_csv(os.path.join(d, "close_quoted.csv"), index_col=0,
                     parse_dates=True).sort_index()
    px.index = pd.DatetimeIndex(px.index)
    cols = list(px.columns)

    # ---- splits: start-anchored adjustment factor, knowable at t ----------
    adj = pd.DataFrame(1.0, index=px.index, columns=cols)
    ca_path = os.path.join(d, "corporate_actions.csv")
    if os.path.exists(ca_path):
        ca = pd.read_csv(ca_path, parse_dates=["date"])
        for _, r in ca.iterrows():
            if r["ticker"] in adj.columns and np.isfinite(r["ratio"]) and r["ratio"] > 0:
                adj.loc[adj.index >= r["date"], r["ticker"]] *= float(r["ratio"])
    apx = px * adj

    lr = np.log(apx).diff()
    lr = lr.replace([np.inf, -np.inf], np.nan)

    # ---- universe: strictly t-1 information ------------------------------
    ok = apx.notna().shift(1).fillna(False) & apx.notna().shift(2).fillna(False)
    lst_path = os.path.join(d, "listings.csv")
    if os.path.exists(lst_path):
        lst = pd.read_csv(lst_path, parse_dates=["listing_date"]).set_index("ticker")
        for t in cols:
            if t in lst.index:
                ld = lst.loc[t, "listing_date"]
                if pd.notna(ld):
                    ok.loc[ok.index <= ld, t] = False

    # ---- news: strip the same-session return it is commentary on ----------
    nf = pd.read_csv(os.path.join(d, "news_feed.csv"), parse_dates=["feed_ts"])
    ns = (nf.pivot_table(index="feed_ts", columns="ticker", values="score",
                         aggfunc="mean")
            .reindex(index=px.index, columns=cols))

    live = apx.notna()
    dm = lr.where(live)
    dm = dm.sub(dm.mean(axis=1), axis=0)          # market-neutral daily log return
    resid = _cs_residual(ns.where(live), dm)

    # ---- blend the persistent part over several horizons ------------------
    sig = None
    for k in WINDOWS:
        m = resid.rolling(k, min_periods=max(2, k // 2)).mean()
        z = _xs_rank(m, ok)
        sig = z if sig is None else sig.add(z, fill_value=0.0)
    sig = sig / float(len(WINDOWS))

    # information through t-1 only
    sig = sig.shift(1)
    vol = lr.rolling(VOL_WINDOW, min_periods=10).std().shift(1)

    raw = (sig / vol).replace([np.inf, -np.inf], np.nan).where(ok)

    # ---- dollar-neutral, unit-gross weights ------------------------------
    def normalize(x: pd.DataFrame) -> pd.DataFrame:
        x = x.sub(x.mean(axis=1), axis=0)
        g = x.abs().sum(axis=1)
        w = x.div(g, axis=0).where(g > 0, 0.0)
        return w.fillna(0.0) * GROSS

    w = normalize(raw)

    # ---- smooth to cut turnover, then re-neutralise and cap gross --------
    w = w.ewm(halflife=SMOOTH_HALFLIFE, adjust=False).mean()
    w = w.where(ok, 0.0)
    n = ok.sum(axis=1)
    w = w.sub((w.sum(axis=1) / n.where(n > 0, np.nan)), axis=0).where(ok, 0.0)
    g = w.abs().sum(axis=1)
    scale = np.minimum(1.0, GROSS / g.where(g > 0, np.nan))
    w = w.mul(scale.fillna(0.0), axis=0).fillna(0.0)

    return w.astype(float)
