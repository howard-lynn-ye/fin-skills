"""Cross-sectional equity signal for the 60-name market in `data/`.

Method
------
The only source with genuine out-of-sample predictive content is `news_feed.csv`.
Its per-session score is ~0.65 rank-correlated with the *same* session's return
(it is commentary published after that close), so using it for session `feed_ts`
is look-ahead; used with a one-session lag it has essentially zero one-day IC.
Averaged over one to three months, however, the slow component of the score is a
persistent, genuinely predictive cross-sectional alpha.

Deliberately NOT used:
  * `llm_score.csv`  - rank IC 0.78 before its 2020-12-31 training cutoff and
    0.016 after: pure in-sample leakage.
  * same-session news  - contemporaneous, not predictive.
  * `fundamentals.csv` amended (10-Q/A) values ahead of their `filed` date, and
    the fundamentals generally (no measurable IC point-in-time).

Prices are de-split with `corporate_actions.csv`; delisted names are kept in the
panel for as long as they quote, so the backtest is not survivorship-filtered.
"""

import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- parameters
WINDOWS = (42, 63)      # trailing-mean horizons for the news score, in sessions
IVOL_WINDOW = 63        # lookback for idiosyncratic-vol risk scaling
IVOL_MINP = 20
MIN_NAMES = 5           # minimum cross-section width to take any position


def _cross_rank(df):
    """Per-row rank transform to [-0.5, 0.5], NaNs preserved."""
    values = df.to_numpy(dtype=float, copy=True)
    out = np.full(values.shape, np.nan)
    for i in range(values.shape[0]):
        row = values[i]
        ok = np.isfinite(row)
        n = int(ok.sum())
        if n < MIN_NAMES:
            continue
        order = np.argsort(np.argsort(row[ok], kind="mergesort"), kind="mergesort")
        out[i, ok] = (order + 0.5) / n - 0.5
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _split_adjusted(close, actions):
    """Undo splits: quoted prices are already divided on and after the split date,
    so multiply each date by the cumulative ratio applied up to and including it."""
    factor = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    if actions is not None and len(actions):
        for _, row in actions.iterrows():
            ticker = row.get("ticker")
            if ticker not in factor.columns:
                continue
            try:
                ratio = float(row.get("ratio"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(ratio) or ratio <= 0:
                continue
            date = pd.Timestamp(row.get("date"))
            factor.loc[factor.index >= date, ticker] *= ratio
    return close * factor


def build_positions(data_dir: str) -> "pandas.DataFrame":
    """Return dates x tickers portfolio weights.

    Row t holds the weights you are IN over session t: they may use information
    available strictly before the close of session t. Weights should be roughly
    dollar-neutral and sum of absolute values <= 1 per row. Missing = 0.
    """
    data_dir = str(data_dir)

    close = pd.read_csv(
        os.path.join(data_dir, "close_quoted.csv"), index_col=0, parse_dates=True
    ).sort_index()
    close.columns = [str(c) for c in close.columns]

    ca_path = os.path.join(data_dir, "corporate_actions.csv")
    actions = (
        pd.read_csv(ca_path, parse_dates=["date"]) if os.path.exists(ca_path) else None
    )

    adj = _split_adjusted(close, actions)
    ret = adj.pct_change()

    # ---- news score panel, aligned to the session it describes -------------
    news = pd.read_csv(os.path.join(data_dir, "news_feed.csv"), parse_dates=["feed_ts"])
    news["ticker"] = news["ticker"].astype(str)
    news = news.drop_duplicates(subset=["feed_ts", "ticker"], keep="last")
    score = (
        news.pivot(index="feed_ts", columns="ticker", values="score")
        .reindex(index=close.index, columns=close.columns)
        .astype(float)
    )

    # Published AFTER the close of feed_ts -> first usable for the next session.
    # A trailing mean through t-1 is therefore fully known before the close of t.
    combo = None
    for w in WINDOWS:
        sm = score.rolling(w, min_periods=max(2, w // 2)).mean().shift(1)
        r = _cross_rank(sm)
        combo = r if combo is None else combo.add(r, fill_value=np.nan)
    combo = combo / len(WINDOWS)

    # ---- tradability: need a usable close-to-close return for session t -----
    tradable = close.notna() & close.shift(1).notna()
    combo = combo.where(tradable)

    # ---- risk scaling by trailing idiosyncratic vol ------------------------
    demeaned = ret.sub(ret.mean(axis=1), axis=0)
    ivol = demeaned.rolling(IVOL_WINDOW, min_periods=IVOL_MINP).std().shift(1)
    ivol = ivol.where(np.isfinite(ivol) & (ivol > 0))
    floor = ivol.median(axis=1, skipna=True) * 0.25
    ivol = ivol.clip(lower=floor, axis=0)

    raw = (combo / ivol).replace([np.inf, -np.inf], np.nan)
    # fall back to unscaled ranks where vol history is missing
    raw = raw.where(raw.notna(), combo)

    # ---- dollar-neutral, gross <= 1 ---------------------------------------
    raw = raw.sub(raw.mean(axis=1), axis=0)
    gross = raw.abs().sum(axis=1)
    weights = raw.div(gross.where(gross > 0), axis=0)
    weights = weights.fillna(0.0)
    weights[~tradable] = 0.0

    # numerical safety: re-enforce the gross budget
    gross = weights.abs().sum(axis=1)
    over = gross > 1.0
    if over.any():
        weights.loc[over] = weights.loc[over].div(gross[over], axis=0)

    return weights


if __name__ == "__main__":  # pragma: no cover
    w = build_positions(os.path.join(os.path.dirname(__file__), "data"))
    print(w.shape, float(w.abs().sum(axis=1).max()))
