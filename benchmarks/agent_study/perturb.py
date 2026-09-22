#!/usr/bin/env python3
"""Build a counterfactual copy of a task workspace: identical up to a cut date, redrawn after.

This is the leakage oracle's instrument and it does not use any guard in this repository.
A pipeline that only looks backwards returns the SAME positions for every session on or
before the cut date when the future is redrawn. A pipeline that peeks - a shifted label, a
scaler fitted on the whole sample, a feed joined on the wrong side, a universe filtered by
who survived - moves. What moves, and by how much, is the measurement.

Perturbation, per file, for rows strictly after the cut:
- `close_quoted.csv`: daily log returns are resampled (seeded permutation, per ticker) and the
  path rebuilt from the last pre-cut price, so the listing pattern and price scale survive.
- `volume.csv`, `llm_score.csv`: values permuted along the date axis, per ticker.
- `news_feed.csv`: scores permuted within each date across tickers.
- `fundamentals.csv`: rows whose `filed` date is after the cut get their value perturbed.
- `listings.csv`, `corporate_actions.csv`: rows dated after the cut are dropped/kept as-is;
  delisting dates after the cut are left untouched (they are a future fact either way).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _read_matrix(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, index_col=0).astype(float)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def perturb_workspace(src: Path, dst: Path, cut: str, seed: int) -> Path:
    """Copy `src/data` to `dst/data`, redrawing everything after `cut`."""
    rng = np.random.default_rng(seed)
    cut_ts = pd.Timestamp(cut)
    dst.mkdir(parents=True, exist_ok=True)
    if (dst / "data").exists():
        shutil.rmtree(dst / "data")
    shutil.copytree(src / "data", dst / "data")
    data = dst / "data"

    close = _read_matrix(data / "close_quoted.csv")
    after = close.index > cut_ts
    logret = np.log(close).diff()
    for c in close.columns:
        col = np.array(logret.loc[after, c].to_numpy(), dtype=float, copy=True)
        finite = np.isfinite(col)
        if finite.sum() > 1:
            vals = col[finite]
            col[finite] = rng.permutation(vals)
        logret.loc[after, c] = col
        base = close.loc[~after, c].dropna()
        if base.empty:                       # whole-column redraw: anchor on its own first print
            first = close.loc[after, c].dropna()
            if first.empty:
                continue
            start = float(first.iloc[0])
        else:
            start = float(base.iloc[-1])
        path = start * np.exp(np.nan_to_num(col, nan=0.0).cumsum())
        keep = close.loc[after, c].notna().to_numpy()
        close.loc[after, c] = np.where(keep, np.round(path, 2), np.nan)
    close.to_csv(data / "close_quoted.csv", index_label="date", float_format="%.2f")

    for name, fmt in (("volume.csv", "%.0f"), ("llm_score.csv", "%.6f")):
        m = _read_matrix(data / name)
        aft = m.index > cut_ts
        for c in m.columns:
            col = np.array(m.loc[aft, c].to_numpy(), dtype=float, copy=True)
            finite = np.isfinite(col)
            if finite.sum() > 1:
                col[finite] = rng.permutation(col[finite])
            m.loc[aft, c] = col
        m.to_csv(data / name, index_label="date", float_format=fmt)

    feed = pd.read_csv(data / "news_feed.csv", parse_dates=["feed_ts"])
    mask = feed["feed_ts"] > cut_ts
    part = feed.loc[mask]
    feed.loc[mask, "score"] = (part.groupby("feed_ts")["score"]
                               .transform(lambda s: rng.permutation(s.to_numpy())))
    feed.to_csv(data / "news_feed.csv", index=False, float_format="%.6f")

    fund = pd.read_csv(data / "fundamentals.csv", parse_dates=["filed"])
    m = fund["filed"] > cut_ts
    fund.loc[m, "val"] = (fund.loc[m, "val"].to_numpy()
                          * rng.normal(1.0, 0.15, int(m.sum()))).round(0)
    fund["filed"] = fund["filed"].dt.strftime("%Y-%m-%d")
    fund.to_csv(data / "fundamentals.csv", index=False)

    shutil.copy(src / "manifest.json", dst / "manifest.json")
    if (src / "TASK.md").exists():
        shutil.copy(src / "TASK.md", dst / "TASK.md")
    return dst
