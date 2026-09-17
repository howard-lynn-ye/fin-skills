#!/usr/bin/env python3
"""Replay a submission the way it would have had to trade: one session at a time.

The held-out year answers "does it generalise". It does not answer "could you have traded
it": a pipeline that joins a session's own after-close commentary keeps earning in any year,
because that file exists in every copy of the market. The difference only shows when the data
is withheld the way the clock withholds it.

So for each session in a contiguous block, the market is truncated at the previous session,
`build_positions` is called on that truncated copy, and the weight it produces for the session
is the weight it actually gets to hold. A strategy that needs data it could not have had
simply returns nothing for that session and earns nothing.

    python benchmarks/agent_study/live_replay.py --task runs/task-11 --submission <dir> \
        --start 2023-01-03 --days 40
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.agent_study.oracle import PERIODS, adjusted_returns, load_positions

COST_BPS = 10.0


def _truncate(task: Path, dst: Path, cut: pd.Timestamp) -> Path:
    (dst / "data").mkdir(parents=True, exist_ok=True)
    for name in ("close_quoted.csv", "volume.csv", "llm_score.csv"):
        m = pd.read_csv(task / "data" / name, index_col=0)
        m.index = pd.to_datetime(m.index)
        m.loc[m.index <= cut].to_csv(dst / "data" / name)
    feed = pd.read_csv(task / "data" / "news_feed.csv", parse_dates=["feed_ts"])
    feed[feed["feed_ts"] <= cut].to_csv(dst / "data" / "news_feed.csv", index=False)
    fund = pd.read_csv(task / "data" / "fundamentals.csv", parse_dates=["filed"])
    fund = fund[fund["filed"] <= cut]
    fund["filed"] = fund["filed"].dt.strftime("%Y-%m-%d")
    fund.to_csv(dst / "data" / "fundamentals.csv", index=False)
    act = pd.read_csv(task / "data" / "corporate_actions.csv", parse_dates=["date"])
    act[act["date"] <= cut].to_csv(dst / "data" / "corporate_actions.csv", index=False)
    lst = pd.read_csv(task / "data" / "listings.csv", parse_dates=["listing_date", "delisting_date"])
    lst.loc[lst["delisting_date"] > cut, "delisting_date"] = pd.NaT
    lst[lst["listing_date"] <= cut].to_csv(dst / "data" / "listings.csv", index=False)
    shutil.copy(task / "manifest.json", dst / "manifest.json")
    return dst


def replay(task: Path, submission: Path, start: str, days: int) -> dict:
    ret = adjusted_returns(task)
    sessions = ret.index[ret.index >= pd.Timestamp(start)][:days]
    weights, failures = {}, 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, d in enumerate(sessions):
            prev = ret.index[ret.index < d][-1]
            box = _truncate(task, Path(tmp) / f"s{i}", prev)
            try:
                pos = load_positions(submission, box / "data")
                row = pos.loc[d] if d in pos.index else (
                    pos.iloc[-1] if len(pos) and pos.index[-1] == prev else None)
            except Exception:
                row, = (None,)
            if row is None:
                failures += 1
                weights[d] = pd.Series(0.0, index=ret.columns)
            else:
                weights[d] = row.reindex(ret.columns).fillna(0.0)
            shutil.rmtree(box, ignore_errors=True)
    w = pd.DataFrame(weights).T.sort_index()
    r = ret.loc[w.index, w.columns].fillna(0.0)
    gross = (w * r).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    net = gross - turn * COST_BPS / 1e4
    sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(PERIODS)) if net.std(ddof=1) else 0.0
    return {
        "submission": submission.name,
        "sessions_replayed": int(len(w)),
        "sessions_without_a_weight": failures,
        "live_mean_daily_bps": float(net.mean() * 1e4),
        "live_cumulative_pct": float(100.0 * ((1 + net).prod() - 1)),
        "live_sharpe": sharpe,
        "live_mean_gross_exposure": float(w.abs().sum(axis=1).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--start", default="2023-01-03")
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = replay(Path(a.task), Path(a.submission), a.start, a.days)
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
