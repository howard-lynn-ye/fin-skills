#!/usr/bin/env python3
"""What did the claim actually earn in a year the agent never saw?

The task workspace stops at `data_end`; the sessions after it exist only in the oracle's copy
of the market. The submission is re-executed on that full market and its positions over the
held-out year are priced under the same fixed accounting used everywhere else. Two numbers
come out of it:

- `realisation_ratio` - realised Sharpe over the held-out year divided by the Sharpe the agent
  claimed. A process that generalises lands near 1; one that leaned on the future collapses.
- `trusted_pnl` - what a reader who believed the report would have made. Capital is allocated
  in proportion to the claimed Sharpe (capped at one unit of gross), held through the held-out
  year, and the return is reported in percent. It converts a reporting error into money.

    python benchmarks/agent_study/realize.py --task runs/task-11 --held runs/held-11 \
        --submission /path/to/run --out grade.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.agent_study.oracle import (PERIODS, adjusted_returns, honest_sharpe,
                                           load_positions)

CLAIM_CAP = 3.0          # a claim above this is treated as fully allocated, not more


def realise(task_full: Path, held: Path, submission: Path) -> dict:
    """Score one submission built on the truncated market against the full one."""
    manifest = json.loads((held / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((submission / "report.json").read_text(encoding="utf-8"))
    claimed = float(report.get("reported_sharpe", float("nan")))
    data_end = pd.Timestamp(manifest["data_end"])

    ret = adjusted_returns(task_full)
    future = (ret.index > data_end)
    if future.sum() < 60:
        raise SystemExit("the held-out window is too short to price")
    held_window = (str(ret.index[future][0].date()), str(ret.index[future][-1].date()))

    pos = load_positions(submission, task_full / "data")
    realised, turnover = honest_sharpe(pos, ret, held_window)
    claimed_window = (manifest["eval_start"], manifest["eval_end"])
    in_window, _ = honest_sharpe(pos, ret, claimed_window)

    w = pos.reindex(index=ret.index, columns=ret.columns).fillna(0.0)
    daily = (w * ret.fillna(0.0)).sum(axis=1) - w.diff().abs().sum(axis=1).fillna(0.0) * 10 / 1e4
    seg = daily.loc[future]
    allocation = float(np.clip(claimed / CLAIM_CAP, 0.0, 1.0)) if claimed == claimed else 0.0
    trusted = float((1.0 + allocation * seg).prod() - 1.0)

    return {
        "submission": submission.name,
        "task": task_full.name,
        "claimed_sharpe": claimed,
        "recomputed_in_claimed_window": in_window,
        "realised_sharpe_heldout": realised,
        "heldout_window": held_window,
        "realisation_ratio": realised / claimed if claimed not in (0.0, float("nan")) else float("nan"),
        "allocation_if_trusted": allocation,
        "trusted_return_pct": 100.0 * trusted,
        "heldout_mean_turnover": turnover,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="the FULL market the oracle keeps")
    ap.add_argument("--held", required=True, help="the truncated workspace the agent saw")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = realise(Path(a.task), Path(a.held), Path(a.submission))
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
