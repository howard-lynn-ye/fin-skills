"""Public intervention checks, separate from the private oracle's grading code.

Run only in a disposable worker. Python submission execution is NOT an OS sandbox.
These probes cover close-price causality, universe selection, sampled same-session
dependencies and accounting. They do not certify absence of every possible leak.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import types
import os
import sys

import numpy as np
import pandas as pd

from fin_skills.api import get

SUPPORTED_GUARDS = ("assert_causal", "survivorship_audit")


def positions(task: Path) -> pd.DataFrame:
    source = task / "submission.py"
    module = types.ModuleType("candidate")
    module.__file__ = str(source)
    # Read current bytes: same-size edits within one second can reuse stale .pyc.
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    result = module.build_positions(str(task / "data"))
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise ValueError("submission must return a non-empty DataFrame")
    result = result.copy()
    result.index = pd.to_datetime(result.index)
    if result.index.has_duplicates or result.columns.has_duplicates:
        raise ValueError("duplicate dates or tickers")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("positions must be finite; do not silently fill invalid output")
    close = pd.read_csv(task / "data/close_quoted.csv", index_col=0, parse_dates=True)
    if not result.index.isin(close.index).all() or not result.columns.isin(close.columns).all():
        raise ValueError("positions contain unknown dates or tickers")
    if (result.abs().sum(axis=1) > 1 + 1e-8).any():
        raise ValueError("gross exposure exceeds task limit 1")
    return result.sort_index().astype(float)


def run_guard(task: Path, name: str) -> dict:
    guard = get(name)  # unknown names fail before any execution receipt exists
    if name not in SUPPORTED_GUARDS:
        raise ValueError(f"no artifact adapter for {name}; supported: {SUPPORTED_GUARDS}")
    close = pd.read_csv(task / "data/close_quoted.csv", index_col=0, parse_dates=True)
    if name == "assert_causal":
        with tempfile.TemporaryDirectory(prefix="causal-probe-") as td:
            copied = Path(td) / "task"
            shutil.copytree(task, copied)

            def signal(frame):
                frame.to_csv(copied / "data/close_quoted.csv")
                return positions(copied).reindex(index=close.index, columns=close.columns).fillna(0)

            result = guard.run(fn=signal, df=close, k=len(close) // 2)
    else:
        weights = positions(task)
        used = weights.columns[weights.abs().sum() > 0]
        if len(used) == 0:
            return {"status": "UNASSESSABLE", "guard": name, "verified_by_runtime": True,
                    "passed": False, "reason": "no active securities in submitted positions",
                    "scope": "empty traded universe; no survivorship conclusion is available"}
        listings = pd.read_csv(task / "data/listings.csv")
        result = guard.run(prices=close.loc[:, used], listings=listings)
    out = asdict(result)
    # Evidence may contain frames; the transport serializes them as diagnostic text.
    return {"status": "EXECUTED", "guard": name, "verified_by_runtime": True,
            "passed": result.passed, "result": out,
            "scope": "final submission and supplied market snapshot"}


def accounting(task: Path) -> dict:
    """Recompute reported performance without importing the independent oracle."""
    report = json.loads((task / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((task / "manifest.json").read_text(encoding="utf-8"))
    close = pd.read_csv(task / "data/close_quoted.csv", index_col=0, parse_dates=True)
    actions = pd.read_csv(task / "data/corporate_actions.csv", parse_dates=["date"])
    for row in actions.itertuples():
        close.loc[close.index < row.date, row.ticker] /= float(row.ratio)
    w = positions(task).reindex(index=close.index, columns=close.columns).fillna(0)
    returns = close.pct_change(fill_method=None).fillna(0)
    cost = float(report["cost_bps_per_side"])
    claimed = float(report["reported_sharpe"])
    if not np.isfinite([cost, claimed]).all() or cost < 0:
        raise ValueError("report requires finite Sharpe and nonnegative costs")
    net = (w * returns).sum(axis=1) - w.diff().abs().sum(axis=1) * cost / 10000
    net = net.loc[manifest["eval_start"]:manifest["eval_end"]]
    if len(net) < 3 or net.std(ddof=1) == 0:
        raise ValueError("insufficient nonconstant evaluation returns")
    sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(252))
    return {"reported_sharpe": claimed, "recomputed_at_stated_cost": sharpe,
            "absolute_gap": abs(claimed - sharpe), "passed": abs(claimed - sharpe) <= 0.05,
            "cost_bps_per_side": cost,
            "scope": "report consistency at stated costs; not proof of tradability"}


def generic_checks(task: Path) -> dict:
    """Ordinary output assertions; deliberately no information-timing interventions."""
    frame = positions(task)
    repeated = positions(task)
    close = pd.read_csv(task / "data/close_quoted.csv", index_col=0, parse_dates=True)
    checks = dict(deterministic=frame.equals(repeated),
                  full_dates=frame.index.equals(close.index),
                  full_tickers=set(frame.columns) == set(close.columns),
                  nonzero=bool((frame.abs().sum(axis=1) > 0).any()))
    return dict(status="EXECUTED", passed=all(checks.values()), assertions=checks,
                scope="schema, finite values, exposure, determinism and nonempty activity; no causal claim")


def same_session_probe(task: Path) -> dict:
    """Public probe uses multiplicative shocks, separate from oracle permutations."""
    baseline = positions(task)
    repeat = positions(task)
    if not baseline.equals(repeat):
        raise ValueError("submission is not deterministic on identical inputs")
    manifest = json.loads((task / "manifest.json").read_text(encoding="utf-8"))
    window = baseline.loc[manifest["eval_start"]:manifest["eval_end"]]
    if len(window) < 3:
        raise ValueError("need three evaluation sessions for the public timing probe")
    dates = window.index[np.linspace(0, len(window) - 1, 3, dtype=int)]
    rates = []
    with tempfile.TemporaryDirectory(prefix="session-probe-") as td:
        for i, day in enumerate(dates):
            copied = Path(td) / str(i)
            shutil.copytree(task, copied)
            for file in ("close_quoted.csv", "llm_score.csv", "volume.csv"):
                path = copied / "data" / file
                frame = pd.read_csv(path, index_col=0, parse_dates=True)
                frame.loc[day] *= np.linspace(0.7, 1.3, len(frame.columns))
                frame.to_csv(path)
            path = copied / "data/news_feed.csv"
            feed = pd.read_csv(path, parse_dates=["feed_ts"])
            feed.loc[feed.feed_ts == day, "score"] *= -2
            feed.to_csv(path, index=False)
            other = positions(copied)
            if day not in other.index:
                raise ValueError("submission omitted probed session")
            diff = (baseline.loc[day] - other.loc[day].reindex(baseline.columns).fillna(0)).abs()
            rates.append(float((diff > 1e-8).mean()))
    return {"passed": max(rates) == 0, "sampled_sessions": len(rates),
            "dependence_rate": float(np.mean(rates)),
            "scope": "three public dates; excludes fundamentals and listing metadata"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        confinement = None
        if os.environ.get("FIN_STUDY_REQUIRE_SANDBOX") == "1":
            from benchmarks.agent_study.linux_sandbox import confine
            from fin_skills.api import registry
            registry()
            root = Path(__file__).resolve().parents[2]
            readonly = [Path(sys.prefix), Path(sys.base_prefix), root / "fin_skills",
                        Path("/usr"), Path("/lib"), Path("/lib64"), Path("/etc/ld.so.cache")]
            shared = os.environ.get("FIN_STUDY_SHARED_RUNTIME")
            if shared:
                readonly.append(Path(shared))
            confinement = confine(args.task.parent, readonly)
        if args.action == "execution":
            frame = positions(args.task)
            value = {"status": "EXECUTED", "passed": True,
                     "rows": len(frame), "columns": len(frame.columns),
                     "scope": "execution and output schema only; no correctness claim"}
        elif args.action == "positions":
            frame = positions(args.task)
            value = {"status": "EXECUTED", "positions": {"index": frame.index.astype(str).tolist(),
                     "columns": frame.columns.tolist(), "data": frame.to_numpy().tolist()}}
        elif args.action == "accounting":
            value = accounting(args.task)
        elif args.action == "generic":
            value = generic_checks(args.task)
        elif args.action == "same_session_probe":
            value = same_session_probe(args.task)
        else:
            value = run_guard(args.task, args.action)
        if confinement:
            value["confinement"] = confinement
    except Exception as exc:
        value = {"status": "ERROR", "passed": False,
                 "error": f"{type(exc).__name__}: {exc}"}
    args.output.write_text(json.dumps(value, default=str, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
