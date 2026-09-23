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


def _normalize_positions_frame(result: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(result).copy()
    date_cols = [c for c in result.columns if str(c).lower() in ("date", "unnamed: 0", "timestamp", "trade_date", "feed_ts", "index")]
    if date_cols and (isinstance(result.index, pd.RangeIndex) or not str(result.index.dtype).startswith("datetime")):
        result = result.set_index(date_cols[0])
    elif isinstance(result.index, pd.RangeIndex):
        first_col = result.columns[0]
        if first_col not in close.columns:
            result = result.set_index(first_col)
        elif len(result) == len(close):
            result.index = close.index
    result.index = pd.to_datetime(result.index, errors="coerce")
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)
    result.index = result.index.normalize()
    result = result.loc[result.index.notna()]
    if result.index.has_duplicates:
        result = result.groupby(level=0).last()
    if result.columns.has_duplicates:
        result = result.loc[:, ~result.columns.duplicated()]
    valid_cols = [c for c in result.columns if c in close.columns]
    valid_idx = result.index.intersection(close.index)
    if not valid_cols or len(valid_idx) == 0:
        raise ValueError("positions contain no valid dates or tickers matching close_quoted.csv")
    result = result.loc[valid_idx, valid_cols]
    result = result.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = result.abs().sum(axis=1)
    over = gross > 1.0
    if over.any():
        result.loc[over] = result.loc[over].div(gross.loc[over], axis=0)
    result = result.reindex(index=close.index, columns=close.columns).fillna(0.0)
    result.index.name = "date"
    return result.sort_index().astype(float)


def positions(task: Path) -> pd.DataFrame:
    source = task / "submission.py"
    module = types.ModuleType("candidate")
    module.__file__ = str(source)
    # Read current bytes: same-size edits within one second can reuse stale .pyc.
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    result = module.build_positions(str(task / "data"))
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise ValueError("submission must return a non-empty DataFrame")
    close = pd.read_csv(task / "data/close_quoted.csv", index_col=0, parse_dates=True)
    result = _normalize_positions_frame(result, close)
    if result.empty:
        raise ValueError("positions contain no valid dates or tickers matching close_quoted.csv")
    return result


def run_guard(task: Path, name: str) -> dict:
    guard = get(name)  # unknown names fail before any execution receipt exists
    if name not in SUPPORTED_GUARDS:
        raise ValueError(f"no artifact adapter for {name}; supported: {SUPPORTED_GUARDS}")
    close = pd.read_csv(task / "data/close_quoted.csv", index_col=0, parse_dates=True)
    close.index.name = "date"
    if name == "assert_causal":
        with tempfile.TemporaryDirectory(prefix="causal-probe-") as td:
            copied = Path(td) / "task"
            shutil.copytree(task, copied)

            def signal(frame):
                frame.index.name = "date"
                frame.to_csv(copied / "data/close_quoted.csv", index_label="date")
                return positions(copied).reindex(index=close.index, columns=close.columns).fillna(0)

            result = guard.run(fn=signal, df=close, k=len(close) // 2)
    else:
        weights = positions(task)
        used = weights.columns[weights.abs().sum() > 0]
        if len(used) == 0:
            return {"status": "UNASSESSABLE", "guard": name, "verified_by_runtime": True,
                    "passed": False, "reason": "no active securities in submitted positions",
                    "scope": "empty traded universe; no survivorship conclusion is available"}
        listings = pd.read_csv(task / "data/listings.csv", parse_dates=["listing_date", "delisting_date"])
        result = guard.run(prices=close.loc[:, used], listings=listings)
        dead_mass = 0.0
        total_mass = float(weights.abs().to_numpy().sum()) or 1.0
        for _, row in listings.dropna(subset=["delisting_date"]).iterrows():
            t = row["ticker"]
            if t in weights.columns:
                dead_mass += float(weights.loc[weights.index > row["delisting_date"], t].abs().sum())
        post_delisting_mass = dead_mass / total_mass
        out = asdict(result)
        out["post_delisting_mass"] = post_delisting_mass
        passed = bool(result.passed and post_delisting_mass <= 1e-12)
        if post_delisting_mass > 1e-12:
            out["remediation"] = "Zero out weights for delisted tickers (date > delisting_date) AFTER cross-sectional demeaning."
        return {"status": "EXECUTED", "guard": name, "verified_by_runtime": True,
                "passed": passed, "result": out,
                "scope": "final submission and supplied market snapshot"}
    out = asdict(result)
    # Evidence may contain frames; the transport serializes them as diagnostic text.
    return {"status": "EXECUTED", "guard": name, "verified_by_runtime": True,
            "passed": result.passed, "result": out,
            "scope": "final submission and supplied market snapshot"}


def accounting(task: Path) -> dict:
    """Recompute reported performance without importing the independent oracle."""
    report_path = task / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    manifest = json.loads((task / "manifest.json").read_text(encoding="utf-8"))
    close = pd.read_csv(task / "data/close_quoted.csv", index_col=0, parse_dates=True)
    actions = pd.read_csv(task / "data/corporate_actions.csv", parse_dates=["date"])
    for row in actions.itertuples():
        close.loc[close.index < row.date, row.ticker] /= float(row.ratio)
    w = positions(task).reindex(index=close.index, columns=close.columns).fillna(0)
    returns = close.pct_change(fill_method=None).fillna(0)
    cost = float(report.get("cost_bps_per_side", 10.0))
    claimed_raw = report.get("reported_sharpe")
    net = (w * returns).sum(axis=1) - w.diff().abs().sum(axis=1) * cost / 10000
    net = net.loc[manifest["eval_start"]:manifest["eval_end"]]
    if len(net) < 3 or net.std(ddof=1) == 0:
        sharpe = 0.0
    else:
        sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(252))
    claimed = float(claimed_raw) if claimed_raw is not None and np.isfinite(float(claimed_raw)) else sharpe
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
                # Fractional shocks also apply to integer-valued volume columns.
                frame = pd.read_csv(path, index_col=0, parse_dates=True).astype(float)
                frame.index.name = "date"
                frame.loc[day] *= np.linspace(0.7, 1.3, len(frame.columns))
                frame.to_csv(path, index_label="date")
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
