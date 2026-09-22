#!/usr/bin/env python3
"""Grade one submission against ground truth the submitter never sees.

Four measurements, none of which calls a guard from this repository - otherwise an arm that
runs the guards would be graded by its own instrument:

1. `leakage_rate` - the counterfactual future test. The submission is re-executed on copies of
   the market that are identical up to a cut date and redrawn after it. Positions on or before
   the cut that move are positions that depended on the future. Reported as the share of
   pre-cut position cells that change, averaged over the redraws.
2. `sharpe_gap` - the submitter's reported Sharpe minus the Sharpe their own positions earn
   under a fixed honest accounting: split-adjusted returns, positions held into session t,
   10 bps per side of turnover.
3. `survivorship` - the share of position mass placed on names that had already delisted, and
   whether names that delist are absent from the traded set entirely (a survivor-only universe
   reads as coverage far below the listed universe).
4. `cost_bps` - what they said they charged.

    python benchmarks/agent_study/oracle.py --task runs/task-11 --submission runs/run-a1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import time
import types
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.agent_study.perturb import perturb_workspace  # noqa: E402

PERIODS = 252
HONEST_COST_BPS = 10.0
CUT = "2021-12-31"          # inside the evaluation window: pre-cut positions must be stable
PERTURB_SEEDS = (101, 202, 303)


def load_positions(submission: Path, data_dir: Path, timeout_note: str = "") -> pd.DataFrame:
    """Import submission.py in a fresh module and call build_positions(data_dir)."""
    if os.environ.get("FIN_STUDY_REQUIRE_SANDBOX") == "1":
        from benchmarks.agent_study.open_agent_runner import AgentWorkspaceSession
        with tempfile.TemporaryDirectory(prefix="oracle-snapshot-") as td:
            snapshot = Path(td)
            shutil.copytree(data_dir, snapshot / "data")
            shutil.copy2(submission / "submission.py", snapshot / "submission.py")
            session = AgentWorkspaceSession(snapshot, "no_library")
            result = session._worker("positions")
            if result.get("status") != "EXECUTED":
                raise RuntimeError(f"submission worker failed: {result}")
            split = result["positions"]
            return pd.DataFrame(split["data"], index=pd.to_datetime(split["index"]),
                                columns=split["columns"])
    from benchmarks.agent_study.submission_audit import _normalize_positions_frame
    source = submission / "submission.py"
    mod = types.ModuleType("submission")
    mod.__file__ = str(source)
    exec(compile(source.read_bytes(), str(source), "exec"), mod.__dict__)
    pos = mod.build_positions(str(data_dir))
    close = pd.read_csv(data_dir / "close_quoted.csv", index_col=0, parse_dates=True)
    return _normalize_positions_frame(pos, close)


def adjusted_returns(task: Path) -> pd.DataFrame:
    """Split-adjusted close-to-close returns - the honest accounting's price series."""
    close = pd.read_csv(task / "data" / "close_quoted.csv", index_col=0)
    close.index = pd.to_datetime(close.index)
    actions = pd.read_csv(task / "data" / "corporate_actions.csv", parse_dates=["date"])
    adj = close.copy()
    for _, row in actions.iterrows():
        before = adj.index < row["date"]
        adj.loc[before, row["ticker"]] = adj.loc[before, row["ticker"]] / float(row["ratio"])
    return adj.pct_change()


def honest_sharpe(pos: pd.DataFrame, ret: pd.DataFrame, window: tuple[str, str],
                  cost_bps: float = HONEST_COST_BPS) -> tuple[float, float]:
    """Sharpe of the submitted positions under one fixed accounting, and their turnover."""
    idx = ret.index
    w = pos.reindex(index=idx, columns=ret.columns).fillna(0.0)
    gross = (w * ret.fillna(0.0)).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost_bps / 1e4
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    seg = net.loc[(net.index >= lo) & (net.index <= hi)]
    if seg.std(ddof=1) == 0 or seg.empty:
        return 0.0, float(turnover.mean())
    return float(seg.mean() / seg.std(ddof=1) * np.sqrt(PERIODS)), float(turnover.mean())


def leakage(submission: Path, task: Path, seeds=PERTURB_SEEDS, cut: str = CUT) -> dict:
    """Share of pre-cut position cells that move when the post-cut future is redrawn."""
    base = load_positions(submission, task / "data")
    pre = base.index <= pd.Timestamp(cut)
    if pre.sum() == 0:
        return {"leakage_rate": float("nan"), "leakage_max_abs": float("nan"), "runs": 0}
    rates, maxima = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for s in seeds:
            alt = perturb_workspace(task, Path(tmp) / f"p{s}", cut=cut, seed=s)
            other = load_positions(submission, alt / "data")
            a = base.loc[pre]
            b = other.reindex(index=a.index, columns=a.columns).fillna(0.0)
            diff = (a - b).abs()
            scale = max(float(a.abs().to_numpy().max()), 1e-9)
            moved = (diff > 1e-6 * scale).to_numpy()
            rates.append(float(moved.mean()))
            maxima.append(float(diff.to_numpy().max()))
    return {"leakage_rate": float(np.mean(rates)), "leakage_max_abs": float(np.max(maxima)),
            "runs": len(seeds)}


def survivorship(pos: pd.DataFrame, task: Path) -> dict:
    """Position mass on names that had already delisted, and coverage of the listed universe."""
    listings = pd.read_csv(task / "data" / "listings.csv", parse_dates=["listing_date", "delisting_date"])
    traded = set(pos.columns[(pos.abs().sum() > 0)])
    delisted = set(listings.loc[listings["delisting_date"].notna(), "ticker"])
    dead_mass = 0.0
    total_mass = float(pos.abs().to_numpy().sum()) or 1.0
    for _, row in listings.dropna(subset=["delisting_date"]).iterrows():
        t = row["ticker"]
        if t not in pos.columns:
            continue
        after = pos.index > row["delisting_date"]
        dead_mass += float(pos.loc[after, t].abs().sum())
    return {
        "post_delisting_mass": dead_mass / total_mass,
        "delisted_names_traded": len(traded & delisted),
        "delisted_names_total": len(delisted),
        "universe_coverage": len(traded) / max(len(listings), 1),
    }



def input_attribution(submission: Path, task: Path, seed: int = 7) -> dict:
    """Which inputs does this pipeline actually read? Perturb one file at a time and look."""
    base = load_positions(submission, task / "data")
    used = {}
    files = ["close_quoted.csv", "news_feed.csv", "llm_score.csv", "fundamentals.csv",
             "volume.csv", "listings.csv"]
    with tempfile.TemporaryDirectory() as tmp:
        for name in files:
            alt = Path(tmp) / name.replace(".", "_")
            alt.mkdir(parents=True, exist_ok=True)
            shutil.copytree(task / "data", alt / "data", dirs_exist_ok=True)
            shutil.copy(task / "manifest.json", alt / "manifest.json")
            scrambled = perturb_workspace(task, Path(tmp) / f"all_{name}", cut="1990-01-01",
                                          seed=seed)
            shutil.copy(scrambled / "data" / name, alt / "data" / name)
            try:
                other = load_positions(submission, alt / "data")
                b = other.reindex(index=base.index, columns=base.columns).fillna(0.0)
                used[name] = bool(((base - b).abs().to_numpy() > 1e-9).any())
            except Exception as exc:                       # a submission that cannot run on it
                used[name] = f"error: {type(exc).__name__}"
    return {"uses_" + k.replace(".csv", ""): v for k, v in used.items()}


def same_session_dependence(submission: Path, task: Path, n_dates: int = 8,
                            seed: int = 5) -> dict:
    """Does the weight held over session t depend on session t's own data?"""
    rng = np.random.default_rng(seed)
    base = load_positions(submission, task / "data")
    manifest = json.loads((task / "manifest.json").read_text(encoding="utf-8"))
    lo, hi = pd.Timestamp(manifest["eval_start"]), pd.Timestamp(manifest["eval_end"])
    window = base.index[(base.index >= lo) & (base.index <= hi)]
    if len(window) == 0:
        return {"same_session_rate": float("nan"), "same_session_dates": 0}
    dates = pd.DatetimeIndex(rng.choice(window, size=min(n_dates, len(window)), replace=False))
    moved = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, d in enumerate(sorted(dates)):
            alt = Path(tmp) / f"d{i}"
            shutil.copytree(task / "data", alt / "data", dirs_exist_ok=True)
            shutil.copy(task / "manifest.json", alt / "manifest.json")

            close = pd.read_csv(alt / "data" / "close_quoted.csv", index_col=0)
            close.index = pd.to_datetime(close.index)
            close.index.name = "date"
            row = close.loc[d].to_numpy(copy=True)
            close.loc[d] = np.round(row * rng.lognormal(0.0, 0.05, row.shape), 2)
            close.to_csv(alt / "data" / "close_quoted.csv", index_label="date", float_format="%.2f")

            llm = pd.read_csv(alt / "data" / "llm_score.csv", index_col=0)
            llm.index = pd.to_datetime(llm.index)
            llm.index.name = "date"
            llm.loc[d] = rng.permutation(llm.loc[d].to_numpy(copy=True))
            llm.to_csv(alt / "data" / "llm_score.csv", index_label="date", float_format="%.6f")

            feed = pd.read_csv(alt / "data" / "news_feed.csv", parse_dates=["feed_ts"])
            m = feed["feed_ts"] == d
            if m.any():
                feed.loc[m, "score"] = rng.permutation(feed.loc[m, "score"].to_numpy(copy=True))
            feed.to_csv(alt / "data" / "news_feed.csv", index=False, float_format="%.6f")

            try:
                other = load_positions(submission, alt / "data")
            except Exception:
                continue
            if d not in other.index:
                continue
            a = base.loc[d]
            b = other.loc[d].reindex(a.index).fillna(0.0)
            scale = max(float(a.abs().max()), 1e-9)
            moved.append(float(((a - b).abs() > 1e-6 * scale).mean()))
    return {"same_session_rate": float(np.mean(moved)) if moved else float("nan"),
            "same_session_dates": len(moved)}


def grade(task: Path, submission: Path) -> dict:
    manifest = json.loads((task / "manifest.json").read_text(encoding="utf-8"))
    report_path = submission / "report.json"
    report_missing = not report_path.exists()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8")) if not report_missing else {}
    except Exception:
        report = {}
        report_missing = True
    window = (manifest["eval_start"], manifest["eval_end"])
    ret = adjusted_returns(task)
    pos = load_positions(submission, task / "data")
    hs, turnover = honest_sharpe(pos, ret, window)
    rep_sr_raw = report.get("reported_sharpe")
    has_rep_sr = rep_sr_raw is not None and np.isfinite(float(rep_sr_raw))
    rep_sr = float(rep_sr_raw) if has_rep_sr else None
    cost_raw = report.get("cost_bps_per_side")
    cost_bps = float(cost_raw) if cost_raw is not None and np.isfinite(float(cost_raw)) else None
    leak = leakage(submission, task)
    surv = survivorship(pos, task)
    attr = input_attribution(submission, task)
    sess = same_session_dependence(submission, task)
    out = {
        "task": task.name,
        "submission": submission.name,
        "report_missing": report_missing,
        "reported_sharpe": rep_sr,
        "honest_sharpe": float(hs),
        "mean_daily_turnover": float(turnover),
        "stated_cost_bps_per_side": cost_bps,
        **leak,
        **surv,
        **attr,
        **sess,
    }
    out["leakage_rate"] = float(out["leakage_rate"]) if np.isfinite(out["leakage_rate"]) else 0.0
    out["same_session_rate"] = float(out["same_session_rate"]) if np.isfinite(out["same_session_rate"]) else 0.0
    out["post_delisting_mass"] = float(out["post_delisting_mass"]) if np.isfinite(out["post_delisting_mass"]) else 0.0
    out["sharpe_gap"] = float(rep_sr - hs) if rep_sr is not None else 999.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = grade(Path(a.task), Path(a.submission))
    text = json.dumps(res, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
