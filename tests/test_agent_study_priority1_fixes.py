"""End-to-end verification of Priority-1 fixes for `benchmarks/agent_study/` and `benchmarks/prediction_audit.py`."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from benchmarks.agent_study.build_task import export
from benchmarks.agent_study.oracle import grade, load_positions
from benchmarks.agent_study.submission_audit import accounting, positions, run_guard, same_session_probe
from benchmarks.agent_study.summarize_matrix import summarize
from benchmarks.prediction_audit import evaluate


def test_build_task_exports_date_column_and_task_md(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    export(11, ws)
    assert (ws / "TASK.md").exists()
    assert (ws / "manifest.json").exists()
    # Verify explicit `date` column header on exported CSVs
    for name in ("close_quoted.csv", "volume.csv", "llm_score.csv"):
        df = pd.read_csv(ws / "data" / name, parse_dates=["date"], index_col="date")
        assert not df.empty
        assert df.index.name == "date"


def test_positions_normalizes_date_column_and_range_index(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    export(11, ws)
    # Write a submission that returns `date` as a regular column with RangeIndex(0, N)
    # and only covers the evaluation window (2021-2022)
    (ws / "submission.py").write_text(
        """
import pandas as pd
import numpy as np

def build_positions(data_dir: str) -> pd.DataFrame:
    close = pd.read_csv(f"{data_dir}/close_quoted.csv")
    listings = pd.read_csv(f"{data_dir}/listings.csv")
    actions = pd.read_csv(f"{data_dir}/corporate_actions.csv")
    dates = pd.to_datetime(close["date"])
    prices = close.drop(columns=["date"]).copy()
    prices.index = dates
    for _, row in actions.iterrows():
        prices.loc[prices.index < pd.Timestamp(row["date"]), row["ticker"]] /= float(row["ratio"])
    # Strictly causal 5-day momentum lagged by 1 session
    ret5 = prices.pct_change(5, fill_method=None).shift(1)
    # Mask post-delisting names as NaN before demeaning so they do not receive -mean weight
    for _, row in listings.dropna(subset=["delisting_date"]).iterrows():
        t = row["ticker"]
        if t in ret5.columns:
            ret5.loc[ret5.index >= pd.Timestamp(row["delisting_date"]), t] = np.nan
    w = ret5.sub(ret5.mean(axis=1), axis=0).fillna(0.0)
    denom = w.abs().sum(axis=1).replace(0.0, 1.0)
    w = w.div(denom, axis=0) * 0.8
    # Return with 'date' as a regular column and RangeIndex(0, N)!
    out = w.reset_index().rename(columns={"index": "date"})
    return out
""",
        encoding="utf-8",
    )

    pos = positions(ws)
    assert isinstance(pos.index, pd.DatetimeIndex)
    assert len(pos) > 500
    assert (pos.abs().sum(axis=1) <= 1.0 + 1e-8).all()

    # Verify run_guard and same_session_probe work cleanly
    causal_res = run_guard(ws, "assert_causal")
    assert causal_res["passed"] is True
    surv_res = run_guard(ws, "survivorship_audit")
    assert surv_res["passed"] is True
    probe_res = same_session_probe(ws)
    assert probe_res["passed"] is True

    # Verify oracle.grade() succeeds EVEN WHEN report.json IS MISSING!
    g_missing = grade(ws, ws)
    assert g_missing["report_missing"] is True
    assert g_missing["leakage_rate"] == 0.0
    assert g_missing["same_session_rate"] == 0.0
    assert g_missing["post_delisting_mass"] == 0.0

    # Now write report.json matching accounting() and verify summarize_matrix grades it as correct=True
    acc = accounting(ws)
    (ws / "report.json").write_text(
        json.dumps({
            "reported_sharpe": acc["recomputed_at_stated_cost"],
            "cost_bps_per_side": 10.0,
            "method": "Causal lagged momentum with split adjustment and delisting mask",
        }),
        encoding="utf-8",
    )
    g_complete = grade(ws, ws)
    assert g_complete["report_missing"] is False
    assert abs(g_complete["sharpe_gap"]) <= 0.05

    # Verify summarize_matrix produces ungradable_accepted == 0
    exp_root = tmp_path / "exp"
    cell_dir = exp_root / "s11-r0-library_with_guards"
    cell_dir.mkdir(parents=True)
    (exp_root / "protocol.json").write_text(
        json.dumps({
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "cells": [{"market_seed": 11, "repetition": 0, "condition": "library_with_guards"}],
        }),
        encoding="utf-8",
    )
    (cell_dir / "result.json").write_text(
        json.dumps({"accepted": True, "termination": "SUBMITTED", "turns": 5, "wall_seconds": 12.3, "usage": []}),
        encoding="utf-8",
    )
    (cell_dir / "grade.json").write_text(json.dumps(g_complete), encoding="utf-8")
    summary = summarize([exp_root])
    grp = summary["groups"][0]
    assert grp["ungradable_accepted"] == 0
    assert grp["correct_among_accepted"] == 1.0


def test_real_timestamped_prediction_audit() -> None:
    pred_csv = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "real_timestamped_predictions.csv"
    res = evaluate(pred_csv, block_days=5, horizon_days=5, draws=200, seed=20260921)
    assert res["status"] == "RECOMPUTED_FROM_PREDICTIONS"
    assert res["rows"] == 35772
    diff = res["paired_comparisons"]["pit_kol_credibility_gated minus naive_follower_volume_weighted"]
    assert diff["mean_daily_ic_difference"] > 0.03
    assert diff["ci95"][0] > 0.0
