"""Frozen holdout comparison on real FX and seeded synthetic worlds; losses are reported honestly."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from fin_skills.algorithms import Algorithm, default_registry, research

ROOT = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
PERIODS = ((2005, 2011), (2012, 2018), (2019, 2025))


def cases():
    manifest = json.loads((ROOT / "data/ecb_fx.provenance.json").read_text())
    fixture = ROOT / "data/ecb_fx.csv"
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != manifest["fixture_sha256"]:
        raise ValueError("ECB fixture hash changed; explicitly refresh/review provenance")
    fx = pd.read_csv(fixture, index_col="Date", parse_dates=True)
    for start, end in PERIODS:
        section = fx.loc[str(start):str(end)]
        for currency in section:
            yield f"ECB_{currency}_{start}_{end}", section[currency], "real_fx"
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=240)
        yield f"stationary_{seed}", pd.Series(noise), "synthetic"
        yield f"drift_{seed}", pd.Series(np.arange(240) * .1 + noise.cumsum()), "synthetic"
        yield f"variance_break_{seed}", pd.Series(np.r_[noise[:120], noise[120:] * 4]), "synthetic"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "ALGORITHM_RESULTS.json")
    args = parser.parse_args()
    rows = []
    for name, series, kind in cases():
        n = len(series)
        initial, holdout = n // 2, n // 5
        horizon = max(5, n // 30)
        kw = dict(initial_train=initial, holdout=holdout, horizon=horizon, gap=1,
                  metric="mae", candidates=("naive", "mean", "drift"))
        tracemalloc.start()
        before = time.perf_counter()
        report = research("forecast", {"series": series}, **kw)
        elapsed = time.perf_counter() - before
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        baseline = research("forecast", {"series": series}, **dict(kw, candidates=("naive",)))
        if report.status != "completed" or report.validation["holdout"] is None:
            raise RuntimeError(f"benchmark case {name} failed: {report.render()}")
        loss = report.validation["holdout"]["score"]
        baseline_loss = baseline.validation["holdout"]["score"]
        rows.append({"case": name, "kind": kind, "task": "forecast", "selected": report.selected,
            "holdout_mae": loss, "naive_holdout_mae": baseline_loss,
            "ratio_to_naive": loss / baseline_loss if baseline_loss else None,
            "improvement_over_naive": baseline_loss - loss,
            "seconds": elapsed, "python_traced_peak_bytes": peak,
            "development_ranking": report.validation["ranking"],
            "split": report.validation["holdout"], "input_hash": report.metadata["input_hash"],
            "environment": report.metadata["environment"]})
    # Additional task panels have separate units, baselines and metrics. Reference-rate
    # changes below are statistical test inputs, not a claim of achievable trading P&L.
    fx = pd.read_csv(ROOT / "data/ecb_fx.csv", index_col="Date", parse_dates=True)
    task_rows = []
    for start, end in PERIODS:
        changes = fx.loc[str(start):str(end)].pct_change(fill_method=None).dropna()
        target = changes.USD
        features = pd.concat({f"lag{i}": target.shift(i) for i in (1, 2, 5)}, axis=1).dropna()
        reg = default_registry()
        reg.register(Algorithm("benchmark_mean", "Training mean", "regression", "benchmark",
            ("X", "y", "X_predict"), ("predict",), source="benchmark definition"),
            lambda d, p: np.full(len(d["X_predict"]), np.mean(d["y"])))
        reg.register(Algorithm("benchmark_majority", "Training majority", "classification", "benchmark",
            ("X", "y", "X_predict"), ("predict",), source="benchmark definition"),
            lambda d, p: np.full(len(d["X_predict"]), int(np.mean(d["y"]) > .5)))
        panels = [
            ("portfolio", {"asset_returns": changes}, ("equal_weight", "inverse_volatility", "min_variance"),
             "variance", "equal_weight", {}, 5),
            ("risk", {"returns": target}, ("historical_var_es", "normal_var_es"),
             "quantile_loss", "historical_var_es", {}, 0),
            ("volatility", {"returns": target}, ("historical_volatility", "ewma_volatility"),
             "qlike", "historical_volatility", {}, 0),
            ("signal", {"returns": target}, ("momentum",),
             "negative_mean_return", "momentum", {}, 5),
            ("regression", {"X": features, "y": target.loc[features.index]},
             ("benchmark_mean", "ridge"), "mae", "benchmark_mean", {}, 0),
            ("classification", {"X": features, "y": (target.loc[features.index] > 0).astype(int)},
             ("benchmark_majority", "logistic"), "error_rate", "benchmark_majority", {}, 0),
        ]
        for task, data, ids, metric, baseline_id, params, cost in panels:
            n = len(next(iter(data.values())))
            kw = dict(initial_train=n // 2, holdout=n // 5, horizon=max(5, n // 30), gap=1,
                      candidates=ids, metric=metric, registry=reg, parameters=params, cost_bps=cost)
            tracemalloc.start()
            before = time.perf_counter()
            report = research(task, data, **kw)
            elapsed = time.perf_counter() - before
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            baseline = research(task, data, **dict(kw, candidates=(baseline_id,)))
            if report.validation["holdout"] is None or baseline.validation["holdout"] is None:
                raise RuntimeError(f"{task} panel failed: {report.to_json()}")
            loss, base_loss = report.validation["holdout"]["score"], baseline.validation["holdout"]["score"]
            task_rows.append({"case": f"ECB_{task}_{start}_{end}", "task": task, "metric": metric,
                "selected": report.selected, "holdout_loss": loss, "baseline": baseline_id,
                "baseline_holdout_loss": base_loss, "improvement_over_baseline": base_loss - loss,
                "seconds": elapsed, "python_traced_peak_bytes": peak,
                "split": report.validation["holdout"], "development_ranking": report.validation["ranking"],
                "input_hash": report.metadata["input_hash"], "environment": report.metadata["environment"],
                "audit_status": report.audit["status"], "cost_bps": cost})
    output = {"schema_version": 1, "cases": rows, "task_panels": task_rows,
        "summary": {"cases": len(rows), "wins": sum(r["holdout_mae"] < r["naive_holdout_mae"] for r in rows),
                    "ties": sum(r["holdout_mae"] == r["naive_holdout_mae"] for r in rows),
                    "losses": sum(r["holdout_mae"] > r["naive_holdout_mae"] for r in rows)},
        "task_panel_summary": {"cases": len(task_rows),
            "wins": sum(r["improvement_over_baseline"] > 0 for r in task_rows),
            "ties": sum(r["improvement_over_baseline"] == 0 for r in task_rows),
            "losses": sum(r["improvement_over_baseline"] < 0 for r in task_rows)},
        "limitations": ["FX series share a EUR base and are not independent markets.",
            "Historical reference-rate prediction, not investable returns or point-in-time vintages.",
            "Task panels compare bounded configurations; not evidence for every adapter or future performance.",
            "Signal panel has one candidate: it checks execution, not selection improvement.",
            "tracemalloc reports Python allocations, not total native-process memory.",
            "No performance threshold is tuned to these holdouts; keep losses visible."]}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"forecast": output["summary"], "task_panels": output["task_panel_summary"]}))


if __name__ == "__main__":
    main()
