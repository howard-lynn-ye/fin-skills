"""Frozen stateful fruit-fly study, including v2 and fixed-strategy references."""
import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.library_utility.evaluate import digest, ledger, paired_blocks
from benchmarks.verified_memory.fly_v2 import PARAMETERS, run_market
from benchmarks.verified_memory.fly_v3 import ACTIONS, ARMS, prepare_market, simulate
from benchmarks.verified_memory.model import EXPERTS, features, prepare, synthetic
from benchmarks.verified_memory.run import dump, manifest, read_ecb, sha
from fin_skills.api import get
from fin_skills.core.trial_ledger import TrialLedger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="11,23,37,53,71")
    parser.add_argument("--market-seeds", default="101,202,303")
    parser.add_argument("--synthetic-length", type=int, default=1600)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    market_seeds = [int(x) for x in args.market_seeds.split(",")]
    if len(seeds) != len(set(seeds)) or len(market_seeds) != len(set(market_seeds)):
        raise ValueError("duplicate seeds")
    args.output.mkdir(parents=True, exist_ok=False)
    datasets = {f"synthetic_{s}": (synthetic(s, args.synthetic_length), {"kind": "synthetic", "seed": s})
                for s in market_seeds}
    if not args.synthetic_only:
        datasets["ecb_proxy"] = read_ecb()
    cells = [{"dataset": name, "arm": arm, "seed": seed}
             for name, arm, seed in itertools.product(datasets, [*ARMS, "fly_v2_reference"], seeds)]
    rng = np.random.default_rng(20260923)
    rng.shuffle(cells)
    hashes = manifest()
    for filename in ("tests/test_fly_v2.py", "tests/test_fly_v3.py"):
        hashes[filename] = sha(ROOT / filename)
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "study": "fly-stateful-v3",
        "arms": ARMS, "actions": ACTIONS, "circuit_parameters": PARAMETERS,
        "primary_cost_bps": 5, "replay_cost_bps": [0, 5, 20], "source_hashes": hashes,
        "initialization_seeds": seeds, "planned_cells": cells,
        "datasets": {name: {"provenance": prov, "rows": len(px), "first": str(px.index[0]),
                            "last": str(px.index[-1]), "frame_sha256": digest(px.to_numpy().tolist())}
                     for name, (px, prov) in datasets.items()},
        "runtime": {"python": sys.version, "numpy": np.__version__,
            "slurm_job_id": os.getenv("SLURM_JOB_ID"), "cwd": str(Path.cwd()),
            "output": str(args.output.resolve()), "tmpdir": os.getenv("TMPDIR"),
            "cache": os.getenv("XDG_CACHE_HOME")},
        "limitations": ["exploratory follow-up on already inspected development markets",
            "rate-based approximation, not complete brain emulation",
            "ECB indicative price-only proxy excludes carry, funding and real fills",
            "v2 comparison changes several components and is not a single-variable contrast",
            "cost sweeps replay frozen actions without retraining or recomputing state-dependent targets",
            "initialization seeds are not independent market samples"]}
    dump(args.output / "protocol.json", protocol)
    trials = TrialLedger(args.output / "trials.jsonl")
    for cell in cells:
        cell["trial_id"] = trials.record("fly_v3", cell.copy())
    for folder in ("datasets", "cells"):
        (args.output / folder).mkdir()
    prepared, old_prepared, bounds = {}, {}, {}
    baseline_rows = []
    began = time.monotonic()
    for name, (prices, _) in datasets.items():
        start = int(np.searchsorted(prices.index.year, 2016)) if name == "ecb_proxy" else int(len(prices) * 0.6)
        end = len(prices) - 1
        bounds[name] = start, end
        guard = get("assert_causal").run(fn=features, df=prices, k=len(prices) // 2)
        if not guard.passed:
            raise RuntimeError(guard.summary())
        prepared[name], old_prepared[name] = prepare_market(prices), prepare(prices, "clean")
        directory = args.output / "datasets" / name
        directory.mkdir()
        prices.to_csv(directory / "prices.csv")
        dump(directory / "checks.json", {"feature_guard": guard.summary(), "start": start, "end": end})
        experts = prepared[name][1]
        schedule = [t for t in experts if start <= t < end]
        for i, label in enumerate([*EXPERTS, "buy_hold_equal"]):
            tid = trials.record("fixed_expert", {"dataset": name, "expert": label})
            if label == "buy_hold_equal":
                decisions = [{"decision_index": schedule[0], "weights": [0.25] * 4}]
            else:
                decisions = [{"decision_index": t, "weights": experts[t][i].tolist()} for t in schedule]
            scores = {str(c): ledger(prices, decisions, start=start, end=end, cost_bps=c) for c in (0, 5, 20)}
            baseline_rows.append({"dataset": name, "arm": label,
                "metrics_by_cost": {c: r["metrics"] for c, r in scores.items()}})
            trials.complete(tid, scores["5"]["metrics"])
        print(json.dumps({"prepared": name}), flush=True)
    dump(args.output / "baselines.json", baseline_rows)
    rows, failures, returns = [], [], {}
    for number, cell in enumerate(cells, 1):
        tid, name, arm = cell["trial_id"], cell["dataset"], cell["arm"]
        directory = args.output / "cells" / tid
        directory.mkdir()
        try:
            start, end = bounds[name]
            if arm == "fly_v2_reference":
                actions, updates, audit = run_market(old_prepared[name], "fly_trace", "verified_update",
                                                    cell["seed"], start=start, end=end)
                result = {"actions": actions, "updates": updates, "audit": audit}
            else:
                result = simulate(datasets[name][0], arm, cell["seed"], start=start, end=end,
                                  prepared=prepared[name])
                actions, audit = result["actions"], result["audit"]
            dump(directory / "trajectory.json", result)
            dump(directory / "actions.json", actions)  # Freeze before independent scoring.
            scores = {str(c): ledger(datasets[name][0], actions, start=start, end=end, cost_bps=c)
                      for c in (0, 5, 20)}
            if arm != "fly_v2_reference":
                if len(result["nav"]) != len(scores["5"]["nav"]):
                    raise AssertionError("NAV length mismatch")
                nav_gap = float(np.max(np.abs(np.array(result["nav"]) - scores["5"]["nav"])))
                turnover_gap = float(np.max(np.abs(np.array(result["daily_turnover"]) - scores["5"]["daily_turnover"])))
                if nav_gap > 1e-10 or turnover_gap > 1e-10:
                    raise AssertionError(f"account reconciliation failed: NAV={nav_gap}, turnover={turnover_gap}")
                eval_rewards = [r["net_reward"] for r in result["receipts"] if r["origin"] >= start]
                reward_gap = abs(float(np.prod(1 + np.asarray(eval_rewards))) - result["nav"][-1])
                if reward_gap > 1e-10:
                    raise AssertionError(f"reward intervals do not reconcile: {reward_gap}")
                audit.update(max_nav_gap=nav_gap, max_turnover_gap=turnover_gap, reward_compounding_gap=reward_gap)
            row = {**cell, "audit": audit, "actions_sha256": digest(actions),
                "metrics_by_cost": {c: r["metrics"] for c, r in scores.items()}}
            dump(directory / "scores.json", {**row, "scores": scores})
            rows.append(row)
            returns[name, arm, cell["seed"]] = scores["5"]["daily_returns"]
            trials.complete(tid, scores["5"]["metrics"])
        except Exception as exc:
            failure = {**cell, "error": repr(exc)}
            dump(directory / "failure.json", failure)
            failures.append(failure)
            trials.abandon(tid, repr(exc))
        if number % 10 == 0 or number == len(cells):
            print(json.dumps({"finished": number, "planned": len(cells), "failures": len(failures),
                              "elapsed_seconds": time.monotonic() - began}), flush=True)
    contrasts = []
    for name, other in itertools.product(datasets, [a for a in ARMS if a != "fly_v3"] + ["fly_v2_reference"]):
        try:
            a = np.mean([returns[name, "fly_v3", s] for s in seeds], axis=0)
            b = np.mean([returns[name, other, s] for s in seeds], axis=0)
        except KeyError:
            continue
        contrasts.append({"dataset": name, "a": "fly_v3", "b": other,
                          "paired_dates": paired_blocks(a, b, block=21, draws=1000, seed=20260923)})
    dump(args.output / "contrasts.json", contrasts)
    summary = {"status": "complete" if not failures else "complete_with_failures",
        "protocol_sha256": sha(args.output / "protocol.json"), "planned_cells": len(cells),
        "rows": rows, "failures": failures, "trial_ledger": trials.summary(),
        "elapsed_seconds": time.monotonic() - began, "limitations": protocol["limitations"]}
    dump(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "completed": len(rows), "failures": len(failures),
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
