"""Freeze and run the verified-memory development matrix without network access."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.library_utility.evaluate import digest, ledger, paired_blocks
from benchmarks.verified_memory.model import (EXPERTS, MODELS, MODES, SCENARIOS,
    features, prepare, simulate, synthetic)
from fin_skills.api import get
from fin_skills.core.trial_ledger import TrialLedger


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ecb():
    path = ROOT / "benchmarks/data/ecb_fx.csv"
    provenance = json.loads(path.with_suffix(".provenance.json").read_text())
    if sha(path) != provenance["fixture_sha256"]:
        raise ValueError("ECB fixture differs from frozen provenance")
    raw = pd.read_csv(path, index_col="Date", parse_dates=True)
    if list(raw.columns) != ["USD", "JPY", "GBP", "CHF"]:
        raise ValueError("unexpected ECB columns")
    prices = 1 / raw
    prices.index = prices.index.tz_localize("UTC") + pd.Timedelta(hours=20)
    prices.columns = [f"asset_{i}" for i in range(4)]
    return prices, provenance


def manifest():
    files = list((ROOT / "fin_skills").rglob("*.py"))
    files += list((ROOT / "benchmarks/verified_memory").glob("*"))
    files += [ROOT / "benchmarks/library_utility/evaluate.py",
              ROOT / "tests/test_verified_memory.py",
              ROOT / "benchmarks/data/ecb_fx.csv",
              ROOT / "benchmarks/data/ecb_fx.provenance.json"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files) if p.is_file()}


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
    if len(set(seeds)) != len(seeds) or len(set(market_seeds)) != len(market_seeds):
        raise ValueError("duplicate seeds")
    args.output.mkdir(parents=True, exist_ok=False)
    datasets = {f"synthetic_{s}": (synthetic(s, args.synthetic_length),
                  {"kind": "synthetic", "market_seed": s}) for s in market_seeds}
    if not args.synthetic_only:
        datasets["ecb_proxy"] = read_ecb()
    cells = [{"dataset": d, "scenario": s, "model": m, "mode": a, "seed": seed}
             for d, s, m, a, seed in itertools.product(datasets, SCENARIOS, MODELS, MODES, seeds)]
    order = np.random.default_rng(20260921).permutation(len(cells))
    cells = [cells[int(i)] for i in order]
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "exploratory development; no unseen holdout or profitability claim",
        "source_hashes": manifest(), "models": MODELS, "modes": MODES,
        "scenarios": SCENARIOS, "initialization_seeds": seeds,
        "market_seeds": market_seeds, "cells": cells,
        "parameters": {"width": 256, "active": 16, "learning_rate": 0.1,
            "softmax_temperature": 0.005, "horizon": 5, "training_cost_bps": 5,
            "evaluation_cost_bps": [0, 5, 20], "payoff_clip": [-0.2, 0.2]},
        "datasets": {name: {"provenance": p, "rows": len(px),
            "first": str(px.index[0]), "last": str(px.index[-1]),
            "frame_sha256": hashlib.sha256(px.to_csv().encode()).hexdigest()}
            for name, (px, p) in datasets.items()},
        "runtime": {"python": sys.version, "numpy": np.__version__,
            "pandas": pd.__version__, "platform": platform.platform(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"), "cwd": str(Path.cwd()),
            "output": str(args.output.resolve()), "tmpdir": os.getenv("TMPDIR"),
            "cache": os.getenv("XDG_CACHE_HOME")}}
    dump(args.output / "protocol.json", protocol)
    trial_ledger = TrialLedger(args.output / "trials.jsonl")
    for cell in cells:
        cell["trial_id"] = trial_ledger.record("verified_memory", cell.copy(),
                                               note="development matrix frozen before evaluation")
    (args.output / "cells").mkdir()
    (args.output / "datasets").mkdir()
    prepared, bounds, completed, failed, returns = {}, {}, [], [], {}
    beginning = time.monotonic()
    for name, (prices, _) in datasets.items():
        start = (int(np.searchsorted(prices.index.year, 2016)) if name == "ecb_proxy"
                 else int(len(prices) * 0.6))
        end = len(prices) - 1
        bounds[name] = (start, end)
        guard = get("assert_causal").run(fn=features, df=prices, k=len(prices) // 2)
        if not guard.passed:
            raise RuntimeError(guard.summary())
        directory = args.output / "datasets" / name
        directory.mkdir()
        prices.to_csv(directory / "prices.csv")
        for scenario in SCENARIOS:
            bundle = prepare(prices, scenario)
            prepared[name, scenario] = bundle
            packets = []
            for packet in bundle[3]:
                record = asdict(packet)
                record["values"] = packet.values.tolist()
                packets.append(record)
            dump(directory / f"receipts-{scenario}.json", packets)
        dump(directory / "data_checks.json", {"feature_causality": guard.summary(),
            "start_index": start, "end_index": end,
            "evaluation_first": str(prices.index[start]),
            "evaluation_last": str(prices.index[end])})
        _, schedule, expert, _ = prepared[name, "clean"]
        baselines = {}
        for i, label in enumerate(EXPERTS):
            params = {"dataset": name, "expert": label}
            tid = trial_ledger.record("fixed_expert", params)
            actions = [{"decision_index": t, "weights": expert[t][i].tolist()}
                       for t in schedule if start <= t < end]
            scores = {str(c): ledger(prices, actions, start=start, end=end, cost_bps=c)
                      for c in (0, 5, 20)}
            baselines[label] = {"actions": actions, "scores": scores}
            trial_ledger.complete(tid, scores["5"]["metrics"])
        dump(directory / "fixed_experts.json", baselines)
    for number, cell in enumerate(cells, 1):
        tid = cell["trial_id"]
        directory = args.output / "cells" / tid
        directory.mkdir()
        began = time.monotonic()
        try:
            name, scenario = cell["dataset"], cell["scenario"]
            start, end = bounds[name]
            actions, updates, audit = simulate(prepared[name, scenario], cell["model"],
                cell["mode"], cell["seed"], start=start, end=end)
            dump(directory / "actions.json", actions)  # freeze before independent scoring
            dump(directory / "updates.json", updates)
            scores = {str(c): ledger(datasets[name][0], actions, start=start, end=end,
                      cost_bps=c) for c in (0, 5, 20)}
            row = {**cell, "audit": audit, "metrics_5bps": scores["5"]["metrics"],
                   "actions_sha256": digest(actions),
                   "elapsed_seconds": time.monotonic() - began}
            dump(directory / "scores.json", {**row, "scores": scores})
            returns[name, scenario, cell["model"], cell["mode"], cell["seed"]] = scores["5"]["daily_returns"]
            trial_ledger.complete(tid, row["metrics_5bps"])
            completed.append(row)
        except Exception as exc:
            failure = {**cell, "error": repr(exc)}
            dump(directory / "failure.json", failure)
            trial_ledger.abandon(tid, repr(exc))
            failed.append(failure)
        if number % 20 == 0 or number == len(cells):
            print(json.dumps({"finished": number, "planned": len(cells),
                "failed": len(failed), "elapsed_seconds": time.monotonic() - beginning}), flush=True)
    contrasts = []
    for name, scenario in itertools.product(datasets, SCENARIOS):
        pairs = [(model, "verified_update", model, "unchecked") for model in MODELS]
        pairs += [("fly_sparse", mode, other, mode) for mode in MODES
                  for other in ("dense", "linear")]
        for ma, aa, mb, ab in pairs:
            try:
                # Average initialization repeats per date; do not treat them as markets.
                a = np.mean([returns[name, scenario, ma, aa, seed] for seed in seeds], axis=0)
                b = np.mean([returns[name, scenario, mb, ab, seed] for seed in seeds], axis=0)
            except KeyError:
                continue
            contrasts.append({"dataset": name, "scenario": scenario,
                "a": [ma, aa], "b": [mb, ab],
                "interpretation": "descriptive; overlapping contrasts, no multiplicity correction",
                "paired_dates": paired_blocks(a, b, block=21, draws=1000, seed=20260921)})
    dump(args.output / "contrasts.json", contrasts)
    summary = {"status": "complete" if not failed else "complete_with_failures",
        "protocol_sha256": sha(args.output / "protocol.json"),
        "planned_cells": len(cells), "completed_cells": len(completed),
        "failures": failed, "rows": completed,
        "elapsed_seconds": time.monotonic() - beginning,
        "trial_ledger": trial_ledger.summary(), "scope": protocol["scope"]}
    dump(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "planned_cells", "completed_cells",
                                             "elapsed_seconds")}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
